#!/usr/bin/env python3
"""Inference with custom trained checkpoint - for Kaggle submission."""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "4RC"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_model_with_checkpoint(base_checkpoint, trained_checkpoint, device):
    """Load base model and apply trained head weights."""
    from arc.models.arc import Arc
    
    log.info("Loading base model from %s", base_checkpoint)
    model = Arc.from_pretrained(base_checkpoint).to(device)
    
    if trained_checkpoint and Path(trained_checkpoint).exists():
        log.info("Loading trained weights from %s", trained_checkpoint)
        ckpt = torch.load(trained_checkpoint, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        
        # Load only the trained heads
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state:
                model_state[k] = v
                loaded += 1
        
        model.load_state_dict(model_state)
        log.info("Loaded %d tensors from checkpoint", loaded)
    
    model.eval()
    return model


def extract_frames_stride(mp4: Path, stride: int, num_frames: int, out_dir: Path):
    """Extract frames with stride sampling."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    indices = list(range(0, total, stride))[:num_frames]
    
    paths = []
    for t in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok:
            break
        p = out_dir / f"frame_{t:04d}.png"
        cv2.imwrite(str(p), frame)
        paths.append((t, p))
    
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return paths, native_h, native_w


def run_inference(model, frame_paths, query_uv, native_h, native_w, device, resolution):
    """Run 4RC inference."""
    from arc.dust3r.utils.image import load_images_for_eval
    
    imgs = load_images_for_eval(
        [str(p) for _, p in frame_paths],
        size=resolution, verbose=False, crop=False, patch_size=14, square_ok=True,
    )
    
    img0 = imgs[0]["img"]
    arc_h, arc_w = int(img0.shape[-2]), int(img0.shape[-1])
    
    uv = query_uv.clone().to(device)
    uv[:, 0] *= float(arc_w) / float(native_w)
    uv[:, 1] *= float(arc_h) / float(native_h)
    
    views = []
    for img in imgs:
        d = dict(img)
        if isinstance(d.get("img"), torch.Tensor):
            ten = d["img"]
            if ten.ndim == 3:
                ten = ten.unsqueeze(0)
            d["img"] = ten.to(device)
        d["track_query_idx"] = torch.tensor([0], device=device)
        d["query_points"] = uv.detach().cpu()
        views.append(d)
    
    with torch.no_grad():
        raw = model(views, force_no_output_conversion=True, inference_track=True)
    
    track = raw["track"]
    if track.ndim == 5:
        track = track[0]
    depth = raw["depth"]
    if depth.ndim == 5:
        depth = depth[0, ..., 0] if depth.shape[-1] == 1 else depth[0]
    elif depth.ndim == 4:
        depth = depth[0]
    
    d0 = depth[0]
    H, W = d0.shape[-2], d0.shape[-1]
    
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    fx = fy = float(max(H, W))
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    z = d0
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts0 = torch.stack([x, y, z], dim=-1)
    
    us = uv[:, 0].round().long().clamp(0, W - 1)
    vs = uv[:, 1].round().long().clamp(0, H - 1)
    base = pts0[vs, us]
    
    # Map frame indices to track indices
    frame_to_track = {fp[0]: i for i, fp in enumerate(frame_paths)}
    
    return track, base, frame_to_track, (H, W), (us, vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", required=True)
    ap.add_argument("--trained_checkpoint", default="")
    ap.add_argument("--test_manifest", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--num_frames", type=int, default=64)
    ap.add_argument("--shard_idx", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = load_model_with_checkpoint(
        args.base_checkpoint,
        args.trained_checkpoint if args.trained_checkpoint else None,
        device
    )
    
    # Load test manifest
    cfg = yaml.safe_load(Path(args.test_manifest).read_text())
    test_root = Path(cfg["root"])
    scored_frames = cfg.get("scored_frames", list(range(0, 192, 6)))
    sequences = cfg["sequences"]
    
    # Shard
    sequences = sequences[args.shard_idx::args.num_shards]
    log.info("Processing %d sequences (shard %d/%d)", len(sequences), args.shard_idx, args.num_shards)
    
    rows = []
    
    for seq_info in sequences:
        scene = seq_info["scene"]
        seq = seq_info["seq"]
        mp4 = test_root / scene / "mp4" / f"{seq}.mp4"
        query_file = test_root / scene / "query" / f"{seq}.npz"
        
        if not mp4.exists() or not query_file.exists():
            log.warning("Missing %s or %s", mp4, query_file)
            continue
        
        log.info("Processing %s/%s", scene, seq)
        
        query_data = np.load(query_file)
        query_uv = torch.from_numpy(query_data["query_uv"].astype(np.float32)).to(device)
        
        with tempfile.TemporaryDirectory(prefix="infer_") as tmp:
            frame_paths, nh, nw = extract_frames_stride(
                mp4, args.stride, args.num_frames, Path(tmp)
            )
            
            track, base, frame_to_track, (H, W), (us, vs) = run_inference(
                model, frame_paths, query_uv, nh, nw, device, args.resolution
            )
        
        # Build predictions for scored frames
        N = query_uv.shape[0]
        
        for t in scored_frames:
            # Find closest frame in our sampled set
            sampled_indices = [fp[0] for fp in frame_paths]
            closest_idx = min(range(len(sampled_indices)), 
                            key=lambda i: abs(sampled_indices[i] - t))
            
            if closest_idx in frame_to_track.values() or closest_idx < track.shape[0]:
                track_idx = closest_idx
            else:
                track_idx = 0
            
            resid = track[track_idx, vs, us]
            pred_xyz = (base + resid).cpu().numpy()
            
            for i in range(N):
                rows.append({
                    "seq_id": f"{scene}/{seq}",
                    "frame_id": t,
                    "point_id": i,
                    "x": float(pred_xyz[i, 0]),
                    "y": float(pred_xyz[i, 1]),
                    "z": float(pred_xyz[i, 2]),
                })
    
    # Write CSV
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_id", "frame_id", "point_id", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)
    
    log.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
