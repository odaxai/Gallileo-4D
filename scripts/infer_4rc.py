#!/usr/bin/env python3
"""4RC + TABA v5 inference for the Syn4D challenge.

Pipeline per sequence:
  1. 4RC feedforward → dense track map (H×W×3) + conf_track for each scored frame
  2. CoTracker3 offline → 2D tracks for query pixels across all scored frames
  3. TABA v5 refinement (two complementary fixes):
     a. Static-point freeze: near-zero 2D motion → freeze at frame-0 4RC prediction
     b. Confidence interpolation: low conf_track frames get linearly interpolated
        from their nearest reliable neighbours, removing trajectory spikes/holes.
     Fix (b) is applied even without CoTracker (confidence from 4RC alone is enough).

Usage:
    python scripts/infer_4rc.py \\
        --checkpoint /path/to/4RC_checkpoint \\
        --benchmark_root /path/to/challenge_eval \\
        --queries_csv data/queries.csv \\
        --out submissions/submission.csv \\
        [--cotracker_ckpt /path/to/cotracker3_offline.pth]
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────
VARIANTS = ("og", "sim", "mixed", "mixed_no_bedlam")
SCENES   = ("antiquity", "dream", "gothic", "office")
# Scored frames: source_frame_index values (0, 6, 12, ..., 186) = 32 frames
SCORED_FRAME_INDICES = list(range(0, 192, 6))   # 32 values
N_QUERIES     = 512
FRAME_H, FRAME_W = 720, 1280


def make_submission_id(variant: str, scene: str, seq_name: str, query_id: int, source_frame: int) -> str:
    """Build the Kaggle submission row ID.

    Format: {variant}-{scene}-{seq_name}-q{query_id:03d}-f{source_frame:03d}
    e.g.    og-antiquity-seq_000000_0-q000-f006
    """
    return f"{variant}-{scene}-{seq_name}-q{query_id:03d}-f{source_frame:03d}"

# ── helpers ──────────────────────────────────────────────────────────────────

def load_queries_for_seq(queries_csv: Path, seq_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load query pixels and intrinsics for a single sequence.

    Args:
        queries_csv: path to challenge queries.csv
        seq_key:     e.g. "og/antiquity/seq_000000_0"

    Returns:
        query_uv: (N, 2) float32 — (u, v) pixel coords
        K:        (3, 3) float32 — camera intrinsics (same for all queries in seq)
    """
    queries: dict[int, tuple[float, float]] = {}
    K = None
    with open(queries_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sequence"] != seq_key:
                continue
            q_idx = int(row["query_id"])
            queries[q_idx] = (float(row["u"]), float(row["v"]))
            if K is None:
                fx, fy = float(row["fx"]), float(row["fy"])
                cx, cy = float(row["cx"]), float(row["cy"])
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    if not queries:
        raise ValueError(f"No queries found for sequence {seq_key!r} in {queries_csv}")

    n = max(queries) + 1
    arr = np.zeros((n, 2), dtype=np.float32)
    for idx, (u, v) in queries.items():
        arr[idx] = (u, v)

    if K is None:
        K = np.array([[726.336, 0, 640], [0, 726.336, 360], [0, 0, 1]], dtype=np.float32)

    return arr, K





def collect_frames(seq_dir: Path, seq_name: str = "") -> list[Path]:
    """Return sorted list of PNG frame paths. seq_dir should point directly to the frames folder."""
    paths = sorted(seq_dir.glob("*.png"))
    if paths:
        return paths
    # Fallback: try seq_dir/png/seq_name/
    if seq_name:
        alt = seq_dir / "png" / seq_name
        paths = sorted(alt.glob("*.png"))
        if paths:
            return paths
    return []


def load_frame_tensor(path: Path, device: torch.device,
                      long_edge: int = 512, patch_size: int = 14) -> torch.Tensor:
    """Load a PNG frame, resize long edge to long_edge (patch-aligned), return (1, 3, H, W) float32."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read frame: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Resize long edge, snap both dims to patch_size multiple
    scale = long_edge / max(h, w)
    new_w = (int(round(w * scale)) // patch_size) * patch_size
    new_h = (int(round(h * scale)) // patch_size) * patch_size
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return t  # (1, 3, H, W)


def unproject(query_uv: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Unproject query pixels using depth and intrinsics.

    Args:
        query_uv: (N, 2) pixel coords (u=col, v=row) in frame-0.
        depth:    (H, W) depth map in metres (frame-0).
        K:        (3, 3) camera intrinsics.

    Returns:
        (N, 3) 3D points in frame-0 camera coords.
    """
    u = query_uv[:, 0].astype(np.float32)
    v = query_uv[:, 1].astype(np.float32)

    H, W = depth.shape
    u_clamped = np.clip(u, 0, W - 1)
    v_clamped = np.clip(v, 0, H - 1)

    # Bilinear sample depth
    u0 = np.floor(u_clamped).astype(int)
    v0 = np.floor(v_clamped).astype(int)
    u1 = np.minimum(u0 + 1, W - 1)
    v1 = np.minimum(v0 + 1, H - 1)
    wu = (u_clamped - u0).astype(np.float32)
    wv = (v_clamped - v0).astype(np.float32)

    d = (depth[v0, u0] * (1 - wu) * (1 - wv)
         + depth[v0, u1] * wu * (1 - wv)
         + depth[v1, u0] * (1 - wu) * wv
         + depth[v1, u1] * wu * wv)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    X = (u - cx) / fx * d
    Y = (v - cy) / fy * d
    Z = d

    return np.stack([X, Y, Z], axis=-1)


def extract_3d_track(
    arc_out: dict,
    query_uv_frame0: np.ndarray,
    frame_idx: int,
    K: np.ndarray,
) -> np.ndarray:
    """Extract 3D position for each query at a given frame index from Arc preds.

    Matches official 4RC track eval: nearest-neighbor integer indexing into the
    dense track map at query-frame UV coordinates.
    """
    preds = arc_out.get("preds", [])
    if not preds:
        raise ValueError("Empty preds from 4RC")
    if frame_idx >= len(preds):
        raise IndexError(f"frame_idx {frame_idx} >= len(preds) {len(preds)}")

    pred = preds[frame_idx]

    def _as_hw3(t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        t = np.asarray(t)
        if t.ndim == 4:
            t = t[0]
        return t  # (H, W, 3)

    field = None
    if "track" in pred:
        field = _as_hw3(pred["track"])
    elif "pts" in pred:
        field = _as_hw3(pred["pts"])
    else:
        log.warning("Falling back to zero predictions for frame %d", frame_idx)
        return np.zeros((len(query_uv_frame0), 3), dtype=np.float32)

    H, W = field.shape[:2]
    us = np.clip(np.rint(query_uv_frame0[:, 0]).astype(np.int64), 0, W - 1)
    vs = np.clip(np.rint(query_uv_frame0[:, 1]).astype(np.int64), 0, H - 1)
    return field[vs, us].astype(np.float32)


# ── CoTracker3 TABA helpers ───────────────────────────────────────────────────

def load_cotracker(ckpt_path: Path | None, device: torch.device):
    """Load CoTracker3 offline model directly from installed package (no internet)."""
    try:
        from cotracker.predictor import CoTrackerPredictor
        if ckpt_path and Path(ckpt_path).exists():
            tracker = CoTrackerPredictor(
                checkpoint=str(ckpt_path),
                offline=True,
                window_len=60,
            )
        else:
            raise FileNotFoundError(f"CoTracker checkpoint not found: {ckpt_path}")
        tracker.eval().to(device)
        log.info("CoTracker3 loaded from %s", ckpt_path)
        return tracker
    except Exception as e:
        log.warning("CoTracker3 not available (%s) — running 4RC-only mode", e)
        return None


@torch.no_grad()
def run_cotracker(
    tracker,
    frame_paths: list[Path],
    query_uv: np.ndarray,          # (N, 2) native pixels
    device: torch.device,
    target_size: int = 512,
) -> np.ndarray | None:
    """Run CoTracker3 on the scored frames and return 2D tracks.

    Returns:
        tracks: (N, T, 2) in native pixel coords, or None on failure.
    """
    if tracker is None:
        return None
    try:
        import cv2
        T = len(frame_paths)
        N = len(query_uv)

        # Load frames at target_size (CoTracker expects square or similar aspect)
        frames = []
        for p in frame_paths:
            img = cv2.imread(str(p))
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            # resize keeping aspect, then pad to square
            scale = target_size / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            # store (native_h, native_w, new_h, new_w) for inverse transform
            frames.append((img, h, w, new_h, new_w))

        # Stack into (1, T, 3, H, W) float tensor [0,255]
        _, h0, w0, nh0, nw0 = frames[0]
        scale_u = float(nw0) / float(w0)
        scale_v = float(nh0) / float(h0)

        video = torch.stack([
            torch.from_numpy(f[0]).permute(2, 0, 1).float()
            for f, *_ in [(x,) for x in frames]
        ], dim=0).unsqueeze(0).to(device)  # (1, T, 3, H, W)

        # Query points in CoTracker format: (1, N, 3) with [time=0, x, y]
        qs_ct = np.zeros((N, 3), dtype=np.float32)
        qs_ct[:, 0] = 0.0                              # query at frame 0
        qs_ct[:, 1] = query_uv[:, 0] * scale_u        # u scaled
        qs_ct[:, 2] = query_uv[:, 1] * scale_v        # v scaled
        queries_t = torch.from_numpy(qs_ct).unsqueeze(0).to(device)

        pred_tracks, _ = tracker(video, queries=queries_t)
        # pred_tracks: (1, T, N, 2) in (x, y) = (u, v) order
        tracks_ct = pred_tracks[0].detach().cpu().numpy()  # (T, N, 2)
        tracks_ct = tracks_ct.transpose(1, 0, 2)            # (N, T, 2)

        # Scale back to native pixels
        tracks_ct[:, :, 0] /= scale_u
        tracks_ct[:, :, 1] /= scale_v
        return tracks_ct.astype(np.float32)
    except Exception as e:
        log.warning("CoTracker run failed: %s", e)
        return None


def _extract_conf_track(preds: list, query_uv_arc: np.ndarray) -> np.ndarray:
    """Extract per-query, per-frame conf_track values from 4RC preds.

    Returns:
        conf: (N, T) float32 in [0, 1]. Falls back to 1.0 if conf_track is absent.
    """
    N = len(query_uv_arc)
    T = len(preds)
    conf = np.ones((N, T), dtype=np.float32)

    for t, pred in enumerate(preds):
        if "conf_track" not in pred:
            continue
        cf = pred["conf_track"]
        if isinstance(cf, torch.Tensor):
            cf = cf.detach().cpu().numpy()
        cf = np.asarray(cf, dtype=np.float32)
        # Remove batch / channel dims: (B, H, W) or (1, H, W) or (H, W)
        while cf.ndim > 2 and cf.shape[0] == 1:
            cf = cf[0]
        if cf.ndim > 2:
            cf = cf[0]
        H, W = cf.shape
        us = np.clip(np.rint(query_uv_arc[:, 0]).astype(int), 0, W - 1)
        vs = np.clip(np.rint(query_uv_arc[:, 1]).astype(int), 0, H - 1)
        raw = cf[vs, us]  # (N,)
        # conf_track is a raw score (inv_log activation) — apply sigmoid to get [0,1]
        conf[:, t] = 1.0 / (1.0 + np.exp(-raw.clip(-20, 20)))

    return conf


def taba_refine(
    arc_preds: list,               # raw preds list from 4RC inference
    tracks_2d: np.ndarray,         # (N, T, 2) — CoTracker3 2D tracks in native pixels
    xyz_4rc: np.ndarray,           # (N, T, 3) — 4RC baseline predictions (frame-0 coords)
    query_uv: np.ndarray,          # (N, 2) original query positions in native pixels
    K: np.ndarray,                 # (3, 3) intrinsics (native resolution)
    arc_h: int, arc_w: int,
    query_uv_arc: np.ndarray | None = None,  # (N, 2) query positions in arc-space pixels
    native_h: int = FRAME_H, native_w: int = FRAME_W,
    static_thresh: float = 5.0,
    conf_thresh: float = 0.4,      # below this, interpolate from reliable neighbours
) -> np.ndarray:
    """TABA v5: static freeze + confidence-guided temporal interpolation.

    Fix 1 — Static freeze (same as v4):
        If CoTracker 2D motion < static_thresh px, the point is static.
        Freeze its 3D position to the frame-0 4RC prediction for all frames.

    Fix 2 — Confidence interpolation (new):
        For dynamic points, 4RC's conf_track tells us which frames are uncertain.
        For low-confidence frames we interpolate linearly from the nearest reliable
        neighbours instead of using the raw (uncertain) prediction.
        This removes temporal spikes and holes in the trajectory.
    """
    N, T = xyz_4rc.shape[:2]
    result = xyz_4rc.copy()

    # Extract per-query, per-frame confidence (N, T)
    conf = None
    if query_uv_arc is not None and arc_preds:
        conf = _extract_conf_track(arc_preds, query_uv_arc)
        n_low = int((conf < conf_thresh).sum())
        log.debug("TABA conf: %d / %d query-frames below thresh %.2f",
                  n_low, N * T, conf_thresh)

    for i in range(N):
        # ── Fix 1: static freeze ─────────────────────────────────────────────
        if tracks_2d is not None:
            max_motion = float(
                np.max(np.linalg.norm(tracks_2d[i] - tracks_2d[i, 0], axis=-1))
            )
            if max_motion < static_thresh:
                xyz_t0 = xyz_4rc[i, 0].copy()
                result[i, :] = xyz_t0
                continue  # skip confidence fix for static points

        # ── Fix 2: confidence-guided temporal interpolation ───────────────────
        if conf is None:
            continue
        conf_i = conf[i]                     # (T,)
        reliable = conf_i >= conf_thresh     # bool mask

        if reliable.all() or not reliable.any():
            continue  # nothing to fix

        reliable_idx = np.where(reliable)[0]
        for t in range(T):
            if reliable[t]:
                continue
            before = reliable_idx[reliable_idx < t]
            after  = reliable_idx[reliable_idx > t]
            if len(before) > 0 and len(after) > 0:
                t0, t1 = int(before[-1]), int(after[0])
                alpha = float(t - t0) / float(t1 - t0)
                result[i, t] = (1.0 - alpha) * xyz_4rc[i, t0] + alpha * xyz_4rc[i, t1]
            elif len(before) > 0:
                result[i, t] = xyz_4rc[i, int(before[-1])]
            elif len(after) > 0:
                result[i, t] = xyz_4rc[i, int(after[0])]

    return result


# ── per-sequence inference ────────────────────────────────────────────────────

def _run_single_pass(
    model,
    paths_scored: list,
    query_uv: np.ndarray,
    K: np.ndarray,
    device: torch.device,
    resolution: int,
    track_query_idx: int = 0,
) -> tuple:
    """Single forward pass through 4RC. Returns (xyz [N,T,3], preds, arc_h, arc_w, query_uv_arc)."""
    from arc.dust3r.inference_multiview import inference
    from arc.dust3r.utils.image import load_images_for_eval

    imgs = load_images_for_eval(
        paths_scored, size=resolution, verbose=False, crop=False, patch_size=14, square_ok=True,
    )
    img0 = imgs[0]["img"]
    arc_h, arc_w = int(img0.shape[-2]), int(img0.shape[-1])
    query_uv_arc = query_uv.copy().astype(np.float32)
    query_uv_arc[:, 0] *= float(arc_w) / float(FRAME_W)
    query_uv_arc[:, 1] *= float(arc_h) / float(FRAME_H)
    qp = torch.tensor(query_uv_arc, dtype=torch.float32)
    q = torch.tensor([track_query_idx])
    for img in imgs:
        img["track_query_idx"] = q
        img["query_points"] = qp

    out = inference(imgs, model, device, dtype="bf16-mixed",
                    verbose=False, profiling=False, use_center_as_anchor=False)
    preds = out["preds"]
    T = len(paths_scored)
    N = len(query_uv)
    xyz = np.zeros((N, T, 3), dtype=np.float32)
    for local_t in range(T):
        if local_t < len(preds):
            xyz[:, local_t, :] = extract_3d_track({"preds": preds}, query_uv_arc, local_t, K)
    return xyz, preds, arc_h, arc_w, query_uv_arc


def infer_sequence(
    model: "Arc",
    seq_dir: Path,
    queries_csv: Path,
    variant: str,
    scene: str,
    seq_name: str,
    device: torch.device,
    cotracker=None,
    resolution: int = 512,
    window_size: int = 48,
    tta_flip: bool = False,
    track_query_idx: int = 0,
) -> dict[str, np.ndarray]:
    """Run 4RC + optional horizontal flip TTA + TABA on challenge scored frames.

    - Uses load_images_for_eval(..., crop=False) like 4RC track eval
    - Scales query UV by pred_size / native_size
    - Optional horizontal flip TTA: runs a second pass with flipped images,
      mirrors X-axis of predictions, then averages with equal weights.
    - Optional TABA refinement with CoTracker3 2D tracks
    """
    seq_key = f"{variant}/{scene}/{seq_name}"
    frame_paths = collect_frames(seq_dir, seq_name=seq_name)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found for {seq_key}")

    query_uv, K = load_queries_for_seq(queries_csv, seq_key)
    n_queries = len(query_uv)
    n_frames = len(frame_paths)

    # Only the frames that Kaggle scores (plus ensure frame 0 is present).
    scored = [t for t in SCORED_FRAME_INDICES if t < n_frames]
    if 0 not in scored:
        scored = [0] + scored
    paths_scored = [str(frame_paths[t]) for t in scored]

    # ── forward pass ─────────────────────────────────────────────────────────
    xyz_fwd, preds, arc_h, arc_w, query_uv_arc = _run_single_pass(
        model, paths_scored, query_uv, K, device, resolution, track_query_idx
    )

    # ── horizontal flip TTA ──────────────────────────────────────────────────
    if tta_flip:
        import tempfile, os
        from PIL import Image as PILImage

        tmp_dir = tempfile.mkdtemp(prefix="tta_flip_")
        flipped_paths = []
        for p in paths_scored:
            img = PILImage.open(p).transpose(PILImage.FLIP_LEFT_RIGHT)
            out_p = os.path.join(tmp_dir, os.path.basename(p))
            img.save(out_p)
            flipped_paths.append(out_p)

        # Mirror query UV: u_flip = (native_W - 1) - u
        query_uv_flip = query_uv.copy().astype(np.float32)
        query_uv_flip[:, 0] = (FRAME_W - 1) - query_uv_flip[:, 0]

        xyz_flip, _, _, _, _ = _run_single_pass(
            model, flipped_paths, query_uv_flip, K, device, resolution, track_query_idx
        )
        # Mirror X axis back to original orientation
        xyz_flip[:, :, 0] = -xyz_flip[:, :, 0]

        for p in flipped_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        xyz_4rc = (xyz_fwd + xyz_flip) * 0.5
        log.debug("Flip TTA applied for %s", seq_key)
    else:
        xyz_4rc = xyz_fwd

    # ── TABA refinement ───────────────────────────────────────────────────────
    xyz_final = xyz_4rc
    if cotracker is not None:
        scored_paths = [frame_paths[t] for t in scored]
        tracks_2d = run_cotracker(cotracker, scored_paths, query_uv, device)
        if tracks_2d is not None:
            xyz_final = taba_refine(
                preds, tracks_2d, xyz_4rc, query_uv, K,
                arc_h=arc_h, arc_w=arc_w,
                query_uv_arc=query_uv_arc,
            )
            log.debug("TABA applied for %s", seq_key)
    elif cotracker is None:
        xyz_final = taba_refine(
            preds, None, xyz_4rc, query_uv, K,
            arc_h=arc_h, arc_w=arc_w,
            query_uv_arc=query_uv_arc,
        )

    # ── build results dict ────────────────────────────────────────────────────
    results: dict[str, np.ndarray] = {}
    for local_t, global_t in enumerate(scored):
        for q_idx in range(n_queries):
            row_id = make_submission_id(variant, scene, seq_name, q_idx, global_t)
            results[row_id] = xyz_final[q_idx, local_t]

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def iter_sequences(benchmark_root: Path) -> Iterator[tuple[str, Path, str, str, str]]:
    """Yield (seq_id, seq_dir, variant, scene, seq_name) for all 128 challenge sequences.

    Challenge layout:
        benchmark_root/{variant}/{scene}/png/{seq_name}/*.png  ← frames
        benchmark_root/{variant}/{scene}/{seq_name}/           ← metadata (may be absent)

    seq_id = "{variant}-{scene}-{seq_name}"
    """
    for variant in VARIANTS:
        variant_dir = benchmark_root / variant
        if not variant_dir.exists():
            log.warning("Missing variant dir: %s", variant_dir)
            continue
        for scene in SCENES:
            scene_dir = variant_dir / scene
            if not scene_dir.exists():
                log.warning("Missing scene dir: %s", scene_dir)
                continue
            png_dir = scene_dir / "png"
            if png_dir.exists():
                seq_names = sorted(d.name for d in png_dir.iterdir() if d.is_dir())
            else:
                seq_names = sorted(d.name for d in scene_dir.iterdir()
                                   if d.is_dir() and d.name.startswith("seq_"))
            for seq_name in seq_names:
                seq_id = f"{variant}-{scene}-{seq_name}"
                # Frames live in scene_dir/png/seq_name/
                seq_dir = scene_dir / "png" / seq_name
                yield seq_id, seq_dir, variant, scene, seq_name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--benchmark_root", type=Path, required=True)
    p.add_argument("--queries_csv", type=Path, default=Path("data/queries.csv"))
    p.add_argument("--out", type=Path, default=Path("submissions/submission.csv"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--cotracker_ckpt", type=Path, default=None,
                   help="Path to CoTracker3 offline checkpoint (.pth). "
                        "If not set, TABA is disabled.")
    p.add_argument("--no_taba", action="store_true",
                   help="Disable TABA even if cotracker_ckpt is provided.")
    p.add_argument("--seq_start", type=int, default=0)
    p.add_argument("--seq_end", type=int, default=128)
    p.add_argument("--no_validate", action="store_true")
    p.add_argument("--head_weights", type=Path, default=None, help="Trained head weights from train_fast.py")
    p.add_argument("--resolution", type=int, default=512,
                   help="Long-edge resolution for 4RC input (patch-aligned). "
                        "Default 512 → 504×280. 728 → 728×406. 1008 → 1008×560.")
    p.add_argument("--window_size", type=int, default=48,
                   help="Temporal window size for 4RC inference. Default 48.")
    p.add_argument("--tta_flip", action="store_true",
                   help="Enable horizontal flip TTA: average forward + flipped inference.")
    p.add_argument("--track_query_idx", type=int, default=0,
                   help="Frame index for query points (official 4RC default is 11)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s  resolution: %d  window_size: %d", device, args.resolution, args.window_size)

    # ── load 4RC ──
    log.info("Loading 4RC from %s", args.checkpoint)
    sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "4RC"))
    from arc.models.arc.arc import Arc
    model = Arc.from_pretrained(str(args.checkpoint))
    model.eval().to(device)
    if hasattr(args, "head_weights") and args.head_weights and args.head_weights.exists():
        import torch as _torch
        log.info("Loading trained head weights from %s", args.head_weights)
        ckpt = _torch.load(args.head_weights, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        ms = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in ms}
        ms.update(loaded)
        model.load_state_dict(ms)
        log.info("Loaded %d trained tensors (soft_apd: %s)", len(loaded), ckpt.get("soft_apd", "?"))
    log.info("4RC loaded — %d parameters", sum(p.numel() for p in model.parameters()))

    # ── load CoTracker3 (optional) ──
    cotracker = None
    if not args.no_taba:
        cotracker = load_cotracker(args.cotracker_ckpt, device)

    # ── collect sequences ──
    all_seqs = list(iter_sequences(args.benchmark_root))
    log.info("Total sequences found: %d", len(all_seqs))
    seqs_to_run = all_seqs[args.seq_start:args.seq_end]
    log.info("Running sequences %d-%d (%d total)", args.seq_start, args.seq_end, len(seqs_to_run))
    log.info("TABA: %s", "enabled" if cotracker is not None else "disabled")

    # ── infer ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.out
    if args.seq_start > 0:
        out_path = args.out.with_stem(f"{args.out.stem}_shard{args.seq_start:04d}")

    written = 0
    with open(out_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "X", "Y", "Z"])

        for i, (seq_id, seq_dir, variant, scene, seq_name) in enumerate(seqs_to_run):
            log.info("[%d/%d] %s", i + 1, len(seqs_to_run), seq_id)
            try:
                results = infer_sequence(
                    model, seq_dir, args.queries_csv,
                    variant, scene, seq_name,
                    device,
                    cotracker=cotracker,
                    resolution=args.resolution,
                    window_size=args.window_size,
                    tta_flip=getattr(args, "tta_flip", False),
                    track_query_idx=getattr(args, "track_query_idx", 0),
                )
                for row_id, xyz in results.items():
                    writer.writerow([row_id, f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"])
                written += len(results)
                log.info("  → %d rows written (total: %d)", len(results), written)
            except Exception as exc:
                import traceback
                log.error("  FAILED %s: %s", seq_id, exc)
                log.error(traceback.format_exc())

    log.info("Done. Output: %s  (%d rows)", out_path, written)

    if not args.no_validate:
        try:
            from src.metrics.validator import validate_submission
            errors = validate_submission(out_path)
            if not errors:
                log.info("Validator: OK")
            else:
                for e in errors:
                    log.warning("Validator: %s", e)
        except Exception as e:
            log.warning("Validator skipped: %s", e)


if __name__ == "__main__":
    main()
