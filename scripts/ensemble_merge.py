#!/usr/bin/env python3
"""
Ensemble fusion: merge outputs from stride=1, stride=3, stride=6 runs.
Each output has format: id,X,Y,Z  where id = scene-seq-qPOINT-fFRAME.

Strategy: for each unique id, take scale-aligned weighted median of X,Y,Z.
Weights: stride=3 (best known) = 0.5, stride=1 = 0.3, stride=6 = 0.2.
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load id -> xyz mapping."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["id"]] = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])])
    return data


def get_seq_key(id_str: str) -> str:
    """Extract sequence key from id for per-sequence scale alignment."""
    # id format: scene-seq-qPOINT-fFRAME  e.g. mixed-antiquity-seq_000000_0-q000-f006
    parts = id_str.split("-")
    # Find the part containing "seq_"
    for i, p in enumerate(parts):
        if p.startswith("seq_"):
            return "-".join(parts[:i+1])
    return id_str


def compute_per_seq_scales(data_list: list[dict]) -> list[dict[str, float]]:
    """Compute per-sequence scale factors to align all configs to config[0]."""
    # Group by sequence
    ref = data_list[0]  # stride=3, our best
    
    scales = [{} for _ in data_list]
    scales[0] = defaultdict(lambda: 1.0)  # reference: scale = 1
    
    # Build seq -> ids mapping for reference
    seq_to_ids = defaultdict(list)
    for id_str in ref:
        seq = get_seq_key(id_str)
        seq_to_ids[seq].append(id_str)
    
    for cfg_idx in range(1, len(data_list)):
        cfg = data_list[cfg_idx]
        for seq, ids in seq_to_ids.items():
            ref_norms = []
            cfg_norms = []
            for id_str in ids:
                if id_str in ref and id_str in cfg:
                    ref_norms.append(np.linalg.norm(ref[id_str]))
                    cfg_norms.append(np.linalg.norm(cfg[id_str]))
            
            if not ref_norms:
                scales[cfg_idx][seq] = 1.0
                continue
            
            ref_med = np.median([n for n in ref_norms if n > 0]) if any(n > 0 for n in ref_norms) else 1.0
            cfg_med = np.median([n for n in cfg_norms if n > 0]) if any(n > 0 for n in cfg_norms) else 1.0
            scales[cfg_idx][seq] = float(ref_med / (cfg_med + 1e-8))
    
    return scales


def ensemble_merge(csv_paths: list[Path], weights: list[float], output: Path):
    """
    Merge multiple CSV files via scale-aligned weighted average.
    Falls back to reference if other configs are missing.
    """
    assert len(csv_paths) == len(weights)
    
    log.info("Loading %d CSV files...", len(csv_paths))
    data_list = [load_csv(p) for p in csv_paths]
    log.info("Sizes: %s", [len(d) for d in data_list])
    
    # Compute per-sequence scales
    log.info("Computing per-sequence scale alignments...")
    scales = compute_per_seq_scales(data_list)
    
    # Normalize weights
    w = np.array(weights, dtype=float)
    w /= w.sum()
    
    # All unique IDs (from reference = first CSV)
    all_ids = list(data_list[0].keys())
    log.info("Total predictions to fuse: %d", len(all_ids))
    
    rows = []
    for id_str in all_ids:
        seq = get_seq_key(id_str)
        
        fused_xyz = np.zeros(3)
        total_w = 0.0
        
        for cfg_idx, (data, scale_map) in enumerate(zip(data_list, scales)):
            if id_str not in data:
                continue
            
            scale = scale_map.get(seq, 1.0) if isinstance(scale_map, dict) else 1.0
            xyz = data[id_str] * scale
            
            fused_xyz += w[cfg_idx] * xyz
            total_w += w[cfg_idx]
        
        if total_w > 0:
            fused_xyz /= total_w
        
        rows.append({"id": id_str, "X": fused_xyz[0], "Y": fused_xyz[1], "Z": fused_xyz[2]})
    
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "X", "Y", "Z"])
        writer.writeheader()
        writer.writerows(rows)
    
    log.info("Wrote %d rows to %s", len(rows), output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    
    ensemble_merge(
        [Path(p) for p in args.csvs],
        args.weights,
        args.output,
    )
