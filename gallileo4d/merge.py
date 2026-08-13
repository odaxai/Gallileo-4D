#!/usr/bin/env python3
"""Merge multiple submission CSVs with scale-aligned weighted averaging.

This script reproduces the exact winning submission (0.58356 APD).

Usage:
    python -m gallileo4d.merge \
        --inputs stride3.csv tta_flip.csv stride1.csv \
        --weights 0.60 0.25 0.15 \
        --out submission.csv

The scale alignment ensures predictions from different configurations
are in the same coordinate frame before averaging.
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


def load_submission(path: Path) -> dict[str, np.ndarray]:
    """Load submission CSV into dictionary."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row["id"]
            xyz = np.array([float(row["X"]), float(row["Y"]), float(row["Z"])], dtype=np.float64)
            data[row_id] = xyz
    return data


def get_seq_key(id_str: str) -> str:
    """Extract sequence key from id for per-sequence scale alignment.
    
    ID format: variant-scene-seq_NNNNNN_N-qNNN-fNNN
    Example: mixed-antiquity-seq_000000_0-q000-f006
    """
    parts = id_str.split("-")
    for i, p in enumerate(parts):
        if p.startswith("seq_"):
            return "-".join(parts[:i+1])
    return id_str


def compute_per_seq_scales(data_list: list[dict]) -> list[dict[str, float]]:
    """Compute per-sequence scale factors to align all configs to config[0].
    
    This aligns predictions from different configurations (stride-3, TTA, stride-1)
    to the same scale before averaging, which is critical for accurate fusion.
    """
    ref = data_list[0]  # First config is reference (stride-3)
    
    scales = [{} for _ in data_list]
    scales[0] = defaultdict(lambda: 1.0)  # Reference: scale = 1
    
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


def merge_submissions(
    submissions: list[dict[str, np.ndarray]], 
    weights: list[float],
    use_scale_alignment: bool = True
) -> dict[str, np.ndarray]:
    """Merge submissions with scale-aligned weighted average.
    
    Args:
        submissions: List of submission dictionaries (id -> xyz)
        weights: List of weights for each submission
        use_scale_alignment: If True, align scales per-sequence before averaging
    
    Returns:
        Merged submission dictionary
    """
    # Normalize weights
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    
    # Compute per-sequence scales if enabled
    if use_scale_alignment:
        log.info("Computing per-sequence scale alignments...")
        scales = compute_per_seq_scales(submissions)
    else:
        scales = [defaultdict(lambda: 1.0) for _ in submissions]
    
    # All unique IDs (from reference = first submission)
    all_ids = list(submissions[0].keys())
    log.info("Total predictions to fuse: %d", len(all_ids))
    
    merged = {}
    for id_str in all_ids:
        seq = get_seq_key(id_str)
        
        fused_xyz = np.zeros(3, dtype=np.float64)
        total_w = 0.0
        
        for cfg_idx, (data, scale_map) in enumerate(zip(submissions, scales)):
            if id_str not in data:
                continue
            
            scale = scale_map.get(seq, 1.0) if isinstance(scale_map, dict) else 1.0
            xyz = data[id_str] * scale
            
            fused_xyz += w[cfg_idx] * xyz
            total_w += w[cfg_idx]
        
        if total_w > 0:
            fused_xyz /= total_w
        
        merged[id_str] = fused_xyz
    
    return merged


def save_submission(data: dict[str, np.ndarray], path: Path) -> None:
    """Save merged submission to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "X", "Y", "Z"])
        for row_id, xyz in data.items():
            writer.writerow([row_id, xyz[0], xyz[1], xyz[2]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge submission CSVs with scale-aligned weighted averaging"
    )
    parser.add_argument("--inputs", nargs="+", type=Path, required=True,
                        help="Input CSV files (first is reference for scale alignment)")
    parser.add_argument("--weights", nargs="+", type=float, required=True,
                        help="Weights for each input (will be normalized)")
    parser.add_argument("--out", type=Path, default=Path("merged.csv"),
                        help="Output CSV path")
    parser.add_argument("--no-scale-align", action="store_true",
                        help="Disable per-sequence scale alignment")
    args = parser.parse_args()
    
    if len(args.inputs) != len(args.weights):
        raise ValueError("Number of inputs must match number of weights")
    
    log.info("Loading %d submissions...", len(args.inputs))
    submissions = [load_submission(p) for p in args.inputs]
    log.info("Sizes: %s", [len(s) for s in submissions])
    
    log.info("Merging with weights: %s", args.weights)
    merged = merge_submissions(
        submissions, 
        args.weights,
        use_scale_alignment=not args.no_scale_align
    )
    
    save_submission(merged, args.out)
    log.info("Saved to %s", args.out)


if __name__ == "__main__":
    main()
