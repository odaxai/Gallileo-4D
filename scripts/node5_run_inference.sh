#!/bin/bash
# ============================================================
# Replica del workflow Leonardo su node5
# Lancia inferenza ensemble con il best model (geofinetune)
# Usage: bash node5_run_inference.sh [OUTPUT_DIR]
# ============================================================
set -e

NODE5_BASE="$HOME/syn4d"
CKPT_DIR="$NODE5_BASE/checkpoints/4RC_geofinetune"
DATA_DIR="$NODE5_BASE/data"
OUT_DIR="${1:-$NODE5_BASE/results/$(date +%Y%m%d_%H%M%S)}"
REPO_DIR="$NODE5_BASE/oda4d_repo/Syn4D_3D"
NGPU=$(nvidia-smi --list-gpus | wc -l)

mkdir -p "$OUT_DIR/stride3" "$OUT_DIR/tta" "$OUT_DIR/stride1"

echo "=================================================="
echo "  node5 Gallileo-4D Inference"
echo "  GPUs: $NGPU x RTX 3090"
echo "  Checkpoint: $CKPT_DIR"
echo "  Output: $OUT_DIR"
echo "=================================================="

# ── 1. Stride-3 (main, 60% weight) ──────────────────────
echo "[1/3] Stride-3 inference..."
torchrun --nproc_per_node=$NGPU \
  "$REPO_DIR/scripts/infer_4rc.py" \
  --checkpoint "$CKPT_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR/stride3" \
  --stride 3 \
  --resolution 512

# ── 2. TTA flip (25% weight) ─────────────────────────────
echo "[2/3] TTA flip inference..."
torchrun --nproc_per_node=$NGPU \
  "$REPO_DIR/scripts/infer_4rc.py" \
  --checkpoint "$CKPT_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR/tta" \
  --stride 3 \
  --resolution 512 \
  --tta_flip

# ── 3. Stride-1 (15% weight) ─────────────────────────────
echo "[3/3] Stride-1 inference..."
torchrun --nproc_per_node=$NGPU \
  "$REPO_DIR/scripts/infer_4rc.py" \
  --checkpoint "$CKPT_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR/stride1" \
  --stride 1 \
  --resolution 512

# ── 4. Ensemble merge ────────────────────────────────────
echo "[4/4] Merging ensemble (60% s3 + 25% tta + 15% s1)..."
python3 "$REPO_DIR/scripts/ensemble_merge.py" \
  --stride3 "$OUT_DIR/stride3" \
  --tta     "$OUT_DIR/tta" \
  --stride1 "$OUT_DIR/stride1" \
  --weights 0.60 0.25 0.15 \
  --output  "$OUT_DIR/submission.csv"

echo ""
echo "DONE. Submission file: $OUT_DIR/submission.csv"
echo "Run pre_submit_gate to verify before uploading."
