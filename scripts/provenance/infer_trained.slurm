#!/bin/bash
#SBATCH --job-name=infer_trained
#SBATCH --account=AIFAC_F02_646
#SBATCH --partition=boost_usr_prod
#SBATCH --array=0-127
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/infer_trained_%A_%a.out
#SBATCH --error=logs/infer_trained_%A_%a.err

set -eo pipefail

module purge
module load python/3.11.7
module load cuda/12.2
module load gcc/12.2.0

PROJECT=/leonardo_scratch/large/userexternal/nsavioli/oda4d
cd "$PROJECT"
source .venv/bin/activate

# Use best checkpoint from fast_train
TRAINED_CKPT="$PROJECT/experiments/fast_train/best.pt"
if [ ! -f "$TRAINED_CKPT" ]; then
    echo "ERROR: No trained checkpoint found at $TRAINED_CKPT"
    echo "Available checkpoints:"
    ls -la $PROJECT/experiments/fast_train/*.pt 2>/dev/null || echo "None"
    exit 1
fi

OUT_DIR="$PROJECT/sweep_results/trained_best"
mkdir -p "$OUT_DIR"

echo "Using trained checkpoint: $TRAINED_CKPT"
echo "Shard $SLURM_ARRAY_TASK_ID / 128"

python -u scripts/infer_trained.py \
    --base_checkpoint external/4RC_geofinetune_checkpoint \
    --trained_checkpoint "$TRAINED_CKPT" \
    --test_manifest configs/test_manifest.yaml \
    --output_csv "$OUT_DIR/shard_${SLURM_ARRAY_TASK_ID}.csv" \
    --resolution 512 \
    --stride 3 \
    --num_frames 64 \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_shards 128

echo "Shard $SLURM_ARRAY_TASK_ID complete"
