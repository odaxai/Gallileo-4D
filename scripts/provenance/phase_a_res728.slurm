#!/bin/bash
#SBATCH --job-name=4rc_sweep_res728
#SBATCH --account=AIFAC_F02_646
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/leonardo/home/userexternal/nsavioli/logs/sweep_res728_%A_%a.out
#SBATCH --error=/leonardo/home/userexternal/nsavioli/logs/sweep_res728_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolo.savioli@odaxai.com

# Phase A sweep: resolution=728
# At 728: 1280×720 → 728×406 (if long-edge = 728)  → patches: 52×29 = 1508 tokens/frame
# Note: 4RC was trained with long-edge up to 504. 728 may be OOD. Cache output; measure proxies.

set -eo pipefail
module load python/3.11.7 gcc/12.2.0 cuda/12.2

SCRATCH=/leonardo_scratch/large/userexternal/nsavioli
PROJECT=$SCRATCH/oda4d
source $PROJECT/.venv/bin/activate

export PYTHONPATH=$PROJECT/external/4RC:$PROJECT/src:${PYTHONPATH:-}
cd $PROJECT

TOTAL_SEQS=128
N_SHARDS=4
SEQ_START=$(( SLURM_ARRAY_TASK_ID * (TOTAL_SEQS / N_SHARDS) ))
SEQ_END=$(( SLURM_ARRAY_TASK_ID == N_SHARDS - 1 ? TOTAL_SEQS : SEQ_START + TOTAL_SEQS / N_SHARDS ))

CONFIG="base_res728"
CACHE_DIR=$PROJECT/cache/phase_a/${CONFIG}
mkdir -p $CACHE_DIR

echo "=== Phase A sweep: ${CONFIG} shard ${SLURM_ARRAY_TASK_ID} (seq ${SEQ_START}-${SEQ_END}) ==="
echo "Account: $SLURM_JOB_ACCOUNT  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Scratch before: $(du -sh $SCRATCH/ 2>/dev/null | cut -f1)"

python -u scripts/infer_4rc.py \
    --checkpoint $PROJECT/external/4RC_checkpoint \
    --benchmark_root $SCRATCH/Syn4D_Benchmark/challenge_eval \
    --queries_csv $PROJECT/data/queries.csv \
    --out $CACHE_DIR/shard_${SLURM_ARRAY_TASK_ID}.csv \
    --device cuda \
    --no_taba \
    --no_validate \
    --seq_start ${SEQ_START} \
    --seq_end ${SEQ_END} \
    --resolution 728

echo "Scratch after: $(du -sh $SCRATCH/ 2>/dev/null | cut -f1)"
echo "=== Shard ${SLURM_ARRAY_TASK_ID} DONE $(date) ==="
