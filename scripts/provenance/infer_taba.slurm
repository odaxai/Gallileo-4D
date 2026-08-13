#!/bin/bash
#SBATCH --job-name=4rc_taba
#SBATCH --account=AIFAC_F02_646
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --array=0-2              # 3 tasks × 4 GPU = 12 GPU max, 128/3 ≈ 43 seq each
#SBATCH --output=/leonardo/home/userexternal/nsavioli/logs/4rc_taba_%A_%a.out
#SBATCH --error=/leonardo/home/userexternal/nsavioli/logs/4rc_taba_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nicolo.savioli@odaxai.com

set -eo pipefail
module load python/3.11.7
module load gcc/12.2.0
module load cuda/12.2

SCRATCH=/leonardo_scratch/large/userexternal/nsavioli
PROJECT=$SCRATCH/oda4d
source $PROJECT/.venv/bin/activate

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH=$PROJECT/external/4RC:$PROJECT/src:$PYTHONPATH
cd $PROJECT

TOTAL_SEQS=128
N_SHARDS=3
SEQ_START=$(( SLURM_ARRAY_TASK_ID * (TOTAL_SEQS / N_SHARDS) ))
SEQ_END=$(( SLURM_ARRAY_TASK_ID == N_SHARDS - 1 ? TOTAL_SEQS : SEQ_START + TOTAL_SEQS / N_SHARDS ))

COTRACKER_CKPT=$PROJECT/external/cotracker3/cotracker3_offline.pth

echo "=== 4RC + TABA shard ${SLURM_ARRAY_TASK_ID} (seq ${SEQ_START}-${SEQ_END}) ==="
echo "Account: $SLURM_JOB_ACCOUNT  Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES  Date: $(date)"
echo "CoTracker ckpt: $COTRACKER_CKPT"

python -u scripts/infer_4rc.py \
    --checkpoint $PROJECT/external/4RC_checkpoint \
    --benchmark_root $SCRATCH/Syn4D_Benchmark/challenge_eval \
    --queries_csv $PROJECT/data/queries.csv \
    --out $PROJECT/submissions/4rc_taba_shard_${SLURM_ARRAY_TASK_ID}.csv \
    --cotracker_ckpt $COTRACKER_CKPT \
    --device cuda \
    --seq_start ${SEQ_START} \
    --seq_end ${SEQ_END}

echo "=== Shard ${SLURM_ARRAY_TASK_ID} DONE $(date) ==="
