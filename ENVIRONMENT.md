# Computational Environment

This document describes the computational environment used to generate the winning submission.

## Hardware

| Component | Specification |
|-----------|---------------|
| **GPU** | NVIDIA A100 80GB |
| **CPU** | AMD EPYC 7742 64-Core |
| **RAM** | 512GB DDR4 |
| **Storage** | NVMe SSD |
| **Cluster** | CINECA Leonardo |

## Software Stack

### Operating System

```
Ubuntu 22.04 LTS
Linux kernel 5.15.0
```

### Python Environment

```
Python 3.10.12
```

### Core Dependencies

| Package | Version |
|---------|---------|
| torch | 2.1.0+cu118 |
| torchvision | 0.16.0+cu118 |
| numpy | 1.24.3 |
| pillow | 10.0.0 |
| transformers | 4.35.0 |

### CUDA Stack

| Component | Version |
|-----------|---------|
| CUDA | 11.8 |
| cuDNN | 8.7.0 |
| NCCL | 2.15.5 |

## External Dependencies

### 4RC Backbone

- **Repository**: https://github.com/facebookresearch/4RC
- **Commit**: (use latest stable)
- **Checkpoint**: `facebook/4RC-geofinetune` from Hugging Face
- **License**: Apache 2.0

### Installation

```bash
git clone https://github.com/facebookresearch/4RC.git
cd 4RC
pip install -e .
```

## Runtime Statistics

| Metric | Value |
|--------|-------|
| **Total GPU Hours** | ~168 |
| **Per-sequence Time** | ~26 minutes |
| **Memory Usage** | ~35GB VRAM |
| **Disk I/O** | ~500GB read |

### Breakdown by Component

| Component | GPU Hours | Sequences | Time/Seq |
|-----------|-----------|-----------|----------|
| Stride-3 | 56 | 128 | ~26 min |
| TTA Flip | 56 | 128 | ~26 min |
| Stride-1 | 56 | 128 | ~26 min |
| **Total** | **168** | — | — |

## Reproducibility Notes

1. **Determinism**: Results may vary slightly (~0.001 APD) due to floating-point non-determinism in CUDA operations.

2. **Memory**: The 4RC backbone requires ~35GB VRAM. GPUs with less memory may need gradient checkpointing.

3. **Multi-GPU**: The code supports single-GPU inference. For multi-GPU, run separate sequence ranges in parallel.

## SLURM Configuration

Example job script for CINECA Leonardo:

```bash
#!/bin/bash
#SBATCH --job-name=gallileo4d
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --account=<account>

module load python/3.10
module load cuda/11.8

source activate gallileo4d

python -m gallileo4d.run \
    --checkpoint facebook/4RC-geofinetune \
    --benchmark_root $WORK/challenge_data \
    --queries_csv $WORK/queries.csv \
    --out $WORK/submission.csv \
    --ensemble
```
