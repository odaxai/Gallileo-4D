# Reproducing the Winning Submission

This document provides step-by-step instructions to reproduce our winning submission (0.58356 APD, 3rd place) for the PhysAI Dynamic 4D Reconstruction Challenge.

## Summary

| Item | Value |
|------|-------|
| **Final Score** | 0.58356 APD |
| **Submission File** | `opt4_r728_116.csv` |
| **Method** | Frozen backbone + 3-component ensemble |
| **Training Required** | None (inference only) |
| **GPU Hours** | ~168 (A100) |

## Prerequisites

### Hardware

- NVIDIA GPU with 40GB+ VRAM (A100 80GB recommended)
- 64GB+ system RAM
- 500GB+ storage

### Software

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

## Step 1: Environment Setup

```bash
# Create conda environment
conda create -n gallileo4d python=3.10
conda activate gallileo4d

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Clone and install Gallileo-4D
git clone https://github.com/odaxai/Gallileo-4D.git
cd Gallileo-4D
pip install -r requirements.txt

# Clone and install 4RC backbone
git clone https://github.com/facebookresearch/4RC.git external/4RC
cd external/4RC
pip install -e .
cd ../..
```

## Step 2: Download Checkpoint

The 4RC geofinetune checkpoint is automatically downloaded from Hugging Face:

```bash
python -c "from arc.models.arc.arc import Arc; Arc.from_pretrained('facebook/4RC-geofinetune')"
```

Or manually download from: https://huggingface.co/facebook/4RC

## Step 3: Prepare Challenge Data

Organize the challenge data as follows:

```
/path/to/challenge_data/
├── og/
│   ├── antiquity/
│   │   └── png/
│   │       ├── seq_000000_0/
│   │       │   ├── 000000.png
│   │       │   ├── 000001.png
│   │       │   └── ...
│   │       └── ...
│   ├── dream/
│   ├── gothic/
│   └── office/
├── sim/
├── mixed/
└── mixed_no_bedlam/
```

Also ensure you have the queries CSV file provided by the challenge organizers.

## Step 4: Generate Submission

### Option A: Full Ensemble (Recommended)

This runs all three components and merges them automatically:

```bash
python -m gallileo4d.run \
    --checkpoint facebook/4RC-geofinetune \
    --benchmark_root /path/to/challenge_data \
    --queries_csv /path/to/queries.csv \
    --out submission.csv \
    --ensemble
```

### Option B: Manual Component Generation

For debugging or partial runs:

```bash
# Component 1: Stride-3 (weight: 0.60)
python -m gallileo4d.run \
    --checkpoint facebook/4RC-geofinetune \
    --benchmark_root /path/to/challenge_data \
    --queries_csv /path/to/queries.csv \
    --out stride3.csv \
    --resolution 512 \
    --frame_stride 3

# Component 2: TTA Flip (weight: 0.25)
python -m gallileo4d.run \
    --checkpoint facebook/4RC-geofinetune \
    --benchmark_root /path/to/challenge_data \
    --queries_csv /path/to/queries.csv \
    --out tta_flip.csv \
    --resolution 512 \
    --frame_stride 3 \
    --tta_flip

# Component 3: Stride-1 (weight: 0.15)
python -m gallileo4d.run \
    --checkpoint facebook/4RC-geofinetune \
    --benchmark_root /path/to/challenge_data \
    --queries_csv /path/to/queries.csv \
    --out stride1.csv \
    --resolution 512 \
    --frame_stride 1

# Merge with exact weights
python -m gallileo4d.merge \
    --inputs stride3.csv tta_flip.csv stride1.csv \
    --weights 0.60 0.25 0.15 \
    --out submission.csv
```

## Step 5: Verify Output

The output CSV should have the format:

```csv
id,X,Y,Z
og-antiquity-seq_000000_0-q000-f000,1.234567,2.345678,3.456789
og-antiquity-seq_000000_0-q000-f006,1.234567,2.345678,3.456789
...
```

Expected statistics:
- **Rows**: 2,097,152 (128 sequences × 512 queries × 32 timestamps)
- **Columns**: 4 (id, X, Y, Z)

## Ensemble Configuration

The exact configuration that produced 0.58356 APD:

| Component | Resolution | Frame Stride | TTA Flip | Weight |
|-----------|------------|--------------|----------|--------|
| stride3 | 512 | 3 | No | 0.60 |
| tta_flip | 512 | 3 | Yes | 0.25 |
| stride1 | 512 | 1 | No | 0.15 |

**Fusion formula**:
```
P_final = 0.60 * P_stride3 + 0.25 * P_tta_flip + 0.15 * P_stride1
```

## Troubleshooting

### Out of Memory

Reduce batch size or use gradient checkpointing:
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Missing Sequences

Check that all 128 sequences are present:
- 32 sequences per variant (og, sim, mixed, mixed_no_bedlam)
- 4 scenes per variant (antiquity, dream, gothic, office)

### Checkpoint Loading Issues

Ensure the 4RC repository is properly installed:
```bash
cd external/4RC
pip install -e . --force-reinstall
```

## Expected Results

| Metric | Value |
|--------|-------|
| Public Score (25%) | 0.55513 |
| Private Score (75%) | 0.58356 |
| Final Rank | 3rd of 27 |

## Contact

For questions about reproducing results, please open an issue on GitHub.
