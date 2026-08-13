# Gallileo-4D Docker

Docker image for reproducing the winning submission `opt4_r728_116.csv`
(**0.58356 private / 0.55513 public — 3rd place**).

## EXACT Reproduction (default, no GPU needed, ~2 minutes)

The image ships the 4 ensemble components. Running the container with **no
arguments** reproduces the winning submission **bit-for-bit** and verifies it
cryptographically (MD5 + SHA256):

```bash
docker pull odaxai/gallileo-4d:latest

mkdir -p results
docker run --rm -v $(pwd)/results:/app/results odaxai/gallileo-4d:latest
```

Expected output:

```
Loading 4 files...
Wrote 2,097,152 rows to /app/results/submission.csv

=== Cryptographic verification ===
MD5    expected: 24d729dedd16de4f65e2d67301455c48
MD5    actual:   24d729dedd16de4f65e2d67301455c48
SHA256 expected: 9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e
SHA256 actual:   9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e

SUCCESS: bit-for-bit reproduction of the winning submission
         opt4_r728_116.csv (0.58356 private / 0.55513 public)
```

The reproduced file is written to `./results/submission.csv` — this is the
exact file submitted to Kaggle.

## Run the Test Suite (includes cryptographic tests)

```bash
docker run --rm odaxai/gallileo-4d:latest python -m pytest tests/ -v
# 75+ tests, including:
#  - test_exact_merge_reproduces_winning_md5 (end-to-end bit-for-bit check)
#  - test_md5_hash_matches / test_sha256_hash (winning submission hashes)
#  - test_component_md5s / test_component_sha256s (all 4 components)
```

## Build Locally

```bash
git clone https://github.com/odaxai/Gallileo-4D.git
cd Gallileo-4D
docker build -t gallileo-4d .
docker run --rm -v $(pwd)/results:/app/results gallileo-4d
```

## Full GPU Inference from Scratch (optional, ~168 GPU-hours)

Only needed if you want to regenerate the 4 components from the raw challenge
data. Requires the 4RC backbone (license restrictions prevent bundling it):

```bash
git clone https://github.com/facebookresearch/4RC.git /path/to/4RC

docker run --gpus all \
    -v /path/to/4RC:/app/external/4RC \
    -v /path/to/Syn4D_Benchmark/challenge_eval:/app/data \
    -v /path/to/output:/app/results \
    odaxai/gallileo-4d:latest \
    bash scripts/node5_run_inference.sh /app/results
```

Note: regenerated components may differ by floating-point noise (< 1e-6);
the APD score is unaffected. For bit-for-bit reproduction use the default
command above.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_DIR` | `/app/checkpoints` | Path to 4RC checkpoint |
| `DATA_DIR` | `/app/data` | Path to challenge data |
| `RESULTS_DIR` | `/app/results` | Output directory |

## GPU Requirements (only for inference from scratch)

- NVIDIA GPU with CUDA 11.8+
- 40GB+ GPU memory recommended
- Docker with NVIDIA Container Toolkit

## Data Layout (only for inference from scratch)

```
/app/data/
├── og/
│   ├── antiquity/
│   ├── dream/
│   ├── gothic/
│   └── office/
├── sim/
├── mixed/
└── mixed_no_bedlam/
```
