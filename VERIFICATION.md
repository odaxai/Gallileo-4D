# Verification Guide for Competition Organizers

This document provides complete information for verifying our winning submission.

## Winning Submission Details

| Metric | Value |
|--------|-------|
| **Final Score (Private 75%)** | 0.58356 APD |
| **Public Score (25%)** | 0.55513 APD |
| **Final Rank** | 3rd Place |
| **Submission File** | `opt4_r728_116.csv` |
| **MD5 Hash** | `24d729dedd16de4f65e2d67301455c48` |
| **SHA256 Hash** | `9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e` |
| **Row Count** | 2,097,152 (+ header) |
| **File Size** | 180,877,574 bytes |

## Ensemble Configuration (EXACT Recipe)

The winning submission is a **plain weighted average of 4 component CSVs**
(no scale alignment), produced by `scripts/plain_merge.py`:

| # | Component | Weight | Description | MD5 Hash |
|---|-----------|--------|-------------|----------|
| 1 | `infer_train_v2_submission.csv` | 52.2% | 4RC train_v2 (LR 5e-6, 1 epoch), stride=3, res=512 | `9ece0f3a0a49a0727f23f2c3a2c967de` |
| 2 | `tta_flip_v2_submission.csv` | 21.8% | 4RC, stride=3, res=512, TTA horizontal flip | `332177d9466adeafd6007536a7a72aca` |
| 3 | `4rc_taba_merged.csv` | 14.4% | 4RC + TABA v5 refinement | `70a5f14420faab4d36a199ac66c549e6` |
| 4 | `res728_component.csv` | 11.6% | 4RC, stride=3, res=728 | `d3210701394e44500bb88271b9d6fd5b` |

All 4 components are included (gzip-compressed) in `reference/components/`.

## Verification Steps

### Option 0: Docker — one command, zero setup (Recommended)

```bash
docker pull odaxai/gallileo-4d:latest
mkdir -p results
docker run --rm -v $(pwd)/results:/app/results odaxai/gallileo-4d:latest
```

The container reproduces the winning submission **bit-for-bit** from the
shipped components and verifies MD5 + SHA256 automatically:

```
=== Cryptographic verification ===
MD5    expected: 24d729dedd16de4f65e2d67301455c48
MD5    actual:   24d729dedd16de4f65e2d67301455c48
SHA256 expected: 9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e
SHA256 actual:   9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e

SUCCESS: bit-for-bit reproduction of the winning submission
         opt4_r728_116.csv (0.58356 private / 0.55513 public)
```

The reproduced file is written to `./results/submission.csv`.

### Option 1: EXACT Reproduction from Components (Recommended, ~1 minute)

The repository ships the 4 component files. Decompress them and run the merge —
the output is **bit-for-bit identical** to the winning submission:

```bash
# 1. Decompress components
cd reference/components
gunzip -k *.csv.gz
cd ../..

# 2. Run the exact merge command
bash scripts/reproduce_winning.sh reference/components submission_reproduced.csv

# The script verifies the MD5 automatically:
# Expected: 24d729dedd16de4f65e2d67301455c48
# Output:   SUCCESS: exact reproduction of the winning submission.
```

Or manually:

```bash
python3 scripts/plain_merge.py \
    --csvs reference/components/infer_train_v2_submission.csv \
           reference/components/tta_flip_v2_submission.csv \
           reference/components/4rc_taba_merged.csv \
           reference/components/res728_component.csv \
    --weights 0.522 0.218 0.144 0.116 \
    --output submission_reproduced.csv

md5sum submission_reproduced.csv
# Expected: 24d729dedd16de4f65e2d67301455c48
```

### Option 2: Verify Reference Sample (Quick)

The repository includes a 1000-row sample from the winning submission:

```bash
# Check the reference sample matches the winning submission
gunzip -k reference/winning_submission.csv.gz
head -1001 reference/winning_submission.csv > /tmp/sample.csv
diff reference/winning_submission_sample.csv /tmp/sample.csv
# Should show no differences
```

### Option 3: GPU End-to-End Verification (input videos → CSV, ~1 minute)

**Verified result**: the inference code in this repository, run on the raw
challenge input videos, regenerates the shipped components **bit-identical**
on the GPU class used for the winning run (NVIDIA RTX 3090).

We verified this by downloading the public repository fresh onto the
inference machine and regenerating two sequences from different eval variants:

| Sequence | Rows regenerated | Bit-identical to shipped component | Max diff |
|----------|-----------------|-----------------------------------|----------|
| `og-antiquity-seq_000000_0` | 16,384 | **16,384 / 16,384** | 0.0 |
| `mixed_no_bedlam-antiquity-seq_000004` | 16,384 | **16,384 / 16,384** | 0.0 |

To run this check yourself (requires GPU ≥ 24 GB, 4RC installed, challenge data):

```bash
bash scripts/verify_gpu_inference.sh \
    /path/to/4RC_geofinetune \
    /path/to/challenge_eval \
    /path/to/queries.csv 2
```

The script runs `scripts/infer_4rc.py` on the first N sequences at the exact
res728 settings and compares every row against
`reference/components/res728_component.csv.gz`. On an RTX 3090 the output is
bit-identical; on a different GPU model tiny differences (< 1e-5) may appear
and do not change the APD score.

### Option 4: Full Reproduction from Raw Data (Requires GPU, ~168 GPU-hours)

To regenerate all 4 components from scratch, download the fine-tuned head weights
from Hugging Face and run inference:

```bash
# Download fine-tuned head weights
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('OdaxAI/gallileo-4d-weights', 'train_v1_heads.pt', local_dir='weights')
hf_hub_download('OdaxAI/gallileo-4d-weights', 'train_v2_heads.pt', local_dir='weights')
"
```

| Weight File | MD5 | Used For |
|-------------|-----|----------|
| `train_v1_heads.pt` | `b55345b0ee52eeca4790c7cbf38a79b8` | `infer_train_v2` component (52.2%) |
| `train_v2_heads.pt` | `b571b5741e15a59a7385018e073b7371` | `tta_flip_v2` component (21.8%) |

**Regenerate each component:**

```bash
# Component 1: infer_train_v2 (52.2% weight)
python scripts/infer_4rc.py \
    --checkpoint /path/to/4RC_geofinetune \
    --head_weights train_v1_heads.pt \
    --benchmark_root /path/to/challenge_eval \
    --queries_csv /path/to/queries.csv \
    --out infer_train_v2_submission.csv \
    --resolution 512 --window_size 48

# Component 2: tta_flip_v2 (21.8% weight)
python scripts/infer_4rc.py \
    --checkpoint /path/to/4RC_geofinetune \
    --head_weights train_v2_heads.pt \
    --benchmark_root /path/to/challenge_eval \
    --queries_csv /path/to/queries.csv \
    --out tta_flip_v2_submission.csv \
    --resolution 512 --window_size 48 --tta_flip

# Component 3: 4rc_taba_merged (14.4% weight)
# Requires CoTracker3 checkpoint
python scripts/infer_4rc.py \
    --checkpoint /path/to/4RC_geofinetune \
    --cotracker_ckpt /path/to/cotracker3.pth \
    --benchmark_root /path/to/challenge_eval \
    --queries_csv /path/to/queries.csv \
    --out 4rc_taba_merged.csv \
    --resolution 512 --window_size 48

# Component 4: res728_component (11.6% weight)
python scripts/infer_4rc.py \
    --checkpoint /path/to/4RC_geofinetune \
    --benchmark_root /path/to/challenge_eval \
    --queries_csv /path/to/queries.csv \
    --out res728_component.csv \
    --resolution 728 --window_size 48
```

**Merge components:**

```bash
python scripts/plain_merge.py \
    --csvs infer_train_v2_submission.csv tta_flip_v2_submission.csv \
           4rc_taba_merged.csv res728_component.csv \
    --weights 0.522 0.218 0.144 0.116 \
    --output submission.csv

md5sum submission.csv
# Expected: 24d729dedd16de4f65e2d67301455c48
```

The full input → output chain is therefore covered:

```
challenge videos ──infer_4rc.py──▶ components   (bit-identical on RTX 3090, Option 3)
components ──plain_merge.py──▶ winning CSV      (bit-identical on ANY CPU, Option 0/1)
```

## Important Notes

1. **The merge is bit-for-bit deterministic on any CPU architecture.**
   `scripts/plain_merge.py` emulates the ARM fused multiply-add (FMA)
   accumulation used for the original winning merge with portable IEEE-754
   error-free transformations. Verified identical output (MD5
   `24d729dedd16de4f65e2d67301455c48`) on macOS arm64 and Linux x86_64,
   with numpy 1.26–2.4 and pandas 2.3–3.0.

2. **GPU inference is deterministic on the same GPU class.** On NVIDIA
   RTX 3090 (the GPU used for the winning run) `scripts/infer_4rc.py`
   regenerates the shipped components bit-identical from the raw input videos
   (verified, see Option 3). On a different GPU model, minor numerical
   differences (typically < 1e-6) may appear due to floating point
   accumulation order; the APD score is unaffected.

3. **No Training Required**: This is an inference-only method using the frozen 4RC backbone.

## Test Suite

Run the test suite to verify code correctness:

```bash
python -m pytest tests/ -v
# Expected: 78 passed, 1 skipped
```

The tests verify:
- **Exact end-to-end reproduction**: decompress components → merge → MD5 + SHA256 must equal the winning submission hashes (`tests/test_reproduction.py::TestExactReproduction::test_exact_merge_reproduces_winning_md5`)
- Winning submission hashes: MD5 `24d729dedd16de4f65e2d67301455c48`, SHA256 `9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e`
- Component MD5 + SHA256 hashes (all 4 components)
- Exact ensemble weights (0.522 / 0.218 / 0.144 / 0.116)
- Reference sample integrity
- Script syntax and constants

## Contact

For verification questions, contact: research@odaxai.com
