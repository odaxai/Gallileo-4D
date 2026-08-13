# Reference Files

This directory contains the **complete winning submission** for verification.

## Winning Submission

| Property | Value |
|----------|-------|
| **File** | `winning_submission.csv.gz` (compressed) |
| **Kaggle Name** | `opt4_r728_116.csv` |
| **Uncompressed Size** | 180,877,574 bytes |
| **MD5 (uncompressed)** | `24d729dedd16de4f65e2d67301455c48` |
| **SHA256 (uncompressed)** | `9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e` |
| **MD5 (compressed)** | `e47264c7602c5e9c4a6ca1ed441898c5` |
| **Rows** | 2,097,152 (+ header) |
| **Private Score (75%)** | **0.58356 APD** |
| **Public Score (25%)** | **0.55513 APD** |
| **Final Rank** | **3rd Place** |

## Quick Verification

```bash
# 1. Decompress the winning submission
gunzip -k reference/winning_submission.csv.gz

# 2. Verify MD5
md5sum reference/winning_submission.csv
# Expected: 24d729dedd16de4f65e2d67301455c48

# 3. Verify row count
wc -l reference/winning_submission.csv
# Expected: 2097153 (header + 2097152 rows)

# 4. Compare with sample (first 1000 rows)
head -1001 reference/winning_submission.csv > /tmp/first_1000.csv
diff reference/winning_submission_sample.csv /tmp/first_1000.csv
# Expected: no differences
```

## For Competition Organizers

To verify this is the actual winning submission:

1. **Submit to the challenge platform** → Should return score **0.58356**
2. **Verify MD5 hash** → `24d729dedd16de4f65e2d67301455c48`
3. **Compare with our Kaggle submission** → `opt4_r728_116.csv`

## Ensemble Configuration (EXACT Recipe)

The winning submission is a **plain weighted average of 4 components**
(no scale alignment), created with `scripts/plain_merge.py`:

| # | Component | Weight | MD5 (uncompressed) |
|---|-----------|--------|--------------------|
| 1 | `infer_train_v2_submission.csv` | 52.2% | `9ece0f3a0a49a0727f23f2c3a2c967de` |
| 2 | `tta_flip_v2_submission.csv` | 21.8% | `332177d9466adeafd6007536a7a72aca` |
| 3 | `4rc_taba_merged.csv` | 14.4% | `70a5f14420faab4d36a199ac66c549e6` |
| 4 | `res728_component.csv` | 11.6% | `d3210701394e44500bb88271b9d6fd5b` |

All 4 components are included (compressed) in `components/`.

## Exact Reproduction

```bash
# Decompress components
cd reference/components && gunzip -k *.csv.gz && cd ../..

# Reproduce the winning submission (bit-for-bit identical)
bash scripts/reproduce_winning.sh reference/components out.csv
# → SUCCESS: exact reproduction of the winning submission.
```

## Sample File

`winning_submission_sample.csv` contains the first 1000 rows for quick verification without decompressing the full file.

## Official Leaderboards

- `public_leaderboard.csv` — official public leaderboard (25% of test data)
- `private_leaderboard.csv` — official private/final leaderboard (75% of test data),
  downloaded from the challenge platform. Row 3: **OdaxAI, 0.58356**.
