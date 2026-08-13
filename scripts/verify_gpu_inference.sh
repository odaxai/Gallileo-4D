#!/bin/bash
# ============================================================================
#  Gallileo-4D — GPU END-TO-END VERIFICATION (input videos → output CSV)
#
#  Proves that the inference code in this repository regenerates the shipped
#  ensemble components directly from the challenge input videos.
#
#  What it does:
#    1. Runs scripts/infer_4rc.py on N test sequences (default: 2) at the
#       exact settings of the res728 component (resolution 728, window 48).
#    2. Compares every generated row against the shipped
#       reference/components/res728_component.csv.gz.
#    3. Reports exact-match count and maximum absolute difference.
#
#  Verified result on NVIDIA RTX 3090 (the GPU used for the winning run):
#    - 16384/16384 rows BIT-IDENTICAL per sequence (max diff = 0.0)
#  On a different GPU model, tiny FP differences (< 1e-5) may appear;
#  they do not change the APD score.
#
#  Usage:
#    bash scripts/verify_gpu_inference.sh <checkpoint_dir> <benchmark_root> \
#         <queries_csv> [num_sequences]
#
#  Example:
#    bash scripts/verify_gpu_inference.sh \
#         checkpoints/4RC_geofinetune \
#         data/Syn4D_Benchmark/challenge_eval \
#         data/queries.csv 2
#
#  Requirements: CUDA GPU (>= 24 GB), 4RC installed (see README), challenge
#  eval data. Runtime: ~30 s per sequence on RTX 3090.
# ============================================================================
set -e
cd "$(dirname "$0")/.."

CHECKPOINT="${1:?usage: verify_gpu_inference.sh <checkpoint> <benchmark_root> <queries_csv> [n_seqs]}"
BENCHMARK_ROOT="${2:?missing benchmark_root}"
QUERIES_CSV="${3:?missing queries_csv}"
N_SEQS="${4:-2}"

OUTDIR=$(mktemp -d)
echo "============================================================================"
echo " GPU end-to-end verification: input videos -> inference -> CSV"
echo " Regenerating $N_SEQS sequence(s) at res728 settings and comparing with"
echo " the shipped component reference/components/res728_component.csv.gz"
echo "============================================================================"

python scripts/infer_4rc.py \
    --checkpoint "$CHECKPOINT" \
    --benchmark_root "$BENCHMARK_ROOT" \
    --queries_csv "$QUERIES_CSV" \
    --out "$OUTDIR/regen.csv" \
    --device cuda --resolution 728 --window_size 48 \
    --seq_start 0 --seq_end "$N_SEQS" --no_validate

python3 - "$OUTDIR/regen.csv" <<'EOF'
import csv, gzip, sys

regen_path = sys.argv[1]
new = {}
with open(regen_path) as f:
    for row in csv.DictReader(f):
        new[row["id"]] = (float(row["X"]), float(row["Y"]), float(row["Z"]))

old = {}
with gzip.open("reference/components/res728_component.csv.gz", "rt") as f:
    for row in csv.DictReader(f):
        if row["id"] in new:
            old[row["id"]] = (float(row["X"]), float(row["Y"]), float(row["Z"]))

n_exact = sum(1 for k in new if k in old and new[k] == old[k])
max_diff = max(max(abs(a - b) for a, b in zip(new[k], old[k])) for k in new if k in old)

print()
print("=== GPU end-to-end verification result ===")
print(f"rows regenerated from input videos: {len(new)}")
print(f"rows bit-identical to shipped component: {n_exact}/{len(new)}")
print(f"max abs difference: {max_diff:.2e}")

if n_exact == len(new):
    print()
    print("SUCCESS: inference from raw input videos is BIT-IDENTICAL to the")
    print("         shipped component used in the winning submission.")
    sys.exit(0)
elif max_diff < 1e-5:
    print()
    print("SUCCESS (within tolerance): differences < 1e-5 are expected on a")
    print("         different GPU model and do not change the APD score.")
    sys.exit(0)
else:
    print()
    print("FAILURE: differences exceed tolerance.")
    sys.exit(1)
EOF
