#!/bin/bash
# ============================================================================
#  Gallileo-4D — ONE-COMMAND VERIFICATION FOR EXAMINERS
#
#  Verifies that this repository reproduces, bit-for-bit, the exact file
#  submitted to the PhysAI Dynamic 4D Reconstruction Challenge:
#
#      opt4_r728_116.csv
#      Private score: 0.58356 APD (3rd place)
#      Public score:  0.55513 APD
#
#  Usage:
#      bash verify.sh
#
#  What it does:
#      [1/4] Verifies the shipped winning submission archive (MD5 + SHA256)
#      [2/4] Verifies all 4 ensemble components (MD5)
#      [3/4] REPRODUCES the winning submission from the components and
#            checks it is bit-for-bit identical (MD5 + SHA256)
#      [4/4] Runs the pytest verification suite (if pytest is installed)
#
#  Requirements: python3 with numpy + pandas (any recent version).
#  No GPU, no external data, no checkpoints needed. Runtime: ~2 minutes.
# ============================================================================
set -e
cd "$(dirname "$0")"

# Expected hashes of the winning submission (uncompressed)
WIN_MD5="24d729dedd16de4f65e2d67301455c48"
WIN_SHA256="9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e"

# Expected MD5 of each ensemble component (uncompressed)
# Format: filename:md5  (portable — works with bash 3.2 on macOS)
COMPONENTS="
infer_train_v2_submission.csv:9ece0f3a0a49a0727f23f2c3a2c967de
tta_flip_v2_submission.csv:332177d9466adeafd6007536a7a72aca
4rc_taba_merged.csv:70a5f14420faab4d36a199ac66c549e6
res728_component.csv:d3210701394e44500bb88271b9d6fd5b
"

md5_of()    { if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}'; else md5 -q "$1"; fi }
sha256_of() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi }

PASS=0; FAIL=0
check() {  # check <label> <actual> <expected>
    if [ "$2" = "$3" ]; then
        echo "  [PASS] $1"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $1"
        echo "         expected: $3"
        echo "         actual:   $2"
        FAIL=$((FAIL+1))
    fi
}

echo "============================================================================"
echo " Gallileo-4D verification — winning submission opt4_r728_116.csv"
echo " Private 0.58356 APD (3rd place) / Public 0.55513 APD"
echo "============================================================================"

# ── [1/4] Verify the shipped winning submission archive ────────────────────
echo ""
echo "[1/4] Verifying shipped winning submission (reference/winning_submission.csv.gz)"
gunzip -c reference/winning_submission.csv.gz > /tmp/g4d_winning.csv
check "winning submission MD5"    "$(md5_of /tmp/g4d_winning.csv)"    "$WIN_MD5"
check "winning submission SHA256" "$(sha256_of /tmp/g4d_winning.csv)" "$WIN_SHA256"
ROWS=$(wc -l < /tmp/g4d_winning.csv | tr -d ' ')
check "row count (2097153 = header + 2097152)" "$ROWS" "2097153"

# ── [2/4] Verify the 4 ensemble components ──────────────────────────────────
echo ""
echo "[2/4] Verifying the 4 ensemble components (reference/components/)"
mkdir -p /tmp/g4d_components
for entry in $COMPONENTS; do
    name="${entry%%:*}"
    expected_md5="${entry##*:}"
    gunzip -c "reference/components/$name.gz" > "/tmp/g4d_components/$name"
    check "$name MD5" "$(md5_of "/tmp/g4d_components/$name")" "$expected_md5"
done

# ── [3/4] Reproduce the winning submission bit-for-bit ──────────────────────
echo ""
echo "[3/4] REPRODUCING the winning submission from components"
echo "      (plain weighted average, weights 0.522 / 0.218 / 0.144 / 0.116)"
python3 -W ignore scripts/plain_merge.py \
    --csvs /tmp/g4d_components/infer_train_v2_submission.csv \
           /tmp/g4d_components/tta_flip_v2_submission.csv \
           /tmp/g4d_components/4rc_taba_merged.csv \
           /tmp/g4d_components/res728_component.csv \
    --weights 0.522 0.218 0.144 0.116 \
    --output /tmp/g4d_reproduced.csv
check "reproduced file MD5    (bit-for-bit)" "$(md5_of /tmp/g4d_reproduced.csv)"    "$WIN_MD5"
check "reproduced file SHA256 (bit-for-bit)" "$(sha256_of /tmp/g4d_reproduced.csv)" "$WIN_SHA256"

# Keep a copy for the examiner
cp /tmp/g4d_reproduced.csv ./submission_reproduced.csv
echo "         reproduced file saved as: ./submission_reproduced.csv"

# ── [4/4] Pytest verification suite (optional) ──────────────────────────────
echo ""
echo "[4/4] Running pytest verification suite"
if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest tests/test_winning_submission.py tests/test_reproduction.py -q || FAIL=$((FAIL+1))
else
    echo "  [SKIP] pytest not installed (pip install pytest to enable)"
fi

# ── Verdict ──────────────────────────────────────────────────────────────────
echo ""
echo "============================================================================"
if [ "$FAIL" -eq 0 ]; then
    echo " VERDICT: ALL CHECKS PASSED ($PASS/$PASS)"
    echo ""
    echo " ./submission_reproduced.csv is BIT-FOR-BIT IDENTICAL to the file"
    echo " submitted to the challenge (opt4_r728_116.csv, score 0.58356)."
    echo "============================================================================"
    exit 0
else
    echo " VERDICT: $FAIL CHECK(S) FAILED"
    echo "============================================================================"
    exit 1
fi
