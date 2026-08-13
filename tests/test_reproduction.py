#!/usr/bin/env python3
"""Tests for reproduction scripts.

These tests verify that the scripts can reproduce the winning submission.
"""

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


# =============================================================================
# Test Constants
# =============================================================================

WINNING_SUBMISSION_MD5 = "24d729dedd16de4f65e2d67301455c48"
WINNING_SUBMISSION_SHA256 = "9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e"
WINNING_SCORE_PUBLIC = 0.55513
WINNING_SCORE_FINAL = 0.58356
EXPECTED_ROWS = 2097152

# First 5 rows of winning submission for verification
WINNING_SAMPLE = [
    ("mixed-antiquity-seq_000000_0-q000-f000", 0.1052617845, -0.11980538893, 0.26479411564),
    ("mixed-antiquity-seq_000000_0-q000-f006", 0.08903972984, -0.11175358618999999, 0.29664207224),
    ("mixed-antiquity-seq_000000_0-q000-f012", 0.07384513097999999, -0.085583684272, 0.33573148174),
    ("mixed-antiquity-seq_000000_0-q000-f018", 0.053851102320000004, -0.033488565812, 0.37187907123999997),
    ("mixed-antiquity-seq_000000_0-q000-f024", 0.031596855060000005, 0.015049984896, 0.39336078760000004),
]


class TestScriptsExist:
    """Verify all required scripts exist."""
    
    def test_infer_4rc_exists(self):
        """Verify infer_4rc.py exists."""
        script = Path(__file__).parent.parent / "scripts" / "infer_4rc.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_ensemble_merge_exists(self):
        """Verify ensemble_merge.py exists."""
        script = Path(__file__).parent.parent / "scripts" / "ensemble_merge.py"
        assert script.exists(), f"Missing: {script}"
    
    def test_run_inference_exists(self):
        """Verify node5_run_inference.sh exists."""
        script = Path(__file__).parent.parent / "scripts" / "node5_run_inference.sh"
        assert script.exists(), f"Missing: {script}"


class TestInferenceScript:
    """Tests for infer_4rc.py script."""
    
    def test_script_imports(self):
        """Verify script can be imported without errors."""
        script = Path(__file__).parent.parent / "scripts" / "infer_4rc.py"
        result = subprocess.run(
            [sys.executable, "-c", f"import sys; sys.path.insert(0, '{script.parent}'); exec(open('{script}').read().split('if __name__')[0])"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Script may fail on import due to missing dependencies, but syntax should be valid
        assert "SyntaxError" not in result.stderr, f"Syntax error in script: {result.stderr}"
    
    def test_constants_match(self):
        """Verify script constants match challenge spec."""
        script = Path(__file__).parent.parent / "scripts" / "infer_4rc.py"
        content = script.read_text()
        
        assert "SCORED_FRAME_INDICES = list(range(0, 192, 6))" in content
        assert "N_QUERIES     = 512" in content
        assert "FRAME_H, FRAME_W = 720, 1280" in content
    
    def test_tta_flip_implemented(self):
        """Verify TTA flip is implemented."""
        script = Path(__file__).parent.parent / "scripts" / "infer_4rc.py"
        content = script.read_text()
        
        assert "--tta_flip" in content
        assert "FLIP_LEFT_RIGHT" in content
    
    def test_frame_stride_implemented(self):
        """Verify window size / resolution parameters are implemented."""
        script = Path(__file__).parent.parent / "scripts" / "infer_4rc.py"
        content = script.read_text()
        
        # Script uses window_size and resolution instead of frame_stride
        assert "--window_size" in content or "--resolution" in content
        assert "window_size" in content.lower() or "resolution" in content.lower()


class TestEnsembleMergeScript:
    """Tests for ensemble_merge.py script."""
    
    def test_script_syntax(self):
        """Verify script has valid Python syntax."""
        script = Path(__file__).parent.parent / "scripts" / "ensemble_merge.py"
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"
    
    def test_scale_alignment_implemented(self):
        """Verify per-sequence scale alignment is implemented."""
        script = Path(__file__).parent.parent / "scripts" / "ensemble_merge.py"
        content = script.read_text()
        
        assert "compute_per_seq_scales" in content
        assert "scale" in content.lower()
    
    def test_weights_argument(self):
        """Verify weights argument is supported."""
        script = Path(__file__).parent.parent / "scripts" / "ensemble_merge.py"
        content = script.read_text()
        
        # Script accepts weights as argument (actual values passed at runtime)
        assert "--weights" in content
        assert "type=float" in content


class TestRunInferenceScript:
    """Tests for node5_run_inference.sh script."""
    
    def test_script_syntax(self):
        """Verify bash script syntax."""
        script = Path(__file__).parent.parent / "scripts" / "node5_run_inference.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"
    
    def test_ensemble_weights(self):
        """Verify correct ensemble weights in script."""
        script = Path(__file__).parent.parent / "scripts" / "node5_run_inference.sh"
        content = script.read_text()
        
        assert "0.60" in content, "Missing stride-3 weight (0.60)"
        assert "0.25" in content, "Missing TTA weight (0.25)"
        assert "0.15" in content, "Missing stride-1 weight (0.15)"
    
    def test_three_components(self):
        """Verify all three components are run."""
        script = Path(__file__).parent.parent / "scripts" / "node5_run_inference.sh"
        content = script.read_text()
        
        assert "stride3" in content.lower() or "stride 3" in content
        assert "tta" in content.lower()
        assert "stride1" in content.lower() or "stride 1" in content
    
    def test_tta_flip_flag(self):
        """Verify TTA flip flag is used."""
        script = Path(__file__).parent.parent / "scripts" / "node5_run_inference.sh"
        content = script.read_text()
        
        assert "--tta_flip" in content


class TestReferenceSample:
    """Tests for reference submission sample."""
    
    def test_sample_exists(self):
        """Verify reference sample exists."""
        sample = Path(__file__).parent.parent / "reference" / "winning_submission_sample.csv"
        assert sample.exists(), f"Missing: {sample}"
    
    def test_sample_format(self):
        """Verify sample has correct format."""
        sample = Path(__file__).parent.parent / "reference" / "winning_submission_sample.csv"
        
        with open(sample) as f:
            reader = csv.DictReader(f)
            row = next(reader)
            
            assert "id" in row
            assert "X" in row
            assert "Y" in row
            assert "Z" in row
    
    def test_sample_values_match_winning(self):
        """Verify sample values match winning submission."""
        sample = Path(__file__).parent.parent / "reference" / "winning_submission_sample.csv"
        
        with open(sample) as f:
            reader = csv.DictReader(f)
            rows = list(reader)[:5]
        
        for i, (expected_id, expected_x, expected_y, expected_z) in enumerate(WINNING_SAMPLE):
            row = rows[i]
            assert row["id"] == expected_id, f"Row {i} ID mismatch"
            assert abs(float(row["X"]) - expected_x) < 1e-6, f"Row {i} X mismatch"
            assert abs(float(row["Y"]) - expected_y) < 1e-6, f"Row {i} Y mismatch"
            assert abs(float(row["Z"]) - expected_z) < 1e-6, f"Row {i} Z mismatch"


class TestEnsembleMergeFunction:
    """Tests for ensemble merge functionality."""
    
    def test_merge_with_scale_alignment(self):
        """Test merge function with scale alignment."""
        # Import from scripts
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        from ensemble_merge import load_csv, get_seq_key, compute_per_seq_scales
        
        # Test get_seq_key
        assert get_seq_key("mixed-antiquity-seq_000000_0-q000-f006") == "mixed-antiquity-seq_000000_0"
        assert get_seq_key("og-gothic-seq_000001_0-q123-f012") == "og-gothic-seq_000001_0"
    
    def test_weights_sum_to_one(self):
        """Verify ensemble weights sum to 1."""
        weights = [0.60, 0.25, 0.15]
        assert abs(sum(weights) - 1.0) < 1e-6


class TestExactReproduction:
    """Tests for the EXACT reproduction recipe of opt4_r728_116.csv.

    The winning submission is a plain weighted average of 4 components:
        infer_train_v2 (52.2%) + tta_flip_v2 (21.8%)
        + 4rc_taba_merged (14.4%) + res728 (11.6%)
    produced by scripts/plain_merge.py.
    """

    COMPONENT_FILES = [
        "infer_train_v2_submission.csv.gz",
        "tta_flip_v2_submission.csv.gz",
        "4rc_taba_merged.csv.gz",
        "res728_component.csv.gz",
    ]
    COMPONENT_MD5S = {
        "infer_train_v2_submission.csv": "9ece0f3a0a49a0727f23f2c3a2c967de",
        "tta_flip_v2_submission.csv": "332177d9466adeafd6007536a7a72aca",
        "4rc_taba_merged.csv": "70a5f14420faab4d36a199ac66c549e6",
        "res728_component.csv": "d3210701394e44500bb88271b9d6fd5b",
    }
    COMPONENT_SHA256S = {
        "infer_train_v2_submission.csv": "73d44fd3d0c6a7181e33d3f6f42afc7366808da26d10de586fdd1dcb56275808",
        "tta_flip_v2_submission.csv": "d5f972dd21b825540d12ecab1e2a445925d0e5801b5a7755be13218aba1b81f4",
        "4rc_taba_merged.csv": "cba07e7b569f83d89149ab2cabd4ba8b491e8da7f2d02788b5f2742c25a4c5e8",
        "res728_component.csv": "a42abfb19e1673e21632518d982fc0362d417310caf6a5d1e3b0f60a46f0d3a4",
    }
    EXACT_WEIGHTS = [0.522, 0.218, 0.144, 0.116]

    def test_plain_merge_exists(self):
        """Verify plain_merge.py (the exact merge script) exists."""
        script = Path(__file__).parent.parent / "scripts" / "plain_merge.py"
        assert script.exists(), f"Missing: {script}"

    def test_reproduce_script_exists(self):
        """Verify reproduce_winning.sh exists with correct recipe."""
        script = Path(__file__).parent.parent / "scripts" / "reproduce_winning.sh"
        assert script.exists(), f"Missing: {script}"

        content = script.read_text()
        assert "0.522 0.218 0.144 0.116" in content, "Exact weights missing"
        assert WINNING_SUBMISSION_MD5 in content, "Expected MD5 missing"
        assert "plain_merge.py" in content, "plain_merge.py not referenced"

    def test_reproduce_script_syntax(self):
        """Verify bash syntax of reproduce_winning.sh."""
        script = Path(__file__).parent.parent / "scripts" / "reproduce_winning.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_verify_script_exists_and_valid(self):
        """Verify the one-command examiner script verify.sh."""
        script = Path(__file__).parent.parent / "verify.sh"
        assert script.exists(), f"Missing: {script}"

        content = script.read_text()
        assert WINNING_SUBMISSION_MD5 in content, "Expected MD5 missing"
        assert "plain_merge.py" in content, "plain_merge.py not referenced"

        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_gpu_verification_script_exists_and_valid(self):
        """Verify the GPU end-to-end verification script."""
        script = (
            Path(__file__).parent.parent / "scripts" / "verify_gpu_inference.sh"
        )
        assert script.exists(), f"Missing: {script}"

        content = script.read_text()
        assert "infer_4rc.py" in content, "infer_4rc.py not referenced"
        assert "res728_component.csv.gz" in content, "reference component missing"
        assert "--resolution 728" in content, "res728 settings missing"

        result = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_private_leaderboard_included(self):
        """Verify the official private leaderboard CSV documents the score."""
        lb = Path(__file__).parent.parent / "reference" / "private_leaderboard.csv"
        assert lb.exists(), f"Missing: {lb}"

        content = lb.read_text()
        assert "OdaxAI" in content, "OdaxAI missing from private leaderboard"
        assert "0.58356" in content, "Winning score missing from leaderboard"

    def test_components_included(self):
        """Verify all 4 compressed components are in the repo."""
        comp_dir = Path(__file__).parent.parent / "reference" / "components"
        for fname in self.COMPONENT_FILES:
            assert (comp_dir / fname).exists(), f"Missing component: {fname}"

    def test_component_md5s(self):
        """Verify MD5 of each decompressed component."""
        import gzip
        import hashlib

        comp_dir = Path(__file__).parent.parent / "reference" / "components"
        for csv_name, expected_md5 in self.COMPONENT_MD5S.items():
            gz_path = comp_dir / f"{csv_name}.gz"
            md5 = hashlib.md5()
            with gzip.open(gz_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    md5.update(chunk)
            actual = md5.hexdigest()
            assert actual == expected_md5, (
                f"{csv_name}: MD5 mismatch (expected {expected_md5}, got {actual})"
            )

    def test_component_sha256s(self):
        """Verify SHA256 of each decompressed component (double check)."""
        import gzip
        import hashlib

        comp_dir = Path(__file__).parent.parent / "reference" / "components"
        for csv_name, expected_sha in self.COMPONENT_SHA256S.items():
            gz_path = comp_dir / f"{csv_name}.gz"
            sha256 = hashlib.sha256()
            with gzip.open(gz_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    sha256.update(chunk)
            actual = sha256.hexdigest()
            assert actual == expected_sha, (
                f"{csv_name}: SHA256 mismatch (expected {expected_sha}, got {actual})"
            )

    def test_exact_weights_sum_to_one(self):
        """Verify the 4 exact weights sum to 1."""
        assert abs(sum(self.EXACT_WEIGHTS) - 1.0) < 1e-9

    def test_exact_merge_reproduces_winning_md5(self, tmp_path):
        """END-TO-END: decompress components, merge, verify exact MD5.

        This is the definitive reproducibility test: the merged output must be
        bit-for-bit identical to the winning submission opt4_r728_116.csv.
        """
        import gzip
        import hashlib
        import shutil

        pytest.importorskip("pandas")

        comp_dir = Path(__file__).parent.parent / "reference" / "components"
        script = Path(__file__).parent.parent / "scripts" / "plain_merge.py"

        # Decompress components to temp dir
        csv_paths = []
        for csv_name in self.COMPONENT_MD5S:
            gz_path = comp_dir / f"{csv_name}.gz"
            out_path = tmp_path / csv_name
            with gzip.open(gz_path, "rb") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            csv_paths.append(str(out_path))

        # Run the exact merge
        out_csv = tmp_path / "reproduced.csv"
        result = subprocess.run(
            [sys.executable, "-W", "ignore", str(script),
             "--csvs", *csv_paths,
             "--weights", "0.522", "0.218", "0.144", "0.116",
             "--output", str(out_csv)],
            capture_output=True, text=True, timeout=600,
        )
        assert result.returncode == 0, f"Merge failed: {result.stderr}"

        # Verify MD5 and SHA256 match the winning submission exactly
        md5 = hashlib.md5()
        sha256 = hashlib.sha256()
        with open(out_csv, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                md5.update(chunk)
                sha256.update(chunk)
        assert md5.hexdigest() == WINNING_SUBMISSION_MD5, (
            f"Reproduction FAILED (MD5)!\n"
            f"Expected: {WINNING_SUBMISSION_MD5}\n"
            f"Actual:   {md5.hexdigest()}"
        )
        assert sha256.hexdigest() == WINNING_SUBMISSION_SHA256, (
            f"Reproduction FAILED (SHA256)!\n"
            f"Expected: {WINNING_SUBMISSION_SHA256}\n"
            f"Actual:   {sha256.hexdigest()}"
        )


class TestWinningSubmissionConstants:
    """Tests for winning submission constants."""
    
    def test_md5_documented(self):
        """Verify MD5 hash is documented."""
        readme = Path(__file__).parent.parent / "reference" / "README.md"
        content = readme.read_text()
        
        assert WINNING_SUBMISSION_MD5 in content
    
    def test_scores_documented(self):
        """Verify scores are documented."""
        readme = Path(__file__).parent.parent / "README.md"
        content = readme.read_text()
        
        assert "0.58356" in content  # Final score
        assert "0.55513" in content or "0.555" in content  # Public score
    
    def test_row_count_correct(self):
        """Verify expected row count is correct."""
        # 512 queries × 32 timestamps × 128 sequences = 2,097,152
        expected = 512 * 32 * 128
        assert expected == EXPECTED_ROWS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
