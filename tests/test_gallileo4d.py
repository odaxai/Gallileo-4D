#!/usr/bin/env python3
"""Tests for Gallileo-4D.

These tests verify that the code correctly reproduces the winning submission.
Architecture: Tests are organized by layer (Domain, Ports, Adapters, Application).
"""

import csv
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


# =============================================================================
# DOMAIN LAYER TESTS
# =============================================================================

class TestEnsembleMerger:
    """Tests for the EnsembleMerger domain service."""
    
    def test_merge_weights_normalized(self):
        """Verify weights are normalized to sum to 1."""
        from gallileo4d.pipeline import EnsembleMerger
        
        pred1 = {"row1": np.array([1.0, 0.0, 0.0])}
        pred2 = {"row1": np.array([0.0, 1.0, 0.0])}
        pred3 = {"row1": np.array([0.0, 0.0, 1.0])}
        
        # Weights that don't sum to 1
        weights = [6.0, 2.5, 1.5]  # sum = 10
        
        merged = EnsembleMerger.merge([pred1, pred2, pred3], weights)
        
        # Should be normalized: 0.6, 0.25, 0.15
        expected = np.array([0.6, 0.25, 0.15])
        np.testing.assert_array_almost_equal(merged["row1"], expected)
    
    def test_merge_exact_weights(self):
        """Verify exact weights 0.60, 0.25, 0.15 produce correct result."""
        from gallileo4d.pipeline import EnsembleMerger
        
        pred_s3 = {"row1": np.array([1.0, 2.0, 3.0])}
        pred_tta = {"row1": np.array([4.0, 5.0, 6.0])}
        pred_s1 = {"row1": np.array([7.0, 8.0, 9.0])}
        
        weights = [0.60, 0.25, 0.15]
        
        merged = EnsembleMerger.merge([pred_s3, pred_tta, pred_s1], weights)
        
        expected = np.array([
            0.60 * 1.0 + 0.25 * 4.0 + 0.15 * 7.0,
            0.60 * 2.0 + 0.25 * 5.0 + 0.15 * 8.0,
            0.60 * 3.0 + 0.25 * 6.0 + 0.15 * 9.0,
        ])
        np.testing.assert_array_almost_equal(merged["row1"], expected)
    
    def test_merge_multiple_rows(self):
        """Verify merge works with multiple rows."""
        from gallileo4d.pipeline import EnsembleMerger
        
        pred1 = {
            "row1": np.array([1.0, 0.0, 0.0]),
            "row2": np.array([0.0, 1.0, 0.0]),
        }
        pred2 = {
            "row1": np.array([0.0, 1.0, 0.0]),
            "row2": np.array([0.0, 0.0, 1.0]),
        }
        
        weights = [0.5, 0.5]
        merged = EnsembleMerger.merge([pred1, pred2], weights)
        
        np.testing.assert_array_almost_equal(merged["row1"], [0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(merged["row2"], [0.0, 0.5, 0.5])
    
    def test_merge_empty_predictions_raises(self):
        """Verify empty predictions raise ValueError."""
        from gallileo4d.pipeline import EnsembleMerger
        
        with pytest.raises(ValueError, match="No predictions"):
            EnsembleMerger.merge([], [])
    
    def test_merge_mismatched_lengths_raises(self):
        """Verify mismatched lengths raise ValueError."""
        from gallileo4d.pipeline import EnsembleMerger
        
        pred1 = {"row1": np.array([1.0, 0.0, 0.0])}
        
        with pytest.raises(ValueError, match="Mismatch"):
            EnsembleMerger.merge([pred1], [0.5, 0.5])


class TestEnsembleConfig:
    """Tests for the EnsembleConfig domain class."""
    
    def test_default_config(self):
        """Verify default config matches winning submission."""
        from gallileo4d.pipeline import EnsembleConfig
        
        config = EnsembleConfig.default()
        
        assert len(config.components) == 3
        
        # Component 1: Stride-3
        assert config.components[0].name == "stride3"
        assert config.components[0].resolution == 512
        assert config.components[0].frame_stride == 3
        assert config.components[0].tta_flip == False
        assert config.components[0].weight == 0.60
        
        # Component 2: TTA Flip
        assert config.components[1].name == "tta_flip"
        assert config.components[1].resolution == 512
        assert config.components[1].frame_stride == 3
        assert config.components[1].tta_flip == True
        assert config.components[1].weight == 0.25
        
        # Component 3: Stride-1
        assert config.components[2].name == "stride1"
        assert config.components[2].resolution == 512
        assert config.components[2].frame_stride == 1
        assert config.components[2].tta_flip == False
        assert config.components[2].weight == 0.15
    
    def test_weights_sum_to_one(self):
        """Verify default weights sum to 1.0."""
        from gallileo4d.pipeline import EnsembleConfig
        
        config = EnsembleConfig.default()
        
        assert abs(config.total_weight() - 1.0) < 1e-10


class TestChallengeConstants:
    """Tests for challenge constants (domain knowledge)."""
    
    def test_variants(self):
        """Verify all 4 variants are defined."""
        from gallileo4d.pipeline import ChallengeConstants
        
        assert len(ChallengeConstants.VARIANTS) == 4
        assert "og" in ChallengeConstants.VARIANTS
        assert "sim" in ChallengeConstants.VARIANTS
        assert "mixed" in ChallengeConstants.VARIANTS
        assert "mixed_no_bedlam" in ChallengeConstants.VARIANTS
    
    def test_scenes(self):
        """Verify all 4 scenes are defined."""
        from gallileo4d.pipeline import ChallengeConstants
        
        assert len(ChallengeConstants.SCENES) == 4
        assert "antiquity" in ChallengeConstants.SCENES
        assert "dream" in ChallengeConstants.SCENES
        assert "gothic" in ChallengeConstants.SCENES
        assert "office" in ChallengeConstants.SCENES
    
    def test_scored_frames(self):
        """Verify scored frame indices are correct."""
        from gallileo4d.pipeline import ChallengeConstants
        
        # 32 frames: 0, 6, 12, ..., 186
        assert len(ChallengeConstants.SCORED_FRAME_INDICES) == 32
        assert ChallengeConstants.SCORED_FRAME_INDICES[0] == 0
        assert ChallengeConstants.SCORED_FRAME_INDICES[-1] == 186
        assert all(
            ChallengeConstants.SCORED_FRAME_INDICES[i+1] - ChallengeConstants.SCORED_FRAME_INDICES[i] == 6
            for i in range(31)
        )
    
    def test_frame_dimensions(self):
        """Verify frame dimensions match challenge spec."""
        from gallileo4d.pipeline import ChallengeConstants
        
        assert ChallengeConstants.FRAME_H == 720
        assert ChallengeConstants.FRAME_W == 1280
    
    def test_n_queries(self):
        """Verify number of queries per sequence."""
        from gallileo4d.pipeline import ChallengeConstants
        
        assert ChallengeConstants.N_QUERIES == 512


class TestSubmissionFormatter:
    """Tests for submission ID formatting."""
    
    def test_make_id_format(self):
        """Verify submission ID format is correct."""
        from gallileo4d.pipeline import SubmissionFormatter
        
        row_id = SubmissionFormatter.make_id(
            variant="og",
            scene="antiquity", 
            seq_name="seq_000000_0",
            query_id=42,
            source_frame=12
        )
        
        assert row_id == "og-antiquity-seq_000000_0-q042-f012"
    
    def test_make_id_padding(self):
        """Verify query and frame IDs are zero-padded."""
        from gallileo4d.pipeline import SubmissionFormatter
        
        row_id = SubmissionFormatter.make_id(
            variant="sim",
            scene="gothic",
            seq_name="seq_000001_0",
            query_id=0,
            source_frame=0
        )
        
        assert row_id == "sim-gothic-seq_000001_0-q000-f000"
    
    def test_parse_id_roundtrip(self):
        """Verify parse_id is inverse of make_id."""
        from gallileo4d.pipeline import SubmissionFormatter
        
        original = ("og", "antiquity", "seq_000000_0", 42, 12)
        row_id = SubmissionFormatter.make_id(*original)
        parsed = SubmissionFormatter.parse_id(row_id)
        
        assert parsed == original


class TestInferenceConfig:
    """Tests for InferenceConfig."""
    
    def test_default_values(self):
        """Verify default configuration values."""
        from gallileo4d.pipeline import InferenceConfig
        
        config = InferenceConfig(
            checkpoint=Path("/test/ckpt"),
            benchmark_root=Path("/test/data"),
            queries_csv=Path("/test/queries.csv"),
            output=Path("/test/out.csv"),
        )
        
        assert config.device == "cuda"
        assert config.resolution == 512
        assert config.frame_stride == 6
        assert config.tta_flip == False
        assert config.seq_start == 0
        assert config.seq_end == 128
    
    def test_immutable(self):
        """Verify config is immutable (frozen dataclass)."""
        from gallileo4d.pipeline import InferenceConfig
        
        config = InferenceConfig(
            checkpoint=Path("/test/ckpt"),
            benchmark_root=Path("/test/data"),
            queries_csv=Path("/test/queries.csv"),
            output=Path("/test/out.csv"),
        )
        
        with pytest.raises(Exception):  # FrozenInstanceError
            config.resolution = 1024


# =============================================================================
# ADAPTER TESTS
# =============================================================================

class TestCSVSubmissionWriter:
    """Tests for CSV submission writer adapter."""
    
    def test_write_creates_file(self):
        """Verify writer creates CSV file with correct format."""
        from gallileo4d.pipeline import CSVSubmissionWriter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.csv"
            writer = CSVSubmissionWriter()
            
            results = {
                "row1": np.array([1.0, 2.0, 3.0]),
                "row2": np.array([4.0, 5.0, 6.0]),
            }
            
            writer.write(results, output_path)
            
            assert output_path.exists()
            
            with open(output_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 2
            assert rows[0]["id"] == "row1"
            assert float(rows[0]["X"]) == pytest.approx(1.0)


class TestMergeScript:
    """Tests for the merge.py script."""
    
    def test_merge_csv_files(self):
        """Test merging CSV files produces correct output."""
        from gallileo4d.merge import load_submission, merge_submissions, save_submission
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test CSV files
            csv1 = tmpdir / "comp1.csv"
            csv2 = tmpdir / "comp2.csv"
            csv3 = tmpdir / "comp3.csv"
            
            with open(csv1, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "X", "Y", "Z"])
                writer.writerow(["row1", "1.0", "0.0", "0.0"])
                writer.writerow(["row2", "0.0", "1.0", "0.0"])
            
            with open(csv2, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "X", "Y", "Z"])
                writer.writerow(["row1", "0.0", "1.0", "0.0"])
                writer.writerow(["row2", "0.0", "0.0", "1.0"])
            
            with open(csv3, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "X", "Y", "Z"])
                writer.writerow(["row1", "0.0", "0.0", "1.0"])
                writer.writerow(["row2", "1.0", "0.0", "0.0"])
            
            # Load and merge
            subs = [load_submission(p) for p in [csv1, csv2, csv3]]
            merged = merge_submissions(subs, [0.60, 0.25, 0.15])
            
            # Verify
            expected_row1 = np.array([0.60, 0.25, 0.15])
            expected_row2 = np.array([0.15, 0.60, 0.25])
            
            np.testing.assert_array_almost_equal(merged["row1"], expected_row1)
            np.testing.assert_array_almost_equal(merged["row2"], expected_row2)


# =============================================================================
# APPLICATION LAYER TESTS
# =============================================================================

class TestGallileoPipeline:
    """Tests for the main pipeline (with mocked dependencies)."""
    
    def test_pipeline_accepts_injected_dependencies(self):
        """Verify pipeline accepts dependency injection."""
        from gallileo4d.pipeline import GallileoPipeline, InferenceConfig
        
        config = InferenceConfig(
            checkpoint=Path("/test/ckpt"),
            benchmark_root=Path("/test/data"),
            queries_csv=Path("/test/queries.csv"),
            output=Path("/test/out.csv"),
        )
        
        # Create mock dependencies
        mock_model = Mock()
        mock_query_loader = Mock()
        mock_frame_loader = Mock()
        mock_writer = Mock()
        
        # Pipeline should accept injected dependencies
        pipeline = GallileoPipeline(
            config,
            model=mock_model,
            query_loader=mock_query_loader,
            frame_loader=mock_frame_loader,
            submission_writer=mock_writer,
        )
        
        assert pipeline._model is mock_model
        assert pipeline._query_loader is mock_query_loader


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests that require actual challenge data."""
    
    @pytest.mark.skip(reason="Requires challenge data")
    def test_reproduce_winning_score(self):
        """Verify we can reproduce the winning 0.58356 score."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
