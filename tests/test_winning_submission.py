"""
Test per verificare la riproduzione ESATTA della submission vincente.

Questo test verifica che:
1. La submission vincente (compressa) nel repo corrisponde all'MD5 atteso
2. Il reference sample corrisponde alle prime 1000 righe della submission vincente
3. Il formato e il numero di righe sono corretti

Per gli organizzatori della challenge:
- Decomprimere winning_submission.csv.gz
- Sottometterlo alla challenge → deve restituire score 0.58356
"""

import csv
import gzip
import hashlib
from pathlib import Path

import pytest


class TestWinningSubmissionVerification:
    """Verifica crittografica della submission vincente."""
    
    # MD5 della submission vincente (opt4_r728_116.csv)
    EXPECTED_MD5 = "24d729dedd16de4f65e2d67301455c48"
    EXPECTED_ROWS = 2097152
    EXPECTED_PRIVATE_SCORE = 0.58356
    EXPECTED_PUBLIC_SCORE = 0.55513
    
    @pytest.fixture
    def reference_dir(self):
        return Path(__file__).parent.parent / "reference"
    
    def test_compressed_submission_exists(self, reference_dir):
        """Verifica che la submission compressa esista."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        assert compressed_path.exists(), f"Missing: {compressed_path}"
        
        # Verifica dimensione ragionevole (30-50 MB compressi)
        size_mb = compressed_path.stat().st_size / (1024 * 1024)
        assert 30 < size_mb < 50, f"Unexpected size: {size_mb:.1f} MB"
    
    def test_md5_hash_matches(self, reference_dir):
        """Verifica MD5 della submission decompressa."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        
        # Calcola MD5 del contenuto decompresso
        md5 = hashlib.md5()
        with gzip.open(compressed_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        
        actual_md5 = md5.hexdigest()
        assert actual_md5 == self.EXPECTED_MD5, (
            f"MD5 mismatch!\n"
            f"Expected: {self.EXPECTED_MD5}\n"
            f"Actual:   {actual_md5}\n"
            f"This means the submission file has been modified."
        )
    
    def test_row_count_correct(self, reference_dir):
        """Verifica il numero di righe."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        
        with gzip.open(compressed_path, 'rt') as f:
            row_count = sum(1 for _ in f) - 1  # exclude header
        
        assert row_count == self.EXPECTED_ROWS, (
            f"Row count mismatch!\n"
            f"Expected: {self.EXPECTED_ROWS:,}\n"
            f"Actual:   {row_count:,}"
        )
    
    def test_sample_matches_submission(self, reference_dir):
        """Verifica che il sample corrisponda alle prime 1000 righe."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        sample_path = reference_dir / "winning_submission_sample.csv"
        
        # Leggi sample
        with open(sample_path) as f:
            sample_rows = list(csv.DictReader(f))
        
        # Leggi prime 1000 righe dalla submission compressa
        with gzip.open(compressed_path, 'rt') as f:
            reader = csv.DictReader(f)
            submission_rows = [next(reader) for _ in range(1000)]
        
        # Confronta
        assert len(sample_rows) == 1000, f"Sample has {len(sample_rows)} rows, expected 1000"
        
        for i, (sample, submission) in enumerate(zip(sample_rows, submission_rows)):
            assert sample['id'] == submission['id'], f"ID mismatch at row {i}"
            assert sample['X'] == submission['X'], f"X mismatch at row {i}: {sample['X']} vs {submission['X']}"
            assert sample['Y'] == submission['Y'], f"Y mismatch at row {i}: {sample['Y']} vs {submission['Y']}"
            assert sample['Z'] == submission['Z'], f"Z mismatch at row {i}: {sample['Z']} vs {submission['Z']}"
    
    def test_csv_format_valid(self, reference_dir):
        """Verifica il formato CSV."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        
        with gzip.open(compressed_path, 'rt') as f:
            reader = csv.DictReader(f)
            
            # Verifica header
            assert reader.fieldnames == ['id', 'X', 'Y', 'Z'], (
                f"Invalid header: {reader.fieldnames}"
            )
            
            # Verifica prime 100 righe
            for i, row in enumerate(reader):
                if i >= 100:
                    break
                
                # Verifica formato ID
                assert '-' in row['id'], f"Invalid ID format at row {i}: {row['id']}"
                
                # Verifica valori numerici
                try:
                    float(row['X'])
                    float(row['Y'])
                    float(row['Z'])
                except ValueError as e:
                    pytest.fail(f"Invalid numeric value at row {i}: {e}")
    
    def test_scores_documented(self, reference_dir):
        """Verifica che gli score siano documentati correttamente."""
        readme_path = reference_dir / "README.md"
        
        with open(readme_path) as f:
            content = f.read()
        
        # Verifica che gli score siano nel README
        assert "0.58356" in content, "Private score not documented"
        assert "0.55513" in content, "Public score not documented"
        assert self.EXPECTED_MD5 in content, "MD5 hash not documented"


class TestReproductionInstructions:
    """Verifica che le istruzioni di riproduzione siano complete."""
    
    def test_verification_md_exists(self):
        """Verifica che VERIFICATION.md esista."""
        path = Path(__file__).parent.parent / "VERIFICATION.md"
        assert path.exists(), "VERIFICATION.md missing"
    
    def test_ensemble_weights_documented(self):
        """Verifica che i pesi ESATTI dell'ensemble vincente siano documentati."""
        path = Path(__file__).parent.parent / "VERIFICATION.md"
        
        with open(path) as f:
            content = f.read()
        
        assert "52.2%" in content or "0.522" in content, "Base weight not documented"
        assert "21.8%" in content or "0.218" in content, "TTA weight not documented"
        assert "14.4%" in content or "0.144" in content, "TABA weight not documented"
        assert "11.6%" in content or "0.116" in content, "Res-728 weight not documented"
    
    def test_scripts_exist(self):
        """Verifica che gli script di riproduzione esistano."""
        scripts_dir = Path(__file__).parent.parent / "scripts"
        
        assert (scripts_dir / "infer_4rc.py").exists(), "infer_4rc.py missing"
        assert (scripts_dir / "ensemble_merge.py").exists(), "ensemble_merge.py missing"
        assert (scripts_dir / "node5_run_inference.sh").exists(), "node5_run_inference.sh missing"


class TestCryptographicVerification:
    """Verifica crittografica avanzata."""
    
    EXPECTED_MD5 = "24d729dedd16de4f65e2d67301455c48"
    EXPECTED_SHA256 = "9bb87a133f7af9d2f814373bc5a7f15b00be188d86619b440432fedd5470006e"
    
    @pytest.fixture
    def reference_dir(self):
        return Path(__file__).parent.parent / "reference"
    
    def test_sha256_hash(self, reference_dir):
        """Verifica SHA256 della submission vincente (doppia verifica oltre MD5)."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        
        sha256 = hashlib.sha256()
        with gzip.open(compressed_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        actual_sha256 = sha256.hexdigest()
        assert actual_sha256 == self.EXPECTED_SHA256, (
            f"SHA256 mismatch!\n"
            f"Expected: {self.EXPECTED_SHA256}\n"
            f"Actual:   {actual_sha256}\n"
            f"This means the submission file has been modified."
        )
    
    def test_file_integrity(self, reference_dir):
        """Verifica integrità del file compresso."""
        compressed_path = reference_dir / "winning_submission.csv.gz"
        
        # Verifica che il file sia un gzip valido
        try:
            with gzip.open(compressed_path, 'rb') as f:
                # Leggi primi bytes per verificare integrità
                f.read(1024)
                # Salta al fondo per verificare CRC
                f.seek(0, 2)  # Seek to end
        except gzip.BadGzipFile:
            pytest.fail("Corrupted gzip file")
