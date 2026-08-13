#!/usr/bin/env python3
"""Gallileo-4D: Frozen Backbone Ensemble for Dynamic 4D Reconstruction.

3rd Place at PhysAI Dynamic 4D Reconstruction Challenge (ECCV 2026)
Final Score: 0.58356 APD

Architecture: Hexagonal (Ports & Adapters)
- Domain: Core business logic (ensemble, merge, config)
- Ports: Abstract interfaces for external dependencies
- Adapters: Concrete implementations (4RC, file I/O)
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# DOMAIN LAYER - Core Business Logic (no external dependencies)
# =============================================================================

@dataclass(frozen=True)
class InferenceConfig:
    """Immutable configuration for inference."""
    checkpoint: Path
    benchmark_root: Path
    queries_csv: Path
    output: Path
    device: str = "cuda"
    resolution: int = 512
    frame_stride: int = 6
    tta_flip: bool = False
    seq_start: int = 0
    seq_end: int = 128


@dataclass(frozen=True)
class ComponentConfig:
    """Configuration for a single ensemble component."""
    name: str
    resolution: int
    frame_stride: int
    tta_flip: bool
    weight: float


@dataclass
class EnsembleConfig:
    """Configuration for ensemble inference."""
    components: list[ComponentConfig] = field(default_factory=list)
    
    @classmethod
    def default(cls) -> "EnsembleConfig":
        """Default ensemble configuration (reproduces 0.58356 APD)."""
        return cls(components=[
            ComponentConfig("stride3", 512, 3, False, 0.60),
            ComponentConfig("tta_flip", 512, 3, True, 0.25),
            ComponentConfig("stride1", 512, 1, False, 0.15),
        ])
    
    def total_weight(self) -> float:
        """Sum of all component weights."""
        return sum(c.weight for c in self.components)


class ChallengeConstants:
    """Challenge-specific constants (domain knowledge)."""
    VARIANTS: tuple[str, ...] = ("og", "sim", "mixed", "mixed_no_bedlam")
    SCENES: tuple[str, ...] = ("antiquity", "dream", "gothic", "office")
    SCORED_FRAME_INDICES: list[int] = list(range(0, 192, 6))
    N_QUERIES: int = 512
    FRAME_H: int = 720
    FRAME_W: int = 1280
    
    @classmethod
    def n_sequences(cls) -> int:
        """Total number of sequences in challenge."""
        return len(cls.VARIANTS) * len(cls.SCENES) * 8  # 8 seqs per scene


class EnsembleMerger:
    """Domain service: merges predictions with weighted average."""
    
    @staticmethod
    def merge(
        predictions: list[dict[str, np.ndarray]], 
        weights: list[float]
    ) -> dict[str, np.ndarray]:
        """Merge predictions with normalized weighted average.
        
        Args:
            predictions: List of prediction dictionaries {row_id: xyz}
            weights: List of weights (will be normalized to sum=1)
            
        Returns:
            Merged predictions dictionary
            
        Raises:
            ValueError: If predictions or weights are empty or mismatched
        """
        if not predictions:
            raise ValueError("No predictions to merge")
        if len(predictions) != len(weights):
            raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(weights)} weights")
        
        w = np.array(weights, dtype=np.float64)
        if w.sum() == 0:
            raise ValueError("Weights sum to zero")
        w = w / w.sum()
        
        ids = list(predictions[0].keys())
        merged = {}
        
        for row_id in ids:
            vals = np.stack([p[row_id] for p in predictions])
            merged[row_id] = np.einsum("c,cd->d", w, vals)
        
        return merged


class SubmissionFormatter:
    """Domain service: formats submission IDs."""
    
    @staticmethod
    def make_id(variant: str, scene: str, seq_name: str,
                query_id: int, source_frame: int) -> str:
        """Build Kaggle submission row ID.
        
        Format: {variant}-{scene}-{seq_name}-q{query_id:03d}-f{source_frame:03d}
        """
        return f"{variant}-{scene}-{seq_name}-q{query_id:03d}-f{source_frame:03d}"
    
    @staticmethod
    def parse_id(row_id: str) -> tuple[str, str, str, int, int]:
        """Parse submission row ID into components."""
        parts = row_id.rsplit("-", 2)
        variant_scene_seq = parts[0]
        query_part = parts[1]  # q000
        frame_part = parts[2]  # f000
        
        # Split variant-scene-seq_name
        v_s_parts = variant_scene_seq.split("-", 2)
        variant = v_s_parts[0]
        scene = v_s_parts[1]
        seq_name = v_s_parts[2]
        
        query_id = int(query_part[1:])
        source_frame = int(frame_part[1:])
        
        return variant, scene, seq_name, query_id, source_frame


# =============================================================================
# PORTS - Abstract Interfaces (Dependency Inversion)
# =============================================================================

@runtime_checkable
class ModelPort(Protocol):
    """Port for 4D reconstruction model."""
    
    def load(self, checkpoint: Path, device: str) -> None:
        """Load model weights."""
        ...
    
    def predict(self, frames: list[str], query_uv: np.ndarray, 
                resolution: int) -> np.ndarray:
        """Run inference. Returns (N, T, 3) predictions."""
        ...


@runtime_checkable
class QueryLoaderPort(Protocol):
    """Port for loading query data."""
    
    def load_for_sequence(self, seq_key: str) -> tuple[np.ndarray, np.ndarray]:
        """Load queries and intrinsics for a sequence."""
        ...


@runtime_checkable
class FrameLoaderPort(Protocol):
    """Port for loading video frames."""
    
    def collect_frames(self, seq_dir: Path, seq_name: str) -> list[Path]:
        """Collect frame paths for a sequence."""
        ...


@runtime_checkable  
class SubmissionWriterPort(Protocol):
    """Port for writing submission output."""
    
    def write(self, results: dict[str, np.ndarray], output_path: Path) -> None:
        """Write results to submission file."""
        ...


class InferenceStrategy(ABC):
    """Abstract strategy for inference augmentation."""
    
    @abstractmethod
    def run(self, model: ModelPort, frames: list[Path], 
            query_uv: np.ndarray, resolution: int) -> np.ndarray:
        """Run inference with this strategy."""
        pass


# =============================================================================
# ADAPTERS - Concrete Implementations
# =============================================================================

class FourRCAdapter:
    """Adapter for 4RC backbone model."""
    
    def __init__(self):
        self._model = None
        self._device = None
    
    def load(self, checkpoint: Path, device: str) -> None:
        """Load 4RC model from checkpoint."""
        import torch
        import sys
        
        # Add 4RC to path
        arc_path = Path(__file__).parent.parent / "external" / "4RC"
        if arc_path.exists():
            sys.path.insert(0, str(arc_path))
        
        from arc.models.arc.arc import Arc
        
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = Arc.from_pretrained(str(checkpoint))
        self._model.eval().to(self._device)
        
        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False
        
        logger.info("Loaded 4RC backbone — %d parameters (frozen)", 
                    sum(p.numel() for p in self._model.parameters()))
    
    def predict(self, frames: list[str], query_uv: np.ndarray,
                resolution: int) -> np.ndarray:
        """Run 4RC inference."""
        import torch
        from arc.dust3r.inference_multiview import inference
        from arc.dust3r.utils.image import load_images_for_eval
        
        imgs = load_images_for_eval(
            frames, size=resolution, verbose=False, crop=False,
            patch_size=14, square_ok=True,
        )
        
        img0 = imgs[0]["img"]
        arc_h, arc_w = int(img0.shape[-2]), int(img0.shape[-1])
        
        # Scale query coordinates
        query_uv_arc = query_uv.copy().astype(np.float32)
        query_uv_arc[:, 0] *= float(arc_w) / float(ChallengeConstants.FRAME_W)
        query_uv_arc[:, 1] *= float(arc_h) / float(ChallengeConstants.FRAME_H)
        
        qp = torch.tensor(query_uv_arc, dtype=torch.float32)
        q = torch.tensor([0])
        for img in imgs:
            img["track_query_idx"] = q
            img["query_points"] = qp
        
        out = inference(imgs, self._model, self._device, dtype="bf16-mixed",
                        verbose=False, profiling=False, use_center_as_anchor=False)
        
        preds = out["preds"]
        T = len(frames)
        N = len(query_uv)
        xyz = np.zeros((N, T, 3), dtype=np.float32)
        
        for t in range(min(T, len(preds))):
            xyz[:, t, :] = self._extract_3d_track(preds[t], query_uv_arc)
        
        return xyz
    
    def _extract_3d_track(self, pred: dict, query_uv_arc: np.ndarray) -> np.ndarray:
        """Extract 3D positions from prediction."""
        import torch
        
        def _as_hw3(t) -> np.ndarray:
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            t = np.asarray(t)
            if t.ndim == 4:
                t = t[0]
            return t

        field = None
        if "track" in pred:
            field = _as_hw3(pred["track"])
        elif "pts" in pred:
            field = _as_hw3(pred["pts"])
        else:
            return np.zeros((len(query_uv_arc), 3), dtype=np.float32)

        H, W = field.shape[:2]
        us = np.clip(np.rint(query_uv_arc[:, 0]).astype(np.int64), 0, W - 1)
        vs = np.clip(np.rint(query_uv_arc[:, 1]).astype(np.int64), 0, H - 1)
        return field[vs, us].astype(np.float32)


class CSVQueryLoader:
    """Adapter for loading queries from CSV."""
    
    def __init__(self, queries_csv: Path):
        self._queries_csv = queries_csv
    
    def load_for_sequence(self, seq_key: str) -> tuple[np.ndarray, np.ndarray]:
        """Load query pixels and intrinsics for a sequence."""
        queries: dict[int, tuple[float, float]] = {}
        K = None
        
        with open(self._queries_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["sequence"] != seq_key:
                    continue
                q_idx = int(row["query_id"])
                queries[q_idx] = (float(row["u"]), float(row["v"]))
                if K is None:
                    fx, fy = float(row["fx"]), float(row["fy"])
                    cx, cy = float(row["cx"]), float(row["cy"])
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        if not queries:
            raise ValueError(f"No queries found for sequence {seq_key!r}")

        n = max(queries) + 1
        arr = np.zeros((n, 2), dtype=np.float32)
        for idx, (u, v) in queries.items():
            arr[idx] = (u, v)

        if K is None:
            K = np.array([[726.336, 0, 640], [0, 726.336, 360], [0, 0, 1]], dtype=np.float32)

        return arr, K


class FileSystemFrameLoader:
    """Adapter for loading frames from filesystem."""
    
    def collect_frames(self, seq_dir: Path, seq_name: str = "") -> list[Path]:
        """Return sorted list of PNG frame paths."""
        paths = sorted(seq_dir.glob("*.png"))
        if paths:
            return paths
        if seq_name:
            alt = seq_dir / "png" / seq_name
            paths = sorted(alt.glob("*.png"))
            if paths:
                return paths
        return []


class CSVSubmissionWriter:
    """Adapter for writing CSV submissions."""
    
    def write(self, results: dict[str, np.ndarray], output_path: Path) -> None:
        """Write results to CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "X", "Y", "Z"])
            for row_id, xyz in results.items():
                writer.writerow([row_id, f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"])


# =============================================================================
# INFERENCE STRATEGIES
# =============================================================================

class StandardInference(InferenceStrategy):
    """Standard forward pass inference."""
    
    def run(self, model: ModelPort, frames: list[Path],
            query_uv: np.ndarray, resolution: int) -> np.ndarray:
        """Single forward pass over the frame window, no augmentation."""
        frame_paths = [str(f) for f in frames]
        return model.predict(frame_paths, query_uv, resolution)


class TTAFlipInference(InferenceStrategy):
    """Test-time augmentation with horizontal flip."""
    
    def run(self, model: ModelPort, frames: list[Path],
            query_uv: np.ndarray, resolution: int) -> np.ndarray:
        """Mirror the window along x, decode, then mirror predictions back.

        The flip acts on the image plane (query u -> 1-u) and on the world
        x axis of the predicted point maps, so the output lives in the
        original coordinate frame and can be averaged with other strategies.
        """
        from PIL import Image as PILImage
        
        frame_paths = [str(f) for f in frames]
        
        # Forward pass
        xyz_fwd = model.predict(frame_paths, query_uv, resolution)
        
        # Flipped pass
        tmp_dir = tempfile.mkdtemp(prefix="tta_flip_")
        flipped_paths = []
        
        try:
            for p in frame_paths:
                img = PILImage.open(p).transpose(PILImage.FLIP_LEFT_RIGHT)
                out_p = os.path.join(tmp_dir, os.path.basename(p))
                img.save(out_p)
                flipped_paths.append(out_p)
            
            query_uv_flip = query_uv.copy()
            query_uv_flip[:, 0] = (ChallengeConstants.FRAME_W - 1) - query_uv_flip[:, 0]
            
            xyz_flip = model.predict(flipped_paths, query_uv_flip, resolution)
            xyz_flip[:, :, 0] = -xyz_flip[:, :, 0]
        finally:
            # Cleanup
            for p in flipped_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass
        
        return (xyz_fwd + xyz_flip) * 0.5


# =============================================================================
# APPLICATION LAYER - Use Cases / Orchestration
# =============================================================================

class SequenceIterator:
    """Iterates over challenge sequences."""
    
    def __init__(self, benchmark_root: Path):
        self._benchmark_root = benchmark_root
    
    def __iter__(self) -> Iterator[tuple[str, Path, str, str, str]]:
        """Yield (seq_id, seq_dir, variant, scene, seq_name)."""
        for variant in ChallengeConstants.VARIANTS:
            variant_dir = self._benchmark_root / variant
            if not variant_dir.exists():
                continue
            for scene in ChallengeConstants.SCENES:
                scene_dir = variant_dir / scene
                if not scene_dir.exists():
                    continue
                png_dir = scene_dir / "png"
                if png_dir.exists():
                    seq_names = sorted(d.name for d in png_dir.iterdir() if d.is_dir())
                else:
                    seq_names = sorted(d.name for d in scene_dir.iterdir()
                                       if d.is_dir() and d.name.startswith("seq_"))
                for seq_name in seq_names:
                    seq_id = f"{variant}-{scene}-{seq_name}"
                    seq_dir = scene_dir / "png" / seq_name
                    yield seq_id, seq_dir, variant, scene, seq_name


class SequenceProcessor:
    """Processes a single sequence (application service)."""
    
    def __init__(
        self, 
        model: ModelPort, 
        query_loader: QueryLoaderPort,
        frame_loader: FrameLoaderPort
    ):
        self._model = model
        self._query_loader = query_loader
        self._frame_loader = frame_loader
    
    def process(
        self, 
        seq_dir: Path, 
        variant: str, 
        scene: str, 
        seq_name: str, 
        resolution: int,
        frame_stride: int,
        tta_flip: bool
    ) -> dict[str, np.ndarray]:
        """Process a single sequence."""
        seq_key = f"{variant}/{scene}/{seq_name}"
        frame_paths = self._frame_loader.collect_frames(seq_dir, seq_name)
        
        if not frame_paths:
            raise FileNotFoundError(f"No frames found for {seq_key}")
        
        query_uv, _ = self._query_loader.load_for_sequence(seq_key)
        n_queries = len(query_uv)
        n_frames = len(frame_paths)
        
        # Select scored frames
        scored = [t for t in ChallengeConstants.SCORED_FRAME_INDICES if t < n_frames]
        if 0 not in scored:
            scored = [0] + scored
        
        # Temporal sampling
        if frame_stride != 6:
            sampled = sorted(set(range(0, n_frames, frame_stride)) | set(scored))
        else:
            sampled = scored
        
        scored_pos = [sampled.index(t) for t in scored]
        paths_sampled = [frame_paths[t] for t in sampled]
        
        # Select inference strategy
        strategy: InferenceStrategy
        if tta_flip:
            strategy = TTAFlipInference()
        else:
            strategy = StandardInference()
        
        xyz = strategy.run(self._model, paths_sampled, query_uv, resolution)
        
        # Build results
        results: dict[str, np.ndarray] = {}
        for local_t, global_t in enumerate(scored):
            pos = scored_pos[local_t]
            for q_idx in range(n_queries):
                row_id = SubmissionFormatter.make_id(variant, scene, seq_name, q_idx, global_t)
                results[row_id] = xyz[q_idx, pos]
        
        return results


class GallileoPipeline:
    """Main application: orchestrates the inference pipeline.
    
    Uses dependency injection for all external dependencies.
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        model: ModelPort | None = None,
        query_loader: QueryLoaderPort | None = None,
        frame_loader: FrameLoaderPort | None = None,
        submission_writer: SubmissionWriterPort | None = None,
    ):
        self._config = config
        
        # Dependency injection with defaults
        self._model = model or FourRCAdapter()
        self._query_loader = query_loader or CSVQueryLoader(config.queries_csv)
        self._frame_loader = frame_loader or FileSystemFrameLoader()
        self._submission_writer = submission_writer or CSVSubmissionWriter()
        
        self._processor = SequenceProcessor(
            self._model, self._query_loader, self._frame_loader
        )
    
    def setup(self) -> None:
        """Initialize the pipeline (load model)."""
        logger.info("Device: %s | Resolution: %d | Stride: %d",
                    self._config.device, self._config.resolution, self._config.frame_stride)
        self._model.load(self._config.checkpoint, self._config.device)
    
    def run_single(self) -> None:
        """Run single-pass inference."""
        self.setup()
        
        sequences = list(SequenceIterator(self._config.benchmark_root))
        seqs_to_run = sequences[self._config.seq_start:self._config.seq_end]
        logger.info("Running %d sequences", len(seqs_to_run))
        
        all_results: dict[str, np.ndarray] = {}
        
        for i, (seq_id, seq_dir, variant, scene, seq_name) in enumerate(seqs_to_run):
            logger.info("[%d/%d] %s", i + 1, len(seqs_to_run), seq_id)
            try:
                results = self._processor.process(
                    seq_dir, variant, scene, seq_name,
                    self._config.resolution,
                    self._config.frame_stride,
                    self._config.tta_flip
                )
                all_results.update(results)
            except Exception as exc:
                logger.error("FAILED %s: %s", seq_id, exc)
        
        self._submission_writer.write(all_results, self._config.output)
        logger.info("Done. Output: %s", self._config.output)
    
    def run_ensemble(self, ensemble_config: EnsembleConfig) -> None:
        """Run ensemble inference."""
        self.setup()
        
        sequences = list(SequenceIterator(self._config.benchmark_root))
        seqs_to_run = sequences[self._config.seq_start:self._config.seq_end]
        logger.info("Running ensemble with %d components on %d sequences",
                    len(ensemble_config.components), len(seqs_to_run))
        
        all_predictions: list[dict[str, np.ndarray]] = []
        weights: list[float] = []
        
        for comp in ensemble_config.components:
            logger.info("=== Component: %s (weight=%.0f%%) ===", 
                        comp.name, comp.weight * 100)
            
            comp_predictions: dict[str, np.ndarray] = {}
            for i, (seq_id, seq_dir, variant, scene, seq_name) in enumerate(seqs_to_run):
                logger.info("[%d/%d] %s", i + 1, len(seqs_to_run), seq_id)
                try:
                    results = self._processor.process(
                        seq_dir, variant, scene, seq_name,
                        comp.resolution,
                        comp.frame_stride,
                        comp.tta_flip
                    )
                    comp_predictions.update(results)
                except Exception as exc:
                    logger.error("FAILED %s: %s", seq_id, exc)
            
            all_predictions.append(comp_predictions)
            weights.append(comp.weight)
        
        # Merge predictions
        logger.info("Merging %d components...", len(all_predictions))
        merged = EnsembleMerger.merge(all_predictions, weights)
        
        self._submission_writer.write(merged, self._config.output)
        logger.info("Done. Output: %s", self._config.output)


# =============================================================================
# CLI - Entry Point
# =============================================================================

def main() -> None:
    """Main entry point."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    parser = argparse.ArgumentParser(
        description="Gallileo-4D: Frozen Backbone Ensemble for Dynamic 4D Reconstruction"
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to 4RC checkpoint")
    parser.add_argument("--benchmark_root", type=Path, required=True,
                        help="Path to challenge data")
    parser.add_argument("--queries_csv", type=Path, default=Path("data/queries.csv"))
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--frame_stride", type=int, default=6)
    parser.add_argument("--tta_flip", action="store_true")
    parser.add_argument("--seq_start", type=int, default=0)
    parser.add_argument("--seq_end", type=int, default=128)
    parser.add_argument("--ensemble", action="store_true",
                        help="Run full ensemble (reproduces 0.58356)")
    args = parser.parse_args()
    
    config = InferenceConfig(
        checkpoint=args.checkpoint,
        benchmark_root=args.benchmark_root,
        queries_csv=args.queries_csv,
        output=args.out,
        device=args.device,
        resolution=args.resolution,
        frame_stride=args.frame_stride,
        tta_flip=args.tta_flip,
        seq_start=args.seq_start,
        seq_end=args.seq_end,
    )
    
    pipeline = GallileoPipeline(config)
    
    if args.ensemble:
        pipeline.run_ensemble(EnsembleConfig.default())
    else:
        pipeline.run_single()


if __name__ == "__main__":
    main()
