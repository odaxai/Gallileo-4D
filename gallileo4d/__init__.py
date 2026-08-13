"""Gallileo-4D: Frozen Backbone Ensemble for Dynamic 4D Reconstruction.

3rd Place at PhysAI Dynamic 4D Reconstruction Challenge (ECCV 2026)
Final Score: 0.58356 APD

Architecture: Hexagonal (Ports & Adapters)
"""

from gallileo4d.pipeline import (
    # Domain Layer
    InferenceConfig,
    ComponentConfig,
    EnsembleConfig,
    ChallengeConstants,
    EnsembleMerger,
    SubmissionFormatter,
    # Ports (Interfaces)
    ModelPort,
    QueryLoaderPort,
    FrameLoaderPort,
    SubmissionWriterPort,
    InferenceStrategy,
    # Adapters (Implementations)
    FourRCAdapter,
    CSVQueryLoader,
    FileSystemFrameLoader,
    CSVSubmissionWriter,
    StandardInference,
    TTAFlipInference,
    # Application Layer
    SequenceIterator,
    SequenceProcessor,
    GallileoPipeline,
)

__version__ = "1.0.0"
__author__ = "OdaxAI Research"

__all__ = [
    # Domain
    "InferenceConfig",
    "ComponentConfig", 
    "EnsembleConfig",
    "ChallengeConstants",
    "EnsembleMerger",
    "SubmissionFormatter",
    # Ports
    "ModelPort",
    "QueryLoaderPort",
    "FrameLoaderPort",
    "SubmissionWriterPort",
    "InferenceStrategy",
    # Adapters
    "FourRCAdapter",
    "CSVQueryLoader",
    "FileSystemFrameLoader",
    "CSVSubmissionWriter",
    "StandardInference",
    "TTAFlipInference",
    # Application
    "SequenceIterator",
    "SequenceProcessor",
    "GallileoPipeline",
]
