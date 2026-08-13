"""Training experiments for Gallileo-4D (negative results).

WARNING: This module documents FAILED training attempts.
12 of 13 configurations degraded the challenge score.
The winning submission uses NO training.

See gallileo4d.pipeline for the actual solution.
"""

from .train import (
    TrainingConfig,
    TRAINING_CONFIGS,
    EXPECTED_RESULTS,
    SoftAPDLoss,
    LoRALayer,
    RefinerHead,
    Trainer,
)

__all__ = [
    "TrainingConfig",
    "TRAINING_CONFIGS",
    "EXPECTED_RESULTS",
    "SoftAPDLoss",
    "LoRALayer",
    "RefinerHead",
    "Trainer",
]
