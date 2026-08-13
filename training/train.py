#!/usr/bin/env python3
"""Training script for Gallileo-4D experiments.

WARNING: This code documents our FAILED training attempts.
12 of 13 configurations degraded the challenge score.
We include it for reproducibility and as a negative result.

The winning submission uses NO training - see gallileo4d/pipeline.py instead.

Training configurations tested (Table 3 from paper):
- train_v1: full network, LR=5e-5, 1 epoch → -0.032 APD
- train_v2: full network, LR=5e-6, 1 epoch → +0.028 APD (only success)
- train_v3: full network, LR=5e-6, 3 epochs → +0.008 APD
- train_v4: full + scale-decoupled, LR=1e-5, 1 epoch → -0.022 APD
- train_v5: full network, LR=1e-7, 1 epoch → -0.012 APD
- paper_A: full network, LR=1e-4, 5 epochs → -0.052 APD (worst)
- paper_B: full network, LR=1e-6, 2 epochs → -0.002 APD
- paper_C: full + cosine, LR=5e-6, 2 epochs → -0.012 APD
- lora_32: LoRA rank 16, LR=1e-4, 1 epoch → -0.042 APD
- lora_64: LoRA rank 32, LR=5e-5, 1 epoch → -0.032 APD
- refiner: added MLP head, LR=1e-3, 1 epoch → -0.062 APD (worst overall)
- decoder: decoder only, LR=5e-5, 1 epoch → -0.022 APD

Key finding: Local validation INVERTED - runs that improved local score
often degraded challenge score. paper_A achieved best local (0.48) but
worst challenge (0.46).
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR

logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    name: str
    learning_rate: float
    epochs: int
    adapted_part: Literal["full", "decoder", "lora", "refiner"]
    lora_rank: int = 0
    scale_decoupled: bool = False
    scheduler: Literal["onecycle", "cosine", "constant"] = "onecycle"
    batch_size: int = 1
    tau: float = 0.05  # Temperature for soft APD loss


# All 13 training configurations from Table 3
TRAINING_CONFIGS = {
    # Naïve fine-tuning
    "train_v1": TrainingConfig("train_v1", 5e-5, 1, "full"),
    "train_v2": TrainingConfig("train_v2", 5e-6, 1, "full"),  # Only success!
    "train_v3": TrainingConfig("train_v3", 5e-6, 3, "full"),
    "train_v4": TrainingConfig("train_v4", 1e-5, 1, "full", scale_decoupled=True),
    "train_v4d": TrainingConfig("train_v4d", 1e-5, 1, "full", scale_decoupled=True),
    "train_v5": TrainingConfig("train_v5", 1e-7, 1, "full"),
    
    # Aggressive schedules
    "paper_A": TrainingConfig("paper_A", 1e-4, 5, "full"),  # Best local, worst challenge!
    "paper_B": TrainingConfig("paper_B", 1e-6, 2, "full"),
    "paper_C": TrainingConfig("paper_C", 5e-6, 2, "full", scheduler="cosine"),
    
    # Parameter-efficient adaptation
    "lora_32": TrainingConfig("lora_32", 1e-4, 1, "lora", lora_rank=16),
    "lora_64": TrainingConfig("lora_64", 5e-5, 1, "lora", lora_rank=32),
    
    # Lightweight refinement
    "refiner": TrainingConfig("refiner", 1e-3, 1, "refiner"),  # Worst overall!
    "decoder": TrainingConfig("decoder", 5e-5, 1, "decoder"),
}

# Results from Table 3 (for validation)
EXPECTED_RESULTS = {
    # name: (local_score, challenge_score, delta_vs_frozen)
    "train_v1": (0.42, 0.48, -0.032),
    "train_v2": (0.44, 0.54, +0.028),
    "train_v3": (0.46, 0.52, +0.008),
    "train_v4": (0.45, 0.49, -0.022),
    "train_v4d": (0.43, 0.48, -0.032),
    "train_v5": (0.40, 0.50, -0.012),
    "paper_A": (0.48, 0.46, -0.052),
    "paper_B": (0.44, 0.51, -0.002),
    "paper_C": (0.45, 0.50, -0.012),
    "lora_32": (0.41, 0.47, -0.042),
    "lora_64": (0.42, 0.48, -0.032),
    "refiner": (0.39, 0.45, -0.062),
    "decoder": (0.43, 0.49, -0.022),
}


# =============================================================================
# Loss Functions
# =============================================================================

class SoftAPDLoss(nn.Module):
    """Differentiable surrogate of the APD metric (Eq. 3 from paper).
    
    L_soft = 1 - (1/|T|) * sum_delta sum_i sigmoid((delta - ||P_aln - P*||) / tau)
    
    where tau=0.05 is the temperature.
    """
    
    def __init__(self, thresholds: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0), tau: float = 0.05):
        super().__init__()
        self.thresholds = thresholds
        self.tau = tau
    
    def forward(
        self, 
        pred: torch.Tensor,  # (N, 3) predicted positions
        target: torch.Tensor,  # (N, 3) ground truth positions
        scale_decoupled: bool = False
    ) -> torch.Tensor:
        """Compute soft APD loss.
        
        Args:
            pred: Predicted 3D positions
            target: Ground truth 3D positions
            scale_decoupled: If True, detach scale computation (Eq. 4)
        """
        # Compute scale factor (Eq. 1)
        if scale_decoupled:
            # Detach scale to stabilize training (Eq. 4)
            with torch.no_grad():
                pred_norm = torch.norm(pred, dim=-1, keepdim=True).clamp(min=1e-6)
                target_norm = torch.norm(target, dim=-1, keepdim=True).clamp(min=1e-6)
                scale = torch.median(target_norm / pred_norm)
            pred_aligned = pred * scale.detach()
        else:
            pred_norm = torch.norm(pred, dim=-1, keepdim=True).clamp(min=1e-6)
            target_norm = torch.norm(target, dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.median(target_norm / pred_norm)
            pred_aligned = pred * scale
        
        # Compute distances
        distances = torch.norm(pred_aligned - target, dim=-1)  # (N,)
        
        # Soft APD over thresholds
        apd = 0.0
        for delta in self.thresholds:
            # Sigmoid approximation of indicator function
            within = torch.sigmoid((delta - distances) / self.tau)
            apd = apd + within.mean()
        
        apd = apd / len(self.thresholds)
        
        # Loss = 1 - APD (we want to maximize APD)
        return 1.0 - apd


# =============================================================================
# LoRA Implementation
# =============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer (Hu et al., 2022).
    
    Adds trainable low-rank matrices A and B to a frozen linear layer:
    W' = W + BA, where B is (out, rank) and A is (rank, in)
    """
    
    def __init__(self, linear: nn.Linear, rank: int = 16, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))
        
        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output + low-rank adaptation
        out = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return out + self.alpha * lora_out


def apply_lora(model: nn.Module, rank: int = 16) -> nn.Module:
    """Apply LoRA to all linear layers in the model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, LoRALayer(module, rank=rank))
    return model


# =============================================================================
# Refiner Head
# =============================================================================

class RefinerHead(nn.Module):
    """MLP refinement head added on top of frozen backbone.
    
    This was our worst-performing configuration (-0.062 APD).
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine 3D predictions."""
        return x + self.mlp(x)  # Residual connection


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Training loop for fine-tuning experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader,
        val_loader,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.criterion = SoftAPDLoss(tau=config.tau)
    
    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer based on adapted part."""
        if self.config.adapted_part == "full":
            params = self.model.parameters()
        elif self.config.adapted_part == "decoder":
            # Only decoder parameters
            params = [p for n, p in self.model.named_parameters() if "decoder" in n]
        elif self.config.adapted_part == "lora":
            # Only LoRA parameters
            params = [p for n, p in self.model.named_parameters() if "lora" in n]
        elif self.config.adapted_part == "refiner":
            # Only refiner head parameters
            params = [p for n, p in self.model.named_parameters() if "refiner" in n]
        else:
            params = self.model.parameters()
        
        return AdamW(params, lr=self.config.learning_rate)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs
        
        if self.config.scheduler == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
            )
        elif self.config.scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=total_steps)
        else:
            return None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            pred = self.model(batch["input"].to(self.device))
            target = batch["target"].to(self.device)
            
            loss = self.criterion(
                pred, target, 
                scale_decoupled=self.config.scale_decoupled
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> float:
        """Compute validation APD."""
        self.model.eval()
        total_apd = 0.0
        
        for batch in self.val_loader:
            pred = self.model(batch["input"].to(self.device))
            target = batch["target"].to(self.device)
            
            # Compute actual APD (not soft)
            apd = self._compute_apd(pred, target)
            total_apd += apd
        
        return total_apd / len(self.val_loader)
    
    def _compute_apd(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute actual APD metric."""
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # Scale alignment
        pred_norm = np.linalg.norm(pred_np, axis=-1, keepdims=True)
        target_norm = np.linalg.norm(target_np, axis=-1, keepdims=True)
        scale = np.median(target_norm / np.clip(pred_norm, 1e-6, None))
        pred_aligned = pred_np * scale
        
        # APD over thresholds
        distances = np.linalg.norm(pred_aligned - target_np, axis=-1)
        apd = 0.0
        for delta in [0.1, 0.3, 0.5, 1.0]:
            apd += (distances < delta).mean()
        
        return apd / 4.0
    
    def train(self) -> dict:
        """Run full training."""
        logger.info(f"Starting training: {self.config.name}")
        logger.info(f"  LR: {self.config.learning_rate}, Epochs: {self.config.epochs}")
        logger.info(f"  Adapted: {self.config.adapted_part}")
        
        best_val = 0.0
        history = {"train_loss": [], "val_apd": []}
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_apd = self.validate()
            
            history["train_loss"].append(train_loss)
            history["val_apd"].append(val_apd)
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                       f"loss={train_loss:.4f}, val_apd={val_apd:.4f}")
            
            if val_apd > best_val:
                best_val = val_apd
        
        logger.info(f"Training complete. Best val APD: {best_val:.4f}")
        return history


# =============================================================================
# Main
# =============================================================================

def main():
    """Run training experiment."""
    parser = argparse.ArgumentParser(description="Gallileo-4D Training (Negative Results)")
    parser.add_argument("--config", type=str, required=True, choices=list(TRAINING_CONFIGS.keys()),
                        help="Training configuration name")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to 4RC checkpoint")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Path to syn4d_sim training data")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"),
                        help="Output directory for checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    config = TRAINING_CONFIGS[args.config]
    logger.info(f"Running configuration: {config.name}")
    logger.info(f"Expected result: {EXPECTED_RESULTS[config.name]}")
    
    # Note: Actual training requires 4RC model and data loaders
    # This script documents the configurations used
    logger.warning("This training code documents FAILED experiments.")
    logger.warning("The winning submission uses NO training.")
    logger.warning("See gallileo4d/pipeline.py for the actual solution.")


if __name__ == "__main__":
    main()
