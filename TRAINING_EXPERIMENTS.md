# Training Experiments Documentation

This document details all 13 training configurations we tested during the PhysAI Dynamic 4D Reconstruction Challenge. **12 of 13 configurations degraded the challenge score**, making this a documented negative result that we believe is valuable for the community.

## Key Finding: Local Validation Inverts

The most important finding is that **local validation scores inverted relative to challenge scores**. Configurations that improved local validation often degraded challenge performance.

![Training Results](assets/fig3_training_results.png)

*Figure 3 from paper: (a) Local validation vs challenge APD changes. 11 runs land in the "local better, challenge worse" quadrant. (b) Absolute challenge scores for all 13 runs.*

## Why Training Failed

The challenge structure created a distribution shift problem:

| Attribute | Training Data (syn4d_sim) | Test Data (challenge) |
|-----------|---------------------------|------------------------|
| Frames/clip | 50 | 192 |
| Resolution | 1024×1024 | 1280×720 |
| Aspect ratio | 1:1 | 16:9 |
| Cameras/clip | 8 (multi-view) | 1 (monocular) |
| Rendering variants | 1 (sim only) | 4 (sim, og, mixed, mixed_no_bedlam) |
| **Test coverage** | **25%** | **100%** |

Only 25% of the test set (sim variant) had a training analogue. The other 75% required cross-variant generalization with zero direct supervision.

## All 13 Training Configurations

### Table 3: Complete Results

| Run | Adapted Part | LR | Epochs | Local APD | Challenge APD | Δ vs Frozen |
|-----|--------------|-----|--------|-----------|---------------|-------------|
| **Naïve Fine-tuning** |
| train_v1 | full network | 5×10⁻⁵ | 1 | 0.42 | 0.48 | -0.032 |
| **train_v2** | full network | 5×10⁻⁶ | 1 | 0.44 | **0.54** | **+0.028** ✓ |
| train_v3 | full network | 5×10⁻⁶ | 3 | 0.46 | 0.52 | +0.008 |
| train_v4 | full, scale-dec. | 1×10⁻⁵ | 1 | 0.45 | 0.49 | -0.022 |
| train_v4d | full, scale-dec. | 1×10⁻⁵ | 1 | 0.43 | 0.48 | -0.032 |
| train_v5 | full network | 1×10⁻⁷ | 1 | 0.40 | 0.50 | -0.012 |
| **Aggressive Schedules** |
| paper_A | full network | 1×10⁻⁴ | 5 | **0.48** | 0.46 | -0.052 ⚠️ |
| paper_B | full network | 1×10⁻⁶ | 2 | 0.44 | 0.51 | -0.002 |
| paper_C | full, cosine | 5×10⁻⁶ | 2 | 0.45 | 0.50 | -0.012 |
| **Parameter-Efficient** |
| lora_32 | LoRA, rank 16 | 1×10⁻⁴ | 1 | 0.41 | 0.47 | -0.042 |
| lora_64 | LoRA, rank 32 | 5×10⁻⁵ | 1 | 0.42 | 0.48 | -0.032 |
| **Lightweight Refinement** |
| refiner | added MLP head | 1×10⁻³ | 1 | 0.39 | 0.45 | **-0.062** ❌ |
| decoder | decoder only | 5×10⁻⁵ | 1 | 0.43 | 0.49 | -0.022 |
| **Frozen Ensemble** |
| none | frozen + ensemble | - | - | - | **0.553** | **+0.041** ✓✓ |

**Legend:**
- ✓ = Only successful training run
- ⚠️ = Best local score but worst challenge score (the inversion)
- ❌ = Worst overall result
- ✓✓ = Final submission (no training)

## Detailed Analysis

### 1. Naïve Fine-tuning

Learning rate behaves monotonically in the wrong direction:
- 5×10⁻⁵ costs -0.032
- 5×10⁻⁶ gains +0.028 (only success)
- 1×10⁻⁷ costs -0.012 (learns nothing, perturbs batch stats)

Extending train_v2 from 1 to 3 epochs (train_v3) raises local from 0.44→0.46 but lowers challenge from 0.54→0.52. This is the cleanest instance of the inversion.

### 2. Aggressive Schedules (paper_A)

**paper_A is the most instructive failure:**
- Learning rate: 1×10⁻⁴ (highest tested)
- Epochs: 5 (most tested)
- Local score: **0.48** (best of all runs)
- Challenge score: **0.46** (second-worst)

If selecting on local validation, this is the checkpoint one would ship. It would have been catastrophic.

### 3. Parameter-Efficient Adaptation (LoRA)

We expected LoRA to help since restricting updates to a low-rank subspace is a standard defense against forgetting. It did not:
- Rank 16 at 1×10⁻⁴: -0.042
- Rank 32 at 5×10⁻⁵: -0.032

Both worse than full fine-tuning at comparable learning rates. The problem is not the capacity of the update but its objective—a low-rank direction that reduces loss on sim is still a direction away from features the other three variants need.

### 4. Lightweight Refinement

- **Refiner (MLP head):** Worst result (-0.062)
- **Decoder only:** -0.022

Both confirm the damage does not require touching the encoder.

## Scale Collapse at High Resolution

![Resolution Study](assets/fig4_resolution.png)

*Figure 4: (a) Local and challenge scores vs inference resolution. (b) At 1008px, median scale collapses from 3.5 to 0.117 (30× smaller).*

Raising inference resolution from 512px to 1008px destroys predictions:
- Local score: 0.398 → 0.121
- Challenge score: 0.540 → 0.476
- Median scale: 3.5 → 0.117 (30× collapse)

The checkpoint was never trained at that resolution. Adding 1008px to our ensemble lowered the final score from 0.55345 to 0.54923.

## Loss Function

We used a differentiable surrogate of the APD metric (Eq. 3):

```python
L_soft = 1 - (1/|T|) * Σ_δ Σ_i σ((δ - ||P_aln - P*||) / τ)
```

With τ=0.05 and scale-decoupled formulation (Eq. 4) to stabilize optimization:

```python
s = <sg(P̂), P*> / ||sg(P̂)||²
P̂_aln = P̂ · sg(s)
```

where sg(·) is stop-gradient.

## Compute Cost

- **Training:** ~1,000 GPU-hours → produced 1 usable artifact (train_v2 heads)
- **Inference:** 168 GPU-hours → produced the submission

The frozen ensemble recovered +0.041 APD, more than any training run achieved, at zero training cost.

## Operating Rule Derived

From our calibration experiments (Table 2 in paper):

> **Local validation is a regression veto, never a promotion gate.** A configuration that scores worse locally is blocked. A configuration that scores equal or better becomes eligible for evaluation on the public split; it is never assumed to be an improvement.

This rule is expensive to follow but correct whenever the validation split covers a minority of the evaluation distribution.

## Code

The training code is available in `training/train.py`. It documents all 13 configurations with their exact hyperparameters. The code is provided for reproducibility of the negative results, not as a recommended approach.

```bash
# Example (documents the failed approach)
python training/train.py --config train_v2 --checkpoint /path/to/4rc.pt --data_root /path/to/syn4d_sim

# The winning approach uses NO training:
python -m gallileo4d.run --query_csv query.csv --frames_dir frames/ --output submission.csv
```

## Conclusion

The lesson is about measurement rather than architecture: **a validation signal covering a minority of the evaluation distribution stops estimating performance and starts estimating overfitting to that minority**, and the two are hard to tell apart from inside.

Our final submission used:
- **0 gradient updates**
- **3 frozen decoding configurations** (stride-3, TTA flip, stride-1)
- **Convex weighting** (0.60, 0.25, 0.15)
- **Result:** 3rd place, 0.58356 APD
