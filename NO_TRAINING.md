# Why No Training Code?

This document explains why our submission contains no training code.

## Summary

**We performed no training.** Our winning submission (0.58356 APD, 3rd place) uses a frozen 4RC backbone with inference-time ensembling only.

## The Experiment

We attempted 13 fine-tuning configurations on the released training data (syn4d_sim). **12 of 13 degraded the challenge score**, while simultaneously improving local validation.

| Run | Method | Local Score | Challenge Score | Δ vs Frozen |
|-----|--------|-------------|-----------------|-------------|
| train_v1 | Full network, LR=5×10⁻⁵ | 0.42 | 0.48 | **-0.032** |
| train_v2 | Full network, LR=5×10⁻⁶ | 0.44 | 0.54 | +0.028 |
| train_v3 | Full network, 3 epochs | 0.46 | 0.52 | +0.008 |
| paper_A | Full network, LR=1×10⁻⁴ | **0.48** | 0.46 | **-0.052** |
| lora_32 | LoRA rank 16 | 0.41 | 0.47 | **-0.042** |
| lora_64 | LoRA rank 32 | 0.42 | 0.48 | **-0.032** |
| refiner | Added MLP head | 0.39 | 0.45 | **-0.062** |
| decoder | Decoder only | 0.43 | 0.49 | **-0.022** |
| ... | ... | ... | ... | ... |
| **frozen** | **No training** | — | **0.553** | **+0.041** |

## Why Fine-Tuning Failed

The training data covers only **25%** of the evaluation distribution:

| Attribute | Training | Evaluation |
|-----------|----------|------------|
| Rendering variants | 1 (sim) | 4 (og, sim, mixed, mixed_no_bedlam) |
| Coverage | 25% | 100% |

Fine-tuning on the sim variant damages the pre-trained features that the other 75% relies on. This is a textbook case of **distribution shift** causing **catastrophic forgetting**.

## The Solution

Instead of training, we spent our compute budget on **inference-time diversity**:

1. **Stride-3**: Long temporal context (144 frames per window)
2. **TTA Flip**: Horizontal flip augmentation
3. **Stride-1**: Dense temporal sampling (48 frames per window)

The ensemble of these three frozen passes achieved **+0.041 APD** over the baseline—more than any training run.

## Conclusion

Our submission demonstrates that **not training** can be the optimal strategy when:

1. Training data covers a minority of the evaluation distribution
2. Local validation cannot detect damage to out-of-distribution performance
3. The pre-trained backbone already has strong general features

For these reasons, we provide inference code only. The "training code" is this document explaining why we didn't train.

## References

- Kumar et al., "Fine-tuning can distort pretrained features and underperform out-of-distribution" (ICLR 2022)
- Wortsman et al., "Robust fine-tuning of zero-shot models" (CVPR 2022)
