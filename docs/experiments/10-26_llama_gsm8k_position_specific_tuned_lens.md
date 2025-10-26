# Position-Specific Tuned Lens (Lite) - LLaMA GSM8K

**Date**: October 26, 2025
**Model**: LLaMA 1B (CODI)
**Dataset**: GSM8K
**Experiment**: Position-specific transformations for critical layers to improve CoT token alignment

## Executive Summary

This experiment tested whether position-specific affine transformations for critical layers [6, 9, 14, 15] could address the 2.5× performance gap observed between CT positions (position 0: 10% vs position 3: 24.78% Top-1 accuracy). **Result: NEGATIVE** - only minimal improvement (~1-2%) achieved, not enough to justify the added complexity.

**Key Finding**: Position-specific transformations for critical layers provide marginal gains but do not solve the fundamental position specialization problem.

## Motivation

Previous CoT alignment experiments (10-26_llama_gsm8k_cot_token_alignment.md) revealed dramatic performance differences across continuous thought positions:

- **Position 0**: 10.00% Top-1 accuracy (poorest)
- **Position 2**: 22.47% Top-1 accuracy
- **Position 3**: 24.78% Top-1 accuracy (best)

Hypothesis: Positions encode different types of information, requiring position-specific transformations to decode properly.

## Approach

### Lite Version Architecture

Instead of full position-specific model (403M params), implemented a hybrid approach:

**Critical Layers** [6, 9, 14, 15]: Position-specific transformations
- 4 layers × 6 positions × 2048² parameters ≈ 100M params
- Separate (W, b) for each (layer, position) pair

**Non-Critical Layers** [0-5, 7-8, 10-13]: Position-agnostic transformations
- 12 layers × 2048² parameters ≈ 50M params
- Shared transformation across all positions

**Total**: ~150M parameters (vs 403M for full, vs ~67M for baseline)

### Rationale for Critical Layers

Selected layers based on previous experiments:
- **Layer 6**: 19.50% Top-1 (mid-layer peak)
- **Layer 9**: 20.50% Top-1 (highest overall)
- **Layer 14**: 20.25% Top-1 (late layer)
- **Layer 15**: 20.17% Top-1 (final layer, 65.5% Top-10)

### Training Configuration

```yaml
Model: LLaMA 1B CODI
Layers: All 16 layers (4 position-specific, 12 position-agnostic)
Hidden Size: 2048
Vocab Size: 32000
Trainable Parameters: 151,072,768

Training:
  - Optimizer: AdamW
  - Learning Rate: 1e-3
  - Weight Decay: 1e-3
  - Batch Size: 32
  - Max Epochs: 50
  - Early Stopping: 5 epochs patience
  - Gradient Clipping: 1.0

Dataset:
  - Training: 76,800 samples (16 layers × 4,800 samples/layer)
  - Test: 19,200 samples (16 layers × 1,200 samples/layer)
  - Same data as baseline CoT alignment experiment
```

## Results

### Training Progress

| Epoch | Val Loss | Top-1 | Top-5 | Top-10 | Notes |
|-------|----------|-------|-------|--------|-------|
| **1** | **3.5909** | **19.76%** | **47.51%** | **61.28%** | **Best Model** ✓ |
| 2 | 3.6316 | 19.43% | 47.11% | 61.18% | No improvement |
| 3 | 3.7568 | 19.54% | 47.15% | 60.90% | No improvement |
| 4 | 3.8846 | 19.71% | 47.09% | 61.06% | No improvement |
| 5 | 4.0578 | 19.03% | 46.59% | 60.60% | No improvement |
| 6 | 4.2297 | 18.93% | 46.55% | 60.27% | Early stopping triggered |

**Training Time**: 11.5 minutes
**Best Model**: Epoch 1 (saved to `models/position_specific/position_specific_best.pt`)

### Comparison with Baseline

| Metric | Position-Specific (Lite) | Baseline (All-Layers) | Improvement |
|--------|-------------------------|----------------------|-------------|
| **Top-1 Accuracy** | 19.76% | 18.43% | **+1.33%** |
| **Top-5 Accuracy** | 47.51% | 45.65% | **+1.86%** |
| **Top-10 Accuracy** | 61.28% | 59.92% | **+1.36%** |
| **Validation Loss** | 3.5909 | 3.9027 | **-8.0%** |
| **Parameters** | 151M | ~67M | +2.25× |

### Analysis

**Minimal Improvement**: Only 1-2% gains across all metrics
- Not enough to justify 2.25× parameter increase
- Far from addressing the 2.5× position performance gap

**Immediate Overfitting**:
- Best model at epoch 1
- Validation loss steadily increased from epoch 1 onward
- Suggests model capacity is not the bottleneck

**Critical Observation**: Position-specific transformations did NOT enable the model to learn the specialization observed in position-level analysis (10% vs 25% gap).

## Why Did This Fail?

### Hypothesis 1: Wrong Critical Layers
Layers 6, 9, 14, 15 may not be the right layers for position specialization. The "best performing" layers in aggregate may not be where position-specific patterns emerge.

### Hypothesis 2: Insufficient Scope
Lite version only applies position-specific transforms to 4/16 layers. The specialization may require transforms across all layers.

### Hypothesis 3: Unembedding Bottleneck
All positions share the same unembedding matrix. Even with position-specific transformations, the final projection to vocabulary may wash out position-specific information.

### Hypothesis 4: Fundamental Data Issue
The uniform split method for assigning CoT tokens to positions may not match how the model actually encodes reasoning. Position 0 may genuinely not encode explicit tokens (hence low accuracy is correct, not a model limitation).

## Implications

1. **Position specialization exists** (10% vs 25% gap is real)
2. **Position-specific transformations alone don't solve it** (only 1-2% improvement)
3. **Problem likely lies elsewhere**:
   - Data assignment strategy (uniform split)
   - Shared unembedding layer
   - Fundamental mismatch between CT encoding and CoT tokens

## Comparison with Related Work

### vs. Layer-15-Only Training
- Layer 15 only: 19.24% Top-1 (severe overfitting)
- Position-specific lite: 19.76% Top-1 (slight improvement)
- Full all-layers baseline: 18.43% Top-1

Position-specific approach performs similarly to single-layer training, suggesting layer diversity matters more than position specialization.

## Conclusions

1. **Negative result**: Position-specific transformations for critical layers provide minimal benefit (~1-2%)

2. **Overfitting from epoch 1**: Suggests model capacity is not the limiting factor

3. **Position gap remains**: Did not address the 2.5× performance difference between positions

4. **Not worth the complexity**: 2.25× parameter increase for 1-2% improvement is poor ROI

5. **Rethink approach needed**: Either:
   - Fix the data assignment strategy (non-uniform split)
   - Add position-specific unembedding
   - Accept that positions encode different abstraction levels (some have few/no explicit tokens)

## Files Generated

### Code
- `src/experiments/tuned_lens/position_specific_model.py` - Hybrid position-specific/agnostic model
- `src/experiments/tuned_lens/train_position_specific.py` - Training script

### Models
- `src/experiments/tuned_lens/models/position_specific/position_specific_best.pt` - Best model checkpoint

### Logs
- `src/experiments/tuned_lens/position_specific_train.log` - Full training log

## Lessons Learned

1. **Test hypotheses cheaply first**: Lite version was right call - saved ~4 hours vs full implementation
2. **Overfitting signals**: Immediate overfitting suggests wrong direction, not insufficient capacity
3. **Marginal gains**: 1-2% improvements often not worth added complexity
4. **Listen to skepticism**: "Not sure this is gonna work" was correct

## Recommendations

**Do NOT pursue**:
- ❌ Full position-specific model (all 16 layers × 6 positions)
- ❌ Hybrid with more critical layers
- ❌ Longer training or different hyperparameters

**Consider instead**:
- ✓ Weighted CoT token assignment (more tokens to positions 2-3)
- ✓ Position-specific unembedding layers
- ✓ Alternative decoding targets (not first token, but all assigned tokens)
- ✓ Causal interventions to understand what positions actually encode

---

**Experiment Status**: ✅ Complete (Negative Result)
**Time Investment**: ~30 minutes (implementation + training)
**Next Steps**: Document and move on to alternative approaches
