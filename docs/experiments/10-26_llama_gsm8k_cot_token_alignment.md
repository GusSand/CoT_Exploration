# CoT Token Alignment Experiment - LLaMA GSM8K

**Date**: October 26, 2025
**Model**: LLaMA 1B (CODI)
**Dataset**: GSM8K
**Experiment**: Training Tuned Lens to decode continuous thought into CoT tokens

## Executive Summary

This experiment trained Tuned Lens transformations to predict Chain-of-Thought (CoT) tokens from continuous thought (CT) hidden states, using a uniform split method to assign CoT tokens to CT positions. Results show that **middle CT positions (2-3) are most aligned with CoT reasoning tokens** (22-25% Top-1 accuracy), while early positions show poor alignment (10% Top-1).

**Key Finding**: Continuous thought positions encode different types of information, with positions 2-3 showing the strongest correspondence to explicit reasoning steps.

## Motivation

Previous tuned lens experiments targeting the first response token achieved only ~18% Top-1 accuracy, suggesting that continuous thought may not be directly predicting the immediate response. This experiment tests whether CT positions instead encode the intermediate reasoning steps (CoT tokens) from the problem-solving process.

## Methodology

### Dataset Creation

**Source Data**: Enriched CODI activations with CoT reasoning steps
- Training: 76,800 samples (16 layers × 4,800 samples/layer)
- Test: 19,200 samples (16 layers × 1,200 samples/layer)

**CoT Token Assignment** (Uniform Split Method):
```python
# For each problem with N CoT tokens and 6 CT positions:
# Position i gets tokens from slice [i*N//6 : (i+1)*N//6]

Example:
  CoT tokens: ['322', '=', '322', '2', '*', '322', '=', '644', ...]
  Position 0: ['322', '=']
  Position 1: ['322', '2']
  Position 2: ['*', '322']
  ...
```

**Target Selection**: Primary target = first CoT token assigned to each position

### Training Configuration

```yaml
Model: LLaMA 1B CODI
Layers: All 16 layers (0-15)
Hidden Size: 2048
Vocab Size: 32000

Training:
  - Optimizer: AdamW
  - Learning Rate: 1e-3
  - Weight Decay: 1e-3
  - Batch Size: 32
  - Max Epochs: 50
  - Early Stopping: 5 epochs patience
  - Gradient Clipping: 1.0

Tuned Lens:
  - Layer Normalization: Yes
  - Initialize Near Identity: Yes
  - One transformation per layer (shared across positions)
```

### Evaluation Metrics

- **Top-1 Accuracy**: Exact match of predicted token
- **Top-5 Accuracy**: Target appears in top 5 predictions
- **Top-10 Accuracy**: Target appears in top 10 predictions
- **Cross-Entropy Loss**: Prediction confidence

## Results

### Training Progress

| Epoch | Val Loss | Top-1 | Top-5 | Top-10 | Notes |
|-------|----------|-------|-------|--------|-------|
| 1 | 3.9957 | 16.15% | 41.90% | 56.47% | |
| 2 | 3.9253 | 17.08% | 44.22% | 58.77% | |
| **3** | **3.9027** | **18.43%** | **45.65%** | **59.92%** | **Best Model** |
| 4 | 3.9690 | 19.03% | 45.73% | 59.96% | |
| 5 | 4.0632 | 18.19% | 44.71% | 59.93% | |
| 6 | 4.1102 | 18.42% | 45.46% | 60.39% | |
| 7 | 4.2314 | 18.62% | 45.64% | 60.31% | |
| 8 | 4.3725 | 17.70% | 44.35% | 59.49% | Early stopping triggered |

**Training Time**: 13m 15s
**Early Stopping**: Triggered after epoch 8 (patience: 5)

### Performance by Layer

| Layer | Samples | Loss | Top-1 | Top-5 | Top-10 | Notes |
|-------|---------|------|-------|-------|--------|-------|
| 0 | 1,200 | 3.76 | 17.83% | 44.67% | 56.17% | Early layer |
| 1 | 1,200 | 3.84 | 17.58% | 45.25% | 58.08% | |
| 2 | 1,200 | 3.78 | 15.92% | 44.92% | 57.33% | |
| 3 | 1,200 | 3.79 | 17.42% | 44.17% | 58.33% | |
| 4 | 1,200 | 3.68 | 18.08% | 46.75% | 60.50% | |
| 5 | 1,200 | 3.71 | 18.25% | 46.42% | 60.17% | |
| 6 | 1,200 | 3.84 | **19.50%** | 46.42% | 61.33% | Mid layer peak |
| 7 | 1,200 | 3.77 | 17.92% | 43.17% | 58.50% | |
| 8 | 1,200 | 3.85 | 17.75% | 41.67% | 57.92% | |
| **9** | 1,200 | 4.07 | **20.50%** | 46.17% | 59.08% | **Highest Top-1** |
| 10 | 1,200 | 4.00 | 18.67% | 42.08% | 57.50% | |
| 11 | 1,200 | 4.27 | 18.58% | 44.67% | 59.08% | |
| 12 | 1,200 | 3.90 | 17.08% | 46.25% | 63.58% | |
| 13 | 1,200 | 4.10 | 19.33% | 47.92% | 62.25% | |
| 14 | 1,200 | 4.02 | 20.25% | 50.25% | 63.42% | |
| **15** | 1,200 | 4.06 | 20.17% | 49.58% | **65.50%** | **Final layer** |

**Key Observations**:
- Layer 9 shows surprisingly high Top-1 accuracy (20.50%)
- Layer 15 (final) achieves best Top-10 accuracy (65.50%)
- Performance relatively consistent across layers (17-20% Top-1)

### Performance by Position ⭐ KEY FINDING

| Position | Samples | Loss | Top-1 | Top-5 | Top-10 | Interpretation |
|----------|---------|------|-------|-------|--------|----------------|
| **0** | 3,200 | 4.48 | **10.00%** | 31.56% | 48.03% | **Poorest** - Early abstraction |
| 1 | 3,200 | 4.12 | 17.53% | 42.16% | 55.00% | Low-moderate |
| **2** | 3,200 | 3.17 | 22.47% | **60.06%** | **74.66%** | **Best Top-5/10** - Core reasoning |
| **3** | 3,200 | 3.58 | **24.78%** | 49.25% | 64.84% | **Highest Top-1** - Peak alignment |
| 4 | 3,200 | 4.25 | 16.72% | 44.66% | 55.00% | Late reasoning |
| 5 | 3,200 | 3.82 | 19.06% | 46.19% | 62.00% | Final position |

**Critical Insight**:
- **Positions 2-3** show dramatically better performance than other positions
- Position 0 performs 2.5× worse than position 3 (10% vs 25%)
- This suggests **CT positions specialize**: early positions hold abstract representations, middle positions encode concrete reasoning steps

### Sample Predictions Analysis

**Sample 1** (Layer 2, Position 4):
```
CoT Steps: ['9*6000=54000']
Target: '=' (ID: 28)
Predicted: '/' (28.42%), '+' (27.11%), '=' (25.09%)
Correct: No, but in Top-5
```
Shows confusion between operators - position 4 may not strongly encode specific operators.

**Sample 2** (Layer 3, Position 3):
```
CoT Steps: ['4-1=3', '3*5=15', '15+2=17', '17*2=34']
Target: ' ' (ID: 220, space token)
Predicted: ' ' (80.20%)
Correct: Yes
```
High confidence on space token - position 3 encodes token boundaries well.

**Sample 3** (Layer 10, Position 0):
```
CoT Steps: ['1000*10*.01=100', '1000-100=900', ...]
Target: '100' (ID: 1041)
Predicted: '100' (66.69%)
Correct: Yes
```
Even position 0 at later layers can decode specific numbers when they're prominent.

## Comparison with Previous Experiments

### Layer-15-Only vs All-Layers Training

| Approach | Training Samples | Top-1 | Top-10 | Notes |
|----------|------------------|-------|--------|-------|
| Layer 15 Only | 4,800 | 19.24% | 64.39% | Severe overfitting (73% train vs 19% val) |
| All Layers | 76,800 | 18.43% | 59.92% | 16× more data, slightly worse |

**Unexpected Result**: More training data from all layers did NOT improve performance. Possible explanations:
1. Different layers encode fundamentally different information
2. Mixing layer representations dilutes the signal from later layers
3. Layer-specific learning might be needed (position-specific transformations)

### Standard Tuned Lens (First Response Token)

| Target | Top-1 | Top-10 | Dataset |
|--------|-------|--------|---------|
| First Response Token | ~18% | ~60% | Layer 15, 4,800 samples |
| CoT Tokens (this exp) | 18.43% | 59.92% | All layers, 76,800 samples |

**Conclusion**: Similar performance suggests continuous thought may encode **reasoning process** rather than specific output tokens.

## Analysis and Interpretation

### Position Specialization Hypothesis

The dramatic performance difference across positions suggests:

```
Position 0 (10% acc):   Abstract problem representation
Position 1 (17.5% acc): Problem decomposition
Position 2 (22.5% acc): Core reasoning operations ⭐
Position 3 (24.8% acc): Intermediate results ⭐
Position 4 (16.7% acc): Result aggregation
Position 5 (19.1% acc): Final answer preparation
```

**Evidence**:
- Positions 2-3 show **2.5× better** performance than position 0
- Position 2 has highest Top-5/Top-10 (60%/75%) - suggests broader reasoning
- Position 3 has highest Top-1 (25%) - suggests specific intermediate values

### Why Uniform Split May Be Suboptimal

The uniform split assumes equal distribution of reasoning across positions, but results suggest:
- Early positions (0-1) handle **abstraction** (few explicit tokens)
- Middle positions (2-3) handle **concrete reasoning** (many explicit tokens)
- Late positions (4-5) handle **consolidation** (intermediate explicit tokens)

**Better approach**: Weight CoT token assignment by position's role in reasoning process.

### Layer Analysis

Layer 9's peak performance (20.5% Top-1) is intriguing:
- Not the final layer (15)
- Not a typical "middle layer" (8)
- May represent optimal balance between abstraction and concreteness

Layer 15's strong Top-10 (65.5%) suggests:
- Final layer still valuable for reasoning
- May encode multiple plausible reasoning paths (hence high Top-10)

## Limitations

1. **Uniform Split Assumption**: May not match actual CT encoding strategy
2. **Single Token Target**: Only uses first CoT token per position
3. **Position Sharing**: All positions use same layer transformation
4. **No Position Context**: Model doesn't know which position it's decoding
5. **Limited Training Data**: Only 1,000 problems total

## Future Directions

### Immediate Next Steps

1. **Position-Specific Transformations**
   - Train separate (W, b) for each (layer, position) pair
   - Expected improvement: Capture position specialization
   - Cost: 16 layers × 6 positions = 96 transformation matrices

2. **Weighted CoT Token Assignment**
   ```python
   # Instead of uniform split:
   position_weights = [0.5, 1.0, 2.0, 2.0, 1.0, 0.5]  # More tokens to positions 2-3
   ```

3. **Multi-Token Target Training**
   - Instead of single token, predict all assigned CoT tokens
   - Use sequence loss or multi-label classification

### Advanced Experiments

4. **Attention Pattern Analysis**
   - Examine which input tokens each CT position attends to
   - Correlate with CoT token alignment

5. **Causal Intervention**
   - Replace CT hidden states with Tuned Lens decoded tokens
   - Measure impact on final answer accuracy

6. **Alternative Target Selection**
   - Try middle token instead of first token
   - Try most informative token (highest TF-IDF)

## Conclusions

1. **Continuous thought positions are specialized**: Positions 2-3 show 2.5× better CoT token prediction than position 0

2. **Middle positions encode concrete reasoning**: Positions 2-3 achieve 22-25% Top-1 accuracy, suggesting they hold explicit reasoning steps

3. **Uniform split is suboptimal**: The assumption that CoT tokens distribute uniformly doesn't match observed position specialization

4. **More data ≠ better performance**: Training on all layers (76,800 samples) performed worse than layer-15-only (4,800 samples)

5. **Layer 9 is interesting**: Shows peak Top-1 accuracy (20.5%), warranting further investigation

6. **Tuned Lens shows promise**: 60-75% Top-10 accuracy for positions 2-3 suggests continuous thought does encode reasoning, just not uniformly

## Next Experiment Recommendation

**Priority 1**: Implement position-specific Tuned Lens transformations to test the position specialization hypothesis. Expected improvement: +5-10% Top-1 accuracy for positions 2-3.

## Files Generated

### Code
- `src/experiments/tuned_lens/create_cot_alignment_dataset.py` - Dataset creation
- `src/experiments/tuned_lens/train_cot_alignment.py` - Training script
- `src/experiments/tuned_lens/evaluate_cot_alignment.py` - Evaluation script

### Data
- `src/experiments/tuned_lens/data/cot_train_all_layers.pt` (603.3 MB)
- `src/experiments/tuned_lens/data/cot_test_all_layers.pt` (150.8 MB)

### Models
- `src/experiments/tuned_lens/models/cot_alignment/tuned_lens_all_layers_best.pt`

### Results
- `src/experiments/tuned_lens/results/cot_alignment/evaluation_results.json`
- `src/experiments/tuned_lens/cot_train_all_layers.log`
- `src/experiments/tuned_lens/evaluate_cot_alignment.log`

---

**Experiment Status**: ✅ Complete
**Time Investment**: ~2.5 hours (dataset creation: 1h, training: 13min, evaluation: 1h)
**Next Steps**: Position-specific transformations or weighted CoT assignment
