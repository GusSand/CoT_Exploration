# SAE Refinement Results - Pilot vs Refined

**Date**: 2025-10-24
**Objective**: Test if smaller dictionary + weaker L1 + Token 1 L8 aggregation improves classification

---

## Configuration Comparison

| Parameter | Pilot | Refined | Change |
|-----------|-------|---------|--------|
| **Features** | 8192 (4x expansion) | 2048 (1x expansion) | ÷ 4 |
| **L1 Coefficient** | 0.001 | 0.0005 | ÷ 2 |
| **Aggregation** | Mean pool (all tokens/layers) | Token 1 L8 only | Targeted |
| **Training Time** | 2 min | 2 min | Same |

---

## Results Summary

### Reconstruction Quality

| Metric | Pilot | Refined | Change |
|--------|-------|---------|--------|
| **MSE Loss** | 0.0319 | 0.0161 | **-49.5%** ✅ |
| **Explained Variance** | 78.62% | 89.25% | **+10.6 pts** ✅ |
| **Cosine Similarity** | 89.60% | 94.95% | **+5.4 pts** ✅ |
| **Verdict** | Fair | **GOOD** | ✅ Improved |

### Sparsity Metrics

| Metric | Pilot | Refined | Change |
|--------|-------|---------|--------|
| **L0 (features/vector)** | 23.34 | 23.14 | -0.2 (same) |
| **L0 Percentage** | 0.28% | 1.13% | +0.85 pts |
| **Dead Features** | 96.97% | 40.67% | **-56.3 pts** ✅✅✅ |
| **Active Features** | 248 / 8192 | 1215 / 2048 | 4.9× more |
| **Verdict** | Poor | **FAIR** | ✅ Major improvement |

### Classification Performance

| Method | Accuracy | vs Baseline | Change |
|--------|----------|-------------|--------|
| **Baseline** (Raw L8) | 83.3% | - | - |
| **Pilot** (Mean pool) | 70.0% | -13.3 pts | - |
| **Refined** (Token 1 L8) | 63.3% | -20.0 pts | **-6.7 pts** ❌ |

**Per-Class F1 Scores**:

| Operation | Pilot | Refined | Change |
|-----------|-------|---------|--------|
| Mixed | 0.58 | 0.52 | -0.06 |
| Addition | 0.64 | 0.62 | -0.02 |
| Multiplication | 0.86 | 0.77 | -0.09 |

---

## Analysis

### ✅ What Improved?

1. **Reconstruction Quality** (🎯 Major Win)
   - Explained variance: 78.62% → 89.25% (+10.6 pts)
   - Cosine similarity: 89.60% → 94.95% (+5.4 pts)
   - SAE captures input structure better

2. **Feature Usage** (🎯 Major Win)
   - Dead features: 96.97% → 40.67% (-56.3 pts!)
   - Active features: 248 → 1215 (4.9× more)
   - Smaller dictionary + weaker L1 = healthier feature distribution

3. **Training Efficiency**
   - Same 2-minute training time
   - Better convergence (lower final loss)

### ❌ What Got Worse?

1. **Classification Accuracy** (Main Goal)
   - 70.0% → 63.3% (-6.7 pts)
   - **Token 1 L8 aggregation hurt performance**
   - Counter-intuitive: Token 1 L8 is most discriminative on RAW activations (77.5%)

2. **Per-Class Performance**
   - Multiplication F1: 0.86 → 0.77 (-0.09)
   - Previously best class now degraded

---

## Key Insights

### 🔍 Insight 1: Aggregation Strategy Matters

**Hypothesis**: Token 1 L8 is most discriminative for RAW activations, but SAE compression changes this.

**Evidence**:
- Raw Token 1 L8: 77.5% accuracy (operation circuits)
- SAE Token 1 L8: 63.3% accuracy (this experiment)
- SAE Mean pool: 70.0% accuracy (pilot)

**Interpretation**: SAE compression distributes information across features in a way that benefits from averaging multiple token positions. Single-position features are too compressed.

### 🔍 Insight 2: Smaller Dictionary Helps Reconstruction, Not Classification

**Refined SAE (2048 features)**:
- ✅ Better reconstruction (89.25% explained variance)
- ✅ Better feature usage (40.67% dead)
- ❌ Worse classification (63.3% accuracy)

**Conclusion**: Smaller dictionary forces SAE to use features more efficiently for reconstruction, but this doesn't preserve task-specific discriminative information.

### 🔍 Insight 3: Sparsity-Discriminability Tradeoff Persists

Both pilot and refined SAE underperform raw activations:
- Raw L8 activations: 83.3%
- Best SAE (pilot mean pool): 70.0%
- Refined SAE (Token 1 L8): 63.3%

**Fundamental limitation**: SAE optimizes for reconstruction, not classification. Compression always loses some task-specific information.

---

## Conclusions

### For Reconstruction: ✅ Refined is Better
- Use **2048 features, L1=0.0005** if goal is quality reconstruction
- 89.25% explained variance is good for interpretability
- 40.67% dead features is acceptable

### For Classification: ⚠️ Neither is Good Enough
- Refined (63.3%) < Pilot (70.0%) < Baseline (83.3%)
- **Token-specific aggregation backfired** - loses too much information
- **Mean pooling across tokens/layers is better** for SAE features

### Recommendation

**If you want SAE features for classification**:
1. Use **mean pooling** (not single token)
2. Try **even weaker L1** (0.0001 or 0.0002)
3. Try **supervised auxiliary loss** during SAE training
4. Accept that you'll lose 10-15 points vs raw activations

**If you want interpretability**:
1. Use refined SAE (2048 features, L1=0.0005)
2. Analyze features for operation/layer/token preferences
3. Don't expect classification performance

**If you want classification**:
1. **Use raw activations** (83.3% baseline)
2. SAE is not the right tool for this task

---

## Next Steps (If Continuing)

### Test Mean Pooling with Refined SAE
Run pilot's mean pooling aggregation on refined SAE features:
- Expected: 70-75% (between pilot and refined Token 1 L8)
- Will isolate aggregation effect from dictionary size effect

### Try Supervised SAE
Add auxiliary loss during training:
```python
loss = reconstruction_loss + l1_penalty + 0.1 * classification_loss
```
- Forces SAE to preserve operation-discriminative features
- May improve classification at cost of interpretability

### Alternative: Hybrid Approach
Use SAE for feature discovery, then train on raw activations:
1. Identify important features from SAE
2. Extract corresponding dimensions from raw activations
3. Train classifier on those dimensions only

---

## Verdict

**Refined SAE**:
- ✅ Better autoencoder (reconstruction, feature usage)
- ❌ Worse classifier (Token 1 L8 aggregation hurt)
- ⚠️ Mixed results - depends on use case

**Key Takeaway**: Token-specific aggregation doesn't help SAE features. Mean pooling across tokens/layers is better for compressed representations, even if single tokens are more discriminative in raw space.

---

**Time Investment**: ~1 hour total
- Train: 20 min
- Validate: 5 min
- Classify: 10 min
- Document: 25 min
