# SAE Features vs Operation Circuits: Comparative Analysis

**Date**: 2025-10-24
**Objective**: Map SAE feature specialization to operation circuits findings

---

## Executive Summary

SAE features show **fundamentally different organization** than raw activations:

| Finding | Operation Circuits (Raw) | SAE Features (Refined 2048) |
|---------|-------------------------|----------------------------|
| **Most Discriminative Location** | Token 1, Layer 8 (77.5% solo) | Layer 14 (97.7% of selective features) |
| **Token 1 L8 Enrichment** | Highest solo accuracy | **0 selective features** (0.0x enrichment) |
| **Layer Distribution** | L8 > L14 > L4 (middle-biased) | L14 >> L8 ≈ L4 (late-biased) |
| **Classification Accuracy** | 83.3% (baseline) | 71.7% (concatenation), 63.3% (Token 1 L8) |

**Key Insight**: SAE compression **redistributes operation-specific information** from middle layers (L8) to late layers (L14), explaining why Token 1 L8 aggregation failed.

---

## Background: Operation Circuits Findings

From `src/experiments/operation_circuits/`:

### Solo Token Performance (Raw L8 Activations)
```
Token 0: 70.0%
Token 1: 77.5% ⭐ (BEST)
Token 2: 73.3%
Token 3: 68.3%
Token 4: 66.7%
Token 5: 70.8%
```

### Layer Performance (Raw Activations)
```
Layer 4 (Early):   75.0%
Layer 8 (Middle):  83.3% ⭐ (BEST)
Layer 14 (Late):   80.0%
```

### Conclusion
**Token 1 × Layer 8** is the most operation-discriminative position in raw activation space.

---

## SAE Feature Analysis Results

### Feature Usage
- **Total features**: 2,048
- **Active features**: 1,215 (59.3%)
- **Operation-selective features** (selectivity ≥ 2.0): 133 (6.5% of total, 10.9% of active)

### Operation Distribution
Highly balanced across operation types:

```
Mixed:            43 features (32.3%)
Addition:         47 features (35.3%)
Multiplication:   43 features (32.3%)
```

No strong bias toward any operation type (unlike classification performance where multiplication was easiest).

### Layer Preferences
Operation-selective features show **extreme late-layer bias**:

```
Layer 4 (Early):    2 features (1.5%)
Layer 8 (Middle):   1 feature  (0.8%)
Layer 14 (Late):  130 features (97.7%) ⭐
```

**Finding**: 97.7% of operation-selective features concentrate in Layer 14, NOT Layer 8.

### Token Preferences
Modest variation across token positions:

```
Token 0:  27 features (20.3%)
Token 1:  34 features (25.6%) ⭐ (slight preference)
Token 2:  23 features (17.3%)
Token 3:  13 features (9.8%)
Token 4:  21 features (15.8%)
Token 5:  15 features (11.3%)
```

Token 1 shows **1.5x enrichment** (25.6% vs 16.7% expected), but this is in **Layer 14**, not Layer 8.

### Token 1 × Layer 8 Hypothesis Test

**Hypothesis** (from operation circuits): Operation-specific SAE features should concentrate in Token 1 L8.

**Result**: **STRONGLY REJECTED**
- Token 1 × Layer 8 features: **0** (0.0%)
- Expected if random: 5.6% (1/18 positions)
- Enrichment: **0.00x**

**Interpretation**: SAE compression completely eliminates the Token 1 L8 advantage observed in raw activations.

---

## Why Did This Happen?

### Hypothesis: Reconstruction Objective Redistributes Information

SAE optimizes for:
```python
loss = reconstruction_loss + l1_penalty
     = ||x - decode(encode(x))||² + λ * ||encode(x)||₁
```

This objective:
1. **Preserves total information** (reconstruction loss)
2. **Forces sparsity** (L1 penalty)
3. **Does NOT preserve task-specific structure** (no classification loss)

Result: Information gets redistributed to maximize reconstruction efficiency, not task discriminability.

### Evidence

1. **Layer 8 → Layer 14 Shift**
   - Raw: Layer 8 most discriminative (83.3%)
   - SAE: Layer 14 most selective (97.7% of features)
   - **Compression pushes operation signals downstream**

2. **Token 1 L8 Failure**
   - Raw Token 1 L8: 77.5% solo accuracy
   - SAE Token 1 L8 aggregation: 63.3% accuracy (WORST)
   - SAE concatenation (all 18): 71.7% accuracy (BEST)
   - **Single-position information is destroyed**

3. **Balanced Operation Features**
   - Classification: Multiplication easiest (89% F1), Mixed hardest (62% F1)
   - Features: Nearly equal (43 mixed, 47 addition, 43 multiplication)
   - **Features don't align with classification difficulty**

---

## Implications

### For Classification

**Raw Activations > SAE Features**:
- Raw baseline: 83.3%
- Best SAE (concatenation): 71.7% (-11.6 pts)
- Token 1 L8 SAE: 63.3% (-20.0 pts)

**Why concatenation works best**: Preserves layer/token position information that SAE distributes across spatial dimensions.

### For Interpretability

**SAE features ARE interpretable**, but represent **compressed/redistributed** version of raw structure:

✅ **What SAE preserves**:
- Operation-specific features exist (133 selective features)
- Balanced representation across operations (43-47 features each)
- Clear layer preferences (97.7% in L14)
- Modest token specialization (Token 1 at 1.5x)

❌ **What SAE loses**:
- Token 1 L8 discriminative advantage
- Middle-layer operation signals
- Task-aligned feature difficulty (multiplication ≠ easier in features)
- 11.6 points of classification accuracy

### For Operation Circuits

**SAE provides complementary view**, NOT confirmation:

| Aspect | Operation Circuits View | SAE Features View |
|--------|------------------------|-------------------|
| **Where** | Token 1 L8 (middle layer) | Layer 14 (late layer) |
| **How** | Concentrated in specific position | Distributed across tokens |
| **What** | Raw operation signals | Compressed operation features |
| **Performance** | High (83.3%) | Lower (71.7%) |

**Interpretation**: Operation circuits identify where raw discriminative power lives. SAE shows how compression redistributes that power.

---

## Top Feature Examples

### Multiplication Features (43 total)

Top 5 by activation strength:
```
Feature 1391: Layer 14, Token 3, max_act=10.928
Feature 1242: Layer 14, Token 0, max_act=10.669
Feature 476:  Layer 14, Token 1, max_act=9.342
Feature 1352: Layer 14, Token 0, max_act=9.332
Feature 1888: Layer 14, Token 4, max_act=9.252
```

All in **Layer 14**, distributed across token positions.

### Addition Features (47 total)

Top 5 by activation strength:
```
Feature 1713: Layer 14, Token 0, max_act=10.540
Feature 1884: Layer 14, Token 0, max_act=10.463
Feature 874:  Layer 14, Token 0, max_act=10.370
Feature 439:  Layer 14, Token 0, max_act=10.337
Feature 1451: Layer 14, Token 1, max_act=9.075
```

Strong preference for **Token 0** in Layer 14.

### Mixed Features (43 total)

Top 5 by activation strength:
```
Feature 1499: Layer 14, Token 4, max_act=11.509
Feature 1062: Layer 14, Token 4, max_act=11.250
Feature 640:  Layer 14, Token 4, max_act=11.046
Feature 340:  Layer 14, Token 2, max_act=9.970
Feature 1120: Layer 8,  Token 3, max_act=9.769 ⭐ (rare L8 feature!)
```

Feature 1120 is one of the **only 1 selective features** in Layer 8.

---

## Conclusions

### 1. SAE Compression Changes Information Topology

Raw activations organize operation signals in **Token 1 × Layer 8**.
SAE reorganizes them to **Layer 14 distributed across tokens**.

This is **not a failure** — it's the natural consequence of optimizing for reconstruction + sparsity without task supervision.

### 2. Token 1 L8 Hypothesis Rejected

The most discriminative position in raw space (Token 1 L8) becomes **the least informative** for SAE features.

This explains why Token 1 L8 aggregation (63.3%) performed worse than mean pooling (70%) and concatenation (71.7%).

### 3. Concatenation Succeeds by Preserving Structure

Concatenating all 18 vectors (3 layers × 6 tokens) preserves the positional structure that SAE distributes information across.

This is why concatenation (71.7%) outperforms both:
- Mean pooling (70%) — loses positional signals
- Token 1 L8 only (63.3%) — discards redistributed information

### 4. SAE Features Are Interpretable, But Different

SAE provides a **valid but compressed** view of operation-specific processing:
- 133 operation-selective features across 3 operations
- Clear layer/token organization (just shifted to L14)
- 59.3% feature usage (good for interpretability)

But this view **differs from raw circuits** and trades 11.6 points of accuracy for compression.

---

## Recommendations

### Use SAE Features For:
✅ Understanding **compressed representations** of operation-specific processing
✅ Identifying **feature-level** patterns (e.g., which features activate for multiplication)
✅ Analyzing **late-layer** (L14) operation specialization
✅ Interpretability when reconstruction quality matters (89.25% explained variance)

### Use Raw Activations For:
✅ **Maximizing classification accuracy** (83.3% vs 71.7%)
✅ Identifying **discriminative positions** (Token 1 L8)
✅ Understanding **mid-layer** (L8) operation circuits
✅ Tasks where losing 11.6 points is unacceptable

### Use Both For:
✅ **Complementary views**: Raw = where signals live, SAE = how they compress
✅ **Compression analysis**: How does sparsity reorganize information?
✅ **Mechanistic interpretability**: Multi-level understanding (raw → compressed → features)

---

## Future Work

### 1. Supervised SAE
Add classification loss during training:
```python
loss = reconstruction + l1_penalty + 0.1 * classification_loss
```

**Expected**: Features concentrate in Token 1 L8, accuracy improves toward baseline.
**Tradeoff**: Less "pure" interpretability, features biased toward task.

### 2. Layer-Specific Analysis
Train separate SAEs for L4, L8, L14 and compare operation selectivity.

**Question**: Does L8 SAE preserve Token 1 preference better than full-dataset SAE?

### 3. Feature Ablation
Surgically remove operation-specific features and measure classification impact.

**Question**: Are those 133 selective features causally important, or just correlated?

---

## Appendix: Selectivity Scores

### Definition
```
selectivity(feature, dimension) = max_mean / avg_mean

where:
  max_mean = highest mean activation for any value in dimension
  avg_mean = average of all mean activations in dimension
```

### Interpretation
- **Selectivity < 1.5**: Feature is general-purpose
- **Selectivity 1.5-2.0**: Feature shows preference
- **Selectivity ≥ 2.0**: Feature is highly selective (threshold used)

### Statistics (Active Features Only)
```
Operation selectivity mean: 1.88
Layer selectivity mean:     2.61 ⭐ (highest)
Token selectivity mean:     2.40
```

Features are **most selective for layer** (mean 2.61), then token (2.40), then operation (1.88).

This suggests **spatial organization** (layer/token) is stronger than **semantic organization** (operation type) in SAE feature space.

---

**Files**:
- Analysis script: `src/experiments/sae_pilot/refined/analyze_feature_specialization.py`
- Results: `src/experiments/sae_pilot/refined/feature_specialization_results.json`
- Visualizations: `src/experiments/sae_pilot/refined/feature_specialization.{png,pdf}`
- Raw operation circuits: `src/experiments/operation_circuits/results/operation_circuits_analysis.json`
- Classification comparison: `src/experiments/sae_pilot/refined/COMPARISON.md`

**Time**: ~45 minutes (analysis + documentation)
