# SAE Pilot - Sparse Autoencoder for Continuous Thought Interpretation

**Date**: 2025-10-24
**Experiment**: Pilot SAE training on continuous thoughts
**Status**: ✅ COMPLETE
**Time**: ~5 hours
**Branch**: `experiment/sae-pilot`

---

## Objective

Train a Sparse Autoencoder (SAE) on CODI continuous thought activations to:
1. Find interpretable sparse features
2. Test if features classify operation types better than raw activations (83.3% baseline)
3. Establish SAE infrastructure for future interpretability work

---

## Methodology

**Data**: Reused 600-problem operation circuits activations (saved 90 min GPU time)
- 10,800 vectors (600 problems × 3 layers × 6 tokens)
- Layers: L4 (early), L8 (middle), L14 (late)
- Hidden dim: 2048 (Llama-3.2-1B)

**SAE Config**:
- Features: 8192 (4x expansion)
- L1 coefficient: 0.001
- Training: 25 epochs, batch size 256, Adam lr=1e-3

---

## Key Results

| Metric | Value | Verdict |
|--------|-------|---------|
| **Training Time** | ~2 minutes | ✅ Fast |
| **Reconstruction Loss** | 0.0319 | ✅ Good |
| **Explained Variance** | 78.62% | ⚠️ Fair |
| **L0 Sparsity** | 23.34 features/vector | ✅ Excellent |
| **Dead Features** | 96.97% (7944/8192) | ❌ Poor |
| **Classification Accuracy** | 70.0% | ❌ Below baseline |
| **Baseline Accuracy** | 83.3% | - |
| **Target Met (>80%)** | NO | ❌ |

---

## Major Findings

### ❌ **Negative Result: SAE Features Underperform Raw Activations**

**Classification Performance**:
- **SAE Features**: 70.0% accuracy (-13.3 pts vs baseline)
- **Raw Activations**: 83.3% accuracy (operation circuits baseline)
- **Target**: 80% (not met)

**Per-Class Results**:
| Operation | Precision | Recall | F1 | Observations |
|-----------|-----------|--------|-----|--------------|
| Mixed | 0.59 | 0.57 | 0.58 | Worst performance |
| Addition | 0.69 | 0.60 | 0.64 | Mid-range |
| Multiplication | **0.80** | **0.93** | **0.86** | Best - SAE captures multiplication patterns well |

### ⚠️ **97% Feature Death Problem**

- Only 248 out of 8192 features ever activate
- Likely causes: Too large dictionary size, L1 penalty too strong
- Active features still meaningful (see top features below)

### ✅ **Features Show Operation/Layer/Token Specificity**

**Top 5 Most-Used Features**:

| Feature | Usage | Operation Preference | Layer | Token |
|---------|-------|---------------------|-------|-------|
| 1072 | 40.4% | Mixed | L4 | T3 |
| 1506 | 41.0% | **Multiplication** | L4 | T1 |
| 413 | 53.7% | **Multiplication** | L4 | T2 |
| 4116 | 48.9% | **Multiplication** | L8 | T2 |
| 4651 | 37.5% | Mixed | L14 | T2 |

**Observations**:
- Multiple features prefer multiplication operations
- Layer specialization: L4 (early), L8 (middle), L14 (late) features distinct
- Token specialization: Different features activate on different token positions
- BUT: Most features very broadly activated (not specific enough for classification)

---

## Analysis

### Why Did Classification Fail?

**Hypothesis 1: Sparsity-Discriminability Tradeoff** ✅ Most Likely
- SAE optimizes for **reconstruction**, not **classification**
- L1 penalty discards task-specific information to achieve sparsity
- Raw activations preserve more operation-discriminative nuance than 23 sparse features

**Hypothesis 2: Information Loss from Compression** ✅ Contributing
- 97% dead features → compressed too aggressively
- Lost subtle operation-specific patterns in pursuit of sparsity

**Hypothesis 3: Aggregation Strategy** ⚠️ Possible
- Mean pooling across tokens/layers may wash out position-specific signals
- Operation circuits showed Token 1 L8 is critical → averaging loses this specificity

### What Worked?

✅ **Infrastructure**: Complete SAE pipeline (extract → train → validate → interpret → classify)
✅ **Sparsity**: Achieved highly sparse representations (0.28% activation rate)
✅ **Interpretability**: Features show clear operation/layer/token preferences
✅ **Multiplication Detection**: 86% F1 for multiplication (better than addition/mixed)
✅ **Speed**: Training completed in 2 minutes (efficient)

---

## Scientific Implications

1. **Negative Results Are Valuable**: Demonstrates fundamental tradeoff between sparsity and task-specific discriminability

2. **SAE Not Universal**: Compressed sparse features don't automatically improve downstream tasks - they trade performance for interpretability

3. **Use Case Matters**: SAE good for "what patterns exist?" (interpretability), bad for "which operation is this?" (classification)

4. **Multiplication More Distinct**: Consistent with operation circuits finding (92.5% recognition for multiplication vs 82.5% for addition)

---

## Next Steps

### Immediate Fixes (If pursuing classification)
1. **Reduce dictionary size**: Try 1024 or 2048 features (vs 8192)
2. **Tune L1 coefficient**: Test 0.0001, 0.0005, 0.001, 0.005
3. **Feature resurrection**: Reinitialize dead features during training
4. **Token-specific aggregation**: Use Token 1 L8 only (best from operation circuits)

### Advanced (If pursuing interpretability)
5. **Token-specific SAEs**: Train separate SAE for each token position
6. **Layer-specific SAEs**: Train per-layer (L4, L8, L14)
7. **Supervised features**: Add operation-type auxiliary loss during SAE training
8. **Feature visualization**: Neuron2Graph-style visualization of what activates each feature

### Alternative Approaches
9. **PCA baseline**: Compare SAE vs PCA features
10. **Probing classifiers**: Train probe on raw activations (different aggregations)
11. **Attention-based aggregation**: Weight tokens by importance before aggregation

---

## Deliverables

### Code (5 scripts, ~800 lines)
- `1_extract_activations.py` - Reuse operation circuits data
- `2_train_sae.py` - Train SAE with WandB
- `3_validate_sae.py` - Reconstruction & sparsity metrics
- `4_visualize_features.py` - Top feature analysis
- `5_classify_operations.py` - Operation classification

### Data & Results
- `data/sae_training_activations.pt` (84.4 MB)
- `results/sae_weights.pt` (128 MB)
- `results/*.json` - Metrics and analysis
- `results/*.{png,pdf}` - Visualizations (6 files)

### Documentation
- `README.md` - Complete methodology & results
- `docs/experiments/10-24_llama_gsm8k_sae_pilot.md` - This file

---

## Conclusions

### For This Pilot
✅ **Infrastructure Success**: End-to-end SAE pipeline works
✅ **Sparsity Success**: Achieved 0.28% activation rate
⚠️ **Interpretability Partial**: Features show preferences but 97% dead
❌ **Classification Failure**: 70% << 83.3% baseline

### For Future Work
**Recommendation**: Use SAE for **interpretability only**, not classification
- Good for: "What general patterns does Token 1 encode?"
- Bad for: "Classify this as addition vs multiplication"

**Key Insight**: Sparsity helps humans understand representations but hurts downstream task performance. This is expected - compression always involves information loss. The question is whether the loss matters for your use case.

---

## Time Breakdown

| Task | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| Story 1.1: Extract activations | 1.5h | 15min | Reused existing data! |
| Story 1.2: Train SAE | 2.5h | 30min | Included failed run (512 features) |
| Story 1.3: Validate SAE | 0.5h | 15min | Including visualizations |
| Story 2.1: Visualize features | 1h | 30min | Top-10 analysis |
| Story 2.2: Classify operations | 1.5h | 30min | Including negative result analysis |
| Documentation | 1h | 2h | README + experiment report |
| **TOTAL** | **8.5h** | **~5h** | Under budget! |

---

## Appendix: WandB Runs

- **Run 1**: `sae_512feat_l10.02` - Failed (feature death, L0 → 0)
- **Run 2**: `sae_8192feat_l10.001` - Success (L0 = 23, final used)

View at: https://wandb.ai/gussand/sae-pilot

---

**Status**: ✅ Pilot complete, infrastructure validated, negative result well-documented, ready for future refinement or alternative approaches.
