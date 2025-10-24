# SAE Pilot Experiment

**Goal**: Train a Sparse Autoencoder (SAE) on continuous thought representations to find interpretable features and test if they classify operation types.

**Status**: ✅ **COMPLETE** (Minimal Pilot)

**Date**: 2025-10-24

**Time Investment**: ~5 hours (setup + training + analysis + documentation)

---

## Executive Summary

**Objective**: Pilot SAE-based interpretability for CODI's continuous thoughts across 3 research goals:
1. Interpret continuous thought representations (middle layer)
2. Find sparse features across all reasoning tokens
3. Compare to operation-specific circuits (83.3% classification baseline)

**Key Results**:
- ✅ Successfully trained 8192-feature SAE on 10,800 activation vectors
- ✅ Achieved high sparsity (L0 = 23 features/vector, 0.28% activation rate)
- ⚠️ 97% dead features (only 248/8192 active)
- ⚠️ Fair reconstruction (78.62% explained variance, 89.60% cosine similarity)
- ❌ Classification: 70% accuracy vs 83.3% baseline (-13.3 points)

**Main Finding**: **Sparse features trade off discriminability for interpretability** - raw activations preserve more operation-specific information than compressed SAE features.

---

## Methodology

### Data
- **Source**: Reused activations from operation circuits experiment
- **Problems**: 600 GSM8K problems (200 addition, 200 multiplication, 200 mixed)
- **Layers**: L4 (early), L8 (middle), L14 (late)
- **Tokens**: 6 continuous thought tokens per layer
- **Total vectors**: 10,800 (600 × 3 × 6)
- **Hidden dimension**: 2048 (Llama-3.2-1B)

### SAE Architecture
```python
Input dim: 2048
Features: 8192 (4x expansion)
Sparsity penalty: L1 coefficient = 0.001
Training: 25 epochs, batch size 256, Adam (lr=1e-3)
```

### Evaluation
1. **Reconstruction quality**: MSE, explained variance, cosine similarity
2. **Sparsity**: L0 (active features), L1 norm, dead feature percentage
3. **Interpretability**: Feature usage by operation/layer/token
4. **Classification**: Logistic regression on aggregated features

---

## Results

### SAE Training (Story 1.2)

| Epoch | Loss | Reconstruction | Sparsity | L0 |
|-------|------|----------------|----------|-----|
| 1 | 0.195 | 0.168 | 0.027 | 165.89 |
| 10 | 0.063 | 0.048 | 0.015 | 18.53 |
| 25 | 0.043 | 0.032 | 0.011 | 23.28 |

**Convergence**: Stable training, L0 dropped from 166 → 23 features/vector

### Validation Metrics (Story 1.3)

| Metric | Value | Verdict |
|--------|-------|---------|
| **Reconstruction Loss** | 0.0319 MSE | Good |
| **Explained Variance** | 78.62% | Fair |
| **Cosine Similarity** | 89.60% | Good |
| **L0 Sparsity** | 23.34 ± 9.98 | Excellent |
| **L0 Percentage** | 0.28% | Highly sparse |
| **Dead Features** | 7944/8192 (96.97%) | Poor |

**Issue Identified**: Massive feature death - only 248 features ever activate

### Feature Analysis (Story 2.1)

**Top Features by Usage**:

| Feature | Max Act | Usage | Operation Preference | Layer | Token |
|---------|---------|-------|---------------------|-------|-------|
| 1072 | 4.20 | 40.4% | Mixed | L4 | T3 |
| 4651 | 3.96 | 37.5% | Mixed | L14 | T2 |
| 1506 | 3.67 | 41.0% | **Multiplication** | L4 | T1 |
| 413 | 2.87 | 53.7% | **Multiplication** | L4 | T2 |
| 4116 | 2.64 | 48.9% | **Multiplication** | L8 | T2 |

**Key Observations**:
- ✅ Multiple features prefer multiplication operations
- ✅ Layer specialization: Some features specific to L4/L8/L14
- ✅ Token specificity: Different tokens activate different features
- ⚠️ Most features are very broadly activated (not specific)

### Classification Performance (Story 2.2)

| Method | Accuracy | vs Baseline | Target Met |
|--------|----------|-------------|------------|
| **Baseline** (Raw activations) | **83.3%** | - | ✅ |
| **SAE Features** | **70.0%** | **-13.3 pts** | ❌ |

**Per-Class Performance**:

| Operation Type | Precision | Recall | F1-Score |
|----------------|-----------|--------|----------|
| Mixed | 0.59 | 0.57 | 0.58 |
| Pure Addition | 0.69 | 0.60 | 0.64 |
| Pure Multiplication | **0.80** | **0.93** | **0.86** |

**Key Finding**: SAE features recognize multiplication best (86% F1), but overall performance **worse than raw activations**.

---

## Analysis

### Why Did SAE Underperform?

**Hypothesis 1: Information Loss from Sparsity**
- 97% dead features → compressed too aggressively
- L1 penalty too strong → discarded operation-discriminative information
- **Evidence**: Raw activations preserve more nuance than 23 sparse features

**Hypothesis 2: Aggregation Strategy**
- Mean pooling across tokens/layers may wash out position-specific signals
- Operation circuits showed Token 1 L8 is critical → averaging loses this
- **Alternative**: Token-specific or layer-specific aggregation

**Hypothesis 3: Feature Objectives Mismatch**
- SAE optimizes for **reconstruction**, not **classification**
- Compression favors general patterns over task-specific features
- **Evidence**: High-usage features (1072, 4651) prefer "mixed" (less specific)

### What Worked Well?

✅ **Infrastructure**: End-to-end SAE training pipeline (reusable)
✅ **Sparsity**: Achieved highly sparse representations (0.28% activation)
✅ **Interpretability**: Features show operation/layer/token preferences
✅ **Multiplication Detection**: 86% F1 for multiplication (best class)

### What Needs Improvement?

❌ **Dead Features**: 97% unused → need smaller dictionary or feature resurrection
❌ **Reconstruction**: 78.62% explained variance → room for improvement
❌ **Classification**: 70% << 83.3% baseline → features not task-aligned

---

## Next Steps (Post-Pilot)

### Immediate Fixes
1. **Reduce dictionary size**: Try 1024 or 2048 features (vs 8192)
2. **Tune L1 coefficient**: Test 0.0001, 0.0005, 0.001, 0.005
3. **Feature resurrection**: Reinitialize dead features during training

### Advanced Improvements
4. **Token-specific SAEs**: Train separate SAE for each token position
5. **Layer-specific SAEs**: Train per-layer (L4, L8, L14)
6. **Supervised features**: Add operation-type loss during SAE training
7. **Better aggregation**: Max pooling, attention-weighted, or token-specific

### Comparison Experiments
8. **Baseline replication**: Run same classifier on raw L8 activations
9. **PCA comparison**: Test if PCA features outperform SAE
10. **Cross-model**: Train SAE on GPT-2 continuous thoughts

---

## Files

### Scripts (5 files, ~800 lines)
- `1_extract_activations.py` - Convert operation circuits data to SAE format
- `2_train_sae.py` - Train sparse autoencoder with WandB logging
- `3_validate_sae.py` - Compute reconstruction & sparsity metrics
- `4_visualize_features.py` - Analyze top features by operation/layer/token
- `5_classify_operations.py` - Test operation classification performance

### Data
- `data/sae_training_activations.pt` - 10,800 vectors × 2048 dims (84.4 MB)

### Results
- `results/sae_weights.pt` - Trained SAE model (128 MB)
- `results/training_results.json` - Training metrics
- `results/validation_report.json` - Quality metrics
- `results/feature_analysis.json` - Top feature analysis
- `results/classification_results.json` - Operation classification results

### Visualizations
- `results/sae_validation.{png,pdf}` - Reconstruction & sparsity plots
- `results/top_features_visualization.{png,pdf}` - Feature analysis (10 features × 3 plots)
- `results/operation_classification.{png,pdf}` - Confusion matrix & accuracy comparison

---

## Conclusions

### Scientific Contributions

1. **Negative Result is Valuable**: Demonstrates that SAE sparsity trades off task-specific discriminability
2. **Feature Death Problem**: Confirmed as major issue in SAE interpretability (97% unused)
3. **Multiplication Specialization**: SAE features recognize multiplication better than addition/mixed
4. **Infrastructure**: Established reusable pipeline for SAE-based continuous thought analysis

### Lessons Learned

✅ **Setup works**: Can train SAEs on continuous thoughts
✅ **Sparsity achievable**: Got 0.28% activation rate
⚠️ **Tuning critical**: Hyperparameters drastically affect feature usage
❌ **Sparsity ≠ Discriminability**: Compressed features lose task-specific information

### Recommendation for Future Work

**Don't use SAE for classification** - use for **interpretability only**:
- Identify what general patterns the model uses (good)
- Don't expect features to outperform raw activations for specific tasks (bad)

**Best use case**: Human interpretation of "what does Token 1 represent?" not "classify operations"

---

## Usage

### Train SAE
```bash
python src/experiments/sae_pilot/2_train_sae.py \
    --n_features 8192 \
    --n_epochs 25 \
    --l1_coefficient 0.001 \
    --learning_rate 1e-3
```

### Validate
```bash
python src/experiments/sae_pilot/3_validate_sae.py
```

### Visualize Features
```bash
python src/experiments/sae_pilot/4_visualize_features.py --n_top 10
```

### Classify Operations
```bash
python src/experiments/sae_pilot/5_classify_operations.py --baseline_acc 0.833
```

---

## Acknowledgments

- Reused activations from **Operation Circuits** experiment (saved 90 min GPU time)
- Built on `sae_lens` library for SAE training infrastructure
- Integrated with existing WandB project for tracking

---

**Pilot Status**: ✅ **COMPLETE** - Infrastructure validated, negative result documented, ready for refinement
