# Deception Detection Experiment - Summary

**Last Updated**: 2025-10-25
**Status**: ❌ **NEGATIVE RESULT** - Continuous thoughts fail at deception detection

---

## Quick Facts

| Metric | Response Tokens | Continuous Thoughts (L4) | Winner |
|--------|----------------|-------------------------|--------|
| **Test Accuracy** | **70.5%** | 48.0% (worse than random) | Response Tokens ✅ |
| **AUROC** | **0.777** | 0.465 (worse than random) | Response Tokens ✅ |
| **F1 Score** | **0.642** | 0.494 | Response Tokens ✅ |

**Conclusion**: Response tokens are superior for deception detection. Continuous thoughts provide no useful signal.

---

## What Happened

### Initial Claim (INVALID ❌)
"98.10% deception detection accuracy using Layer 4 Token 1 continuous thoughts"

### Actual Result (VALID ✅)
**47.98%** test accuracy (worse than 50% random baseline)

### The Error
**Data leakage**: Testing on training data instead of using proper train/test split

```python
# BUGGY CODE in train_probes.py
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)  # Testing on TRAINING data!
accuracy = accuracy_score(y, y_pred)  # Reports training accuracy as if it's test
```

The 98% was **training set accuracy**, not test set accuracy. The actual cross-validation score was 51.4% (random chance).

---

## Corrected Results

### Model & Dataset
- **Model**: GPT-2 CODI (124M parameters)
- **Dataset**: Liars-Bench (Apollo Research), 784 samples, perfectly balanced
- **Split**: 80/20 train/test (627 train, 157 test)
- **Evaluation**: Proper holdout test set, 3 random seeds averaged

### Performance by Method

**Response Tokens (WINNER)**:
- Test Accuracy: 70.50%
- AUROC: 0.777
- F1: 0.642
- Precision: 0.70
- Recall (deceptive): 0.88

**Continuous Thoughts (Layer 4, ALL tokens)**:
| Token | Accuracy | AUROC | F1 | Status |
|-------|----------|-------|----|----|
| 0 | 47.35% | 0.456 | 0.489 | ❌ Below random |
| 1 | 47.98% | 0.466 | 0.494 | ❌ Below random (claimed "98%") |
| 2 | 48.20% | 0.465 | 0.491 | ❌ Below random |
| 3 | 47.77% | 0.465 | 0.487 | ❌ Below random |
| 4 | 48.62% | 0.464 | 0.500 | ❌ Below random |
| 5 | 48.83% | 0.465 | 0.498 | ❌ Below random |

**Mean**: 47.06% ± 1.02% (ALL worse than random 50%)

---

## Scientific Implications

### For CODI Research
1. **Negative result**: Continuous thoughts don't encode all cognitive properties equally
2. **Task-specific encoding**: Deception may require explicit language representations
3. **Compression losses information**: Self-distillation may lose deception-relevant features

### For Deception Detection
1. **Monitor response tokens**: 70.5% accuracy is decent for a small model
2. **Don't rely on latent space**: Continuous thoughts provide no signal
3. **Scale matters**: Larger models might perform better

### For ML Methodology
1. **Always use train/test split**: Never evaluate on training data
2. **Sanity check results**: 98% should trigger verification
3. **Negative results are valuable**: Document failures honestly

---

## Files

### Valid Results
- `src/experiments/liars_bench_codi/results/probe_results_proper_split_gpt2.json` ✅
- `src/experiments/liars_bench_codi/results/probe_results_response_gpt2.json` ✅

### Invalid Results (DO NOT USE)
- `src/experiments/liars_bench_codi/results/probe_results_gpt2.json` ❌ (data leakage)
- `src/experiments/liars_bench_codi/results/probe_results_balanced_gpt2.json` ❌ (data leakage)

### Documentation
- **Current (valid)**: `gpt2_liars_bench_deception_METHODOLOGY_ERRORS_2025-10-25.md` ✅
- **Old (invalid)**: `gpt2_liars_bench_deception_INVALID_DATA_LEAKAGE_2025-10-25.md` ❌

---

## Next Steps

Before re-running this experiment, discuss with PM:

1. **Is it worth re-running?**
   - Current result is clear: continuous thoughts fail
   - May not change with better methodology

2. **What would a proper experiment look like?**
   - Balanced dataset ✅ (already have)
   - Correct labels ✅ (already have)
   - Proper train/test split ✅ (need to implement in scripts)
   - Multiple random seeds ✅ (already done in proper_split)
   - Cross-validation OR holdout test ✅ (already done)

3. **Alternative approaches?**
   - Try larger model (LLaMA 1B)
   - Try different layers (8, 11)
   - Try different probe architectures
   - Try different deception tasks

4. **Accept the negative result?**
   - Document that continuous thoughts fail for deception
   - Focus on tasks where CODI succeeds (GSM8k, etc.)
   - Investigate why deception fails (attention analysis, etc.)

---

## Timeline

- **2025-10-24**: Initial experiment, claimed 72% accuracy
- **2025-10-25a**: Found class imbalance bug, claimed 59% accuracy
- **2025-10-25b**: Claimed 98% accuracy after "fixing" extraction bug
- **2025-10-25c**: Found data leakage, RETRACTED, actual result ~48%

---

**Bottom Line**: The deception detection experiment **failed**. Continuous thoughts do not encode deception. Response tokens are superior (70.5% vs 48%). This is a valid **negative result** for CODI.
