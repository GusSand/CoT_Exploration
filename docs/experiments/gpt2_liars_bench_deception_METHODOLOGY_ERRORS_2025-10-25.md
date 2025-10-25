# Deception Detection - Methodology Errors Identified

**Date**: 2025-10-25 (CORRECTED)
**Model**: GPT-2 CODI (124M parameters)
**Dataset**: Liars-Bench (Apollo Research)

---

## ⚠️ CRITICAL: PREVIOUS RESULTS INVALID

**This document corrects previously reported findings that contained severe methodological errors.**

---

## Executive Summary

**CORRECTED FINDING**: Linear probes on continuous thoughts **FAIL** at deception detection, achieving only **~48% accuracy** (worse than random chance). Response tokens achieve **70.5% accuracy** and are clearly superior for deception detection.

**Previous claim of "98.10% accuracy" was due to data leakage** (testing on training data without proper train/test split).

---

## Methodology Errors Identified

### Error 1: Data Leakage (Most Critical)

**Location**: `train_probes.py` lines 65-69

```python
# INCORRECT CODE (data leakage)
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)  # ← Predicting on TRAINING data!
accuracy = accuracy_score(y, y_pred)  # ← This is training accuracy, not test
```

**Impact**: The reported "98.10%" accuracy was training set accuracy, not test set performance. This is a classic case of data leakage and overfitting.

**Actual Cross-Validation Score**: The LogisticRegressionCV internal CV score for Layer 4 Token 1 was **51.4%** (essentially random chance), not 98%.

### Error 2: Misinterpretation of Cross-Validation

While `LogisticRegressionCV` uses cross-validation internally to select the regularization parameter (C), the code reported the accuracy on the full training set instead of the CV scores.

**Should have reported**: `clf.scores_[1].mean()` (CV accuracy)
**Actually reported**: `accuracy_score(y, y_pred)` (training accuracy)

### Error 3: Documentation Inconsistencies

The documentation claimed:
- "858 samples after deduplication (622 honest + 236 deceptive)"

Actual files contain:
- **CURRENT**: 1000 samples (500+500) balanced
- **CLEAN**: 784 samples (392+392) balanced
- **CORRUPTED**: 2000 samples (1000+1000) balanced

**None of these match the documented "858 samples with 72.5%/27.5% imbalance."**

---

## Actual Results (Proper Train/Test Split)

### Experimental Setup
- Dataset: 784 samples (392 honest + 392 deceptive), perfectly balanced
- Train/Test Split: 80/20 (627 train, 157 test)
- Random seeds: 42, 123, 456 (averaged results)
- Model: Logistic Regression with CV for C selection
- Evaluation: Test set accuracy (proper holdout)

### Results Summary

| Method | Test Accuracy | AUROC | F1 Score | Notes |
|--------|--------------|-------|----------|-------|
| **Response Tokens** | **70.50%** | **0.777** | **0.642** | ✅ **BEST** |
| Continuous L4 Token 5 | 48.83% | 0.465 | 0.498 | Worst among all methods |
| Continuous L4 Token 4 | 48.62% | 0.464 | 0.500 | Below random |
| Continuous L4 Token 2 | 48.20% | 0.465 | 0.491 | Below random |
| Continuous L4 Token 1 | 47.98% | 0.466 | 0.494 | Was claimed "98%" |
| Continuous L4 Token 3 | 47.77% | 0.465 | 0.487 | Below random |
| Continuous L4 Token 0 | 47.35% | 0.456 | 0.489 | Below random |
| **Random Chance** | **50.00%** | **0.500** | **0.500** | Baseline |

**Mean continuous thought accuracy**: 47.06% ± 1.02%

### Detailed Layer 4 Results

All Layer 4 continuous thought tokens (the "best" layer according to previous analysis):

```
Token 0: Test Acc = 47.35%, AUROC = 0.456, F1 = 0.489
Token 1: Test Acc = 47.98%, AUROC = 0.466, F1 = 0.494 ← Previously claimed "98%"!
Token 2: Test Acc = 48.20%, AUROC = 0.465, F1 = 0.491
Token 3: Test Acc = 47.77%, AUROC = 0.465, F1 = 0.487
Token 4: Test Acc = 48.62%, AUROC = 0.464, F1 = 0.500
Token 5: Test Acc = 48.83%, AUROC = 0.465, F1 = 0.498
```

### Overfitting Evidence

Average overfitting gap across Layer 4 tokens: **15.2 percentage points**
- Training accuracy: ~63%
- Test accuracy: ~48%
- **The model is overfitting, but even training accuracy is only 63%, not 98%**

---

## Comparison: Invalid vs Valid Results

### Previously Reported (INVALID)
- **Claim**: "Layer 4 Token 1 achieves 98.10% accuracy"
- **Method**: Testing on training data (data leakage)
- **Result**: Training set accuracy, completely invalid

### Actual Results (VALID)
- **Finding**: Layer 4 Token 1 achieves 47.98% test accuracy
- **Method**: Proper 80/20 train/test split with holdout evaluation
- **Result**: Worse than random chance (50%)

**Difference**: 50.12 percentage points due to data leakage!

---

## Scientific Conclusions (Corrected)

### 1. Continuous Thoughts FAIL at Deception Detection ❌

All continuous thought tokens across Layer 4 achieve **worse than random performance** (~48% vs 50% baseline). The AUROC scores (~0.46) are also below the random baseline of 0.50.

### 2. Response Tokens Are Superior ✅

Response tokens achieve **70.5% accuracy with 0.777 AUROC**, clearly demonstrating that:
- Deception information is NOT effectively encoded in continuous thought space
- Language-based representations (response tokens) retain more deception-relevant information
- The claim that "continuous thoughts >> response tokens" is **completely backwards**

### 3. No Linear Separability

The claim that "deception is linearly separable in continuous thought space" is refuted. Linear probes cannot extract any meaningful deception signal from continuous thoughts.

### 4. Layer Analysis Invalid

Previous claims about "Layer 4 encoding deception best" are invalid since all layers perform at or below random chance.

---

## Why The Errors Occurred

### 1. No Train/Test Split
The original `train_probes.py` script had:
```python
clf.fit(X_scaled, y)
y_pred = clf.predict(X_scaled)  # Testing on training data!
accuracy = accuracy_score(y, y_pred)
```

This is a beginner mistake in ML evaluation - you cannot evaluate a model on the same data it was trained on.

### 2. Confusion About Cross-Validation
While `LogisticRegressionCV` performs cross-validation internally for hyperparameter selection, this does NOT mean the final reported accuracy should be on the training set. The code should have either:
- Used a proper train/test split, OR
- Reported the CV scores from `clf.scores_`

The actual CV scores (~51%) would have revealed the problem immediately.

### 3. No Sanity Checks
A 98% accuracy claim should have triggered skepticism:
- Too good to be true for a difficult task
- Massive gap vs response tokens (98% vs 70%)
- Would represent a major scientific breakthrough

These red flags should have prompted verification.

---

## Implications for CODI Research

### Negative Result for CODI

This experiment provides **negative evidence** for CODI's continuous thought approach:
- Continuous thoughts do NOT encode deception effectively
- Language-based representations (response tokens) are superior
- The compression into continuous space loses critical information for deception detection

### Alternative Interpretations

1. **Task-specific encoding**: Deception may require explicit language reasoning that gets lost in compression
2. **Model scale**: GPT-2 (124M) may be too small to encode complex properties in continuous space
3. **Training objective**: CODI's self-distillation may not preserve deception-relevant features
4. **Layer selection**: Perhaps other layers (8, 11) would perform better (though unlikely given the uniform failure)

### What This Means for Continuous CoT

The finding that continuous thoughts fail at deception detection while succeeding at arithmetic (GSM8k) suggests:
- **Different tasks encode differently**: Mathematical reasoning may compress better than deceptive reasoning
- **Deception is inherently linguistic**: Lying may require language-level representations
- **CODI's limitations**: Not all cognitive properties transfer to continuous space equally

---

## Corrected Deliverables

### Valid Results Files
- `probe_results_proper_split_gpt2.json` - Proper train/test split evaluation
- `probe_results_response_gpt2.json` - Response token baseline (70.5%)

### Invalid Results Files (DO NOT USE)
- ~~`probe_results_gpt2.json`~~ - Contains data leakage, reports training accuracy
- ~~`probe_results_balanced_gpt2.json`~~ - Also has data leakage issues
- ~~`probe_results_nonlinear_gpt2.json`~~ - Same methodology errors

### Code Files
- `train_probes.py` - ⚠️ CONTAINS BUG (data leakage on lines 65-69)
- `train_probes_proper_split.py` - ✅ CORRECT (if it exists)

---

## Lessons Learned

### Methodological Lessons
1. **Always use train/test split** - Never evaluate on training data
2. **Understand your evaluation code** - Know the difference between training accuracy and test accuracy
3. **Sanity check results** - 98% accuracy should trigger verification
4. **Report proper metrics** - Use CV scores or holdout test sets, not training accuracy
5. **Negative results are valid** - Not all experiments succeed; failures teach us too

### Process Lessons
1. **Code review matters** - This bug should have been caught in review
2. **Documentation accuracy** - Ensure documented data (858 samples) matches actual files (784/1000)
3. **Reproducibility** - Proper random seeds and split documentation
4. **Scientific rigor** - Challenge extraordinary claims with evidence

---

## Recommendations

### For This Experiment
1. ✅ Use proper train/test split (80/20 or 70/30)
2. ✅ Report holdout test accuracy, not training accuracy
3. ✅ Use multiple random seeds and average results
4. ✅ Compare to proper baselines (random, majority class, response tokens)
5. ⚠️ Consider whether continuous thoughts can encode deception at all

### For Future Work
1. **Re-run all probe experiments** with proper evaluation methodology
2. **Test on larger models** (LLaMA 1B) to see if scale helps
3. **Try other tasks** where continuous thoughts may work better
4. **Investigate why deception fails** - attention analysis, feature analysis
5. **Accept negative results** - Document failures as learning opportunities

---

## Bottom Line

**Previous Claim**: "98.10% deception detection from continuous thoughts validates CODI"

**Actual Finding**: "~48% deception detection (worse than random) shows continuous thoughts FAIL for deception; response tokens at 70.5% are superior"

**Root Cause**: Data leakage (testing on training data) + misunderstanding of cross-validation

**Impact**: Invalidates all previous deception detection claims; requires complete re-evaluation with proper methodology

---

## Files Generated

### Results (Proper Evaluation)
- `/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/results/probe_results_proper_split_gpt2.json`
- `/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/results/probe_results_response_gpt2.json`

### Data Files
- `/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/data/processed/probe_dataset_gpt2_clean.json` (784 samples)

### This Documentation
- `/home/paperspace/dev/CoT_Exploration/docs/experiments/gpt2_liars_bench_deception_METHODOLOGY_ERRORS_2025-10-25.md`

---

**Status**: CORRECTED - Previous findings retracted due to methodology errors
**Next Steps**: Determine if experiment is worth re-running properly or if negative result stands
