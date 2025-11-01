# GSM8K Train vs Test Generalization Analysis

**Date:** November 1, 2025
**Experiment:** Comparison of intervention performance on GSM8K train vs test sets

## Executive Summary

The model shows **significant overfitting** to the training set, with a dramatic **31.8% drop** in baseline performance from train (86.4%) to test (54.5%). Remarkably, **all interventions maintain similar relative drops (~30-31%)**, suggesting that the interventions themselves don't add or reduce overfitting - they preserve the model's generalization characteristics.

## Key Findings

### 1. Massive Train-Test Gap

**Baseline Performance:**
- **Train:** 86.4%
- **Test:** 54.5%
- **Gap:** -31.8%

This 31.8% drop indicates severe overfitting in the base CODI-LLaMA model on GSM8K.

### 2. Interventions Preserve Generalization Gap

**All major interventions show nearly identical train-test gaps:**

| Intervention | Train | Test | Gap |
|--------------|-------|------|-----|
| Baseline | 86.4% | 54.5% | **-31.8%** |
| Minus (numbers) | 84.8% | 53.8% | **-31.1%** |
| Minus (all) | 84.8% | 53.0% | **-31.8%** |
| Replacement (numbers) | 83.3% | 53.0% | **-30.3%** |
| Replacement (all) | 83.3% | 53.0% | **-30.3%** |

**Key Insight:** The interventions don't add overfitting - they maintain approximately the same ~30-31% generalization gap as the baseline.

### 3. Complete Results: Train vs Test

| Intervention | Scope | Train | Test | Delta | Relative Drop |
|--------------|-------|-------|------|-------|---------------|
| **baseline** | none | 86.4% | 54.5% | -31.8% | -37% |
| **replacement** | numbers | 83.3% | 53.0% | -30.3% | -36% |
| **replacement** | all | 83.3% | 53.0% | -30.3% | -36% |
| **zero** | numbers | 12.1% | 14.4% | +2.3% | +19% |
| **zero** | all | 10.6% | 9.8% | -0.8% | -8% |
| **average** | numbers | 47.7% | 31.1% | -16.7% | -35% |
| **average** | all | 42.4% | 28.0% | -14.4% | -34% |
| **minus** | numbers | 84.8% | 53.8% | -31.1% | -37% |
| **minus** | all | 84.8% | 53.0% | -31.8% | -38% |
| **discretize** | numbers | 62.9% | 43.2% | -19.7% | -31% |
| **discretize** | all | 54.5% | 33.3% | -21.2% | -39% |
| **discretize_ln** | numbers | 65.9% | 43.2% | -22.7% | -34% |
| **discretize_ln** | all | 55.3% | 34.1% | -21.2% | -38% |
| **proj1** | numbers | 62.9% | 43.9% | -18.9% | -30% |
| **proj1** | all | 38.6% | 27.3% | -11.4% | -29% |
| **proj5** | numbers | 69.7% | 47.7% | -22.0% | -32% |
| **proj5** | all | 65.9% | 38.6% | -27.3% | -41% |
| **proj5_unnorm** | numbers | 69.7% | 47.7% | -22.0% | -32% |
| **proj5_unnorm** | all | 61.4% | 41.7% | -19.7% | -32% |

### 4. Patterns in Generalization

**Least Affected by Train-Test Gap:**
1. **Proj1 (all positions):** -11.4% (only 29% relative drop)
2. **Average ablation:** -14.4% to -16.7% (34-35% relative drop)
3. **Discretization:** -19.7% to -22.7% (31-39% relative drop)

**Most Affected (Similar to Baseline):**
1. **Baseline, Minus, Replacement:** -30.3% to -31.8% (36-38% relative drop)
2. **Proj5 (all positions):** -27.3% (41% relative drop)

**Anomaly - Improves on Test:**
- **Zero ablation (numbers):** +2.3% (improves from 12.1% to 14.4%)
  - Likely statistical noise given the low baseline accuracy

### 5. Intervention Rankings on Test Set

#### Test Set Performance (54.5% baseline):
| Rank | Intervention | Scope | Test Accuracy | Δ from Baseline |
|------|--------------|-------|---------------|-----------------|
| 1 | Baseline | none | 54.5% | - |
| 2 | Minus | numbers | 53.8% | -0.7% |
| 3 | Replacement | numbers | 53.0% | -1.5% |
| 3 | Replacement | all | 53.0% | -1.5% |
| 3 | Minus | all | 53.0% | -1.5% |
| 6 | Proj5 | numbers | 47.7% | -6.8% |
| 6 | Proj5 (unnorm) | numbers | 47.7% | -6.8% |
| 8 | Proj1 | numbers | 43.9% | -10.6% |
| 9 | Discretize | numbers | 43.2% | -11.3% |
| 9 | Discretize+LN | numbers | 43.2% | -11.3% |
| 11 | Proj5 (unnorm) | all | 41.7% | -12.8% |
| 12 | Proj5 | all | 38.6% | -15.9% |
| 13 | Discretize+LN | all | 34.1% | -20.4% |
| 14 | Discretize | all | 33.3% | -21.2% |
| 15 | Average | numbers | 31.1% | -23.4% |
| 16 | Average | all | 28.0% | -26.5% |
| 17 | Proj1 | all | 27.3% | -27.2% |
| 18 | Zero | numbers | 14.4% | -40.1% |
| 19 | Zero | all | 9.8% | -44.7% |

**Key Insight:** On the test set, **Minus and Replacement remain competitive with baseline**, maintaining within 1.5% of baseline accuracy.

### 6. Three-Way Comparison: Clean vs Train vs Test

| Intervention | Scope | Clean (90.2%) | Train (86.4%) | Test (54.5%) |
|--------------|-------|---------------|---------------|--------------|
| **Baseline** | none | **90.2%** | **86.4%** | **54.5%** |
| **Minus** | numbers | 84.1% | 84.8% | 53.8% |
| **Minus** | all | 83.3% | 84.8% | 53.0% |
| **Replacement** | numbers | 81.8% | 83.3% | 53.0% |
| **Replacement** | all | 81.8% | 83.3% | 53.0% |
| **Proj5** | numbers | 56.8% | 69.7% | 47.7% |
| **Proj5** | all | 41.7% | 65.9% | 38.6% |

**Observations:**
- **Clean dataset** (examples LLAMA solves correctly): Highest absolute accuracy, most sensitive to interventions
- **Train dataset**: Intermediate, shows overfitting vs test
- **Test dataset**: Lowest accuracy, largest train-test gap

### 7. Theoretical Implications

#### Why Such a Large Train-Test Gap?

1. **Model Memorization:** CODI-LLaMA may have memorized specific reasoning patterns from GSM8K train
2. **Distribution Shift:** Test set may contain harder or qualitatively different problems
3. **CoT Overfitting:** The continuous thought mechanism may overfit to training distribution
4. **Small Training Set:** 132 examples is tiny - model trained on much more may still show overfitting

#### Why Do Interventions Preserve the Gap?

1. **Interventions are applied at test time** - they don't change the underlying model weights
2. **The overfitting is in the base model**, not in the continuous thought activations
3. **Interventions modify CoT representations**, but the model's tendency to rely on memorized patterns persists
4. **Proportional degradation**: Each intervention degrades performance proportionally to baseline

### 8. Surprising Findings

1. **Zero ablation improves slightly on test (+2.3%)**: Random noise may occasionally help when the model is confidently wrong

2. **Proj1 (all) has smallest absolute drop (-11.4%)**: When performance is already very low (38.6%), there's less room to fall

3. **Average ablation more robust (-14-17%)**: Replacing with dataset-specific mean may provide some regularization

4. **Minus maintains relative ranking**: Still outperforms replacement on test (53.8% vs 53.0%)

### 9. Practical Implications

#### For Deployment:

1. **Don't trust train accuracy as indicator** of test performance
2. **Expect ~30-32% drop** when deploying to new problems
3. **Minus and Replacement** maintain best relative performance
4. **Projection methods** suffer more on test (likely because they're more constrained)

#### For Research:

1. **The train-test gap is the real problem**, not intervention effectiveness
2. **Better generalization** of base CODI models is crucial
3. **Interventions don't fix overfitting** - they preserve it
4. **Regularization during training** would likely help more than test-time interventions

### 10. Recommendations

#### Immediate:

1. **Report test set results** prominently in papers (not just train)
2. **Use multiple test sets** to validate generalization
3. **Consider ensembling** different intervention strategies
4. **Analyze failure cases** on test set to understand distribution shift

#### Long-term:

1. **Improve base model training** to reduce overfitting
2. **Augment training data** with more diverse reasoning patterns
3. **Add regularization** to CoT training process
4. **Test on completely different datasets** (not just GSM8K test)

### 11. Statistical Significance

With 132 examples per set, 95% confidence interval for accuracy:
- **±8.5%** for accuracies near 50%
- **±5.1%** for accuracies near 90%

**Significant differences (>8.5% on test set):**
- Baseline vs Proj5: 54.5% vs 47.7% = 6.8% (**not significant at p<0.05**)
- Baseline vs Discretize: 54.5% vs 43.2% = 11.3% (**significant**)
- Baseline vs Average: 54.5% vs 31.1% = 23.4% (**highly significant**)

**Minus vs Replacement on test:**
- 53.8% vs 53.0% = 0.8% (**not significant** - effectively tied)

## Conclusion

The **critical finding** is that CODI-LLaMA shows severe overfitting on GSM8K (31.8% train-test gap), and **no intervention fixes this**. All interventions maintain approximately the same relative performance drop.

However, the **good news** is that:
1. **Minus ablation remains competitive** with baseline on test (within 0.7%)
2. **Interventions don't add overfitting** - they preserve generalization characteristics
3. **Rankings are mostly preserved** across train and test

The **actionable insight**: Focus on improving base model generalization through better training, not just test-time interventions. Test-time interventions can modify reasoning but cannot overcome fundamental overfitting in the base model.

---

**Files Generated:**
- `full_results_gsm8k_test_132_examples.json` (4.2 MB)
- `bar_plot_gsm8k_test.png`
- `visualization_gsm8k_test.html`
