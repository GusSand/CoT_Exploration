# Data Quality Report - Deception Detection Dataset

**Date**: 2025-10-25
**Analyst**: PM Role
**Purpose**: Validate data before re-running corrected deception detection experiment

---

## Executive Summary

**Recommendation**: Use **probe_dataset_gpt2.json** (1000 samples) after removing 5 duplicates → 995 unique samples.

| Criteria | Status |
|----------|--------|
| **Duplicates** | ⚠️ 5 duplicates (0.5%), all with matching labels - SAFE TO REMOVE |
| **Labels** | ✅ All valid booleans, no contradictions |
| **Balance** | ✅ Perfect 50/50 split |
| **Sample Size** | ✅ Sufficient for linear models with regularization |
| **Data Completeness** | ✅ All fields present and valid |
| **Ready for Use** | ✅ YES - after removing duplicates |

---

## Dataset Comparison

### Available Datasets

| File | Total | Unique | Balance | Duplicates | Tokens/Layer | Test Set | Feature/Sample Ratio | Verdict |
|------|-------|--------|---------|------------|--------------|----------|---------------------|---------|
| **probe_dataset_gpt2.json** | 1000 | 995 | 500+500 | 5 (0.5%) | 6 | 200 | 0.96 | ✅ **RECOMMENDED** |
| probe_dataset_gpt2_clean.json | 784 | 784 | 392+392 | 0 | 6 | 157 | 1.23 | ⚠️ Small, overfitting risk |
| probe_dataset_gpt2_CORRUPTED.json | 2000 | 1979 | 1000+1000 | 21 (1.1%) | 16 | 400 | 0.48 | ⚠️ Different structure |

### Detailed Analysis

#### ✅ **probe_dataset_gpt2.json** (RECOMMENDED)
- **Samples**: 1000 total, 995 unique (after deduplication)
- **Balance**: 500 honest + 500 deceptive (perfect balance)
- **Duplicates**: 5 pairs (0.5%) - all have matching labels ✅
- **80/20 Split**: 800 train / 200 test
  - Train: ~400 per class ✅ SUFFICIENT
  - Test: ~100 per class ✅ SUFFICIENT
- **Feature/Sample Ratio**: 768 features / 800 train = **0.96**
  - Status: ⚠️ CAUTION - regularization critical but acceptable
- **Power**: Can detect effects >7% from chance (95% CI)
- **Recommendation**: **USE THIS** - Best balance of size and quality

#### ⚠️ probe_dataset_gpt2_clean.json
- **Samples**: 784 total (all unique)
- **Balance**: 392 + 392 (perfect balance)
- **Duplicates**: None ✅
- **80/20 Split**: 627 train / 157 test
  - Train: ~313 per class ✅ Acceptable
  - Test: ~78 per class ⚠️ Marginal
- **Feature/Sample Ratio**: 768 / 627 = **1.23**
  - Status: ❌ DANGEROUS - underdetermined system, severe overfitting risk
- **Power**: Can detect effects >8% from chance
- **Recommendation**: **AVOID** - Too small, feature/sample ratio > 1

#### ⚠️ probe_dataset_gpt2_CORRUPTED.json
- **Samples**: 2000 total, 1979 unique
- **Balance**: 1000 + 1000 (perfect balance)
- **Duplicates**: 21 (1.1%)
- **80/20 Split**: 1600 train / 400 test
  - Train: ~800 per class ✅ EXCELLENT
  - Test: ~200 per class ✅ EXCELLENT
- **Feature/Sample Ratio**: 768 / 1600 = **0.48**
  - Status: ✅ EXCELLENT - ample samples, minimal overfitting risk
- **BUT**: Has **16 tokens per layer** instead of 6 - different structure!
- **Recommendation**: **INVESTIGATE** - Why different token count? If valid, this is best option.

---

## Data Quality Checks - Detailed Results

### ✅ Check 1: Duplicate Detection

**probe_dataset_gpt2.json** - 5 duplicate pairs found:

| Index Pair | Label Match | Status |
|------------|-------------|--------|
| 87, 133 | True = True | ✅ SAFE |
| 130, 172 | True = True | ✅ SAFE |
| 49, 205 | True = True | ✅ SAFE |
| 12, 379 | True = True | ✅ SAFE |
| 180, 406 | True = True | ✅ SAFE |

**Verdict**: All duplicates have matching labels. Safe to remove without data loss.

### ✅ Check 2: Label Validation

For all datasets:
- ✅ All labels are valid booleans (True/False)
- ✅ No None or missing labels
- ✅ No contradictions (same Q+A always has same label)

### ✅ Check 3: Data Completeness

For all datasets:
- ✅ All samples have 'question' field
- ✅ All samples have 'answer' field
- ✅ All samples have 'thoughts' field
- ✅ All samples have 'is_honest' label

### ✅ Check 4: Structure Consistency

For all datasets:
- ✅ All samples have consistent layer structure
- ✅ All samples have consistent tensor shapes within each file
- ✅ Layers present: layer_4, layer_8, layer_11 (early, middle, late)

---

## Sample Size Sufficiency

### Guidelines for Binary Classification
- **Minimum**: 100-200 samples per class
- **Good practice**: 300-500 samples per class
- **Research standard**: 500-1000 samples per class

### Our Data (probe_dataset_gpt2.json after deduplication)
- **Total**: 995 samples
- **Per class**: ~497-498 per class
- **Train (80%)**: ~400 per class ✅ **GOOD**
- **Test (20%)**: ~100 per class ✅ **SUFFICIENT**

**Assessment**: ✅ Meets "good practice" standard for linear classification

---

## Statistical Power Analysis

### Effect Size Detection (probe_dataset_gpt2.json)

With 200 test samples (100 per class):
- **Standard Error**: 3.54%
- **Minimum Detectable Difference** (95% CI): ±7.07%
- **Reliable Range**: 42.9% to 57.1%

**Interpretation**:
- Can reliably detect if accuracy is significantly different from random (50%)
- Effects smaller than 7% may not reach significance
- Current results (~48%) are NOT significantly different from random ✅
- Response tokens (70.5%) are clearly and significantly better ✅

---

## Feature-to-Sample Ratio Analysis

### Critical Threshold
For linear models: Feature/Sample ratio should be **< 1.0** (ideally < 0.5)

### Our Data
- **Feature dimensionality**: 768 (GPT-2 hidden size)
- **Train samples** (80% of 995): 796
- **Ratio**: 768 / 796 = **0.965**

**Assessment**: ⚠️ **BORDERLINE** - Right at the edge of safe zone

**Mitigation Required**:
- ✅ Use L2 regularization (already using LogisticRegressionCV with C grid)
- ✅ Cross-validation for hyperparameter selection (already doing)
- ⚠️ Consider L1 regularization (Lasso) for feature selection
- ⚠️ Monitor train/test gap for overfitting

**Current Observed Overfitting**:
- Train accuracy: ~63%
- Test accuracy: ~48%
- Gap: 15pp ⚠️ Confirms overfitting concern

---

## Question: Why is "CORRUPTED" file larger and different?

The file labeled "CORRUPTED" has:
- **2000 samples** (vs 1000)
- **16 tokens per layer** (vs 6)

**Hypothesis**: This might be from a different extraction configuration or includes additional data. The "CORRUPTED" label may be misleading.

**Action Required**: Investigate why this file has different structure. If valid, it's actually the BEST option (feature/sample ratio = 0.48).

---

## Recommendations

### For Immediate Use

1. ✅ **USE**: `probe_dataset_gpt2.json` (1000 samples)
2. ✅ **PREPROCESS**: Remove 5 duplicates → 995 unique samples
3. ✅ **SPLIT**: 80/20 train/test (~796 train, ~199 test)
4. ✅ **REGULARIZE**: Use LogisticRegressionCV with C grid search
5. ✅ **VALIDATE**: Report both train and test accuracy to monitor overfitting

### For Future Improvement

1. 🔍 **INVESTIGATE**: Why does "CORRUPTED" file have 16 tokens? Is it usable?
2. 📊 **COLLECT MORE DATA**: Aim for 2000+ samples to improve feature/sample ratio
3. 🧪 **TRY FEATURE SELECTION**: Use L1 regularization or PCA to reduce dimensionality
4. 🔄 **K-FOLD CV**: Consider 5-fold or 10-fold CV instead of single train/test split

---

## Data Deduplication Script

To remove duplicates from `probe_dataset_gpt2.json`:

```python
import json
import hashlib

# Load data
with open('probe_dataset_gpt2.json', 'r') as f:
    data = json.load(f)

# Remove duplicates
seen = {}
unique_samples = []

for sample in data['samples']:
    key = f"{sample['question']}|||{sample['answer']}"
    qa_hash = hashlib.md5(key.encode()).hexdigest()

    if qa_hash not in seen:
        seen[qa_hash] = True
        unique_samples.append(sample)

# Update counts
honest = sum(1 for s in unique_samples if s['is_honest'] == True)
deceptive = sum(1 for s in unique_samples if s['is_honest'] == False)

data['samples'] = unique_samples
data['n_honest'] = honest
data['n_deceptive'] = deceptive

# Save deduplicated data
with open('probe_dataset_gpt2_deduplicated.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Original: {len(data['samples'])} samples")
print(f"Deduplicated: {len(unique_samples)} samples")
print(f"Removed: {len(data['samples']) - len(unique_samples)} duplicates")
```

---

## Conclusion

**Data Quality**: ✅ **ACCEPTABLE** for linear classification with proper regularization

**Recommended Dataset**: `probe_dataset_gpt2.json` after removing 5 duplicates (995 unique samples)

**Key Concerns**:
- ⚠️ Feature/sample ratio = 0.965 (borderline, requires strong regularization)
- ⚠️ Observed 15pp overfitting gap (train 63%, test 48%)
- ✅ Sample size sufficient for detecting moderate effects (>7%)
- ✅ Current negative result (~48% test accuracy) is statistically reliable

**Ready to Proceed**: ✅ YES - Data is clean, balanced, and sufficient for proper evaluation

**Next Step**: Define experiment requirements with PM to design proper methodology with train/test split.
