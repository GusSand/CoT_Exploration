# Next Steps: Contrastive CODI Deception Detection

**Date**: October 31, 2025
**Status**: 🚨 **CRITICAL FIXES REQUIRED BEFORE RE-RUN**

## Summary

The contrastive CODI deception detection experiment **FAILED** due to critical experimental design flaws, but successfully built reusable infrastructure. Before the research question can be answered, fundamental methodological issues must be fixed.

## Critical Issues Discovered

### 1. **Data Leakage (CRITICAL)**
**Problem**: Probes trained and tested on identical data
```python
# WRONG: What we did
probe.fit(X_scaled, y)           # Train on all 288 samples
y_pred = probe.predict(X_scaled) # Test on same 288 samples → Invalid 100% accuracy
```

**Fix Required**:
```python
# CORRECT: Proper train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
probe.fit(X_train, y_train)      # Train on 80% (230 samples)
y_pred = probe.predict(X_test)   # Test on 20% (58 samples)
accuracy = accuracy_score(y_test, y_pred)  # Report THIS as final result
```

**File to Fix**: `train_deception_probes.py:60-63`

### 2. **Identical Features Problem (CRITICAL)**
**Problem**: Used same base model activations for both "CT tokens" and "regular hidden states"

**Fix Required**:
- Debug CODI model loading in `extract_activations.py`
- Extract actual CT token representations from trained CODI model
- Verify features are actually different between conditions

**File to Fix**: `extract_activations.py:181` - CODI model loading

### 3. **Missing Model Validation (CRITICAL)**
**Problem**: Never verified that actual CT tokens were extracted

**Fix Required**:
- Add explicit checks that CT tokens ≠ regular hidden states
- Verify CODI model contains trained continuous thought components
- Add feature difference validation before probe training

## Immediate Action Plan

### Step 1: Fix Probe Evaluation (HIGHEST PRIORITY)
```bash
# Edit train_deception_probes.py
# Lines 60-70: Replace train-on-all with proper train/test split
# Lines 35-40: Add stratified sampling
# Add explicit data leakage prevention checks
```

### Step 2: Debug CODI Model Loading
```bash
# Investigate model checkpoint structure:
find ~/codi_ckpt/contrastive_liars_llama1b_smoke_test/ -name "*.bin" -o -name "*.pt"

# Fix extract_activations.py line 181:
# Replace CODI.from_pretrained() with proper loading method
# Verify CT tokens are extracted correctly
```

### Step 3: Add Validation Checks
```python
# Add to pipeline:
assert not np.array_equal(ct_activations, regular_activations), "Features are identical!"
assert len(set(train_hashes) & set(test_hashes)) == 0, "Data leakage detected!"
```

### Step 4: Re-run with Proper Methodology
```bash
# After fixes:
cd /home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi
bash run_full_pipeline.sh
```

## Expected Results After Fixes

- **Realistic Accuracy**: 60-80% (not 100%)
- **Meaningful Comparison**: CT tokens vs regular hidden states should differ
- **Proper Evaluation**: Test accuracy on held-out data only
- **Cross-Validation**: Primary evaluation metric

## Research Question Status

**"Can continuous thought (CT) tokens encode deception intent when trained with contrastive examples?"**

**Current Answer**: **UNANSWERED** - Experiment invalid due to methodological flaws

**Timeline to Answer**:
- Fix implementation: ~2 hours
- Re-run experiment: ~1 hour
- Analysis: ~30 minutes
- **Total**: ~3.5 hours

## Reusable Components

✅ **What Works and Can Be Reused**:
- CODI training pipeline: `train_contrastive_codi.sh`
- Dataset creation: `create_contrastive_dataset.py`
- Model successfully trained (loss: 4.2 → 0.9)
- WandB integration: https://wandb.ai/gussand/huggingface/runs/6qkgea7f

❌ **What Needs Fixing**:
- Probe evaluation methodology
- CODI model loading for activation extraction
- Feature validation and comparison

## Validation Checklist

Before declaring experiment complete:
- [ ] **Different Features**: CT tokens ≠ regular hidden states (verify numerically)
- [ ] **Proper Split**: Probe trains on subset, tests on held-out data
- [ ] **Realistic Results**: Accuracy 60-80%, not 100%
- [ ] **No Data Leakage**: Zero question overlap between train/test
- [ ] **Multiple Seeds**: Test robustness across random seeds
- [ ] **Cross-Validation**: Use as primary evaluation metric

## Files to Modify

1. **`train_deception_probes.py`** (CRITICAL):
   - Lines 60-70: Add proper train/test split
   - Lines 35-40: Add stratified sampling
   - Add data leakage checks

2. **`extract_activations.py`** (CRITICAL):
   - Line 181: Fix CODI model loading
   - Add CT token validation
   - Verify feature differences

3. **`run_full_pipeline.sh`** (OPTIONAL):
   - Add validation steps
   - Improve error handling

## Documentation Updates Needed

After successful re-run:
- Update `docs/experiments/10-31_llama_liars_bench_contrastive_codi_smoke_test.md`
- Update `docs/research_journal.md`
- Update `docs/DATA_INVENTORY.md` (remove INVALID status)

---

**Key Takeaway**: This experiment demonstrates the critical importance of rigorous experimental design validation. The infrastructure is solid, but scientific methodology must be fixed before any conclusions can be drawn.