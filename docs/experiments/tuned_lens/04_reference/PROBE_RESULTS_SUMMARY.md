# Tuned Lens Probe Training Results Summary

## Training Completed Successfully

### Balanced Training Approach
- **Training Script**: 
- **Approach**: Hierarchical/balanced sampling to address token frequency imbalance
- **Training Examples Scanned**: 10,000
- **Raw Samples Collected**: 60,000 (10K examples × 6 iterations)
- **Balanced Samples Used**: 28,899 (downsampled with inverse frequency weighting)
- **Training Epochs**: 50
- **Training Accuracy**: 30.14%
- **Test Accuracy**: 30.00% (on 100 test examples)

### Token Distribution Before/After Balancing

**Before Balancing** (highly imbalanced):
- '>>': 15.03%
- 'The': 8.80%
- '-': 5.79%
- '/': 4.12%
- '*': 3.04%

**After Balancing** (much more uniform):
- Most tokens at ~0.10% each
- Rare tokens oversampled (e.g., 'ESH': 1 → 28 samples)
- Frequent tokens undersampled (e.g., '>>': 9018 → 1 sample)

### Model & Probe Files

**Trained Probes**:
1. **1K Training (unbalanced)**: 
   - File: 
   - Training: 61.03% accuracy (6,000 samples)
   - Test: 36.33% accuracy

2. **Balanced Training (30K samples)**:
   - File: 
   - Training: 30.14% accuracy (28,899 samples)
   - Test: 30.00% accuracy

### Evaluation & Inspection Logs

**Test Evaluation**:
- File: 
- Test Accuracy: 30.00% (180/600 correct)
- Per-Iteration Accuracy:
  - Iteration 1: 26.00%
  - Iteration 2: 35.00%
  - Iteration 3: 34.00%
  - Iteration 4: 30.00%
  - Iteration 5: 26.00%
  - Iteration 6: 29.00%

**Detailed Inspection** (50 examples with full metrics):
- File: 
- Contains for each example:
  - Full question text
  - All 6 continuous thought iterations
  - Top-5 predictions from:
    - Direct Layer 10 decoding
    - Probed Layer 10 → 11
    - Ground Truth Layer 11
  - Metrics: cosine similarity, L2 distance, norms
  - Match indicators (✓/✗)

### Key Findings

1. **Balanced vs Unbalanced Training**:
   - Unbalanced (1K): 36.33% test accuracy (biased toward frequent tokens)
   - Balanced (30K): 30.00% test accuracy (more uniform across token vocabulary)

2. **Iteration Performance**:
   - Iteration 2 shows best performance (35% accuracy)
   - Early iterations (1, 5) slightly lower (~26%)

3. **Probe Behavior**:
   - Works well on structured patterns (e.g., Example 2: 6/6 matches)
   - Struggles with complex reasoning (e.g., Example 3: 0/6 matches)
   - Negative cosine similarity doesn't prevent correct predictions
   - High confidence targets easier to predict

### Manual Inspection Recommendations

For detailed analysis, examine:
1.  - Full 50-example inspection
2.  - Complete training log with token distributions
3.  - Test evaluation with first 5 examples

### Scripts Available

**Training**:
-  - Balanced sampling approach (✓ USED)
-  - Correct CODI loading, unbalanced
-  - Alternative loading method

**Evaluation**:
-  - Test set evaluation
-  - Detailed example inspection

### Memory Efficiency

Successfully avoided OOM by:
- Collecting then downsampling (60K → 29K samples)
- Using inverse frequency weighting for balance
- Training with 50 epochs instead of 100
- Freeing memory after collection phase

---

**Date**: 2025-10-28
**Server**: ssh root@213.173.105.6 -p 30129
**Location**: 
