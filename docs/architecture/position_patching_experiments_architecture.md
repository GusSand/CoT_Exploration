# Architecture Review: Position-wise CoT Activation Patching Experiments

**Date:** 2025-10-30
**Role:** Architect
**Status:** ✅ APPROVED WITH RECOMMENDATIONS

---

## Executive Summary

Architecture review for two position-wise patching experiments extending the layer-wise patching baseline. After thorough review, the approach is **APPROVED** with filtered dataset.

**Key Findings:**
- ✅ Data quality: **57 pairs** (filtered from 66), no duplicates, all properly labeled
- ✅ **Filtered out 9 pairs** with same clean/corrupted answers (integer rounding edge cases)
- ✅ Technical architecture is sound and reuses existing infrastructure
- ✅ Metrics are appropriate, new metric (answer logit diff) is well-designed
- ✅ Computational estimates updated for 57 pairs: **~2 hours total on A100**
- ✅ Memory management is safe (~4 GB peak)

---

## 1. Data Quality Assessment

### 1.1 Dataset: LLaMA Clean/Corrupted Pairs

**Source:** `corrected_llama_cot_clean_corrupted_pairs/llama_clean_corrupted_pairs.json`
**Original Data:** `src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs.json` (66 pairs)
**Filtered Data:** `src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs_filtered.json` (57 pairs)

**Statistics:**
- Total pairs: **57** (filtered from 66)
- Format: Clean/corrupted question pairs with known numeric answers
- Model: LLaMA-3.2-1B on GSM8K arithmetic problems
- All pairs have valid numeric answers (no nulls)
- No duplicate questions found
- **All pairs have different clean/corrupted answers** (same-answer pairs removed)

### 1.2 Data Quality Issues

#### Issue 1: Same-Answer Pairs (9 pairs, 13.6%) - **RESOLVED**

Nine pairs had identical clean and corrupted answers despite different questions. **These have been filtered out:**

| Pair ID | Clean Answer | Corrupted Answer | Reason |
|---------|-------------|------------------|---------|
| 12 | 13 | 13 | Integer division rounds both to same result |
| 55 | 14 | 14 | (30-2)/2 = 14, (31-2)/2 = 14 (floor division) |
| 229 | 21 | 21 | Percentage rounding |
| 287 | 8 | 8 | Integer division |
| 334 | 6 | 6 | (8+10+1+5)/4 = 6, (9+10+1+5)/4 = 6 (floor) |
| 397 | 25 | 25 | (300-50)/10 = 25, (300-51)/10 = 24.9 → 25 |
| 465 | 30 | 30 | Integer division |
| 475 | 20 | 20 | 240/(4×3) = 20, 241/(4×3) = 20 (floor) |
| 555 | 2 | 2 | Different calculation paths yield same result |

**Example:**
```
Clean: Jean has 30 lollipops. Eats 2, packages in bags of 2.
Answer: (30-2)/2 = 14 bags

Corrupted: Jean has 31 lollipops. Eats 2, packages in bags of 2.
Answer: (31-2)/2 = 14.5 → 14 bags (integer division)
```

**Resolution:**
✅ **FILTERED OUT** - Removed all 9 pairs with same answers to ensure clean experimental design.

**Removed pair IDs:** 12, 55, 229, 287, 334, 397, 465, 475, 555

**Rationale:**
- These pairs would contaminate the "answer logit difference" metric
- Unclear whether null effects are due to ineffective patching or inherent same-answer constraint
- Cleaner experimental design focuses on pairs where answers definitively differ
- Still have 57 high-quality pairs for robust analysis

### 1.3 Data Sufficiency

**Is 57 pairs enough?**

For exploratory research: **YES**
- Previous layer-patching experiment used 66 pairs successfully; 57 is comparable
- Focus is on understanding mechanisms, not statistical power
- Per-pair analysis is valuable for case studies
- Can identify patterns even with smaller N
- Answer differences range from 1 to 3600 (mean: 112.7) - good variation

For statistical conclusions:
- 57 pairs provide reasonable power for aggregate statistics
- Critical (layer, position) combinations should show strong effects if they exist
- 13.6% reduction from filtering is acceptable
- May need larger dataset for weak effects or interaction analyses

**Recommendation:** ✅ **57 pairs is SUFFICIENT** for these exploratory experiments.

### 1.4 Train/Test Split

**Current approach:** Using all 57 filtered pairs for analysis (no train/test split)

**Justification:**
- This is mechanistic analysis, not prediction/generalization
- We're measuring causal effects (patching), not training models
- Same data used in baseline layer-patching experiment
- All pairs from LLaMA CoT-dependent subset (already filtered for quality)

**Recommendation:** ✅ **No train/test split needed** - appropriate for causal intervention studies.

---

## 2. Technical Architecture Review

### 2.1 Experiment 1: Position-wise Patching

#### Architecture

```
For each of 66 pairs:
  1. Extract clean activations at all 16 layers × 5 positions
     └─> Use existing extract_activations_at_layer()
     └─> Store: clean_acts[layer_idx][position_idx] = activation tensor

  2. Run baseline corrupted (no patching)
     └─> Get baseline answer logits

  3. For each layer (0-15):
       For each position (0-4):
         a. Patch ONLY this (layer, position)
            └─> Modify ActivationPatcher to patch single position
         b. Run forward pass
         c. Compute metrics:
            - KL divergence (vs baseline)
            - L2 difference (vs baseline)
            - Prediction change rate
            - Answer logit difference (NEW - vs clean/corrupted answers)
         d. Store: results[pair_id][layer_idx][position_idx] = metrics
```

**Forward passes per pair:**
- Baseline: 1
- Activation extraction: 16 layers
- Position patching: 16 layers × 5 positions = 80
- **Total: 97 forward passes per pair**

**Total compute: 57 pairs × 97 passes = 5,529 forward passes**

#### Code Changes Needed

**1. Extend ActivationPatcher for single position:**
```python
# Current: patches all positions in a list
self.positions = positions  # e.g., [0, 1, 2, 3, 4]

# New: patch single position
self.position = position  # e.g., 2 (just one integer)
```

**Recommendation:** ✅ Create new `SinglePositionPatcher` class inheriting from `ActivationPatcher` for clarity.

**2. Add answer logit difference metric:**
```python
def compute_answer_logit_difference(
    patched_logits,      # [batch, seq_len, vocab]
    clean_answer_tokens, # [num_answer_tokens]
    corrupted_answer_tokens, # [num_answer_tokens]
    answer_start_pos     # int
):
    """
    Measure how much patching shifts logits toward clean vs corrupted answer

    Returns:
        mean_diff: Mean(logits[clean_tokens]) - Mean(logits[corrupted_tokens])
        clean_score: Mean logit for clean answer tokens
        corrupted_score: Mean logit for corrupted answer tokens
    """
```

**Concern:** What if clean/corrupted answers have different token lengths?

**Solution:** Average logits across tokens, then compare averages (not sum, which would favor longer answers).

**Recommendation:** ✅ Implement as proposed, averaging across token positions.

#### Memory Management

**Per pair memory:**
- Clean activations: 16 layers × 5 positions × 2048 dim × 4 bytes = **655 KB**
- Baseline output: Minimal (just logits)
- Patched output: Generated on-the-fly, not stored

**Peak memory:**
- Model: ~2.5 GB (LLaMA-1B + LoRA)
- Activations: 0.66 MB per pair
- Working memory: ~1 GB
- **Total: ~4 GB** (well within A100's 40 GB)

**Activation cache size:**
- 57 pairs × 16 layers × 5 positions × 2048 dim × 4 bytes = **~37 MB**

**Recommendation:** ✅ Memory usage is safe. Use `torch.cuda.empty_cache()` between pairs for good measure.

---

### 2.2 Experiment 2: Iterative Patching with Generation

#### Architecture

```
For each of 57 pairs:
  1. Extract clean activations at all layers (reuse from Exp 1 if available)

  2. Run baseline corrupted (no patching)
     └─> Store baseline answer logits

  3. For each layer (0-15):
       # Iterative generation loop
       corrupted_input = prepare_corrupted_input()

       For iteration in [0, 1, 2, 3, 4]:  # One per CoT position
         a. Patch position[iteration] with clean activation
         b. Run forward pass
         c. Extract activation at position[iteration+1] from output
            └─> This becomes input for next iteration
         d. Measure metrics at answer tokens:
            - KL divergence (vs baseline)
            - Answer logit difference
            - Activation similarity (cos_sim to clean/corrupted)
         e. Store: results[pair_id][layer_idx][iteration] = {
              metrics,
              intermediate_activation
            }

       After iteration 4:
         f. Final forward pass for answer generation
         g. Compute final metrics
```

**Forward passes per layer:**
- 5 iterations (one per position)
- Each iteration: patch position N → run model → extract position N+1

**Total compute: 57 pairs × 16 layers × 5 iterations = 4,560 forward passes**

#### Key Technical Challenge: Activation Chaining

**Problem:** How to ensure activation at position N+1 from iteration N is correctly used in iteration N+1?

**Current architecture:**
```python
# Iteration 0:
patch_position_0(clean_act_0)
output = model.forward()
extracted_act_1 = extract_from_output(position=1)

# Iteration 1:
patch_position_1(extracted_act_1)  # Use extracted, not clean!
output = model.forward()
extracted_act_2 = extract_from_output(position=2)
```

**Critical Question:** Should we patch position N+1 with:
- **Option A:** Extracted activation from previous iteration (model's own generation)
- **Option B:** Clean activation from clean question (external injection)

**Based on PM requirements:** "Patch position N, then generate position N+1 using patched value"

**Interpretation:** We patch position N with clean activation, then let the model naturally generate position N+1, then patch that generated position N+1 in the next iteration.

**Recommendation:** ✅ **Use Option A** - chain the model's own generated activations. This tests compounding effects.

**Implementation:**
```python
def iterative_patching_at_layer(model, corrupted_input, clean_acts_all_positions, layer_idx):
    """
    Run iterative patching at a single layer
    """
    current_input = corrupted_input
    trajectory = []

    for iteration in range(num_latent):
        # Patch position[iteration] with clean activation
        with ActivationPatcher(model, layer_idx, position=iteration,
                              replacement=clean_acts_all_positions[iteration]):
            output = model.forward(current_input)

        # Extract activation at position[iteration+1] (if not last)
        if iteration < num_latent - 1:
            next_act = extract_activation_at_position(
                output, layer_idx, position=iteration+1
            )

        # Compute metrics
        metrics = compute_all_metrics(output, baseline_output,
                                      clean_answer_tokens,
                                      corrupted_answer_tokens)

        trajectory.append({
            'iteration': iteration,
            'metrics': metrics,
            'next_activation': next_act if iteration < num_latent - 1 else None
        })

    return trajectory
```

**Concern:** This approach requires modifying input activations on-the-fly, not just patching during forward pass.

**Alternative approach:** Store sequence of activations and use them in next forward pass.

**Recommendation:** ⚠️ **ARCHITECT FLAG**: This requires careful implementation. Developer should choose between:
1. **Simpler:** Patch all previous positions with generated activations in each iteration
2. **Complex but cleaner:** Modify model state to use chained activations

**Suggested:** Use approach 1 (simpler) for initial implementation.

#### Memory Management

**Per layer memory:**
- Clean activations: 5 positions × 2048 dim × 4 bytes = **41 KB**
- Generated activations: 5 iterations × 2048 × 4 = **41 KB**
- Trajectory results: 5 iterations × metrics (~1 KB) = **5 KB**

**Peak memory:**
- Model: ~2.5 GB
- Per-pair working memory: ~0.1 MB
- **Total: ~3 GB** (safe)

**Recommendation:** ✅ Memory is not a concern.

---

### 2.3 Metrics Validation

#### Existing Metrics (Reused)

**1. KL Divergence** ✅
- Measures distribution shift at answer tokens
- Well-established in mechanistic interpretability
- Sensitive to small changes in probability mass

**2. L2 Logit Difference** ✅
- Measures raw logit vector distance
- Complementary to KL (captures magnitude, not just distribution)

**3. Prediction Change Rate** ✅
- Binary metric: did top-1 prediction change?
- Useful for understanding when effects are strong enough to flip predictions

#### New Metric: Answer Logit Difference

**Purpose:** Directly measure how much patching shifts logits toward clean vs corrupted answer.

**Formula:**
```python
clean_logits = patched_output[:, answer_start_pos:, vocab_size][clean_answer_token_ids]
corrupted_logits = patched_output[:, answer_start_pos:, vocab_size][corrupted_answer_token_ids]

# Average across answer tokens
mean_clean = clean_logits.mean()
mean_corrupted = corrupted_logits.mean()

answer_logit_diff = mean_clean - mean_corrupted
```

**Interpretation:**
- **Positive:** Patching shifts toward clean answer (recovery)
- **Negative:** Patching shifts toward corrupted answer (worsening)
- **Zero:** No change in answer preference

**Concern:** What if answers are the same (9 pairs)?

**Solution:** For same-answer pairs, this metric will be near-zero regardless of patching. Flag these in analysis.

**Recommendation:** ✅ **APPROVED** - This metric is well-designed and answers the key question: "Does patching recover the correct answer?"

#### Additional Metric for Exp 2: Activation Similarity

**Purpose:** Track how much generated activations resemble clean vs corrupted.

**Formula:**
```python
def compute_activation_similarity(generated_act, clean_act, corrupted_act):
    cos_sim_clean = cosine_similarity(generated_act, clean_act)
    cos_sim_corrupted = cosine_similarity(generated_act, corrupted_act)

    return {
        'similarity_to_clean': cos_sim_clean,
        'similarity_to_corrupted': cos_sim_corrupted,
        'bias': cos_sim_clean - cos_sim_corrupted  # Positive = closer to clean
    }
```

**Recommendation:** ✅ **ADD THIS** to Experiment 2 for richer analysis of iterative effects.

---

## 3. Computational Cost Validation

### 3.1 Experiment 1: Position-wise Patching

**Updated Calculation (57 pairs):**
- Baseline: 57 × 1 = 57 passes
- Activation extraction: 57 × 16 = 912 passes
- Position patching: 57 × 16 × 5 = 4,560 passes
- **Total: 5,529 passes**

**Corrected estimate:**
- Forward pass time: ~0.65s per pass on A100
- Total time: 5,529 × 0.65s / 60 = **60 minutes (1 hour)**

**Storage:** ~60 MB (results + activations)

### 3.2 Experiment 2: Iterative Patching

**Updated Calculation (57 pairs):**
- Baseline: 57 × 16 = 912 passes
- Activation extraction: 57 × 16 = 912 passes (reuse from Exp 1)
- Iterative patching: 57 × 16 × 5 = 4,560 passes
- **Total: 5,472 passes** (reusing Exp 1 activations)

**Corrected estimate:**
- Forward pass time: ~0.65s per pass on A100
- Total time: 5,472 × 0.65s / 60 = **59 minutes (~1 hour)**

**Storage:** ~95 MB (results + trajectory data)

**Optimization:** Run Exp 1 first, save extracted clean activations, reuse in Exp 2.

### 3.3 Total Cost Summary (Updated for 57 pairs)

| Experiment | Forward Passes | Compute Time | Storage |
|-----------|---------------|-------------|---------|
| Exp 1 | 5,529 | **~60 min (1.0 hr)** | 60 MB |
| Exp 2 (reusing acts) | 5,472 | **~59 min (1.0 hr)** | 95 MB |
| **Total** | **11,001** | **~2.0 hours** | **155 MB** |

**Savings from filtering 9 pairs:**
- Forward passes reduced: 12,738 → 11,001 (13.6% reduction)
- Time saved: ~16 minutes
- Storage saved: ~25 MB

**Recommendation:** ✅ **Estimates are accurate for 57 pairs.**

---

## 4. Architecture Decisions

### 4.1 Code Organization

**Recommendation:** ✅ **TWO SEPARATE EXPERIMENTS** as PM proposed

**Directory structure:**
```
src/experiments/
├── 10-30_llama_gsm8k_position_patching/
│   ├── config.py
│   ├── core/
│   │   ├── model_loader.py (symlink or copy from layer_patching)
│   │   ├── activation_patcher.py (extend for single position)
│   │   ├── metrics.py (add answer_logit_diff)
│   ├── scripts/
│   │   ├── 1_load_data.py
│   │   ├── 2_run_position_patching.py
│   │   ├── 3_visualize_heatmaps.py
│   ├── results/
│
├── 10-30_llama_gsm8k_iterative_patching/
│   ├── config.py
│   ├── core/
│   │   ├── model_loader.py (reuse)
│   │   ├── iterative_patcher.py (NEW - handles chaining)
│   │   ├── metrics.py (add activation_similarity)
│   ├── scripts/
│   │   ├── 1_load_data.py
│   │   ├── 2_run_iterative_patching.py
│   │   ├── 3_visualize_trajectories.py
│   │   ├── 4_compare_to_exp1.py
│   ├── results/
```

**Rationale:**
- Clear separation of concerns
- Each experiment is self-contained
- Easy to run independently or in sequence
- Separate W&B experiments for clear tracking

### 4.2 Code Reuse Strategy

**Reuse from layer-patching:**
- ✅ `model_loader.py` - identical, symlink or copy
- ✅ `metrics.py` - extend with new metrics
- ⚠️ `activation_patcher.py` - needs modification for single position

**Recommendation:** Use **git submodules** or **shared core directory** if more experiments planned. For now, **copy and extend** is simpler.

### 4.3 Data Flow

**Optimization opportunity:**

```
Exp 1: Extract clean activations → Run position patching → Save results
                ↓
         (Cache activations)
                ↓
Exp 2: Load cached activations → Run iterative patching → Save results
```

**Recommendation:** ✅ **SAVE EXTRACTED ACTIVATIONS** from Exp 1 to disk, reuse in Exp 2.

**File format:**
```python
# save_activations.py
torch.save({
    'clean_activations': {
        pair_id: {
            layer_idx: {
                position_idx: activation_tensor
            }
        }
    },
    'metadata': {
        'num_pairs': 57,
        'num_layers': 16,
        'num_positions': 5,
        'hidden_dim': 2048
    }
}, 'results/clean_activations_cache.pt')
```

**Size:** 57 pairs × 16 layers × 5 positions × 2048 dim × 4 bytes = **~37 MB** (reasonable)

---

## 5. Risk Assessment & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Same-answer pairs contaminate results** | Medium | Medium | Flag and analyze separately, don't exclude |
| **Activation chaining incorrect (Exp 2)** | Medium | High | Extensive logging, visual inspection of activations, unit tests |
| **Memory overflow on A100** | Low | High | Batch size=1, clear cache between pairs, monitor with nvidia-smi |
| **Compute time exceeds estimate** | Medium | Low | Run in tmux/screen, use TEST_MODE first, can continue if interrupted |
| **Position patching doesn't show effects** | Medium | Medium | Still scientifically interesting (null result), cross-validate with layer-patching |
| **Iterative effects saturate immediately** | Medium | Low | Still learn about information flow, compare to parallel |
| **W&B logging fails** | Low | Medium | Also save local JSON, can upload retroactively |

**Additional Risks:**

**Exp 2 specific - Activation extraction in middle of forward pass:**
- Current `extract_activations_at_layer` runs full forward pass with hook
- For iterative patching, we need activations from DURING the patched forward pass
- **Solution:** Modify hook to also store next position's activation, not just patch current

**Recommendation:** ⚠️ **DEVELOPER MUST CAREFULLY IMPLEMENT** activation chaining with thorough testing.

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Critical functions to test:**

1. **SinglePositionPatcher:**
```python
def test_single_position_patching():
    # Verify only one position is modified
    # Verify other positions remain unchanged
    # Compare to multi-position patching baseline
```

2. **Answer logit difference:**
```python
def test_answer_logit_difference():
    # Test with same answers (should be near-zero)
    # Test with different answers (should be positive after patching)
    # Test with different token lengths
```

3. **Iterative activation chaining:**
```python
def test_activation_chaining():
    # Verify activation at position N from iteration N is used in iteration N+1
    # Verify patching doesn't affect other positions
    # Compare to manual chaining
```

### 6.2 Integration Tests

**TEST_MODE validation (5 pairs):**

1. **Exp 1:**
   - Run on 5 pairs
   - Verify results structure is correct
   - Check that sum of single-position KL ≈ all-positions KL (from layer-patching)
   - Visual inspection of one heatmap

2. **Exp 2:**
   - Run on 5 pairs, 1-2 layers
   - Verify activation chaining is working (log activations)
   - Check trajectory plots make sense
   - Compare final metrics to Exp 1

### 6.3 Validation Checks

**Cross-experiment consistency:**

1. **Baseline metrics match:**
   - Exp 1 baseline == Exp 2 iteration 0 baseline
   - Both match layer-patching baseline

2. **Approximate equivalence:**
   - Exp 1 "all 5 positions patched" ≈ Exp 2 "iteration 4 final result"
   - (May not be exact due to chaining effects)

3. **Critical layer correlation:**
   - Layers identified as critical in layer-patching should show strong effects in both new experiments

**Recommendation:** ✅ **IMPLEMENT ALL THREE LEVELS** of testing before full run.

---

## 7. Final Architecture Recommendations

### ✅ APPROVED

1. **Overall approach:** Sound, builds properly on existing work
2. **Data quality:** Adequate for exploratory research, interesting edge cases
3. **Metrics:** Well-designed, answer critical research questions
4. **Code organization:** Clean separation, good reuse strategy
5. **Computational cost:** Reasonable, fits within A100 budget

### ⚠️ RECOMMENDATIONS

1. **Update compute estimates:** ~70 min for Exp 1, ~68 min for Exp 2 (total ~2.3 hrs)
2. **Cache clean activations:** Save after Exp 1, reuse in Exp 2 (saves ~15 min + ensures consistency)
3. **Flag same-answer pairs:** Analyze separately in results, document as scientifically interesting
4. **Add activation similarity metric:** Enhances Exp 2 analysis
5. **Thorough testing of activation chaining:** Critical for Exp 2 validity
6. **Memory monitoring:** Add `torch.cuda.empty_cache()` between pairs

### 🔍 DEVELOPER ATTENTION REQUIRED

1. **Activation chaining (Exp 2):** Most complex part, needs careful implementation and testing
2. **Single-position patching:** Should create clean abstraction (new class or modify existing)
3. **Answer tokenization:** Handle variable-length numeric answers correctly
4. **Logging:** Extensive logging for trajectory analysis in Exp 2

---

## 8. Architecture Approval

**Status:** ✅ **APPROVED FOR DEVELOPMENT**

**Conditions:**
1. Implement TEST_MODE validation before full run
2. Add activation similarity metric to Exp 2
3. Update PM cost estimates in documentation
4. Thoroughly test activation chaining

**Next Steps:**
1. Developer reviews this architecture document
2. Implements user stories from PM specification
3. Runs TEST_MODE validation
4. Gets architect approval on test results before full run
5. Proceeds with full experiments

---

**Document Created:** 2025-10-30
**Architect:** Claude Code
**Approved For:** Development Phase
**Next Review:** After TEST_MODE validation
