# Developer Handoff: Position-wise Patching Experiments

**Date:** 2025-10-30
**From:** Product Manager + Architect
**To:** Developer
**Status:** ✅ READY FOR DEVELOPMENT

---

## Quick Summary

Implement two experiments to understand position-specific effects in continuous thought (CoT) tokens:

1. **Experiment 1:** Position-wise patching - Test all (layer, position) combinations individually
2. **Experiment 2:** Iterative patching - Patch position N, generate N+1 using patched value

**Dataset:** 57 filtered pairs (removed 9 same-answer edge cases from original 66)
**Timeline:** ~11 hours (8.5h dev + 2h compute + 0.5h doc)
**Compute:** 2 hours on A100

---

## Key Documents

📄 **PM Specification:** `docs/project/position_patching_experiments.md`
- Complete user stories (28 total across 7 epics)
- Acceptance criteria for each story
- Original cost estimates

📄 **Architecture Review:** `docs/architecture/position_patching_experiments_architecture.md`
- Data quality assessment
- Technical architecture for both experiments
- Metrics validation
- Risk assessment and mitigations
- Testing strategy

---

## Data Filtering Completed

✅ **Dataset filtered by Architect:** 66 → 57 pairs

**Location:**
- **Original:** `src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs.json` (66 pairs)
- **Filtered:** `src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs_filtered.json` (57 pairs) ← **USE THIS**

**Removed pairs:** 12, 55, 229, 287, 334, 397, 465, 475, 555
**Reason:** These had identical clean/corrupted answers due to integer rounding, which would contaminate the answer logit difference metric.

**Validation:**
- All 57 pairs have different clean/corrupted answers
- Answer differences range from 1 to 3600 (mean: 112.7)
- No duplicate questions
- All pairs have valid numeric answers

---

## Updated Cost Estimates (57 pairs)

### Experiment 1: Position-wise Patching
- **Forward passes:** 5,529 (57 baseline + 912 extraction + 4,560 patching)
- **Compute time:** 60 minutes (1 hour)
- **Storage:** 60 MB

### Experiment 2: Iterative Patching
- **Forward passes:** 5,472 (reusing Exp 1 activations)
- **Compute time:** 59 minutes (~1 hour)
- **Storage:** 95 MB

### Total
- **Forward passes:** 11,001
- **Compute time:** ~2 hours on A100
- **Storage:** 160 MB

---

## Implementation Priority

### Phase 1: Experiment 1 (Position-wise Patching)
**Why first:**
- Simpler implementation (no activation chaining complexity)
- Generates clean activations cache that Exp 2 can reuse
- Validates base approach before tackling iterative logic

**Key deliverables:**
1. Single-position patching infrastructure
2. Answer logit difference metric
3. Position-wise heatmap visualizations
4. Clean activations cache (~37 MB)

### Phase 2: Experiment 2 (Iterative Patching)
**Why second:**
- Reuses activations from Exp 1
- More complex (activation chaining requires careful implementation)
- Builds on validated approach from Exp 1

**Key deliverables:**
1. Iterative generation infrastructure
2. Activation chaining logic (most complex part!)
3. Trajectory visualizations
4. Comparison analysis with Exp 1

---

## Critical Implementation Details

### 1. Use Filtered Dataset
```python
# CORRECT
pairs_path = "results/prepared_pairs_filtered.json"  # 57 pairs

# INCORRECT - Don't use
# pairs_path = "results/prepared_pairs.json"  # 66 pairs (has same-answer pairs)
```

### 2. New Metric: Answer Logit Difference
```python
def compute_answer_logit_difference(
    patched_logits,           # [batch, seq_len, vocab]
    clean_answer_tokens,      # [num_tokens] - tokenized clean answer
    corrupted_answer_tokens,  # [num_tokens] - tokenized corrupted answer
    answer_start_pos          # int - position where answer starts
):
    """
    Measure how much patching shifts logits toward clean vs corrupted answer

    Returns:
        mean_diff: Mean(logits[clean_tokens]) - Mean(logits[corrupted_tokens])
        clean_score: Mean logit for clean answer tokens
        corrupted_score: Mean logit for corrupted answer tokens
    """
    # Extract logits at answer positions
    answer_logits = patched_logits[:, answer_start_pos:, :]

    # Get logits for clean answer tokens
    clean_logits = []
    for token in clean_answer_tokens:
        clean_logits.append(answer_logits[..., token].mean())

    # Get logits for corrupted answer tokens
    corrupted_logits = []
    for token in corrupted_answer_tokens:
        corrupted_logits.append(answer_logits[..., token].mean())

    # Average and compute difference
    clean_score = torch.tensor(clean_logits).mean()
    corrupted_score = torch.tensor(corrupted_logits).mean()
    mean_diff = clean_score - corrupted_score

    return float(mean_diff), float(clean_score), float(corrupted_score)
```

### 3. Activation Chaining (Experiment 2)
**Most complex part - requires careful attention!**

**Approach:** Patch position N, extract activation at position N+1, use that to patch position N+1 in next iteration.

```python
def iterative_patching_at_layer(model, corrupted_input, clean_acts, layer_idx, num_positions=5):
    """
    Run iterative patching at a single layer

    The key: Each iteration uses the GENERATED activation from previous iteration,
    not the pre-extracted clean activation.
    """
    trajectory = []

    for iteration in range(num_positions):
        # Patch position[iteration] with clean activation
        if iteration == 0:
            # First iteration: use clean activation
            patch_activation = clean_acts[layer_idx][iteration]
        else:
            # Later iterations: use generated activation from previous iteration
            patch_activation = trajectory[iteration-1]['next_activation']

        # Run forward pass with patching
        with ActivationPatcher(model, layer_idx, position=iteration,
                              replacement=patch_activation):
            output = model.forward(corrupted_input)

        # Extract activation at NEXT position (if not last iteration)
        next_act = None
        if iteration < num_positions - 1:
            next_act = extract_activation_at_position(
                model, corrupted_input, layer_idx, position=iteration+1
            )

        # Compute metrics
        metrics = compute_all_metrics(output, baseline_output, ...)

        trajectory.append({
            'iteration': iteration,
            'metrics': metrics,
            'next_activation': next_act
        })

    return trajectory
```

**Testing strategy for chaining:**
- Log activation norms at each step to verify they're changing
- Visual inspection: Print first few values of activations to ensure they differ
- Sanity check: Iteration 0 baseline should match no-patching baseline

### 4. Cache Clean Activations
**Save after Exp 1, reuse in Exp 2:**

```python
# After Exp 1 completes
torch.save({
    'clean_activations': clean_acts_dict,  # {pair_id: {layer_idx: {pos_idx: tensor}}}
    'metadata': {
        'num_pairs': 57,
        'num_layers': 16,
        'num_positions': 5,
        'hidden_dim': 2048
    }
}, 'results/clean_activations_cache.pt')

# In Exp 2
cache = torch.load('results/clean_activations_cache.pt')
clean_acts_dict = cache['clean_activations']
```

**Size:** ~37 MB (reasonable)

### 5. Memory Management
```python
# Between pairs, clear cache
torch.cuda.empty_cache()

# Use batch_size=1 throughout
config.BATCH_SIZE = 1

# Enable FP16 for efficiency
config.USE_FP16 = True
```

---

## Testing Requirements

### 1. TEST_MODE First
**Run on 5 pairs before full dataset:**

```python
# In config.py
TEST_MODE = True
TEST_SUBSET_SIZE = 5
```

**Validation checklist:**
- [ ] All metrics compute without errors
- [ ] Results structure is correct
- [ ] W&B logging works
- [ ] Memory stays under 10 GB
- [ ] Single pair completes in ~60 seconds

### 2. Experiment 1 Validation
- [ ] Sum of single-position KL ≈ all-positions KL (from layer-patching baseline)
- [ ] Sanity check: Layer 0, position 0 should have minimal effect
- [ ] Heatmap shows non-uniform patterns
- [ ] Pair 0: clean_answer=18, corrupted_answer=20 (verify answer logit diff is positive after patching)

### 3. Experiment 2 Validation
- [ ] Iteration 0 baseline matches no-patching baseline
- [ ] Activations change between iterations (log norms to verify)
- [ ] Trajectory plots show interpretable patterns (not flat lines or NaNs)
- [ ] Final metrics similar to (but not identical to) Exp 1 all-positions patching

---

## W&B Logging

### Experiment 1
```python
wandb.init(
    project="cot-exploration",
    name="10-30_llama_gsm8k_position_patching_20251030_HHMM"
)

# Per (pair, layer, position)
wandb.log({
    f"pair_{pair_id}/layer_{layer}/pos_{pos}/kl_div": ...,
    f"pair_{pair_id}/layer_{layer}/pos_{pos}/answer_logit_diff": ...,
})

# Aggregate
wandb.log({
    f"aggregate/layer_{layer}/pos_{pos}/mean_kl": ...,
})
```

### Experiment 2
```python
wandb.init(
    project="cot-exploration",
    name="10-30_llama_gsm8k_iterative_patching_20251030_HHMM"
)

# Per (pair, layer, iteration)
wandb.log({
    f"pair_{pair_id}/layer_{layer}/iter_{iter}/kl_div": ...,
    f"pair_{pair_id}/layer_{layer}/iter_{iter}/activation_sim_clean": ...,
})
```

---

## Directory Structure

```
src/experiments/
├── 10-30_llama_gsm8k_position_patching/
│   ├── config.py
│   ├── core/
│   │   ├── model_loader.py (copy from layer_patching)
│   │   ├── single_position_patcher.py (NEW)
│   │   ├── metrics.py (add answer_logit_diff)
│   ├── scripts/
│   │   ├── 1_load_data.py
│   │   ├── 2_run_position_patching.py
│   │   ├── 3_save_activations_cache.py (NEW)
│   │   ├── 4_visualize_heatmaps.py
│   ├── results/
│   │   ├── position_patching_results.json (output)
│   │   ├── clean_activations_cache.pt (output - for Exp 2)
│   │   ├── heatmaps/ (output)
│
├── 10-30_llama_gsm8k_iterative_patching/
│   ├── config.py
│   ├── core/
│   │   ├── model_loader.py (reuse)
│   │   ├── iterative_patcher.py (NEW - handles chaining)
│   │   ├── metrics.py (add activation_similarity)
│   ├── scripts/
│   │   ├── 1_load_data.py
│   │   ├── 2_load_activation_cache.py (NEW)
│   │   ├── 3_run_iterative_patching.py
│   │   ├── 4_visualize_trajectories.py
│   │   ├── 5_compare_to_exp1.py
│   ├── results/
│   │   ├── iterative_patching_results.json (output)
│   │   ├── trajectories/ (output)
```

---

## Documentation Requirements

After completing experiments:

1. **Update research journal** (`docs/research_journal.md`):
   - High-level summary of findings
   - Top 3-5 critical (layer, position) combinations
   - Key insights about iterative vs parallel patching

2. **Create detailed reports:**
   - `docs/experiments/10-30_llama_gsm8k_position_patching.md`
   - `docs/experiments/10-30_llama_gsm8k_iterative_patching.md`
   - Include: methodology, results, visualizations, error analysis

3. **Update DATA_INVENTORY.md:**
   - Add entries for filtered dataset
   - Add entries for both result files
   - Document clean activations cache

4. **Git commit:**
   ```bash
   git add src/experiments/10-30_llama_gsm8k_position_patching/
   git add src/experiments/10-30_llama_gsm8k_iterative_patching/
   git add docs/experiments/10-30_llama_gsm8k_*.md
   git add docs/research_journal.md
   git add docs/DATA_INVENTORY.md
   git commit -m "feat: Complete position-wise CoT patching experiments

   - Experiment 1: Position-wise patching across 16 layers × 5 positions
   - Experiment 2: Iterative patching with activation chaining
   - Dataset: 57 filtered pairs (removed 9 same-answer edge cases)
   - Results: Position-specific heatmaps and trajectory analysis
   - Compute: 2 hours on A100, 11,001 forward passes

   🤖 Generated with [Claude Code](https://claude.com/claude-code)

   Co-Authored-By: Claude <noreply@anthropic.com>"
   git push
   ```

---

## Questions/Issues?

If you encounter any issues during implementation:

1. **Data issues:** Check architecture doc section 1.2 for edge cases
2. **Memory issues:** Reduce to batch_size=1, add more frequent cache clearing
3. **Activation chaining unclear:** See architecture doc section 2.2 for detailed explanation
4. **Metrics computation:** Reference existing layer-patching code

**Ready to proceed?** Start with Experiment 1, validate with TEST_MODE, then move to Experiment 2.

---

**Approval Chain:**
- ✅ PM: User stories and cost estimates approved
- ✅ Architect: Technical design approved with 57 filtered pairs
- ⏳ Developer: Ready to implement

**Next Step:** Begin development of Experiment 1 (position-wise patching)
