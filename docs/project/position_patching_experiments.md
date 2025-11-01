# Position-wise CoT Activation Patching Experiments

**Date:** 2025-10-30
**Role:** Product Manager
**Status:** ✅ READY FOR IMPLEMENTATION

---

## Executive Summary

Two new experiments extending the layer-wise patching work to understand position-specific effects in continuous thought (CoT) tokens:

1. **Position-wise Patching**: Test all (layer, position) combinations individually
2. **Iterative Patching**: Patch position N, then generate position N+1 using patched value

**Dataset:** 57 pairs (filtered from 66, removing same-answer edge cases)
**Timeline:** 10-12 hours (8h 40m dev + 2h compute + documentation)
**Compute:** ~2 hours on A100
**Storage:** 155 MB

---

## Background

Building on `src/experiments/10-30_llama_gsm8k_layer_patching`, which identified critical layers for CoT reasoning by patching all CoT positions simultaneously at each layer. Now we need finer granularity to understand:

- Are certain CoT positions more critical than others?
- Do effects compound when patching sequentially vs. in parallel?
- How do position-specific effects vary across layers?

---

## Requirements

### Dataset
- **Source:** Filtered 57 clean/corrupted pairs from layer-patching experiment
- **Original:** `src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs.json` (66 pairs)
- **Filtered:** `src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs_filtered.json` (57 pairs)
- **Format:** Each pair has clean/corrupted questions with known numeric answers (all different)
- **Filtering:** Removed 9 pairs where clean and corrupted answers were identical due to integer rounding

### Model
- **Model:** LLaMA-3.2-1B-Instruct
- **Checkpoint:** `/home/paperspace/codi_ckpt/llama_gsm8k/pytorch_model.bin`
- **Architecture:** 16 layers, 5 CoT positions, 2048 hidden dim
- **Task:** GSM8K arithmetic reasoning

### Metrics
1. **KL Divergence:** Between patched and baseline answer distributions
2. **L2 Difference:** Logit vector distance
3. **Prediction Change Rate:** How often top-1 prediction changes
4. **Answer Logit Difference (NEW):** Mean logit difference between clean and corrupted answers

### Infrastructure
- **Hardware:** A100 GPU
- **Framework:** PyTorch + Transformers
- **Logging:** Weights & Biases
- **Priority:** ASAP, both experiments equally important

---

## Experiment 1: Position-wise Patching

### Objective
Isolate the effect of patching individual (layer, position) combinations to create a fine-grained map of position-specific importance.

### Methodology
For each of 66 pairs:
1. Extract clean activations at all 16 layers × 5 positions
2. Run baseline corrupted (no patching)
3. For each of 16 layers:
   - For each of 5 positions:
     - Patch ONLY that single position at that layer
     - Run forward pass
     - Compute metrics comparing to baseline
4. Total: 80 patched runs + 1 baseline per pair = 5,529 forward passes

### Expected Output
- **Heatmap:** 16 (layers) × 5 (positions) showing mean KL divergence across all pairs
- **Top-K Critical Combinations:** Ranked list of most impactful (layer, position) pairs
- **Per-pair Analysis:** Individual heatmaps for detailed case studies
- **JSON Results:** Structured as `results[pair_id][layer_idx][position_idx] = {metrics}`

### User Stories

#### Epic 1.1: Position-wise Activation Patching Infrastructure

**Story 1.1.1: Create position-wise patching configuration**
- **Time:** 15 minutes
- **Acceptance Criteria:**
  - [ ] New directory: `src/experiments/10-30_llama_gsm8k_position_patching/`
  - [ ] Config inherits from layer-patching but adds position-specific params
  - [ ] W&B experiment name: `10-30_llama_gsm8k_position_patching`

**Story 1.1.2: Extend ActivationPatcher for single-position patching**
- **Time:** 20 minutes
- **Acceptance Criteria:**
  - [ ] Modify `ActivationPatcher` to accept single position (not list)
  - [ ] Or create `SinglePositionPatcher` subclass
  - [ ] Verify only one position is modified per forward pass

**Story 1.1.3: Add clean/corrupted answer logit difference metric**
- **Time:** 30 minutes
- **Acceptance Criteria:**
  - [ ] Extract logits for clean answer tokens from patched output
  - [ ] Extract logits for corrupted answer tokens from patched output
  - [ ] Compute mean logit difference (clean - corrupted)
  - [ ] Add to `metrics.py` as `compute_answer_logit_difference()`
  - [ ] Takes ground truth clean/corrupted answers as input
  - [ ] Log to W&B alongside KL divergence

**Story 1.1.4: Implement position-wise patching loop**
- **Time:** 45 minutes
- **Acceptance Criteria:**
  - [ ] Nested loop: for each pair → for each layer → for each position
  - [ ] Extract clean activations at all layers (reuse existing code)
  - [ ] Patch single position at single layer
  - [ ] Compute all 4 metrics
  - [ ] Progress bar shows (pair, layer, position) progress
  - [ ] Total: 5,346 forward passes

**Story 1.1.5: Save position-wise results with proper structure**
- **Time:** 20 minutes
- **Acceptance Criteria:**
  - [ ] JSON structure: `results[pair_id][layer_idx][position_idx] = {metrics}`
  - [ ] Include baseline metrics for comparison
  - [ ] Save to `results/position_patching_results.json`
  - [ ] Log all metrics to W&B

**Story 1.1.6: Create position-wise heatmap visualization**
- **Time:** 40 minutes
- **Acceptance Criteria:**
  - [ ] Aggregate KL divergence across pairs for each (layer, position)
  - [ ] Create heatmap: rows=layers (0-15), cols=positions (0-4)
  - [ ] Annotate top-5 critical combinations
  - [ ] Save as PNG and log to W&B
  - [ ] Create individual heatmaps for each pair

#### Epic 1.2: Testing and Validation

**Story 1.2.1: Test mode validation**
- **Time:** 15 minutes
- **Acceptance Criteria:**
  - [ ] TEST_MODE runs on 5 pairs
  - [ ] Verify metrics are reasonable
  - [ ] Check A100 memory usage
  - [ ] Verify W&B logging works

**Total Time:** 3 hours 5 minutes
**Compute Time:** 60 minutes on A100
**Storage:** 60 MB

---

## Experiment 2: Iterative Patching with Generation

### Objective
Test whether patching position N and then generating position N+1 using the patched value creates compounding or different effects compared to parallel patching.

### Methodology
For each of 57 pairs:
1. Run baseline corrupted (no patching)
2. For each of 16 layers:
   - **Iteration 0:** Patch position 0 (from clean), run model → get activation at position 1
   - **Iteration 1:** Patch position 1 (from previous iteration), run model → get position 2
   - **Iteration 2:** Patch position 2, run model → get position 3
   - **Iteration 3:** Patch position 3, run model → get position 4
   - **Iteration 4:** Patch position 4, run model → get final answer logits
   - Measure metrics after each iteration AND at final answer
3. Total: 5 iterations × 16 layers × 57 pairs + 912 baselines = 5,472 forward passes

### Expected Output
- **Trajectory Plots:** Line plots showing how KL divergence evolves position 0→1→2→3→4→answer
- **Per-layer Comparison:** Which layers show compounding vs. saturating effects
- **Exp 1 vs Exp 2 Comparison:** Side-by-side effectiveness analysis
- **JSON Results:** `results[pair_id][layer_idx][iteration] = {metrics, intermediate_activations}`

### User Stories

#### Epic 2.1: Iterative Generation Infrastructure

**Story 2.1.1: Create iterative patching experiment directory**
- **Time:** 10 minutes
- **Acceptance Criteria:**
  - [ ] New directory: `src/experiments/10-30_llama_gsm8k_iterative_patching/`
  - [ ] Config specifies iterative approach
  - [ ] W&B experiment name: `10-30_llama_gsm8k_iterative_patching`

**Story 2.1.2: Implement iterative position generation**
- **Time:** 1 hour
- **Acceptance Criteria:**
  - [ ] For each layer, iterate through 5 positions sequentially
  - [ ] Patch position N with clean activation
  - [ ] Run forward pass to generate position N+1
  - [ ] Extract activation at position N+1 for next iteration
  - [ ] Total: 5 forward passes per layer
  - [ ] Compare to baseline (1 forward pass, no patching)
  - [ ] Verify activations are correctly chained

**Story 2.1.3: Measure both intermediate and final effects**
- **Time:** 45 minutes
- **Acceptance Criteria:**
  - [ ] After each position, measure:
    - KL divergence at answer tokens
    - Answer logit difference (clean vs corrupted)
    - Intermediate activation cosine similarity to clean/corrupted
  - [ ] Store time series for all 5 iterations
  - [ ] Log trajectories to W&B

**Story 2.1.4: Run iterative patching for all layers**
- **Time:** 30 minutes
- **Acceptance Criteria:**
  - [ ] For each pair → for each layer → 5-step iteration
  - [ ] Total: 6,336 forward passes
  - [ ] Save results: `results[pair_id][layer_idx][iteration] = {metrics}`
  - [ ] Progress bar shows pair/layer/iteration

**Story 2.1.5: Create trajectory visualizations**
- **Time:** 40 minutes
- **Acceptance Criteria:**
  - [ ] Line plot: X=position (0-4), Y=KL divergence
  - [ ] One line per layer
  - [ ] Highlight critical layers from layer-patching experiment
  - [ ] Create aggregate (mean) and per-pair plots
  - [ ] Save as PNG and log to W&B

#### Epic 2.2: Comparison Analysis

**Story 2.2.1: Compare iterative vs. parallel patching**
- **Time:** 45 minutes
- **Acceptance Criteria:**
  - [ ] Load results from both experiments
  - [ ] For each layer, compare:
    - Final KL divergence (iterative vs. all-at-once)
    - Answer logit difference
    - Prediction accuracy
  - [ ] Side-by-side comparison plots
  - [ ] Statistical significance testing (t-test or Wilcoxon)
  - [ ] Document which approach is more effective

**Total Time:** 3 hours 50 minutes
**Compute Time:** 59 minutes on A100
**Storage:** 95 MB

---

## Epic 3: Documentation and Deliverables

**Story 3.1: Document results in research journal**
- **Time:** 20 minutes
- **Acceptance Criteria:**
  - [ ] Update `docs/research_journal.md` with TLDR for each experiment
  - [ ] Include top findings (critical positions/layers)
  - [ ] Link to detailed reports

**Story 3.2: Create detailed experiment reports**
- **Time:** 1 hour
- **Acceptance Criteria:**
  - [ ] `docs/experiments/10-30_llama_gsm8k_position_patching.md`
  - [ ] `docs/experiments/10-30_llama_gsm8k_iterative_patching.md`
  - [ ] Include: methodology, results, visualizations, error analysis, implications

**Story 3.3: Update DATA_INVENTORY.md**
- **Time:** 15 minutes
- **Acceptance Criteria:**
  - [ ] Add entries for position_patching_results.json
  - [ ] Add entries for iterative_patching_results.json
  - [ ] Document structure and usage

**Story 3.4: Commit and push all results**
- **Time:** 10 minutes
- **Acceptance Criteria:**
  - [ ] Git commit all new scripts
  - [ ] Git commit all documentation
  - [ ] Git commit result JSON files
  - [ ] Push to GitHub

**Total Time:** 1 hour 45 minutes

---

## Cost Summary

| Category | Experiment 1 | Experiment 2 | Documentation | **Total** |
|----------|-------------|-------------|---------------|-----------|
| **Dev Time** | 3h 5m | 3h 50m | 1h 45m | **8h 40m** |
| **Compute Time** | 60m | 59m | 0m | **2h 0m** |
| **Storage** | 60 MB | 95 MB | 5 MB | **160 MB** |

**Dataset:** 57 pairs (filtered from 66)
**Wall Time:** 10-12 hours (with parallel development + compute)
**A100 GPU Time:** ~2 hours total
**Cost:** Negligible (using existing infrastructure)

---

## Success Criteria

### Experiment 1
1. ✅ All 80 (layer, position) combinations tested per pair (57 pairs total)
2. ✅ Heatmap reveals position-specific patterns (not uniform)
3. ✅ Statistical validation: Some positions significantly more critical
4. ✅ Results reproducible with same random seed
5. ✅ All metrics logged to W&B without errors

### Experiment 2
1. ✅ Iterative generation completes for all 16 layers × 57 pairs
2. ✅ Trajectory plots show interpretable patterns
3. ✅ Comparison with Exp 1 shows measurable difference
4. ✅ Intermediate activations correctly chained
5. ✅ Final metrics align with or exceed parallel patching

### Cross-Experiment Validation
- Baseline metrics match across both experiments
- Exp 1 "all positions" ≈ Exp 2 "final iteration" (approximately)
- Critical layers from previous work show strong effects in both experiments

---

## Key Research Questions

### Experiment 1
- Are certain CoT positions more critical than others across all layers?
- Do different layers rely on different positions?
- Is there a pattern (early vs. late positions)?

### Experiment 2
- Does iterative patching compound effects or saturate quickly?
- Are effects layer-dependent (early vs. late layers)?
- Is iterative patching more/less effective than parallel patching?

### Cross-Experiment
- How do position-specific effects combine?
- Can we predict layer-wise effects from position-wise effects?
- What does this reveal about CoT token specialization?

---

## Validation Approach

### Experiment 1
- Compare to layer-wise baseline: sum of single-position effects ≈ all-positions effect
- Sanity check: Patching position 0 at layer 0 should have minimal effect
- Cross-validate with attention analysis from previous experiments

### Experiment 2
- Verify iteration 0 matches no-patching baseline
- Check activations at position N are actually used to generate N+1
- Compare final KL to Exp 1 all-positions patching
- Measure correlation between layer criticality and iterative effectiveness

### Documentation
- Unexpected findings documented with error analysis
- Failure cases analyzed
- Comparison table showing Exp 1 vs Exp 2
- Statistical significance testing

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OOM on A100 | Low | High | Use batch_size=1, FP16, clear cache between runs |
| Compute time exceeds estimate | Medium | Medium | Run overnight, use TEST_MODE for validation first |
| Position effects too small to detect | Medium | High | Use multiple metrics, aggregate across pairs |
| Iterative chaining incorrect | Medium | High | Add extensive logging, visual inspection of activations |
| Results not reproducible | Low | High | Set random seeds, document all hyperparameters |

---

## Dependencies

### Code Dependencies
- Existing layer-patching experiment codebase
- CODI model checkpoint
- Prepared pairs dataset

### Research Dependencies
- Must understand layer-patching results to interpret position-wise findings
- Attention analysis provides context for position specialization

### Infrastructure Dependencies
- A100 GPU availability
- W&B account with sufficient storage

---

## Next Steps (After PM Phase)

1. **Architect Review:**
   - Validate data quality (66 pairs sufficient?)
   - Confirm no data duplication
   - Review train/test split approach (using existing pairs)
   - Approve technical architecture

2. **Development:**
   - Implement stories in order
   - Use TEST_MODE for validation
   - Track actual vs. estimated costs
   - Keep user informed of progress

3. **Deployment:**
   - Run full experiments on A100
   - Monitor for errors
   - Validate results

4. **Documentation:**
   - Update all documentation
   - Commit and push to GitHub
   - Share findings with team

---

## Implementation Status

- [ ] **PM Phase:** COMPLETE
- [ ] **Architect Phase:** Not Started
- [ ] **Development Phase:** Not Started
- [ ] **Execution Phase:** Not Started
- [ ] **Documentation Phase:** Not Started

---

**Document Created:** 2025-10-30
**Last Updated:** 2025-10-30
**Next Review:** After Architect Phase
