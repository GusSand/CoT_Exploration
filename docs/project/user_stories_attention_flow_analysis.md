# User Stories: CODI Attention Flow Analysis

**Created**: 2025-10-27
**Product Manager**: Claude Code
**Status**: Approved
**Total Estimated Cost**: 19.5 dev hours + 3.0 GPU hours = 22.5 hours

---

## Epic Overview

**Epic**: Attention Flow Analysis for CODI Continuous Thought Tokens

**Goal**: Extract and analyze attention patterns between the 6 continuous thought positions to understand information flow during reasoning, comparing GPT-2 and LLaMA models.

**Context**:
- We have trained CODI models (GPT-2 124M and LLaMA-3.2-1B) with 6 continuous thought tokens
- Previous work showed attention-importance correlation in LLaMA (r=0.367 at L8)
- We need to understand detailed attention flow patterns between positions
- We want to compare how model capacity affects attention circuits

**Models**:
- `~/codi_ckpt/gpt2_gsm8k/` - GPT-2 124M (12 layers, 12 heads)
- `~/codi_ckpt/llama_gsm8k/` - LLaMA-3.2-1B (16 layers, 32 heads)

**Dataset**: GSM8K training set (100 problems initially, scale to ~7,473 for validation)

**Success Criteria**: See detailed criteria in each story below

---

## Phase 1: Basic Attention Pattern Visualization (LLaMA)

### Story 1.1: Dataset Preparation
**Priority**: P0
**Estimated Cost**: 0.5 hours
**Dependencies**: None

**As a** researcher
**I want to** select and validate 100 GSM8k training problems
**So that** I have a reliable dataset for attention analysis

**Acceptance Criteria**:
- [ ] Load 100 problems randomly sampled from GSM8K training set (~7,473 total)
- [ ] Use random sampling with seed=42 for reproducibility
- [ ] Verify data quality: all problems have questions, answers, and solutions
- [ ] Verify no duplicates in the dataset
- [ ] Confirm source is training set only (not test set)
- [ ] Save dataset with metadata to `src/experiments/codi_attention_flow/data/attention_dataset_100_train.json`
- [ ] Update DATA_INVENTORY.md with new dataset entry including:
  - Source: "GSM8K training set"
  - Sampling method: "Random with seed=42"
  - Size: 100 problems
  - Purpose: "Attention flow analysis between continuous thought tokens"

**Success Metrics**:
- 100 unique training problems loaded
- No data quality issues
- Reproducible sampling (same seed = same problems)

---

### Story 1.2: Extract 6×6 Attention Patterns (LLaMA)
**Priority**: P0
**Estimated Cost**: 1.5 hours
**Dependencies**: Story 1.1
**GPU Time**: 0.25 hours (15 minutes)

**As a** researcher
**I want to** extract attention weights specifically for the 6 continuous thought token positions
**So that** I can analyze information flow between positions

**Acceptance Criteria**:
- [ ] Load LLaMA model from `~/codi_ckpt/llama_gsm8k/` with attention caching enabled
- [ ] Run inference on 100 problems
- [ ] Extract attention for all 16 layers and 32 heads
- [ ] Filter attention to only the 6 continuous thought positions
- [ ] Output shape: `[100, 16, 32, 6, 6]`
- [ ] Verify attention weights sum to 1.0 ± 0.01 across source positions (columns)
- [ ] Save as `results/llama/attention_patterns_raw.npy`
- [ ] Include metadata JSON: `results/llama/attention_metadata.json` with:
  - model_name: "llama-3.2-1b"
  - n_layers: 16
  - n_heads: 32
  - n_problems: 100
  - continuous_thought_positions: [list of token indices]
- [ ] Add WandB logging for extraction progress

**Success Metrics**:
- Attention sums to 1.0 ± 0.01 per position
- Max attention weight in 10%+ of heads should be > 0.4 (non-random patterns)
- Standard deviation within heads should be > 0.15 (showing structure)

**Reuse**: `src/experiments/codi_attention_interp/scripts/2_extract_attention.py`

---

### Story 1.3: Compute Averaged Attention Patterns
**Priority**: P0
**Estimated Cost**: 0.5 hours
**Dependencies**: Story 1.2

**As a** researcher
**I want to** average attention patterns across all 100 problems
**So that** I can identify consistent information flow patterns

**Acceptance Criteria**:
- [ ] Load raw attention patterns from Story 1.2
- [ ] Compute mean attention across problems: `[16, 32, 6, 6]`
- [ ] Compute standard deviation to check consistency (should be < 0.2)
- [ ] Identify top 20 heads by maximum attention weight
- [ ] Save averaged patterns as `results/llama/attention_patterns_avg.npy`
- [ ] Save statistics as `results/llama/attention_stats.json` including:
  - Mean/std/min/max per position
  - Variance across problems (stability check)
  - Top 20 heads ranked by attention strength
  - Per-head max attention value

**Success Metrics**:
- Patterns have low variance (std < 0.2) = consistent across problems
- At least 3-4 heads show clear structure (not uniform ~0.167)
- Top heads have max attention > 0.4

---

### Story 1.4: Visualize Attention Heatmaps
**Priority**: P0
**Estimated Cost**: 1.0 hour
**Dependencies**: Story 1.3

**As a** researcher
**I want to** create heatmap visualizations of attention patterns
**So that** I can visually identify information flow structures

**Acceptance Criteria**:
- [ ] Create grid of heatmaps for top 20 heads (strongest patterns)
- [ ] Layout: 4×5 grid showing 6×6 heatmaps
- [ ] Each heatmap is 6×6 (rows=destination position, cols=source position)
- [ ] Use consistent color scale across all heatmaps (0-1.0)
- [ ] Label axes clearly: "Source Position (0-5)" and "Destination Position (0-5)"
- [ ] Title each subplot with: "L{layer}H{head} (max={max_attn:.2f})"
- [ ] Save as `figures/llama/1_top_heads_attention.png` (high res, 300 DPI)
- [ ] Create aggregated heatmap per layer (averaged across all heads)
- [ ] Save as `figures/llama/2_attention_by_layer.png`
- [ ] Pass "squint test": structure visible from distance

**Success Metrics**:
- Clear patterns visible in heatmaps (bright cells, structure)
- Not all heatmaps look identical (different heads do different things)
- At least one shows obvious hub/bottleneck
- Visual differences between early/middle/late layers

**Reuse**: `src/experiments/codi_attention_interp/scripts/3_analyze_and_visualize.py` patterns

---

### Story 1.5: Identify Hub Positions and Flow Patterns
**Priority**: P0
**Estimated Cost**: 1.0 hour
**Dependencies**: Story 1.3

**As a** researcher
**I want to** compute summary statistics about information flow
**So that** I can answer key questions about the reasoning circuit

**Acceptance Criteria**:
- [ ] Compute "hub score" for each position (average incoming attention across all heads/layers)
- [ ] Identify which position receives most attention (should be 1-2 clear winners)
- [ ] Test for sequential flow:
  - Measure attention from i→i+1 for all positions
  - Compute sequential_flow_score = avg(attention[i→i+1]) / avg(attention[i→all])
  - High score (>0.5) = strong sequential pattern
- [ ] Test for skip connections:
  - Measure attention from pos 5→0, 5→1, 5→2
  - Compute skip_score = avg(attention[5→{0,1,2}])
  - High score (>0.3) = skip connections present
- [ ] Save summary as `results/llama/attention_summary.json` including:
  - Hub positions ranked by incoming attention
  - Per-position hub scores
  - Sequential flow score (0-1)
  - Skip connection score (0-1)
  - Answers to: "Does pos 5 attend to pos 4?", "Which position is the hub?", "Are there skip connections?"
- [ ] Create visualization showing hub connectivity
- [ ] Save as `figures/llama/3_hub_analysis.png`

**Success Metrics**:
- Can identify 1-2 clear hub positions
- Can answer YES/NO with confidence (>80%) to sequential flow and skip connections
- Hub position has 2-3× higher attention than uniform baseline (0.167)
- Non-random patterns visible

---

## Phase 2: Identify Critical Attention Heads (LLaMA)

### Story 2.1: Compute Information Flow Scores
**Priority**: P0
**Estimated Cost**: 0.5 hours
**Dependencies**: Story 1.3

**As a** researcher
**I want to** measure how much each head moves information forward through reasoning
**So that** I can identify heads implementing sequential computation

**Acceptance Criteria**:
- [ ] For each head, compute flow score:
  - `flow_score = sum(attention[i→j] for all i>j) / total_attention`
  - Measures forward information flow (later positions attending to earlier)
  - High score (>0.6) = head moves information forward
- [ ] Compute for all 16 layers × 32 heads = 512 heads
- [ ] Rank heads by flow score (descending)
- [ ] Save rankings as `results/llama/heads_ranked_by_flow.csv` with columns:
  - layer, head, flow_score, layer_type (early/mid/late)
- [ ] Identify top 10 heads

**Success Metrics**:
- Top heads have flow_score > 0.6
- Clear gap between top heads and random (2× difference minimum)
- Top 10 heads account for >40% of forward flow

---

### Story 2.2: Compute Hub Connectivity Scores
**Priority**: P0
**Estimated Cost**: 0.5 hours
**Dependencies**: Story 1.3

**As a** researcher
**I want to** measure which heads create hub positions
**So that** I can identify heads that aggregate information

**Acceptance Criteria**:
- [ ] For each head, compute variance in attention distribution across 6 positions
- [ ] High variance (>0.2) = creates hubs (some positions heavily attended)
- [ ] Low variance (<0.1) = distributes attention evenly
- [ ] Rank heads by hub score (descending)
- [ ] Save rankings as `results/llama/heads_ranked_by_hub.csv` with columns:
  - layer, head, hub_score (variance), max_attention_position, max_attention_value
- [ ] Identify top 10 heads

**Success Metrics**:
- Top heads have hub variance > 0.2
- Can identify which position each head treats as hub
- Top 10 heads show >50% concentration on hub positions

---

### Story 2.3: Compute Skip Connection Scores
**Priority**: P0
**Estimated Cost**: 0.5 hours
**Dependencies**: Story 1.3

**As a** researcher
**I want to** measure which heads implement shortcuts in reasoning
**So that** I can identify heads that bypass intermediate steps

**Acceptance Criteria**:
- [ ] For each head, measure attention from position 5 to positions 0-2 (skipping 3-4)
- [ ] Compute `skip_score = avg(attention[5→0], attention[5→1], attention[5→2])`
- [ ] High score (>0.3) = head implements long-range dependencies
- [ ] Rank heads by skip score (descending)
- [ ] Save rankings as `results/llama/heads_ranked_by_skip.csv` with columns:
  - layer, head, skip_score, attention_5to0, attention_5to1, attention_5to2
- [ ] Identify top 10 heads

**Success Metrics**:
- Can identify if skip connections exist (any head with skip_score > 0.3)
- Top heads show 2× higher skip attention than random baseline (0.167)
- Skip heads are in later layers (L10+)

---

### Story 2.4: Identify Top Critical Heads
**Priority**: P0
**Estimated Cost**: 1.0 hour
**Dependencies**: Stories 2.1, 2.2, 2.3

**As a** researcher
**I want to** combine all metrics to identify the most critical heads
**So that** I can understand which 2-3 heads implement the core reasoning circuit

**Acceptance Criteria**:
- [ ] Load rankings from Stories 2.1, 2.2, 2.3
- [ ] Identify top 5 heads for each metric
- [ ] Find heads that rank high on multiple metrics (multi-purpose heads)
- [ ] Compute composite score = 0.4×flow_score + 0.4×hub_score + 0.2×skip_score
- [ ] Create master ranking: `results/llama/ranked_heads.csv` with columns:
  - layer, head, flow_score, hub_score, skip_score, composite_score, functional_type
- [ ] Assign functional types based on highest metric:
  - "Forward Flow" (flow_score dominant)
  - "Hub Aggregator" (hub_score dominant)
  - "Skip Connection" (skip_score dominant)
  - "Multi-Purpose" (high on 2+ metrics)
- [ ] Save findings as `results/llama/critical_heads_findings.txt` including:
  - Top 10 heads with descriptions
  - Which layers are most critical
  - Functional specialization patterns

**Success Metrics**:
- Identify 2-3 heads as most critical (composite_score > 0.7)
- Different heads specialize in different functions
- Rankings stable (if run twice on different 100 samples, >70% overlap in top 10)
- Can make statements like: "L8H5 is a forward flow head with 0.82 flow score"

---

### Story 2.5: Visualize Critical Head Patterns
**Priority**: P0
**Estimated Cost**: 1.0 hour
**Dependencies**: Story 2.4

**As a** researcher
**I want to** visualize attention patterns for the top critical heads
**So that** I can understand what makes them critical

**Acceptance Criteria**:
- [ ] Extract top 12 heads from composite ranking
- [ ] Create 3×4 grid of attention heatmaps (one per head)
- [ ] Label each with: "L{layer}H{head}: {functional_type} (score={composite:.2f})"
- [ ] Use color-coding to show functional types:
  - Red border = Forward Flow
  - Blue border = Hub Aggregator
  - Green border = Skip Connection
  - Purple border = Multi-Purpose
- [ ] Save as `figures/llama/4_top_heads_visualization.png`
- [ ] Create separate detailed views for top 3 heads:
  - Larger 6×6 heatmap
  - Annotate brightest cells with values
  - Show incoming/outgoing attention bar charts
- [ ] Save as `figures/llama/5_critical_head_detail_L{layer}H{head}.png` (3 files)

**Success Metrics**:
- Visual patterns clearly show different specializations
- Top heads have visually distinct patterns from random heads
- Can explain "why" each head is critical from visualization alone
- Patterns match assigned functional types

---

### Story 2.6: Cross-Model Comparison (GPT-2 vs LLaMA)
**Priority**: P0
**Estimated Cost**: 3.5 hours
**Dependencies**: Stories 1.1-2.5 completed for LLaMA
**GPU Time**: 0.25 hours (15 minutes)

**As a** researcher
**I want to** compare attention flow patterns between GPT-2 and LLaMA
**So that** I can understand how model capacity affects continuous thought circuits

**Acceptance Criteria**:
- [ ] Run **same pipeline (Stories 1.2-2.5)** on GPT-2 model using `~/codi_ckpt/gpt2_gsm8k/`
- [ ] Use **exact same 100 training problems** (from Story 1.1) for fair comparison
- [ ] Save GPT-2 results in separate directory: `results/gpt2/`
- [ ] Create comparison analysis including:
  - Hub positions: Does GPT-2 have same hubs as LLaMA?
  - Sequential flow: Compare flow scores between models
  - Skip connections: Compare skip scores between models
  - Critical heads: Do both models use same number of critical heads?
  - Attention distribution: Is GPT-2 more/less uniform than LLaMA?
- [ ] Create side-by-side comparison visualizations:
  - `figures/comparison/6_model_comparison_hubs.png` - Hub scores compared (bar chart)
  - `figures/comparison/7_model_comparison_flow.png` - Flow patterns compared (line plot by layer)
  - `figures/comparison/8_model_comparison_critical_heads.png` - Top 5 heads per model (heatmaps)
- [ ] Save comparison report as `results/gpt2_vs_llama_comparison.json` including:
  - Hub position differences
  - Flow/hub/skip score comparisons
  - Number of critical heads per model
  - Specialization vs distribution analysis
- [ ] Answer key questions:
  - "Which model has more specialized attention patterns?" (hypothesis: GPT-2)
  - "Does GPT-2 use fewer critical heads?" (hypothesis: yes, due to capacity constraints)
  - "Do patterns validate the specialized vs distributed encoding hypothesis?"

**Success Metrics**:
- Can identify clear differences in attention strategies between models
- Findings align with previous work (GPT-2 specialized, LLaMA distributed)
- Visualizations show side-by-side patterns clearly
- Can make comparative statements like: "GPT-2 has 3 critical heads vs LLaMA's 7"

**Reuse**: All code from Stories 1.2-2.5, just change model path

---

## Phase 3: Documentation & Integration

### Story 3.1: Document Experiment Results
**Priority**: P0
**Estimated Cost**: 1.0 hour
**Dependencies**: Stories 1.5, 2.6

**As a** researcher
**I want to** document all findings in the standard format
**So that** results are reproducible and accessible

**Acceptance Criteria**:
- [ ] Update `docs/research_journal.md` with high-level summary (TLDR format):
  - Date, experiment name, models, dataset size
  - Key findings (3-5 bullet points)
  - Links to detailed reports
- [ ] Create `docs/experiments/10-27_llama_gsm8k_attention_flow_analysis.md` with:
  - Objective, methodology, results, key findings
  - Answer all questions from Prompt 1 success criteria
  - Include all metrics and statistics
  - Link to figures and data files
  - Validation of hypotheses
- [ ] Create `docs/experiments/10-27_gpt2_gsm8k_attention_flow_analysis.md` (same structure)
- [ ] Create `docs/experiments/10-27_both_gsm8k_model_comparison.md` for comparison findings
- [ ] Update `docs/DATA_INVENTORY.md` with new dataset entry:
  - Link to `attention_dataset_100_train.json`
  - Document how to recreate it
  - Document usage in this experiment
  - Stratification: N/A (random sampling)
  - Size: 100 problems
- [ ] Create `src/experiments/codi_attention_flow/README.md` with:
  - Quick start guide
  - How to reproduce the experiment
  - File structure
  - Command-line instructions
  - Expected outputs
  - Expected runtime

**Success Metrics**:
- Can answer all key questions: "Which position receives most attention?", "Is there sequential flow?", "Are there skip connections?", "Which 2-3 heads are most critical per model?", "How do GPT-2 and LLaMA differ?"
- Documentation follows CLAUDE.md guidelines
- All links work
- Experiment is reproducible from documentation alone

---

### Story 3.2: Create WandB Dashboard
**Priority**: P1 (Nice to have)
**Estimated Cost**: 0.5 hours
**Dependencies**: Stories 1.2, 2.6

**As a** researcher
**I want to** track experiment metrics in WandB
**So that** I can monitor progress and compare runs

**Acceptance Criteria**:
- [ ] Log attention extraction progress (problems processed, ETA)
- [ ] Log attention statistics per layer (mean, std, variance)
- [ ] Log head rankings (top 10 by each metric)
- [ ] Log comparison metrics (GPT-2 vs LLaMA differences)
- [ ] Create WandB report with:
  - Attention heatmaps for top heads (both models)
  - Line plots of metrics across layers
  - Summary table of critical heads
  - Comparison charts
- [ ] Tag run with: "attention-flow", "gsm8k", "llama", "gpt2", "100-samples", "phase1"
- [ ] Log hyperparameters: model_name, n_problems, seed

**Success Metrics**:
- WandB dashboard shows real-time progress during extraction
- All key metrics logged and visualized
- Can compare runs easily

---

### Story 3.3: Commit and Push Results
**Priority**: P0
**Estimated Cost**: 0.5 hours
**Dependencies**: Story 3.1

**As a** researcher
**I want to** version control all code, results, and documentation
**So that** work is preserved and shareable

**Acceptance Criteria**:
- [ ] Update `.gitignore` to exclude large files:
  - `*.npy` > 100MB
  - `results/*/attention_patterns_raw.npy`
  - Keep averaged patterns and small JSON files
- [ ] Stage all new files:
  - Scripts in `src/experiments/codi_attention_flow/scripts/`
  - Results (small files only) in `src/experiments/codi_attention_flow/results/`
  - Figures in `src/experiments/codi_attention_flow/figures/`
  - Documentation in `docs/`
- [ ] Create descriptive commit message following CLAUDE.md guidelines:
  ```
  feat: CODI attention flow analysis - identify critical reasoning heads

  Complete attention pattern extraction and analysis for LLaMA and GPT-2
  CODI models on 100 GSM8K training problems.

  Key findings:
  - [Top hub position and score]
  - [Sequential/skip flow findings]
  - [Top 3 critical heads per model]
  - [GPT-2 vs LLaMA comparison insight]

  Results: Phase 1+2 experiments (100 samples)
  - LLaMA: 16 layers, 32 heads analyzed
  - GPT-2: 12 layers, 12 heads analyzed
  - Identified [N] critical heads per model

  Time: ~13.5 hours (on budget)

  🤖 Generated with [Claude Code](https://claude.com/claude-code)

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
- [ ] Push to GitHub
- [ ] Verify all links in documentation work (test on GitHub web interface)

**Success Metrics**:
- All code and documentation committed
- GitHub shows latest results
- Documentation links work
- Large files excluded (repo size reasonable)

---

## Phase 4: Scale to Full Training Set (OPTIONAL - Run after Phase 1-3)

### Story 4.1: Full Training Set Dataset Preparation
**Priority**: P1 (Only after Phase 1-3 success)
**Estimated Cost**: 0.5 hours
**Dependencies**: Story 1.5 (validate 100-problem findings first)

**As a** researcher
**I want to** prepare the complete GSM8K training set for attention analysis
**So that** I can validate findings with stronger statistical power

**Acceptance Criteria**:
- [ ] Load **all ~7,473 GSM8K training problems**
- [ ] Verify data quality: check for corrupt/incomplete problems
- [ ] Remove any duplicates
- [ ] Save as `src/experiments/codi_attention_flow/data/attention_dataset_full_train.json`
- [ ] Update DATA_INVENTORY.md with entry:
  - Source: "GSM8K training set (complete)"
  - Size: ~7,473 problems
  - Purpose: "Full-scale attention flow analysis for validation"
  - Stratification: None
- [ ] Document expected GPU time: ~2.5 hours for extraction

**Success Metrics**:
- All ~7,473 training problems loaded
- No quality issues
- Documentation updated

---

### Story 4.2: Full-Scale Attention Extraction (LLaMA only)
**Priority**: P1
**Estimated Cost**: 0.5 hours development + 2.5 hours GPU time
**Dependencies**: Story 4.1

**As a** researcher
**I want to** extract attention patterns for all ~7,473 training problems
**So that** I have robust statistics for pattern analysis

**Acceptance Criteria**:
- [ ] Run attention extraction on full dataset using LLaMA model only
- [ ] Use batching to handle large dataset efficiently (batch_size=10)
- [ ] Implement checkpointing:
  - Save every 1,000 problems to `results/llama_full/checkpoints/attention_checkpoint_{N}.npy`
  - Resume from last checkpoint if interrupted
- [ ] Log progress to WandB (problems processed, ETA, memory usage)
- [ ] Save final results as `results/llama_full/attention_patterns_full_raw.npy`
- [ ] Output shape: `[7473, 16, 32, 6, 6]`
- [ ] Verify attention weights sum to 1.0
- [ ] Create summary stats comparing 100-sample vs full-dataset patterns:
  - Correlation between averaged attention patterns (should be > 0.9)
  - Variance reduction in full dataset
- [ ] Save comparison as `results/llama_full/100_vs_full_comparison.json`

**Success Metrics**:
- All 7,473 problems processed successfully
- No memory issues or crashes (checkpointing works)
- Patterns highly correlated with 100-sample findings (r > 0.9)
- Extraction completes in < 3 hours

---

### Story 4.3: Full-Scale Analysis & Validation
**Priority**: P1
**Estimated Cost**: 1.5 hours
**Dependencies**: Story 4.2

**As a** researcher
**I want to** re-run all analyses on the full dataset
**So that** I can validate findings with stronger statistical confidence

**Acceptance Criteria**:
- [ ] Run Stories 1.3-2.5 pipeline on full dataset (LLaMA only)
- [ ] Compute all metrics (flow, hub, skip scores) with full data
- [ ] Compare rankings: Do top critical heads remain the same?
- [ ] Compute statistical significance with larger sample:
  - All correlation p-values should be << 0.001
  - Confidence intervals should be tighter
  - Hub scores should have smaller standard errors
- [ ] Create comparison report: `results/llama_full/100_vs_7473_validation.json` including:
  - Top 10 heads from 100-sample vs full dataset
  - Overlap percentage (should be >70% for validation)
  - Confidence intervals on all metrics
  - Statistical test results (paired t-tests for hub scores)
- [ ] Generate updated visualizations with "N=7473" labels
- [ ] Save as `figures/llama_full/*` directory

**Success Metrics**:
- Top heads stable (>70% overlap with 100-sample findings)
- All p-values highly significant (p < 0.001)
- Findings confirmed with stronger evidence
- Can make statements with confidence intervals

---

### Story 4.4: Document Full-Scale Results
**Priority**: P1
**Estimated Cost**: 1.0 hour
**Dependencies**: Story 4.3

**As a** researcher
**I want to** document the full-scale experiment results
**So that** findings are preserved and reproducible

**Acceptance Criteria**:
- [ ] Update `docs/research_journal.md` with full-scale findings
- [ ] Create `docs/experiments/10-27_llama_gsm8k_attention_flow_full_scale.md` with:
  - Comparison of 100-sample vs full dataset
  - Validation of all hypotheses
  - Statistical confidence improvements
  - Any new findings from larger dataset
  - Final conclusions
- [ ] Update `docs/DATA_INVENTORY.md` with full dataset entry
- [ ] Update `src/experiments/codi_attention_flow/README.md` with:
  - Instructions for running on full dataset
  - Expected compute time (2.5 hours)
  - How to use checkpointing
  - How to resume interrupted runs
- [ ] Commit and push all results (exclude large .npy files via .gitignore)
- [ ] Create commit message highlighting validation results

**Success Metrics**:
- Documentation complete per CLAUDE.md standards
- All findings validated with full dataset
- GitHub updated with latest results
- Can reproduce full experiment from documentation

---

## Summary Tables

### Phase 1-3: Initial Analysis (100 problems)

| Phase | Stories | Description | Dev Hours | GPU Hours | Models |
|-------|---------|-------------|-----------|-----------|--------|
| 1 | 1.1-1.5 | Basic attention patterns (LLaMA) | 4.5 | 0.25 | LLaMA |
| 2 | 2.1-2.6 | Critical heads + GPT-2 comparison | 4.5 | 0.25 | Both |
| 3 | 3.1-3.3 | Documentation & integration | 2.0 | 0 | - |
| **Total** | **13 stories** | | **11.0** | **0.5** | |

### Phase 4: Full-Scale Validation (7,473 problems, OPTIONAL)

| Phase | Stories | Description | Dev Hours | GPU Hours | Models |
|-------|---------|-------------|-----------|-----------|--------|
| 4 | 4.1-4.4 | Full dataset validation | 3.5 | 2.5 | LLaMA |

### Grand Total

- **Development**: 14.5 hours (11.0 base + 3.5 optional)
- **GPU Time**: 3.0 hours (0.5 base + 2.5 optional)
- **Total**: 17.5 hours (or 11.5 without Phase 4)

---

## Execution Strategy

### Stage 1: Pilot Analysis (Stories 1.1-3.3) - 11.0 hours
1. **LLaMA Analysis** (Stories 1.1-2.5): 6.5 hours
   - Extract and analyze attention patterns
   - Identify critical heads
   - Create visualizations

2. **GPT-2 Comparison** (Story 2.6): 3.5 hours
   - Run same pipeline on GPT-2
   - Compare with LLaMA findings

3. **Documentation** (Stories 3.1-3.3): 2.0 hours
   - Document all findings
   - Commit and push

**Checkpoint**: Review findings before proceeding to Stage 2

### Stage 2: Full-Scale Validation (Stories 4.1-4.4) - 3.5 hours [OPTIONAL]
4. **Only proceed if**:
   - Stage 1 shows clear patterns (pass success criteria)
   - Need stronger statistical validation
   - Findings are promising enough for publication

5. **Full Dataset Analysis**: 3.5 hours
   - Extract ~7,473 problems
   - Validate rankings and findings
   - Document with confidence intervals

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Attention patterns are noise/uniform | Low | High | Use existing code that showed r=0.367 correlation; have clear pass/fail criteria |
| 100 problems insufficient for stable rankings | Medium | Medium | Phase 4 available to scale to full dataset |
| Metrics don't capture meaningful structure | Low | Medium | Metrics validated in prior work; clear thresholds defined |
| GPU memory issues with large dataset | Low | Low | Batching + checkpointing implemented in Story 4.2 |
| GPT-2 vs LLaMA comparison shows no difference | Low | Low | Prior work shows clear differences; would still be informative |

---

## Success Criteria Summary

### Phase 1 Success (Prompt 1)
✅ **PASS if**:
- Non-random patterns visible (max attention > 0.4 in 10%+ heads)
- Patterns consistent across problems (std < 0.2)
- Can identify hub positions, sequential flow, skip connections (YES/NO with confidence)
- Heatmaps pass "squint test"

### Phase 2 Success (Prompt 2)
✅ **PASS if**:
- Top 10 heads clearly identified (composite score > 0.6)
- Heads show functional specialization
- Rankings stable across runs (>70% overlap)
- Can name 2-3 most critical heads per model

### Comparison Success (Story 2.6)
✅ **PASS if**:
- Clear differences between GPT-2 and LLaMA identified
- Findings align with capacity hypothesis
- Side-by-side visualizations show distinct patterns

### Full-Scale Success (Phase 4)
✅ **PASS if**:
- Top heads stable (>70% overlap with 100-sample)
- All p-values << 0.001
- Findings validated with stronger evidence

---

## Dependencies & Reuse

### Existing Code to Reuse
- `src/experiments/codi_attention_interp/scripts/2_extract_attention.py` - Attention extraction logic
- `src/experiments/codi_attention_interp/scripts/3_analyze_and_visualize.py` - Visualization patterns
- Stratified GSM8K datasets in DATA_INVENTORY.md (for reference)

### External Dependencies
- torch, numpy, matplotlib, seaborn, wandb, pandas
- CODI model checkpoints:
  - `~/codi_ckpt/llama_gsm8k/` (LLaMA-3.2-1B)
  - `~/codi_ckpt/gpt2_gsm8k/` (GPT-2 124M)
- GSM8K training set (~7,473 problems)

---

## Deliverables

### Phase 1-3 Deliverables
- ✅ 100-problem training dataset
- ✅ Attention patterns extracted for LLaMA and GPT-2
- ✅ Hub, flow, and skip connection analysis
- ✅ Top 10-12 critical heads identified per model
- ✅ Heatmap visualizations (20+ figures)
- ✅ Comparison analysis (GPT-2 vs LLaMA)
- ✅ Complete documentation (3 detailed reports + research journal entry)
- ✅ WandB dashboard (optional)
- ✅ All code committed to GitHub

### Phase 4 Deliverables (Optional)
- ✅ Full training set extraction (~7,473 problems)
- ✅ Validated rankings with confidence intervals
- ✅ Statistical validation report
- ✅ Updated documentation

---

**Status**: Ready for implementation
**Next Step**: Transition to Architect role for implementation planning
**Approval Date**: 2025-10-27
