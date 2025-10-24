# Token Threshold & Criticality Experiments

**Date**: 2025-10-24
**Status**: 🔄 **IN PROGRESS** - Pilot (10 problems) running
**Branch**: `experiment/token-threshold`

---

## Executive Summary

**Objective**: Determine minimum token thresholds for reasoning and identify which continuous thought tokens are most critical in LLaMA CODI through systematic corruption and enhancement experiments.

**Motivation**: The CODI paper claims that 4/6 token corruption (67%) causes catastrophic failure, and suggests middle tokens (z₃, z₄) may be special. This experiment tests these claims empirically with LLaMA-3.2-1B CODI and takes a data-driven approach to identify which tokens are actually critical.

**Key Innovation**: Multi-method token criticality assessment combining:
1. **Threshold degradation** (1→6 token corruption)
2. **Critical token identification** (skip tests)
3. **Enhancement responsiveness** (amplification tests)
4. **Convergent validity** (do different methods agree?)

---

## Research Questions

1. **RQ1 (Threshold)**: What is the accuracy degradation curve as we corrupt 1→6 tokens? Does 4/6 corruption (67%) cause catastrophic failure?

2. **RQ2 (Critical Tokens)**: Which token position(s) are most critical? Data-driven identification without preset hypotheses.

3. **RQ3 (Enhancement)**: Can enhancing specific tokens improve performance? Which positions are most enhancement-responsive?

4. **RQ4 (Convergence)**: Do corruption and enhancement measures agree on which tokens are critical?

---

## Methodology

### Dataset

- **Source**: Stratified GSM8K dataset (10-problem pilot from `test_dataset_10.json`)
- **Difficulty distribution**: Balanced across reasoning complexity
- **Problems**: All require continuous thought tokens for LLaMA to solve correctly
- **Model**: LLaMA-3.2-1B CODI (`~/codi_ckpt/llama_gsm8k/`)
- **Test layer**: Middle (L8)

### Experiment 1: Threshold Degradation Test

**Goal**: Map accuracy degradation as function of # corrupted tokens

**Design**:
- **Corruption levels**: 1, 2, 3, 4, 5, 6 tokens
- **Sampling strategy**:
  - Level 1: All 6 positions individually (6 configs)
  - Level 2: 3 strategic samples - sequential pairs
  - Level 3: 3 samples - first/second half, distributed
  - Level 4: 6 samples - skip each token individually (**critical for RQ2**)
  - Level 5: 6 samples - keep each token individually
  - Level 6: Complete ablation (1 config)
- **Total configurations**: 25 position combinations
- **Corruption methods**: Zero ablation, Gaussian σ=1.0
- **Total experiments**: 10 problems × 50 configs = **500 experiments**

**Measurements**:
- Accuracy (correct/incorrect answer)
- Predicted answer vs expected answer
- Baseline comparison

### Experiment 2: Token Enhancement Test

**Goal**: Test if amplifying specific tokens improves reasoning

**Design**:
- **Positions tested**: All 6 tokens individually
- **Multipliers**: [0.5x, 1.0x (baseline), 1.5x, 2.0x, 3.0x]
- **Mode**: Standalone (no corruption, pure enhancement)
- **Total experiments**: 10 problems × 30 configs = **300 experiments**

**Measurements**:
- Accuracy change from baseline
- Improvement rate (baseline wrong → enhanced correct)
- Degradation rate (baseline correct → enhanced wrong)

### Experiment 3: Combined Analysis

**Goal**: Synthesize findings to create comprehensive token criticality ranking

**Analyses**:
1. Token importance from corruption (failure rates)
2. Token responsiveness from enhancement (mean effects)
3. Combined criticality ranking
4. Convergent validity (correlation between methods)
5. Comparison to paper claims (z₃/z₄ special?)

---

## Results

### Experiment 1: Threshold Degradation

#### Degradation Curve

**[PENDING - Results will be added after experiment completes]**

- Baseline accuracy: X%
- Level 1 (1 token): X%
- Level 2 (2 tokens): X%
- Level 3 (3 tokens): X%
- Level 4 (4 tokens): X% ← **67% threshold test**
- Level 5 (5 tokens): X%
- Level 6 (6 tokens): X%

#### 67% Threshold Test

**[PENDING]**

- Baseline → Level 4 drop: X percentage points
- P-value: X
- Cohen's d: X
- Result: CATASTROPHIC / Degraded but functional

#### Critical Token Identification (Skip Tests)

**[PENDING]**

Level 4 skip test results (which single token is sufficient?):
- Skip Token 0 (keep 1-5): X% accuracy → Token 0 is CRITICAL / non-critical
- Skip Token 1 (keep 0,2-5): X% accuracy → Token 1 is CRITICAL / non-critical
- Skip Token 2 (keep 0-1,3-5): X% accuracy → Token 2 is CRITICAL / non-critical
- Skip Token 3 (keep 0-2,4-5): X% accuracy → Token 3 is CRITICAL / non-critical
- Skip Token 4 (keep 0-3,5): X% accuracy → Token 4 is CRITICAL / non-critical
- Skip Token 5 (keep 0-4): X% accuracy → Token 5 is CRITICAL / non-critical

**Critical tokens** (accuracy >50% when skipped): [PENDING]

### Experiment 2: Token Enhancement

#### Enhancement Heatmap

**[PENDING]**

Accuracy by position × multiplier:
```
             0.5x  1.0x  1.5x  2.0x  3.0x
Token 0:      X%    X%    X%    X%    X%
Token 1:      X%    X%    X%    X%    X%
Token 2:      X%    X%    X%    X%    X%
Token 3:      X%    X%    X%    X%    X%
Token 4:      X%    X%    X%    X%    X%
Token 5:      X%    X%    X%    X%    X%
```

#### Optimal Multipliers

**[PENDING]**

- Token 0: X.Xx (accuracy: X%)
- Token 1: X.Xx (accuracy: X%)
- Token 2: X.Xx (accuracy: X%)
- Token 3: X.Xx (accuracy: X%)
- Token 4: X.Xx (accuracy: X%)
- Token 5: X.Xx (accuracy: X%)

#### Position Criticality (ANOVA)

**[PENDING]**

- F-statistic: X
- P-value: X
- Significant: Yes/No
- Interpretation: [PENDING]

### Experiment 3: Combined Analysis

#### Token Criticality Ranking

**[PENDING]**

Ranked from most to least critical:
1. Token X: corruption=X%, enhancement=X, combined=X
2. Token X: corruption=X%, enhancement=X, combined=X
3. Token X: corruption=X%, enhancement=X, combined=X
4. Token X: corruption=X%, enhancement=X, combined=X
5. Token X: corruption=X%, enhancement=X, combined=X
6. Token X: corruption=X%, enhancement=X, combined=X

#### Convergent Validity

**[PENDING]**

Correlation between corruption and enhancement:
- Pearson r: X (p=X)
- Spearman r: X (p=X)
- Interpretation: Convergent / Divergent

#### Comparison to Paper Claims

**[PENDING]**

**Claim 1**: 67% threshold (4/6 corruption) causes catastrophic failure
- **Result**: VALIDATED / REFUTED
- **Evidence**: [PENDING]

**Claim 2**: Middle tokens (z₃, z₄) are special
- **Result**: VALIDATED / REFUTED / PARTIALLY VALIDATED
- **Evidence**: [PENDING]
- **Data-driven finding**: Tokens [X, Y] are most critical

---

## Key Findings

**[PENDING - Will be filled after analysis completes]**

1. **Threshold**: [PENDING]
2. **Critical Tokens**: [PENDING]
3. **Enhancement**: [PENDING]
4. **Convergence**: [PENDING]

---

## Visualizations

Generated figures (PDF + PNG):
1. `degradation_curve.{pdf,png}` - Accuracy vs # corrupted tokens
2. `critical_tokens.{pdf,png}` - Skip test results by position
3. `enhancement_heatmap.{pdf,png}` - Position × multiplier heatmap
4. `enhancement_effects.{pdf,png}` - Enhancement responsiveness by position
5. `combined_ranking.{pdf,png}` - Overall token criticality ranking
6. `convergent_validity.{pdf,png}` - Corruption vs enhancement correlation

---

## Statistical Power

- **Sample size**: 10 problems (pilot)
- **Experiments per problem**: 80 (50 threshold + 30 enhancement)
- **Total experiments**: 800
- **Runtime**: ~15 minutes (estimated)
- **Statistical tests**: T-tests, ANOVA, correlation
- **Note**: Pilot for methodology validation. 100-problem expansion planned.

---

## Technical Details

### Configuration

```json
{
  "model": "LLaMA-3.2-1B CODI",
  "model_path": "~/codi_ckpt/llama_gsm8k/",
  "test_layer": "middle (L8)",
  "dataset": "test_dataset_10.json",
  "corruption_methods": ["zero", "gauss_1.0"],
  "enhancement_multipliers": [0.5, 1.0, 1.5, 2.0, 3.0],
  "wandb_project": "codi-token-threshold",
  "git_branch": "experiment/token-threshold"
}
```

### Infrastructure

- **Scripts**: `src/experiments/token_threshold/scripts/`
- **Results**: `src/experiments/token_threshold/results/`
- **Figures**: `src/experiments/token_threshold/figures/`
- **WandB tracking**: Yes
- **Version control**: All committed to GitHub

---

## Limitations

1. **Small sample size**: 10 problems for pilot (expansion to 100 planned)
2. **Single model**: LLaMA-3.2-1B only (GPT-2 comparison future work)
3. **Single layer**: Middle layer (L8) only
4. **Corruption methods**: Only zero and Gaussian (could add more)
5. **Enhancement mode**: Standalone only (combined scenarios future work)

---

## Future Work

### Immediate Next Steps
1. **Expand to 100 problems** (~2 hours runtime)
2. **Stratify by difficulty** (easy/medium/hard analysis)
3. **Compare to CCTA results** (integrate with existing attention analysis)

### Phase 2 Experiments
1. **Combined scenarios**: Enhance middle + corrupt edges
2. **Layer sweep**: Test early/middle/late layers
3. **Token pairs**: Test compositional importance (z₃+z₄ together)
4. **Cross-model**: Compare LLaMA vs GPT-2

### Phase 3 Analysis
1. **Residual stream decomposition**: How do tokens build computation?
2. **Attention flow**: Which question tokens route to which continuous thoughts?
3. **Difficulty interaction**: Are critical tokens different for easy vs hard problems?

---

## Deliverables

**Code**:
- ✅ `scripts/utils.py` - Shared utilities and WandB integration
- ✅ `scripts/corruption_utils.py` - Multi-token corruption framework
- ✅ `scripts/1_run_threshold_test.py` - Threshold experiment runner
- ✅ `scripts/2_analyze_threshold.py` - Threshold analysis script
- ✅ `scripts/3_run_enhancement_test.py` - Enhancement experiment runner
- ✅ `scripts/4_analyze_enhancement.py` - Enhancement analysis script
- ✅ `scripts/5_combined_analysis.py` - Combined criticality ranking

**Data**:
- 🔄 `results/threshold_test_10.json` - Threshold experiment results (running)
- ⏳ `results/enhancement_test_10.json` - Enhancement experiment results (pending)
- ⏳ `results/threshold_analysis.json` - Threshold statistics (pending)
- ⏳ `results/enhancement_analysis.json` - Enhancement statistics (pending)
- ⏳ `results/combined_analysis.json` - Combined criticality ranking (pending)

**Figures**:
- ⏳ All 6 figures (PDF + PNG) (pending)

**Documentation**:
- ✅ `README.md` - Experiment overview and usage
- 🔄 `docs/experiments/token_threshold_2025-10-24.md` - This detailed report
- ⏳ `docs/research_journal.md` - High-level entry (pending)

---

## Time Investment

- **Infrastructure setup**: 35 minutes
- **Experiment development**: 205 minutes (~3.4 hours)
- **Experiment runtime**: 15 minutes (pilot)
- **Analysis & visualization**: TBD
- **Documentation**: TBD
- **Total**: ~7 hours (estimated)

---

## Conclusion

**[PENDING - Will be written after results are complete]**

---

## References

1. CODI Paper: Continuous Chain-of-Thought via Self-Distillation
2. CCTA Experiment: `src/experiments/codi_attention_interp/`
3. Activation Patching: `src/experiments/activation_patching/`

---

**Last Updated**: 2025-10-24 01:15 UTC
**Experiment Status**: Threshold test running, enhancement test pending
