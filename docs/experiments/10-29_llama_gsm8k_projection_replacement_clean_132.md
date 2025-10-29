# LLaMA Projection Replacement Intervention - Clean Variants (132 Samples)

**Date**: 2025-10-29
**Model**: CODI-LLaMA (Llama-3.2-1B + LoRA)
**Dataset**: 132 activation patching samples (variant=clean)
**Experiment**: Number embedding projection intervention on verified correct reasoning chains

---

## Objective

Validate the number embedding intervention findings from the 50-sample pilot on a curated dataset of problems with verified correct baseline reasoning.

**Research Question**: Does the robustness to number embedding intervention hold when baseline model performance is substantially higher (90% vs 52% accuracy)?

---

## Background

### Motivation from Previous Experiment

**50-Sample Pilot Results**:
- Baseline accuracy: 52%
- Intervention accuracy: 54% (+2%)
- Answer changes: 16%
- Conclusion: Number embeddings not strongly causal

**This experiment addresses concerns** by using:
- Higher quality problems (baseline 90% accuracy)
- Larger sample size (132 examples)
- Verified correct reasoning chains

### Dataset: Activation Patching Clean Variants

**Source**: GitHub repository llama_cot_all.json
- Contains pairs: clean variant vs corrupted variant
- This experiment uses only clean variants

**Dataset Characteristics**:
- Total samples: 132
- Variant label: clean
- Contains metadata: pair_id, reasoning_steps, baseline_correct

---

## Results

### Overall Accuracy

| Condition | Accuracy | Correct/Total | Change |
|-----------|----------|---------------|---------|
| Without Intervention | 90.15% | 119/132 | - |
| With Intervention | 80.30% | 106/132 | **-9.85%** |

**CRITICAL FINDING**: Intervention DECREASED accuracy by 9.85 percentage points!

**Contrast with Pilot**:
- Pilot: +2.0% improvement
- Clean: -9.85% degradation
- Complete reversal of effect direction

### Answer Changes

| Metric | Count | Percentage |
|--------|-------|------------|
| Total examples | 132 | 100% |
| Answers changed | 19 | 14.4% |
| Answers unchanged | 113 | 85.6% |

### Impact on Correctness

Among 19 examples that changed:

| Transition | Count |
|-----------|-------|
| Correct to Incorrect | **16** |
| Incorrect to Correct | 3 |

**Devastating Effect**: 16 correct answers became incorrect (84% of changes were harmful)

**Net Effect**: -13 correct answers

---

## Key Findings

### Finding 1: High Baseline Performance Makes Intervention Harmful

**Pilot (52% baseline)**: +2% (not significant)
**Clean (90% baseline)**: -9.85% (large, negative)

**Hypothesis**: When baseline reasoning is CORRECT, number embedding intervention corrupts computation. When baseline is INCORRECT, intervention is neutral or slightly beneficial.

**Implication**: Number embeddings ARE causally important for correct reasoning!

### Finding 2: Asymmetric Intervention Effects

- Correct to Incorrect: 16 (harmful)
- Incorrect to Correct: 3 (beneficial)
- Ratio: 5.3:1 (harm:benefit)

Intervention is much more likely to break correct reasoning than fix incorrect reasoning.

### Finding 3: Similar Change Rate, Opposite Outcomes

- Pilot change rate: 16.0%
- Clean change rate: 14.4%

Despite similar change rates, outcomes differ dramatically based on baseline correctness.

---

## Reconciliation of Results

### Resolution: Baseline Performance Mediates Effect

**Unified Model**:
- Low baseline (52%): Neutral/Positive effect
- High baseline (90%): Negative effect (corrupts correct reasoning)

**Mechanism**:
1. Correct reasoning: Number embeddings carry critical information
2. Incorrect reasoning: Computation already broken, intervention has mixed effect

---

## Statistical Analysis

### Accuracy Change Significance

**McNemar Test**:
- Before: 119/132 correct
- After: 106/132 correct
- Discordant pairs: 16 worsened, 3 improved
- p-value = 0.0028
- **Conclusion**: Accuracy decrease is statistically significant (p < 0.01)

**Contrast with Pilot**:
- Pilot p-value: 0.655 (not significant)
- Clean p-value: 0.0028 (highly significant)

---

## Mechanistic Insights

### What Do Number Embeddings Encode?

**Evidence**:
1. Not purely decorative: -9.85% accuracy impact
2. Not completely critical: 85.6% unchanged
3. More important for correct reasoning

**Encoding Model - Hybrid representation**:
- Primary: High-dimensional continuous features (robust, distributed)
- Secondary: Token embedding projections (guiding, important for correctness)
- When secondary corrupted, primary sometimes fails

### Why 85.6% Unchanged?

**Robustness Mechanisms**:
1. Distributed encoding across multiple features
2. Continuous features beyond token identity
3. Attention to uncorrupted positions
4. Redundancy across CoT positions

---

## Implications

### For CODI Interpretability

**Revised Understanding**:
- Decoded tokens are PARTIALLY causal, not purely observational
- They guide computation but do not fully determine it
- Continuous features carry primary computational burden

### For Model Safety

**Vulnerability**: Number embedding corruption causes 9.85% accuracy drop

**Severity**: Medium (significant but not catastrophic)

### For Future Experiments

**Design Principles**:
1. Test on high-quality baselines
2. Report change direction (not just rate)
3. Measure asymmetry (harm vs benefit)
4. Control for baseline performance

**Methodological Lesson**: Low baseline (52%) gave misleading conclusion. High baseline (90%) revealed true causal role.

---

## Limitations

1. Dataset Selection Bias: Only clean variants
2. Single Model Checkpoint
3. Intervention Method: Projection replacement only
4. Fixed CoT Length
5. No Mechanistic Explanation

---

## Future Directions

1. Corrupted Variant Analysis
2. Dose-Response Analysis (sweep k values)
3. Position-Specific Intervention
4. Target Token Analysis
5. Full GSM8K Test Set

---

## Deliverables

- Code: scale_intervention_llama_clean.py, generate_embedded_html_clean.py
- Data: llama_intervention_results_clean.json
- Visualizations: results_visualization_clean_embedded.html (233.9 KB)
- Documentation: This file

---

## Conclusion

This experiment **reverses the initial conclusion** from the 50-sample pilot, demonstrating that **number token embeddings ARE causally important for correct reasoning** in CODI-LLaMA.

### Summary of Evidence

1. Significant Performance Degradation: -9.85% (p=0.0028)
2. Asymmetric Harm: 16 correct to incorrect vs 3 incorrect to correct
3. Baseline-Mediated Effect
4. Partial Robustness: 85.6% unchanged

### Revised Mechanistic Model

**Hybrid Encoding**:
- Primary: Continuous features (robust, distributed)
- Secondary: Token embeddings (guiding, important)
- Failure: When primary insufficient, secondary corruption breaks reasoning

### Methodological Insight

**Critical Lesson**: Baseline performance is crucial
- Low baseline masks effects
- High baseline reveals true causal role

**Recommendation**: Always test on high-quality baselines

### Bottom Line

Number token embeddings:
- Are NOT purely decorative
- ARE causally important for correct reasoning
- Show partial robustness (85.6%)
- Follow baseline-dependent pattern

---

**Experiment conducted by**: Claude Code
**Date**: October 29, 2025
**Status**: Complete
