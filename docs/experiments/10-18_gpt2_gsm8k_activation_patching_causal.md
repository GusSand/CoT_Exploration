# Activation Patching Causal Analysis Experiment

**Date**: 2025-10-18
**Status**: Planning
**Researcher**: Product Manager
**Goal**: Test whether CODI's decoded intermediate results are causally involved in reasoning or merely epiphenomenal correlates

---

## Research Question

**Do CODI's continuous thought representations causally determine downstream reasoning steps, or are decoded intermediates merely epiphenomenal?**

This directly addresses the central interpretability question for latent reasoning systems.

---

## Background

CODI (Continuous Chain-of-Thought via Self-Distillation) compresses natural language CoT into continuous space. The model uses special [THINK] tokens to represent reasoning steps without explicit language. While we can decode these representations using the existing `probe_latent_token.py`, it remains unclear whether these decoded values are:

1. **Causally involved**: The representations drive downstream computation
2. **Epiphenomenal**: The representations merely correlate with reasoning but don't causally influence it

---

## Experimental Design

### Experiment 1: Direct Activation Patching

**Hypothesis**: If decoded intermediate results are causal, patching clean activations into corrupted problems should restore correct computation.

#### Protocol

1. **Clean Run**:
   - Input: "John has 3 bags with 7 apples each. How many apples total?"
   - Expected intermediate: 3×7=21 (decode shows "21")
   - Final answer: 21

2. **Corrupted Run**:
   - Input: "John has 3 bags with **8** apples each. How many apples total?"
   - Expected intermediate: 3×8=24 (decode shows "24")
   - Final answer: 24

3. **Patched Run**:
   - Input: Corrupted problem (8 apples)
   - Intervention: Replace [THINK] token activation at multiplication step with clean activation (7 apples)
   - Measure: Does final answer shift back toward 21?

#### Measurements
- **Primary**: Final answer accuracy recovery rate
- **Secondary**:
  - Intermediate decode consistency
  - Downstream [THINK] token decodes
  - Output distribution KL divergence

#### Success Criteria
- ✅ >50% accuracy recovery with patching
- ✅ Intermediate decodes shift toward patched value
- ✅ Effect is specific to reasoning layers (not uniform across all layers)

---

### Experiment 2: Counterfactual Patching

**Hypothesis**: Patching representations from different problems should predictably alter reasoning.

#### Protocol

1. **Source Problem**:
   - Input: "What is 2×6?"
   - [THINK] decodes to "12"
   - Extract activation at multiplication step

2. **Target Problem**:
   - Input: "What is 3×5?"
   - Expected: [THINK] should decode to "15"

3. **Patched Run**:
   - Input: "What is 3×5?"
   - Intervention: Replace [THINK] activation with "12" from source
   - Measure: Does answer shift toward 12-based reasoning?

#### Measurements
- **Primary**: Distribution shift in final answers
- **Secondary**:
  - Systematic vs random errors
  - Decode values at subsequent [THINK] tokens
  - Magnitude of distribution shift

#### Success Criteria
- ✅ Predictable shift in answer distribution toward patched value
- ✅ Systematic errors (not random noise)
- ✅ Effect propagates to downstream reasoning steps

---

### Experiment 3: Ablation + Substitution

**Hypothesis**: Ablating continuous thoughts should impair reasoning; substituting them should be sufficient to alter computation.

#### Protocols

**A) Ablation**:
1. Zero out [THINK] token activation at step *i*
2. Measure impact on step *i+1* and final answer
3. Test at different positions (early vs late steps)

**B) Substitution**:
1. Replace [THINK] activation with:
   - Mean activation across dataset
   - Random sample from other problems
   - Gaussian noise
2. Measure degradation in reasoning quality

#### Measurements
- **Primary**: Accuracy drop by ablation position
- **Secondary**:
  - Recovery capability (can model compensate?)
  - Error patterns (systematic vs random)
  - Layer-specific effects

#### Success Criteria
- ✅ Systematic accuracy drop with ablation
- ✅ Early ablations have larger impact than late ablations
- ✅ Model cannot fully compensate for ablations

---

## Control Conditions

### 1. Random Patching
- Patch random activations (not from valid reasoning steps)
- Should have no systematic effect if model is truly using causal structure

### 2. Layer Controls
- Patch at different layers: early (layers 1-4), middle (5-8), late (9-12)
- Identify which layers are causally important for reasoning

### 3. Explicit CoT Baseline
- Run identical experiments on explicit CoT model
- Compare causal structure: implicit vs explicit reasoning

### 4. Token Position Controls
- Patch non-[THINK] tokens to verify effect is specific to continuous thought
- E.g., patch regular word embeddings and verify no systematic effect

---

## Dataset Specification

### Primary Dataset: GSM8K Subset

**Selection Criteria**:
- 2-3 reasoning steps (interpretable intermediates)
- Clear arithmetic operations (multiplication, addition)
- Verifiable intermediate values

**Size**: 500 problem pairs per condition

**Problem Pair Types**:
1. **Clean/Corrupted**: Same problem structure, one number changed
2. **Counterfactual**: Different problems, matched intermediate values
3. **Ablation**: Single problem, multiple ablation positions

### Secondary Dataset: CommonsenseQA

**Purpose**: Test generalization beyond arithmetic
**Size**: 200 problems
**Use**: Validate findings aren't arithmetic-specific

---

## Implementation Requirements

### Infrastructure Needed

#### 1. Activation Hooks (NEW - Must Build)
```python
class ActivationCache:
    """Cache and retrieve activations from specific layers/positions"""
    - register_hooks(model, layer_indices, token_positions)
    - extract_activations(forward_pass)
    - inject_activations(forward_pass, cached_activations)
```

#### 2. Intervention Framework (NEW - Must Build)
```python
class InterventionManager:
    """Manage activation patching experiments"""
    - patch_activation(source, target, layer, position)
    - ablate_activation(target, layer, position, method='zero')
    - substitute_activation(target, layer, position, replacement)
```

#### 3. Enhanced Decoder (EXTEND EXISTING)
- Current: `probe_latent_token.py` does top-k probing
- Add: Confidence scores, attention weights, layer-wise decoding

#### 4. WandB Integration (NEW - Must Build)
- Experiment tracking
- Hyperparameter logging
- Metric visualization
- Comparative analysis across runs

#### 5. Visualization Module (NEW - Must Build)
```python
class ActivationVisualizer:
    - plot_attention_maps()
    - plot_activation_trajectories()
    - plot_intervention_effects()
    - plot_layer_importance()
    - plot_error_distributions()
```

### Metrics to Track

#### Accuracy Metrics
- Final answer accuracy (exact match)
- Intermediate decode accuracy
- Recovery rate (clean→corrupted patching)

#### Distribution Metrics
- KL divergence (output distributions)
- Jensen-Shannon divergence
- Cosine similarity (activations)

#### Causal Metrics
- Intervention effect size (Cohen's d)
- Layer-wise importance scores
- Position-wise importance scores

#### Interpretability Metrics
- Decode confidence scores
- Attention entropy
- Activation consistency

---

## Expected Outcomes

### If Causal Hypothesis Supported:
1. Clean→Corrupted patching restores >50% accuracy
2. Counterfactual patching shifts answers predictably
3. Ablation causes systematic downstream errors
4. Effects are layer-specific and position-specific
5. Explicit CoT shows similar causal structure

**Implication**: Continuous thoughts are genuinely involved in reasoning, supporting interpretability claims for latent CoT.

### If Epiphenomenal Hypothesis Supported:
1. Patching has minimal effect (<10% change)
2. Ablation doesn't prevent correct answers
3. Effects are uniform across layers
4. No systematic patterns in errors
5. Explicit CoT shows different causal structure

**Implication**: Decoded values are correlates, not causes. Reasoning may occur through different mechanisms.

---

## Timeline Estimate

### Phase 1: Infrastructure (Week 1)
- Day 1-2: Activation hooks and caching
- Day 3-4: Intervention framework
- Day 5: WandB integration
- Day 6-7: Visualization module

### Phase 2: Experiment 1 (Week 2)
- Day 1-2: Generate problem pairs
- Day 3-4: Run experiments
- Day 5-7: Analysis and visualization

### Phase 3: Experiments 2-3 (Week 3)
- Day 1-3: Experiment 2 (Counterfactual)
- Day 4-6: Experiment 3 (Ablation)
- Day 7: Cross-experiment analysis

### Phase 4: Controls & Documentation (Week 4)
- Day 1-2: Control conditions
- Day 3-4: Explicit CoT baseline comparison
- Day 5-7: Final analysis and write-up

**Total Estimate**: 4 weeks

---

## Compute Resources

### GPU Requirements
- Single A100/V100 GPU
- ~4GB VRAM per forward pass with activation caching
- Estimated: 2-3 hours per condition (500 problems)

### Storage
- Cached activations: ~5GB per experiment
- Visualizations: ~500MB
- Logs and metrics: ~100MB

### Total Compute
- ~20 hours GPU time
- ~50GB storage

---

## Risk Assessment

### Technical Risks
1. **Activation extraction overhead**: May slow inference significantly
   - Mitigation: Batch processing, selective layer caching

2. **Intervention artifacts**: Patching may create distribution shifts
   - Mitigation: Careful normalization, control conditions

3. **Decode ambiguity**: Multiple tokens may decode to similar probabilities
   - Mitigation: Use confidence thresholds, analyze top-k not just top-1

### Scientific Risks
1. **Null results**: May find no causal effects
   - Mitigation: Test multiple hypotheses, check explicit CoT baseline

2. **Confounding variables**: Other factors may explain results
   - Mitigation: Extensive controls, layer/position ablations

---

## Success Metrics

### Research Success
- Clear answer to causal vs epiphenomenal question
- Quantitative effect sizes with confidence intervals
- Actionable insights for interpretability

### Technical Success
- Reusable intervention framework
- Comprehensive visualization suite
- Well-documented experiments for reproducibility

### Documentation Success
- Detailed write-up with all results
- Code documented and version controlled
- Findings shared in research journal

---

## References

### Related Work
- **Causal Mediation Analysis**: Vig et al. (2020) - Investigating causal attention patterns
- **Activation Patching**: Meng et al. (2022) - Locating and editing factual associations
- **Mechanistic Interpretability**: Olsson et al. (2022) - In-context learning circuits

### Internal Documentation
- CODI Paper: `/home/paperspace/dev/CoT_Exploration/docs/codi.pdf`
- Lit Review: `/home/paperspace/dev/CoT_Exploration/docs/lit_review.pdf`
- Model Implementation: `/home/paperspace/dev/CoT_Exploration/codi/src/model.py`
- Probe Implementation: `/home/paperspace/dev/CoT_Exploration/codi/probe_latent_token.py`

---

## Next Steps

1. ✅ Complete experimental design (this document)
2. ⏳ Create detailed user stories
3. ⏳ Cost estimation
4. ⏳ Get approval for implementation
5. ⏳ Begin infrastructure development
