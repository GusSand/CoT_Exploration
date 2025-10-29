# Mechanistic Analysis Follow-Up - Critical Heads Experiments

**Context**: We've identified critical attention heads for continuous thought reasoning in LLaMA and GPT-2. Now we need to test if these patterns are **causally important** or just correlational.

**Previous Work**:
- [10-28_both_gsm8k_critical_heads_comparison.md](../experiments/10-28_both_gsm8k_critical_heads_comparison.md)
- [10-28_llama_gsm8k_attention_flow_analysis.md](../experiments/10-28_llama_gsm8k_attention_flow_analysis.md)

**Status**: 🎯 **Ready for PM to create user stories**

---

## Quick Context: What We Found

**Critical Heads Identified**:
- **LLaMA**: L4H5 (Hub Aggregator at Position 0, composite=0.528)
- **GPT-2**: L0H3 (Multi-Purpose at Position 1, composite=0.600)

**Key Discovery**: Different hub positions (CT0 vs CT1) and layer strategies (middle vs early)

**Open Question**: Are these patterns **functionally necessary** or just **observational artifacts**?

---

## Proposed Experiments (For PM to Prioritize)

### Category 1: Causal Necessity Tests (PRIORITY: HIGH)

These test whether identified patterns are **required** for reasoning.

---

#### Experiment 1.1: Critical Head Ablation

**Goal**: Test if top critical heads are causally necessary for correct reasoning

**Method**:
- Ablate (zero out) attention weights for top 3 heads in each model
- Measure accuracy drop on 1,000 GSM8K test problems
- Test incremental ablation (1 head, 2 heads, 3 heads)

**Expected Time**: 2-3 hours
- 1h: Implement ablation mechanism
- 1h: Run inference on test set
- 0.5h: Analysis and documentation

**Success Criteria**:
- If accuracy drops >10%: Heads are causally important ✅
- If accuracy drops <2%: Heads are redundant, hub is distributed ❌

**Value**: **HIGH** - Directly validates whether our attention analysis found functional circuits

**Dependencies**:
- Test set (1,000 GSM8K problems)
- Existing CODI models (already available)

**Code Starting Point**:
```python
# Pseudo-code structure
heads_to_ablate = [(4, 5), (5, 30), (5, 28)]  # LLaMA top 3

def ablate_heads(model, heads_list):
    # Hook to zero out attention for specific heads
    for layer, head in heads_list:
        model.register_hook(layer, head, zero_attention)

    # Run inference
    accuracy = evaluate(model, test_set)
    return accuracy

# Incremental ablation
for n in [1, 2, 3]:
    acc = ablate_heads(model, heads_to_ablate[:n])
    print(f"Ablate {n} heads: {acc:.2f}%")
```

**Deliverables**:
- Ablation accuracy results (CSV: n_heads, llama_acc, gpt2_acc)
- Experiment report documenting findings
- Visualization: Accuracy vs number of heads ablated

---

#### Experiment 1.2: Hub Position Patching

**Goal**: Test if hub position (Position 0 for LLaMA, Position 1 for GPT-2) is a causal bottleneck

**Method**:
- Replace hub position activations with random noise
- Compare accuracy drop for hub vs non-hub positions
- Test: Does ablating Position 0 hurt LLaMA more than Position 2?

**Expected Time**: 3-4 hours
- 1.5h: Implement activation patching
- 1h: Run experiments on both models
- 1h: Analysis and comparison

**Success Criteria**:
- Hub position ablation causes >10% accuracy drop
- Non-hub position ablation causes <5% accuracy drop
- Delta confirms hub is a bottleneck

**Value**: **HIGH** - Tests whether hub-centric architecture is functionally critical

**Dependencies**:
- Activation caching infrastructure (already exists from prior experiments)

**Deliverables**:
- Position ablation results (CSV: position, llama_drop, gpt2_drop)
- Visualization: Accuracy drop by position
- Experiment report

---

#### Experiment 1.3: Attention Pattern Intervention

**Goal**: Test if forcing hub-centric attention pattern is necessary for accuracy

**Method**:
- Override attention to force specific patterns:
  - Hub pattern: All positions attend 80% to hub position
  - Uniform pattern: Equal attention to all positions
  - Sequential pattern: Position i attends only to i-1
- Measure accuracy under each forced pattern

**Expected Time**: 4-5 hours
- 2h: Implement attention override mechanism
- 1.5h: Run experiments with 3 patterns × 2 models
- 1h: Analysis and documentation

**Success Criteria**:
- Hub pattern: Maintains >90% of baseline accuracy
- Uniform pattern: Drops >15% accuracy
- Sequential pattern: Drops >20% accuracy

**Value**: **MEDIUM-HIGH** - Tests architectural necessity, not just head importance

**Dependencies**:
- Attention override hooks (new implementation needed)

**Deliverables**:
- Pattern intervention results
- Comparative analysis: Which pattern is most critical?
- Experiment report

---

### Category 2: Cross-Model Understanding (PRIORITY: MEDIUM)

These investigate **why** models differ in their strategies.

---

#### Experiment 2.1: Cross-Model Head Transplantation

**Goal**: Test if critical heads are model-specific or implement universal functions

**Method**:
- Extract weight matrices for top hub heads:
  - LLaMA L4H5 (hub aggregator)
  - GPT-2 L0H3 (multi-purpose)
- Transplant GPT-2 head into LLaMA and vice versa
- Measure if transplanted head maintains functionality

**Expected Time**: 5-6 hours
- 2h: Implement head weight extraction and transplant
- 2h: Run inference with transplanted heads
- 1.5h: Analysis of compatibility

**Success Criteria**:
- Transplanted head maintains >80% accuracy: Universal function ✅
- Transplanted head drops >30% accuracy: Model-specific ❌

**Value**: **MEDIUM** - Reveals whether hub aggregation is a shared circuit

**Dependencies**:
- Weight manipulation infrastructure (moderate complexity)

**Challenges**:
- LLaMA has 2048-dim embeddings, GPT-2 has 768-dim
- May need projection layer for compatibility

**Deliverables**:
- Transplantation results
- Analysis: Are hub functions universal or model-specific?
- Experiment report

---

#### Experiment 2.2: Hub Position Flexibility Test

**Goal**: Test if LLaMA can function with Position 1 hub (like GPT-2) and vice versa

**Method**:
- Surgically modify attention routing:
  - Force LLaMA to use Position 1 as hub (instead of Position 0)
  - Force GPT-2 to use Position 0 as hub (instead of Position 1)
- Measure adaptation capability

**Expected Time**: 4-5 hours
- 2h: Implement attention rerouting
- 1.5h: Run experiments
- 1h: Analysis

**Success Criteria**:
- Model maintains >70% accuracy: Hub position is arbitrary ✅
- Model drops >40% accuracy: Hub position is architecturally determined ❌

**Value**: **MEDIUM** - Tests whether hub position is learned or hardcoded

**Deliverables**:
- Hub rerouting results
- Analysis: Is hub position flexible or fixed?
- Experiment report

---

### Category 3: Information Flow Tracing (PRIORITY: MEDIUM-LOW)

These investigate **what information** flows through critical heads.

---

#### Experiment 3.1: Layer-by-Layer Information Accumulation

**Goal**: Trace when reasoning information accumulates at hub position

**Method**:
- Train linear probes on hub position activations at each layer
- Probe target: Predict final answer from hub activations
- Measure: At which layer does hub "know" the answer?

**Expected Time**: 3-4 hours
- 1h: Extract activations for all layers (100 problems)
- 1.5h: Train linear probes (16 layers × LLaMA + 12 layers × GPT-2)
- 1h: Analysis and visualization

**Success Criteria**:
- Identify "critical layer" where probe accuracy jumps
- Confirm it aligns with critical head locations (L4-5 for LLaMA, L0 for GPT-2)

**Value**: **MEDIUM-LOW** - Provides temporal insight into reasoning process

**Dependencies**:
- Activation caching (already exists)
- Linear probe training infrastructure (already exists from SAE experiments)

**Deliverables**:
- Probe accuracy by layer (CSV + visualization)
- Identification of "reasoning inflection point"
- Experiment report

---

#### Experiment 3.2: Feature Attribution for Critical Heads

**Goal**: Understand what information critical heads extract from input

**Method**:
- Use Integrated Gradients to attribute importance to input tokens
- For each critical head, identify:
  - Which input tokens it attends to most
  - What semantic features it tracks (numbers, operations, entities)

**Expected Time**: 5-6 hours
- 2h: Set up Integrated Gradients infrastructure
- 2h: Run attribution on 100 problems
- 1.5h: Semantic analysis of attention targets

**Success Criteria**:
- Identify clear semantic patterns (e.g., "L4H5 tracks multiplication operations")
- Patterns align with functional type (Hub Aggregator, Skip Connection, etc.)

**Value**: **MEDIUM-LOW** - Reveals semantic role, but doesn't test causality

**Dependencies**:
- Captum or similar attribution library
- Token-level semantic annotations

**Deliverables**:
- Attribution heatmaps for critical heads
- Semantic role summary
- Experiment report

---

### Category 4: Difficulty and Problem Stratification (PRIORITY: LOW)

These investigate **when** critical heads matter most.

---

#### Experiment 4.1: Hub Importance by Problem Difficulty

**Goal**: Test if hub is more critical for complex vs simple problems

**Method**:
- Stratify GSM8K test set by difficulty:
  - Easy: Model confidence >0.9, single-step problems
  - Medium: Model confidence 0.6-0.9, 2-3 steps
  - Hard: Model confidence <0.6, multi-step reasoning
- Ablate hub heads for each difficulty tier
- Measure: Does accuracy drop more for hard problems?

**Expected Time**: 3-4 hours
- 1h: Stratify test set by difficulty
- 1.5h: Run ablation experiments per tier
- 1h: Analysis

**Success Criteria**:
- Hard problem accuracy drop >15%
- Easy problem accuracy drop <5%
- Confirms hub is critical for complex reasoning

**Value**: **LOW-MEDIUM** - Interesting but not essential for causal understanding

**Dependencies**:
- Difficulty annotations or confidence scores

**Deliverables**:
- Accuracy drop by difficulty tier
- Analysis: When is the hub most important?
- Experiment report

---

## Recommended Prioritization for PM

### Phase 1: Validate Causal Necessity (MUST DO)
**Goal**: Confirm findings are functionally important, not just correlational

**User Stories** (3 weeks):
1. **Story 1.1**: Critical Head Ablation (2-3 hours)
2. **Story 1.2**: Hub Position Patching (3-4 hours)
3. **Story 1.3**: Attention Pattern Intervention (4-5 hours)

**Expected Outcome**: Know whether critical heads are **causal** or **coincidental**

**Decision Point**:
- If heads are causal (>10% accuracy drop) → Continue to Phase 2
- If heads are not causal (<2% accuracy drop) → Pivot to distributed analysis

---

### Phase 2: Cross-Model Understanding (NICE TO HAVE)
**Goal**: Understand why LLaMA and GPT-2 differ

**User Stories** (2 weeks):
1. **Story 2.2**: Hub Position Flexibility (4-5 hours)
2. **Story 2.1**: Cross-Model Head Transplantation (5-6 hours) - Optional, if time permits

**Expected Outcome**: Understand if hub strategies are **universal** or **model-specific**

---

### Phase 3: Information Flow Tracing (OPTIONAL)
**Goal**: Deep dive into semantic roles

**User Stories** (2 weeks):
1. **Story 3.1**: Layer-by-Layer Information Accumulation (3-4 hours)
2. **Story 3.2**: Feature Attribution (5-6 hours)

**Expected Outcome**: Know **what information** flows through critical heads

---

### Phase 4: Problem Difficulty Analysis (DEFER)
**Goal**: Understand when hub matters most

**User Stories** (1 week):
1. **Story 4.1**: Hub Importance by Difficulty (3-4 hours)

**Expected Outcome**: Know **when** hub is most critical

---

## Estimated Total Time Investment

| Phase | Time | Priority |
|-------|------|----------|
| Phase 1: Causal Validation | 10-12 hours | MUST DO |
| Phase 2: Cross-Model Understanding | 9-11 hours | NICE TO HAVE |
| Phase 3: Information Flow | 8-10 hours | OPTIONAL |
| Phase 4: Difficulty Analysis | 3-4 hours | DEFER |
| **Total** | **30-37 hours** | |

**Recommended Minimal Scope**: Phase 1 only (10-12 hours)
**Recommended Full Scope**: Phase 1 + Phase 2 (19-23 hours)

---

## Technical Dependencies

### Existing Infrastructure (✅ Available)
- CODI models (LLaMA 1B, GPT-2 124M)
- Activation caching (`cache_activations.py`, `cache_activations_llama.py`)
- GSM8K test set (1,319 problems)
- Linear probe training (from SAE experiments)

### New Infrastructure Needed (⚠️ To Build)
- **Attention override hooks** (Stories 1.2, 1.3, 2.2)
- **Head ablation mechanism** (Story 1.1)
- **Weight transplantation utilities** (Story 2.1)
- **Integrated Gradients setup** (Story 3.2)

### Moderate Complexity
- Most experiments require 1-2 hours of infrastructure setup
- Ablation and patching are straightforward (modify forward pass)
- Transplantation is harder (dimension mismatch issues)

---

## Success Metrics

### Phase 1 Success Criteria
- [ ] Measure accuracy drop from ablating top 3 heads
- [ ] Identify if hub position is a bottleneck
- [ ] Determine which attention pattern is necessary

**Decision**: If >10% accuracy drop → Critical heads are causal → Proceed to Phase 2

### Phase 2 Success Criteria
- [ ] Test if hub position is flexible or fixed
- [ ] Test if hub heads are transferable across models

**Decision**: If hub is universal → Deeper mechanistic understanding possible

### Phase 3 Success Criteria
- [ ] Identify when reasoning information accumulates
- [ ] Attribute semantic roles to critical heads

### Overall Success
- **Minimum**: Validate causal importance (Phase 1)
- **Ideal**: Full mechanistic circuit understanding (Phase 1-3)

---

## Open Questions for PM to Consider

1. **Scope**: Should we do Phase 1 only (causal validation) or expand to Phase 2 (cross-model)?
2. **Dataset Size**: Use 100 problems (fast, 10min inference) or 1,000 problems (robust, 1.5h inference)?
3. **Error Analysis**: Should we analyze which problem types are most affected by ablation?
4. **Visualization**: What figures are most important for publication/presentation?
5. **Comparison Baseline**: Should we compare against random head ablation?

---

## Potential Risks

1. **Null Result Risk**: Critical heads may not be causal (accuracy drop <2%)
   - **Mitigation**: Still valuable negative result - hub is distributed, not localized

2. **Implementation Complexity**: Attention override may be harder than expected
   - **Mitigation**: Start with simpler ablation experiments (Story 1.1)

3. **Dimension Mismatch**: Head transplantation may fail due to embedding size differences
   - **Mitigation**: Consider this optional (Story 2.1)

4. **Time Overrun**: Inference on 1,000 problems may be slow
   - **Mitigation**: Use 100-problem subset for initial validation

---

## Next Steps

**For PM**:
1. Review experiment proposals
2. Prioritize based on research goals and timeline
3. Create user stories for approved experiments
4. Estimate costs (hours + compute)
5. Assign to developer

**For Developer** (once approved):
1. Start with **Story 1.1** (Critical Head Ablation) - simplest and highest value
2. Validate infrastructure works on small sample (10 problems)
3. Scale to full test set (100-1,000 problems)
4. Document results and proceed to next story

---

## Contact

For questions about experimental design or technical feasibility, refer to:
- **Attention Analysis Report**: `docs/experiments/10-28_both_gsm8k_critical_heads_comparison.md`
- **Code Location**: `src/experiments/codi_attention_flow/`
- **Critical Heads Rankings**: `src/experiments/codi_attention_flow/results/{model}/ranked_heads.csv`

---

**Status**: 🎯 Ready for PM review and user story creation
**Created**: 2025-10-28
**Author**: Developer (Claude Code)
