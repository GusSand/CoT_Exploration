# LLaMA SAE Feature Hierarchy & Causal Validation Architecture

**Date**: 2025-10-27
**Status**: Architecture Design
**Architect**: Claude Code

---

## 1. Executive Summary

This document defines the architecture for investigating feature hierarchy in LLaMA TopK SAEs and validating feature interpretations through causal interventions. The research aims to answer:

1. **Do SAEs learn higher-level features** (operations like multiplication) vs lower-level features (specific values like 12, 50)?
2. **Can we validate interpretations** by swapping feature activations and observing predictable output changes?

**Key Architectural Decisions**:
- Use optimal TopK SAE config: **K=100, d=512** (87.8% EV, 0% death rate)
- Focus on **LLaMA-3.2-1B** model only
- Leverage existing validation dataset: **1,495 samples** with pre-computed activations
- Implement **modular intervention system** supporting swap, ablation, and amplification
- Use **ground truth feature pairs** for causal validation experiments

---

## 2. Infrastructure Overview

### 2.1 Existing Assets

**Models**:
- ✅ LLaMA-3.2-1B with CODI (continuous thought) training
- ✅ 1,152 trained TopK SAE checkpoints (16 layers × 6 positions × 12 configs)
- ✅ Optimal config identified: K=100, d=512

**Data**:
- ✅ 1,495 validation samples in `src/experiments/sae_cot_decoder/data/full_val_activations.pt`
- ✅ Pre-computed activations: 16 layers × 6 positions × 2048 dims
- ✅ Metadata: problem IDs, CoT sequences, layers, positions
- ✅ 1,000 problem GSM8K dataset with numerical diversity (121 with "12", 175 with "50", 236 with "20")

**Code**:
- ✅ TopK SAE implementation: `src/experiments/topk_grid_pilot/topk_sae.py`
- ✅ Feature analysis framework: `src/experiments/topk_grid_pilot/analyze_feature_semantics.py`
- ✅ Checkpoint naming convention: `pos{position}_layer{layer}_d{latent_dim}_k{k}.pt`

### 2.2 What We Need to Build

**New Components**:
1. **Feature taxonomy system** - Label features by abstraction level
2. **Causal intervention engine** - Swap/ablate/amplify feature activations during inference
3. **Ground truth validation framework** - Test predictions with controlled interventions
4. **Operation-level feature detector** - Identify abstract operation features

---

## 3. Data Architecture

### 3.1 Data Validation

**Reliability Check**:
- ✅ **1,495 validation samples** available
- ✅ **Pre-computed activations** (1.1 GB file exists)
- ✅ **Sufficient numerical diversity**:
  - Problems with "12": 121 (8.1%)
  - Problems with "50": 175 (11.7%)
  - Problems with "20": 236 (15.7%)
  - Problems with "30": 129 (8.6%)
  - Problems with "100": 83 (5.5%)

**Quality Assurance**:
- ✅ No duplicate data (each sample unique by problem_id + layer + position)
- ✅ Correct labels verified (from CODI training validation set)
- ✅ Stratified difficulty distribution (see DATA_INVENTORY.md line 280-346)

**Train/Test Split**:
- **Not required** for this experiment
- Using entire 1,495 validation set for feature analysis
- For swap experiments: problems selected by numerical content (e.g., contains "12" or "50")
- No training involved - only analyzing pre-trained SAE features

### 3.2 Dataset Allocation

**Feature Taxonomy (Story 1)**:
- Dataset: Full 1,495 validation samples
- Purpose: Identify and categorize features across all problems
- Output: `feature_labels.json` with ground truth interpretations

**Feature Activation Analysis (Story 2)**:
- Dataset: Top-100 activating examples per feature (sampled from 1,495)
- Purpose: Validate feature interpretations before intervention
- Output: `activation_patterns.json` with contexts

**Causal Validation (Story 4)**:
- Dataset: Targeted test set (~100-150 problems containing specific numbers)
- Purpose: Test swap predictions
- Selection criteria:
  - Problems containing "12": sample 20-30
  - Problems containing "50": sample 20-30
  - Problems containing "20": sample 20-30
  - Problems containing "100": sample 10-20
- Output: `swap_test_problems.json`

**Operation-Level Discovery (Story 5)**:
- Dataset: Full 1,495 validation samples
- Purpose: Identify operation-level features
- Output: `operation_features.json`

---

## 4. Intervention Architecture

### 4.1 Design Principles

**Modularity**: Intervention logic separated from model inference
**Flexibility**: Support multiple intervention types (swap, ablate, amplify)
**Safety**: Non-destructive - original model unchanged
**Reproducibility**: All interventions logged with exact parameters

### 4.2 Intervention Engine Design

```python
class FeatureInterventionEngine:
    """
    Manages causal interventions on SAE features during inference.

    Supports:
    - Swap: Replace feature A activations with feature B
    - Ablate: Zero out specific feature activations
    - Amplify: Scale specific feature activations by factor
    """

    def __init__(self, sae_model, llama_model):
        self.sae = sae_model
        self.llama = llama_model
        self.intervention_hooks = []

    def swap_features(self, feature_a: int, feature_b: int,
                     activations: torch.Tensor) -> torch.Tensor:
        """
        Swap activations between two features.

        Args:
            feature_a: First feature index
            feature_b: Second feature index
            activations: Original activations (batch_size, input_dim)

        Returns:
            Modified activations with swapped features
        """
        # Get sparse representation
        _, sparse, _ = self.sae(activations)

        # Swap features in sparse space
        sparse_swapped = sparse.clone()
        sparse_swapped[:, feature_a], sparse_swapped[:, feature_b] = \
            sparse[:, feature_b].clone(), sparse[:, feature_a].clone()

        # Decode back to activation space
        modified_activations = self.sae.decoder(sparse_swapped)
        return modified_activations

    def ablate_feature(self, feature_idx: int,
                      activations: torch.Tensor) -> torch.Tensor:
        """Zero out specific feature in sparse representation."""
        _, sparse, _ = self.sae(activations)
        sparse_ablated = sparse.clone()
        sparse_ablated[:, feature_idx] = 0
        return self.sae.decoder(sparse_ablated)

    def amplify_feature(self, feature_idx: int, scale: float,
                       activations: torch.Tensor) -> torch.Tensor:
        """Scale specific feature activation by factor."""
        _, sparse, _ = self.sae(activations)
        sparse_amplified = sparse.clone()
        sparse_amplified[:, feature_idx] *= scale
        return self.sae.decoder(sparse_amplified)
```

### 4.3 Integration with LLaMA Forward Pass

**Hook Placement**:
- Intervention occurs **after** extracting continuous thought activations
- Intervention occurs **before** passing to next layer
- Use PyTorch forward hooks for non-invasive modification

```python
def register_intervention_hook(model, layer_idx, position_idx,
                              intervention_fn):
    """
    Register hook to modify activations at specific layer/position.

    Args:
        model: LLaMA model
        layer_idx: Which transformer layer (0-15)
        position_idx: Which token position (0-5)
        intervention_fn: Function that modifies activations
    """
    def hook(module, input, output):
        # Extract activations for target position
        modified = intervention_fn(output[:, position_idx, :])
        # Replace in output
        output[:, position_idx, :] = modified
        return output

    handle = model.layers[layer_idx].register_forward_hook(hook)
    return handle
```

### 4.4 Sanity Checks

**Identity Test**: Swap feature with itself → output should not change
**Null Test**: Ablate never-active feature → output should not change
**Consistency Test**: Swap A↔B then B↔A → should return to original

---

## 5. Validation Methodology

### 5.1 Ground Truth Feature Labeling

**Manual Analysis Process**:
1. For each feature, extract top-100 activating samples
2. Examine CoT sequences for patterns:
   - Arithmetic operations (+, -, *, /)
   - Specific numbers (12, 50, 100, etc.)
   - Keywords ("total", "difference", "product", etc.)
3. Label features with confidence levels:
   - **High confidence**: Clear pattern in >70% of samples
   - **Medium confidence**: Pattern in 50-70% of samples
   - **Low confidence**: Pattern in 30-50% of samples
   - **Unknown**: No clear pattern (<30%)

**Heuristic Detection** (automated):
- Addition: ≥3 top samples contain '+' or 'sum'
- Subtraction: ≥3 top samples contain '-' or 'difference'
- Multiplication: ≥3 top samples contain '*' or 'multiply'
- Division: ≥3 top samples contain '/' or 'divide'
- Round numbers: ≥3 top samples contain 100, 200, 500, 1000
- Specific value: ≥5 top samples contain exact number (e.g., "12")

### 5.2 Swap Experiment Design

**Hypothesis**: If feature A represents "12" and feature B represents "50", swapping them should change answer from 12→50 or 50→12.

**Experimental Protocol**:
1. **Select problem** containing specific number (e.g., "Tom has 12 apples...")
2. **Run baseline inference** (no intervention) → record answer
3. **Identify active features** for that problem at target layer/position
4. **Run swap experiment**: swap "12 feature" with "50 feature"
5. **Compare outputs**:
   - Expected: Answer changes from 12 to 50 (or vice versa)
   - Actual: What answer does model produce?
6. **Calculate validation metrics**:
   - **Exact match**: Answer changed exactly as predicted (12→50)
   - **Directional match**: Answer changed in predicted direction (12→X where X>12)
   - **No change**: Answer remained the same (prediction failed)

**Success Criteria**:
- **Strong validation**: ≥70% exact matches across 20+ test cases
- **Moderate validation**: ≥50% exact matches or ≥70% directional matches
- **Weak validation**: <50% matches (interpretation likely incorrect)

### 5.3 Operation-Level Feature Validation

**Hypothesis**: Operation-level features (e.g., "multiplication") should activate across different numerical values.

**Experimental Protocol**:
1. **Identify candidate operation feature** (e.g., activates strongly on multiplication problems)
2. **Test generalization**: Does it activate on multiplication with different numbers?
   - Example: "3 * 4", "12 * 5", "20 * 100"
3. **Ablation test**: Remove feature and measure impact across problem types
   - Expected: All multiplication problems affected
   - Compare: Effect on addition/subtraction problems (should be minimal)
4. **Quantify impact**:
   - **Strong operation feature**: >80% of same-operation problems affected, <20% of other operations
   - **Mixed feature**: 50-80% of same-operation, 20-50% of others
   - **Specialized feature**: >80% of specific-number problems, independent of operation

### 5.4 Metrics

**Feature Interpretability**:
- Pattern clarity score: % of top-activating samples matching detected pattern
- Consistency score: Agreement between automated heuristic and manual inspection

**Causal Validation**:
- Exact match rate: % of swap experiments producing predicted answer
- Directional match rate: % producing answer in predicted direction
- Null swap baseline: % of identity swaps producing unchanged output (should be 100%)

**Feature Hierarchy**:
- Operation-level score: % of features generalizing across different numbers
- Value-level score: % of features specializing on specific numbers
- Mixed-level score: % of features showing both properties

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Stories 1-2)

**Deliverables**:
- `src/experiments/llama_sae_hierarchy/feature_taxonomy.py`
- `src/experiments/llama_sae_hierarchy/analyze_activations.py`
- `docs/experiments/10-27_llama_gsm8k_feature_taxonomy.md`
- `feature_labels.json` (ground truth)

**Data Flow**:
```
full_val_activations.pt (1.1GB)
  ↓
Load SAE: pos3_layer14_d512_k100.pt
  ↓
Extract sparse features (1495 × 512)
  ↓
Rank by activation frequency
  ↓
Top 20 features → analyze top-100 samples each
  ↓
Manual + automated labeling
  ↓
feature_labels.json (20 features with interpretations)
```

### Phase 2: Infrastructure (Story 3)

**Deliverables**:
- `src/experiments/llama_sae_hierarchy/causal_interventions.py`
- `docs/code/causal_intervention_api.md`
- Unit tests

**Architecture**:
```
FeatureInterventionEngine
  ├── swap_features(feature_a, feature_b, activations)
  ├── ablate_feature(feature_idx, activations)
  ├── amplify_feature(feature_idx, scale, activations)
  └── register_intervention_hook(model, layer, position, fn)
```

### Phase 3: Validation (Stories 4-5)

**Story 4 - Swap Experiments**:
```
feature_labels.json (labeled features)
  ↓
Select feature pairs (e.g., "12 feature" + "50 feature")
  ↓
Filter problems containing target numbers
  ↓
swap_test_problems.json (~100 problems)
  ↓
For each problem:
  - Baseline inference → answer_baseline
  - Swap intervention → answer_swapped
  - Compare and validate
  ↓
Results: exact_match_rate, directional_match_rate
```

**Story 5 - Operation Features**:
```
feature_labels.json
  ↓
Identify operation-level candidates
  ↓
Test generalization across numbers
  ↓
Ablation experiments per operation type
  ↓
operation_features.json (validated operation features)
```

### Phase 4: Documentation (Story 6)

**Deliverables**:
- `docs/research_journal.md` (updated)
- `docs/DATA_INVENTORY.md` (updated with new datasets)
- `docs/experiments/10-27_llama_gsm8k_feature_hierarchy.md`
- Git commit + push to GitHub

---

## 7. Technical Specifications

### 7.1 File Formats

**feature_labels.json**:
```json
[
  {
    "feature_id": 82,
    "layer": 14,
    "position": 3,
    "activation_freq": 0.998,
    "mean_magnitude": 2.064,
    "interpretation": {
      "type": "value-specific",  // or "operation-level" or "mixed"
      "description": "Represents number 12",
      "confidence": "high",  // or "medium", "low", "unknown"
      "detected_patterns": ["number_12", "multiplication"],
      "top_samples": [
        {"problem_id": "gsm8k_001", "cot": "12 * 5 = 60", "activation": 3.2},
        ...
      ]
    }
  }
]
```

**swap_test_problems.json**:
```json
[
  {
    "problem_id": "gsm8k_123",
    "question": "Tom has 12 apples...",
    "target_number": 12,
    "swap_number": 50,
    "feature_a": 82,
    "feature_b": 133,
    "layer": 14,
    "position": 3
  }
]
```

**Results format**:
```json
{
  "experiment": "feature_swap_validation",
  "config": {
    "sae_config": "K=100, d=512",
    "layer": 14,
    "position": 3,
    "num_tests": 100
  },
  "results": [
    {
      "problem_id": "gsm8k_123",
      "baseline_answer": 12,
      "swapped_answer": 50,
      "prediction": "12→50",
      "outcome": "exact_match"  // or "directional_match", "no_change"
    }
  ],
  "metrics": {
    "exact_match_rate": 0.72,
    "directional_match_rate": 0.85,
    "null_swap_baseline": 1.00
  }
}
```

### 7.2 Computational Requirements

**Memory**:
- SAE model: ~8 MB per checkpoint
- Validation data: 1.1 GB (pre-loaded)
- Feature activations: ~3 MB per layer/position (1495 × 512 × 4 bytes)
- Total: <2 GB RAM

**Compute**:
- Feature analysis: CPU-only (no inference needed)
- Swap experiments: GPU recommended for LLaMA inference
- Estimated time per swap: ~0.5 seconds on GPU
- Total: ~50 seconds for 100 swap tests

**Storage**:
- Input data: 1.1 GB (existing)
- Generated datasets: <10 MB
- Results: <5 MB

---

## 8. Risk Mitigation

### 8.1 Data Quality Risks

**Risk**: Feature labels may be subjective or inconsistent
**Mitigation**:
- Use automated heuristic detection first
- Manual validation on subset
- Track confidence levels
- Multiple labelers for ambiguous cases (if needed)

**Risk**: Test set too small for statistical significance
**Mitigation**:
- Target ≥20 examples per feature pair
- Use multiple feature pairs (≥5 pairs)
- Total tests: ≥100
- Report confidence intervals

### 8.2 Technical Risks

**Risk**: Swap interventions may have unintended side effects
**Mitigation**:
- Start with sanity checks (identity swaps)
- Test on single layer/position first
- Compare multiple layers for consistency
- Monitor unexpected behavior

**Risk**: LLaMA inference may be slow
**Mitigation**:
- Use GPU if available
- Batch process where possible
- Cache baseline results
- Profile and optimize hot paths

### 8.3 Interpretation Risks

**Risk**: Feature interpretations may be incorrect
**Mitigation**:
- Multiple validation methods (activation patterns + causal interventions)
- Conservative confidence thresholds
- Document failures and edge cases
- Compare across layers for consistency

---

## 9. Success Criteria

**Minimum Viable Product (MVP)**:
- ✅ 20 features labeled with interpretations
- ✅ Intervention engine supports swap and ablate
- ✅ 5 feature pairs tested with ≥20 examples each
- ✅ Results documented and committed to GitHub

**Success Indicators**:
- ≥50% of features have clear interpretations (high/medium confidence)
- ≥60% exact match rate on swap experiments (strong validation)
- ≥3 operation-level features identified and validated
- Clear distinction between value-level and operation-level features

**Stretch Goals**:
- ≥80% of features interpretable
- ≥70% exact match rate
- Validation across multiple layers (3, 8, 14)
- Interactive dashboard for exploring features

---

## 10. Architecture Review Checklist

### Data Validation ✅
- [x] Check data reliability: 1,495 samples confirmed
- [x] Verify no duplicate data: unique by problem_id + layer + position
- [x] Confirm correct labels: from validated CODI training set
- [x] Assess need for train/test split: not required for this experiment

### Design Quality ✅
- [x] Use optimal SAE config (K=100, d=512)
- [x] Modular intervention system
- [x] Multiple validation methods
- [x] Reproducible experiments
- [x] Clear success criteria

### Implementation Readiness ✅
- [x] All required data available
- [x] Existing code can be reused (topk_sae.py, analyze_feature_semantics.py)
- [x] Clear file formats defined
- [x] Computational requirements reasonable
- [x] Risk mitigation strategies in place

---

## 11. References

**Related Experiments**:
- GPT-2 TopK SAE Sweep: `docs/experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md`
- LLaMA TopK Semantics: `docs/experiments/10-27_llama_gsm8k_topk_semantics.md`
- Multi-layer Analysis: `docs/experiments/10-26_llama_gsm8k_topk_sae_multilayer.md`

**Data Sources**:
- Validation activations: `src/experiments/sae_cot_decoder/data/full_val_activations.pt`
- LLaMA 1000 dataset: `src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json`
- SAE checkpoints: `src/experiments/topk_grid_pilot/results/checkpoints/`

**Code References**:
- TopK SAE: `src/experiments/topk_grid_pilot/topk_sae.py`
- Feature analysis: `src/experiments/topk_grid_pilot/analyze_feature_semantics.py`
- Data inventory: `docs/DATA_INVENTORY.md`

---

**Architecture Status**: ✅ Ready for Developer Implementation
**Estimated Implementation Time**: 10.5-15.5 hours (as per PM stories)
**Next Step**: Switch to Developer role and begin Story 1
