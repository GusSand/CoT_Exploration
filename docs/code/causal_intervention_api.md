# Causal Intervention API Documentation

**Module**: `src/experiments/llama_sae_hierarchy/causal_interventions.py`
**Class**: `FeatureInterventionEngine`

---

## Overview

The `FeatureInterventionEngine` provides a clean API for performing causal interventions on SAE features. It supports three types of interventions:

1. **Ablation**: Zero out specific features to test necessity
2. **Swap**: Exchange activations between features to test content specificity
3. **Amplification**: Scale feature activations to test sufficiency

---

## Quick Start

```python
from causal_interventions import FeatureInterventionEngine
from topk_sae import TopKAutoencoder

# Load your trained SAE
sae = TopKAutoencoder(input_dim=2048, latent_dim=512, k=100)
sae.load_state_dict(checkpoint['model_state_dict'])

# Create intervention engine
engine = FeatureInterventionEngine(sae)

# Run interventions
ablated = engine.ablate_feature(activations, feature_idx=449)
swapped = engine.swap_features(activations, feature_a=332, feature_b=194)
amplified = engine.amplify_feature(activations, feature_idx=156, scale=2.0)
```

---

## API Reference

### Constructor

#### `FeatureInterventionEngine(sae_model)`

Initialize the intervention engine with a trained SAE model.

**Parameters**:
- `sae_model` (`TopKAutoencoder`): Trained TopK SAE model

**Example**:
```python
engine = FeatureInterventionEngine(sae_model)
```

---

### Core Methods

#### `ablate_feature(activations, feature_idx) -> Tensor`

Zero out a specific feature in the sparse representation.

**Parameters**:
- `activations` (`Tensor`): Input activations, shape `(batch_size, input_dim)`
- `feature_idx` (`int`): Index of feature to ablate `(0 to latent_dim-1)`

**Returns**:
- `modified_activations` (`Tensor`): Activations with feature ablated

**Use Case**: Test if a feature is necessary for correct computation

**Example**:
```python
# Ablate general feature 449 (99.87% activation)
modified = engine.ablate_feature(activations, feature_idx=449)

# Compare outputs with/without feature
original_output = model(activations)
ablated_output = model(modified)
```

---

#### `ablate_features(activations, feature_indices) -> Tensor`

Zero out multiple features simultaneously.

**Parameters**:
- `activations` (`Tensor`): Input activations
- `feature_indices` (`List[int]`): List of feature indices to ablate

**Returns**:
- `modified_activations` (`Tensor`): Activations with all features ablated

**Use Case**: Test combined effect of ablating multiple features

**Example**:
```python
# Ablate top 5 most frequent features
top_features = [449, 168, 131, 261, 288]
modified = engine.ablate_features(activations, top_features)
```

---

#### `swap_features(activations, feature_a, feature_b) -> Tensor`

Exchange activations between two features.

**Parameters**:
- `activations` (`Tensor`): Input activations
- `feature_a` (`int`): First feature index
- `feature_b` (`int`): Second feature index

**Returns**:
- `modified_activations` (`Tensor`): Activations with features swapped

**Use Case**: Test if features encode specific content (e.g., swap "addition" with "subtraction")

**Example**:
```python
# Swap operation-specialized features
addition_feat = 332  # addition specialist
subtraction_feat = 194  # subtraction specialist
modified = engine.swap_features(activations, addition_feat, subtraction_feat)
```

---

#### `amplify_feature(activations, feature_idx, scale) -> Tensor`

Scale a feature's activation by a multiplicative factor.

**Parameters**:
- `activations` (`Tensor`): Input activations
- `feature_idx` (`int`): Feature to amplify
- `scale` (`float`): Scaling factor (e.g., `2.0` = double, `0.5` = halve, `10.0` = amplify 10x)

**Returns**:
- `modified_activations` (`Tensor`): Activations with feature amplified

**Use Case**: Test if amplifying a feature is sufficient to bias model behavior

**Example**:
```python
# Amplify rare multiplication feature 5x
mult_feat = 156  # multiplication specialist (24.1% activation)
modified = engine.amplify_feature(activations, mult_feat, scale=5.0)

# Check if model becomes more likely to use multiplication
```

---

### Utility Methods

#### `measure_feature_impact(activations, feature_idx, metric='all') -> Dict`

Quantify the impact of ablating a feature.

**Parameters**:
- `activations` (`Tensor`): Input activations
- `feature_idx` (`int`): Feature to measure
- `metric` (`str`): Metric type - `'reconstruction_diff'`, `'l2_norm'`, or `'all'`

**Returns**:
- `impact_metrics` (`Dict[str, float]`): Dictionary with measurements:
  - `mean_abs_diff`: Mean absolute difference in reconstruction
  - `max_abs_diff`: Maximum absolute difference
  - `mean_l2_diff`: Mean L2 norm of difference vector
  - `max_l2_diff`: Maximum L2 norm
  - `feature_activation_freq`: How often feature is active (%)
  - `feature_mean_magnitude`: Average magnitude when active

**Use Case**: Quantify feature importance without running full model inference

**Example**:
```python
# Measure impact of ablating a general feature
impact = engine.measure_feature_impact(activations, feature_idx=449, metric='all')

print(f"Activation freq: {impact['feature_activation_freq']:.1%}")
print(f"Mean impact: {impact['mean_abs_diff']:.4f}")
print(f"Max impact: {impact['max_abs_diff']:.4f}")
```

---

### Sanity Check Methods

#### `identity_test(activations, feature_idx, tolerance=1e-5) -> bool`

Test that swapping a feature with itself doesn't change output.

**Parameters**:
- `activations` (`Tensor`): Test activations
- `feature_idx` (`int`): Feature to test
- `tolerance` (`float`): Maximum allowed difference (default: `1e-5`)

**Returns**:
- `passed` (`bool`): `True` if test passed

**Example**:
```python
passed = engine.identity_test(activations, feature_idx=0)
assert passed, "Identity test failed!"
```

---

#### `null_ablation_test(activations, feature_idx, tolerance=1e-5) -> bool`

Test that ablating an inactive feature doesn't change output.

**Parameters**:
- `activations` (`Tensor`): Test activations
- `feature_idx` (`int`): Feature to test (must be inactive)
- `tolerance` (`float`): Maximum allowed difference

**Returns**:
- `passed` (`bool`): `True` if test passed (or `False` if feature was active)

**Example**:
```python
# Find inactive feature
_, sparse = engine.encode(activations)
inactive = (sparse != 0).sum(dim=0) == 0
if inactive.any():
    feat_idx = int(inactive.nonzero()[0])
    passed = engine.null_ablation_test(activations, feat_idx)
```

---

#### `run_sanity_checks(activations, verbose=True) -> Dict[str, bool]`

Run all sanity checks and report results.

**Parameters**:
- `activations` (`Tensor`): Test activations
- `verbose` (`bool`): Print detailed results (default: `True`)

**Returns**:
- `results` (`Dict[str, bool]`): Test results:
  - `identity_swap`: Identity test result
  - `null_ablation`: Null ablation test result (or `None` if skipped)
  - `reconstruction_fidelity`: Reconstruction quality test

**Example**:
```python
results = engine.run_sanity_checks(test_activations, verbose=True)
# Prints detailed test results

assert all(v for v in results.values() if v is not None)
```

---

## Encode/Decode Helpers

#### `encode(activations) -> Tuple[Tensor, Tensor]`

Encode activations to sparse feature space.

**Returns**:
- `reconstruction`: Reconstructed activations
- `sparse`: Sparse feature activations `(batch_size, latent_dim)`

---

#### `decode(sparse) -> Tensor`

Decode sparse features back to activation space.

**Parameters**:
- `sparse` (`Tensor`): Sparse activations

**Returns**:
- `reconstruction` (`Tensor`): Reconstructed activations

---

## Usage Examples

### Example 1: Ablation Study - Measure Feature Importance

```python
import torch
from causal_interventions import FeatureInterventionEngine
from topk_sae import TopKAutoencoder

# Load SAE
sae = TopKAutoencoder(input_dim=2048, latent_dim=512, k=100)
sae.load_state_dict(torch.load('checkpoint.pt')['model_state_dict'])
engine = FeatureInterventionEngine(sae)

# Load validation data
val_data = torch.load('val_activations.pt')
activations = val_data['activations'][:100]  # First 100 samples

# Measure importance of top 10 features
top_features = [449, 168, 131, 261, 288, 273, 286, 4, 152, 212]

for feat_idx in top_features:
    impact = engine.measure_feature_impact(activations, feat_idx, metric='all')
    print(f"Feature {feat_idx}:")
    print(f"  Activation: {impact['feature_activation_freq']*100:.1f}%")
    print(f"  Impact: {impact['mean_abs_diff']:.4f}")
```

**Output**:
```
Feature 449:
  Activation: 99.0%
  Impact: 0.1234
Feature 168:
  Activation: 99.0%
  Impact: 0.1189
...
```

---

### Example 2: Operation Swap Experiment

```python
# Swap addition ↔ subtraction features
addition_feat = 332  # Rank 496, 0.3% activation
subtraction_feat = 194  # Rank 505, 0.1% activation

# Find samples where addition feature is active
_, sparse = engine.encode(activations)
addition_active_mask = sparse[:, addition_feat] != 0
addition_samples = activations[addition_active_mask]

print(f"Found {len(addition_samples)} samples with active addition feature")

# Swap features
swapped_acts = engine.swap_features(addition_samples, addition_feat, subtraction_feat)

# Run model inference
original_outputs = model(addition_samples)
swapped_outputs = model(swapped_acts)

# Compare
for i in range(len(addition_samples)):
    print(f"Original: {original_outputs[i]}")
    print(f"Swapped: {swapped_outputs[i]}")
```

---

### Example 3: Amplification Experiment - Bias Toward Multiplication

```python
# Amplify multiplication-specialized feature
mult_feat = 156  # Rank 138, 24.1% activation, multiplication specialist

# Try different amplification scales
scales = [1.0, 2.0, 5.0, 10.0]

for scale in scales:
    amplified = engine.amplify_feature(activations, mult_feat, scale=scale)
    outputs = model(amplified)

    # Count how many problems use multiplication
    mult_count = count_multiplication_steps(outputs)
    print(f"Scale {scale}x: {mult_count} problems used multiplication")
```

**Expected**: Higher amplification → more multiplication in reasoning

---

## Testing

Run the demo to verify installation:

```bash
python src/experiments/llama_sae_hierarchy/causal_interventions.py
```

This runs sanity checks and demos all three intervention types.

---

## Design Notes

### Why This API?

**Modularity**: Intervention logic separated from model inference
- Easy to test interventions without full model
- Can mock SAE for unit testing

**Safety**: Non-destructive operations
- Original activations never modified in-place
- All operations use `.clone()`
- SAE always in eval mode

**Reproducibility**: Deterministic operations
- No randomness in interventions
- All changes explicit and logged

### Performance

**Memory**: Each intervention creates 2 copies of activations
- Original: `(batch_size, input_dim)`
- Sparse: `(batch_size, latent_dim)`
- Modified: `(batch_size, input_dim)`

**Recommendation**: Process in batches of 10-100 samples

**Compute**: All operations are fast (<1ms per sample on CPU)
- Encoder: Linear + TopK selection
- Decoder: Linear projection
- Interventions: In-place modifications on sparse representation

---

## Related Files

- **Implementation**: `src/experiments/llama_sae_hierarchy/causal_interventions.py`
- **SAE Model**: `src/experiments/topk_grid_pilot/topk_sae.py`
- **Architecture Doc**: `docs/architecture/llama_sae_feature_hierarchy_architecture.md`
- **Experiment Results**: `docs/experiments/10-27_llama_gsm8k_*.md`

---

## Version History

- **2025-10-27**: Initial implementation with ablation, swap, amplification support
