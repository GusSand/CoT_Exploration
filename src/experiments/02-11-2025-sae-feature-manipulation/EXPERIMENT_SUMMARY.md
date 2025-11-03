# Feature 2203 Manipulation Experiment - Summary
## Overview
Successfully implemented and executed SAE feature manipulation experiment using activation-space steering approach (based on SAELens methodology).

## Experimental Setup

### Problem Variants
1. **Original**: "Janet eats **three** for breakfast... bakes muffins with **four**" (answer: 18)
2. **Variant A**: "Janet eats **two** for breakfast... bakes muffins with **four**" (answer: 20)
3. **Variant B**: "Janet eats **three** for breakfast... bakes muffins with **three**" (answer: 20)

### Interventions
- **Baseline**: No intervention (control)
- **Ablation**: Subtract Feature 2203's contribution from original problem
- **Addition**: Add Feature 2203 to variants with magnitudes [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

### Technical Approach
- **Method**: Activation-space steering (directly add/subtract decoder weight vectors)
- **Model**: CODI-LLAMA 3.2-1B with 6 continuous thought iterations
- **SAE**: 2048→8192 features, trained on CoT latent states
- **Layers tracked**: Early (4), Middle (8), Late (14)

## Key Findings

### 1. Feature 2203 Naturally Discriminates Problem Variants

**Baseline Activations:**
- **Original (three...four)**:
  - Early: 1.836 (position 1), 0.471 (position 2)
  - Middle: 2.234, 0.832
  - Late: 3.469, 1.305

- **Variant A (two...four)**: Zero activation across all layers
- **Variant B (three...three)**: Zero activation across all layers

**Interpretation**: Feature 2203 appears to specifically respond to the "three...four" number pair pattern in the original problem but not to the modified variants.

### 2. Ablation Successfully Reduces Feature Activity

**Original Problem with Ablation:**
- Position 1 activation: 1.836 (same - measured before intervention)
- **Middle layer**: Drops from 2.234 → 0.0 at position 1
- **Late layer**: Drops from 3.469 → 0.165 at position 1

**Effect**: The intervention successfully prevents Feature 2203 from propagating through deeper layers, demonstrating causal control over the feature's downstream influence.

### 3. Addition Successfully Induces Feature Activity

**Variant A with Added Feature 2203:**

| Magnitude | Early Layer | Middle Layer |
|-----------|-------------|--------------|
| 0.0       | 0.0         | 0.0          |
| 0.5       | 0.0         | 0.044-0.303  |
| 1.0       | 0.0-0.097   | 0.578-1.344  |
| 2.0       | 0.0-0.797   | 1.719-3.766  |
| 5.0       | 0.0-0.609   | 5.375-6.844  |
| 10.0      | 0.0-0.303   | 11.438-12.5  |

**Effect**: Adding the feature successfully induces activation that was absent in baseline. At magnitude 1.0, activation levels approach natural baseline values from the original problem.

## Technical Achievements

1. ✅ Implemented activation-space steering for SAE features
2. ✅ Successfully applied interventions during CODI latent iterations
3. ✅ Tracked feature activations across 3 layers and 6 CoT steps
4. ✅ Demonstrated causal control over feature propagation

## Data Files

- **Results**: `results/feature_2203_manipulation_results_20251103_083652.json`
- **Script**: `feature_2203_manipulation_experiment_v2.py`
- **Analysis**: `analyze_manipulation_results.py`
