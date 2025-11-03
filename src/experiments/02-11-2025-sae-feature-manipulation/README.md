# SAE Feature 2203 Manipulation Experiment

## Overview

This experiment manipulates SAE Feature 2203 instead of just measuring it.

## Experiment Design

### Problems
- **Original**: Janet's ducks, 16 eggs, eats 3, bakes with 4 → Answer: 8
- **Variant A**: Janet's ducks, 16 eggs, eats 2, bakes with 4 → Answer: 0  
- **Variant B**: Janet's ducks, 16 eggs, eats 3, bakes with 3 → Answer: 0

### Interventions
1. **Baseline** - No intervention (all 3 problems)
2. **Ablation** - Zero out Feature 2203 (original problem only)
3. **Addition** - Add Feature 2203 with different magnitudes (variants only)
   - Magnitudes tested: 0.0, 0.5, 1.0, 2.0, 5.0, 10.0

### Tracking
- Layers: early (4), middle (8), late (14)
- At each position:
  - Original Feature 2203 activation
  - Decoded token
  - Intervention applied

## Running the Experiment

```bash
cd /workspace/CoT_Exploration/src/experiments/02-11-2025-sae-feature-manipulation
python3 feature_2203_manipulation_experiment.py
```

## Expected Runtime
- Model loading: ~30s
- SAE loading: ~5s  
- Per experiment run: ~10-20s
- Total: ~5-10 minutes (3 baseline + 1 ablation + 12 additions = 16 runs)

## Output
- JSON results saved to: 
- Console summary with accuracy comparison

## Key Research Questions
1. Does ablating Feature 2203 change the answer for the original problem?
2. Does adding Feature 2203 make the variants behave like the original?
3. What magnitude of intervention is needed to see an effect?
4. How do decoded tokens change at each CoT step?
