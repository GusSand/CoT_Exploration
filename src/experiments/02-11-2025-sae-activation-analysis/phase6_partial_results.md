# Phase 6: Multiplication Feature Ablation - Partial Results

## Experiment Setup

**Objective**: Test if ablating multiplication-selective SAE features affects final answer logits.

**Test Example**: Janet's duck eggs problem (correct answer: 18)

**Features Ablated**:
- Early Layer (L4): 37 features with >70% precision for '*'
- Middle Layer (L8): 40 features  
- Late Layer (L14): 45 features

## Baseline Results (No Ablation)

**Generated Output**: 


**Extracted Answer**: 144.0 (incorrect - model got stuck in repetition loop)

**Final Position Logits**:
- Token '18': -0.7852
- Token ' 18': 13.3750  ← HIGHEST
- Token '9': -1.7969
- Token ' 9': 13.3750   ← HIGHEST

**Analysis**: 
- The model initially generated 18 (correct), but then entered a repetition/doubling pattern
- At the final position, logits for ' 18' and ' 9' are tied at 13.375
- The logits for '18' and '9' without space prefix are much lower

## Status

**Completed**: Baseline generation and log it extraction  
**Incomplete**: Ablation run (encountered tensor shape mismatch in hook)

## Key Findings

Even without the ablation comparison, the baseline results show:
1. Model can generate correct answer initially but lacks proper stopping
2. High logits for both possible answers (' 18' and ' 9') at final position
3. The space-prefixed tokens dominate the distribution

