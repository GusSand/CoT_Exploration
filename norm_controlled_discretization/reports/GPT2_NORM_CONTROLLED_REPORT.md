# CODI-GPT2 Norm-Controlled Discretization Analysis

**Date:** October 22, 2025  
**Model:** CODI-GPT2 (117M parameters)  
**Dataset:** GSM8K Test Set (1,319 examples)  
**Method:** L2 Norm-Controlled Discretization  
**Batch Size:** 32  
**Processing Time:** 0.7 minutes  
**Speed:** 1,856 examples/minute  

---

## Executive Summary

GPT-2 Norm-Controlled Results:
- **Vanilla**: 40.33% (baseline)
- **Alternating**: 33.21% (-7.12pp from vanilla)
- **Full**: 11.14% (-29.19pp from vanilla)

Compared to Previous (No Norm Control):
- Vanilla: 42.53% (similar, within variation)
- Alternating: 25.93% → **33.21% (+7.28pp improvement with norm control)**
- Full: 24.64% → **11.14% (-13.50pp worse with norm control)**

**Key Finding**: Norm control HELPS alternating discretization but HURTS full discretization!

---

## Analysis

### Alternating Discretization: Norm Control Helps
- Improvement: +7.28pp absolute
- Reduces degradation by 57%
- Suggests norm preservation helps when some positions remain continuous

### Full Discretization: Norm Control Hurts
- Degradation: -13.50pp absolute  
- Makes performance WORSE than no norm control
- Suggests magnitude carries semantic meaning that amplifies when preserved incorrectly

---

## Generated Files

- Results: `/workspace/CoT_Exploration/gpt2_norm_controlled_results/final_results.json`
- Plots: `/workspace/CoT_Exploration/norm_controlled_discretization/plots/gpt2/`
  - `accuracy_comparison.png`
  - `confidence_by_position.png`
  - `token_diversity.png`
  - `scale_factor_distribution.png`
