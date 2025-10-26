# Position-wise Token Ablation Experiment

**Date:** 2025-10-24

## Overview

Investigate whether continuous thought positions decode to numerical information and if "number-encoding" positions are causally important for reasoning accuracy.

## Research Questions

1. Do certain continuous thought positions decode to numbers more often than others?
2. Are "number positions" more important for correct predictions?
3. Do GPT-2 and LLaMA show different position specialization patterns?

## Experiments

1. **Token Decoding**: Decode final layer activations to identify which positions output numbers
2. **Position Ablation**: Selectively ablate number vs non-number positions and measure accuracy impact
3. **Cross-Model Comparison**: Compare patterns between GPT-2 and LLaMA

## Datasets

- **GPT-2**: 1000 GSM8k problems (all CoT-dependent)
- **LLaMA**: ~450 CoT-dependent problems from 532 pairs

## Directory Structure

```
gpt2_token_ablation/
├── scripts/          # Experiment scripts
├── data/            # Filtered datasets and unembedding matrices
├── results/         # Experimental results (JSON)
├── analysis/        # Analysis notebooks and visualizations
├── utils/           # Utility functions (WandB, decoding, etc.)
└── README.md        # This file
```

## Time Tracking

| Task | Estimated | Actual | Variance |
|------|-----------|--------|----------|
| Setup | 20 min | TBD | TBD |
| Filter datasets | 20 min | TBD | TBD |
| Decode tokens | 45 min | TBD | TBD |
| Analysis | 1 hour | TBD | TBD |
| GPT-2 ablation | 1.5 hours | TBD | TBD |
| LLaMA ablation | 1 hour | TBD | TBD |
| Comparison | 1 hour | TBD | TBD |
| Documentation | 20 min | TBD | TBD |
| **TOTAL** | **6-7 hours** | **TBD** | **TBD** |
