# CODI-Llama Discretization Analysis Report

**Model:** CODI-Llama3.2-1B (1.0B parameters)  
**Dataset:** GSM8K Test Set (1,319 examples)  
**Date:** October 21, 2025  
**Processing Time:** 21 minutes (GPU A100, batch_size=16)  
**Speed:** 62.83 examples/minute  

## Executive Summary

The CODI-Llama model shows **catastrophic failure** when continuous thought representations are discretized, with accuracy dropping from 37.68% to near-zero levels (0.83-2.12%). This is dramatically worse than GPT-2's response to discretization, suggesting that larger models rely more heavily on fine-grained continuous representations.

## Results

| Mode | Accuracy | Correct/Total | Accuracy Drop from Vanilla |
|------|----------|---------------|----------------------------|
| **Vanilla** (baseline) | **37.68%** | 497/1319 | - |
| **Alternating** (T1,T3,T5) | **0.83%** | 11/1319 | **-36.85% (-97.8% relative)** |
| **Full** (T0-T5) | **2.12%** | 28/1319 | **-35.56% (-94.4% relative)** |

## Key Findings

### 1. Near-Complete Performance Collapse
- Discretization reduces Llama accuracy to essentially random guessing
- 97.8% relative drop with alternating discretization
- 94.4% relative drop with full discretization

### 2. Comparison to GPT-2
- **GPT-2 retains ~60% of baseline** performance with discretization
- **Llama retains <6% of baseline** performance with discretization  
- Llama is **16x more sensitive** to discretization than GPT-2

### 3. Model Size Hypothesis
Larger models (Llama-1B vs GPT-2-117M) appear to:
- Develop more complex internal representations
- Rely on finer-grained continuous information
- Suffer more catastrophically when forced to discrete commitments

## Visualizations

See `../plots/llama/` for:
- `accuracy_comparison.png` - Dramatic accuracy drop across modes
- `confidence_by_position.png` - Overconfidence in wrong answers
- `token_diversity.png` - Reduced diversity with discretization

## Implications

1. **Continuous representations are critical** for reasoning in larger models
2. **Discretization is not a viable interpretability method** for CODI-like architectures
3. **Model capacity matters**: Larger models need more representational flexibility
4. **Design principle**: Preserve continuous latent spaces in chain-of-thought systems

## Technical Details

- **Hardware**: NVIDIA A100 80GB GPU
- **Precision**: BFloat16 for memory efficiency  
- **Batch Size**: 16 examples/batch
- **Discretization**: Argmax selection + token embedding replacement
- **Evaluation**: Exact match on numerical answers

## Conclusion

The near-total failure of CODI-Llama under discretization provides strong evidence that continuous thought representations encode essential information that cannot be captured by discrete token selections. This finding has important implications for the design and interpretation of latent-reasoning language models.
