# Position-wise Token Ablation Experiment

**Date:** 2025-10-24
**Duration:** ~2 hours
**Models:** GPT-2 CODI, LLaMA-3.2-1B CODI

---

## Research Question

Do continuous thought positions specialize in encoding numerical information, and are "number-encoding" positions causally important for reasoning accuracy?

---

## Hypothesis

User hypothesis: GPT-2's final position (position 5) decodes to numbers and plays a special role.

---

## Methods

### 1. Token Decoding
- Extracted final layer activations (L11 for GPT-2, L15 for LLaMA)
- Applied unembedding matrix: `decoded_token = argmax(W_U @ activation)`
- Classified positions as "number" vs "non-number" based on digit presence

### 2. Position Ablation
- **Condition A**: Zero-ablate all number-decoding positions
- **Condition B**: Zero-ablate all non-number-decoding positions
- **Baseline**: No ablation
- Measure accuracy impact

### 3. Cross-Model Comparison
- Compare position specialization patterns
- Compare causal importance via ablation

---

## Datasets

- **GPT-2**: 1000 GSM8k problems (all CoT-dependent, 43.2% baseline accuracy)
- **LLaMA**: 424 CoT-dependent problems (85.4% baseline accuracy)

---

## Key Results

### Finding 1: Different Position Specialization Patterns

**GPT-2:**
| Position | % Decodes to Number |
|----------|---------------------|
| 0        | 23.8%               |
| 1        | 0.0%  ❌            |
| 2        | 29.2%               |
| 3        | 0.0%  ❌            |
| 4        | 14.6%               |
| 5        | 0.0%  ❌  (contradicts hypothesis!) |

**LLaMA:**
| Position | % Decodes to Number |
|----------|---------------------|
| 0        | 55.2%               |
| 1        | 85.8%  ✅           |
| 2        | 62.3%               |
| 3        | 4.7%                |
| 4        | 83.3%  ✅           |
| 5        | 54.0%               |

### Finding 2: Statistical Significance
- **GPT-2**: χ² = 870.85, p < 1e-150 (highly significant position effect)
- **LLaMA**: χ² = 745.11, p < 1e-150 (highly significant position effect)

### Finding 3: Ablation Impact
*(Results will be added when experiments complete)*

---

## Conclusions

1. **User hypothesis REJECTED**: GPT-2 position 5 does NOT decode to numbers (0%) and does not have a special numerical role
2. **Architectural differences**: GPT-2 shows alternating pattern (odd positions never decode to numbers); LLaMA shows strong specialization (positions 1 & 4 are 85%+ numerical)
3. **Position specialization is real**: Both models show highly significant position effects (p < 1e-150)

---

## Implications

- Different CODI training procedures or model architectures lead to different position specialization strategies
- Position-level interpretability may be model-specific
- Numerical decoding at final layer does not guarantee causal importance (pending ablation results)

---

## Limitations

1. Different datasets (GPT-2: 1000 samples, LLaMA: 424 CoT-dependent)
2. Final layer decoding only (not tracking emergence across layers)
3. Simple number classification (digit-based regex)

---

## Future Work

1. **Logit lens**: Track number emergence across all layers
2. **Intermediate layer ablation**: Test at early/middle layers
3. **Gradient attribution**: Complement ablation with gradient-based methods
4. **Token-specific analysis**: What specific numbers are being decoded?

---

**Files Generated:**
- `results/gpt2_final_layer_decoding.json`
- `results/llama_final_layer_decoding.json`
- `results/position_analysis_summary.md`
- `results/gpt2_position_ablation.json` *(in progress)*
- `results/llama_position_ablation.json` *(in progress)*
- `results/cross_model_comparison_summary.md` *(pending)*
