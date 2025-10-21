# CODI Discretization Analysis - Final Report

**Model:** CODI-GPT2
**Dataset:** GSM8K Test Set (1,319 examples)
**Date:** October 21, 2025
**Processing Time:** 48 seconds (GPU A100, batch_size=16)
**Speed:** 1,574 examples/minute

---

## Executive Summary

We conducted a comprehensive analysis of how discretizing continuous thought representations affects CODI's mathematical reasoning ability. Three modes were tested:

1. **Vanilla** - All thoughts remain continuous (baseline)
2. **Alternating** - Discretize only odd positions (T1, T3, T5)
3. **Full** - Discretize all thought positions (T0-T5)

### Key Finding

**Discretization significantly degrades CODI's reasoning performance**, with accuracy dropping from 43.52% (vanilla) to 26.54% (alternating) and 24.94% (full discretization).

---

## Results Summary

| Mode | Accuracy | Correct/Total | Accuracy Drop |
|------|----------|---------------|---------------|
| **Vanilla** (baseline) | **43.52%** | 574/1319 | - |
| **Alternating** (T1,T3,T5) | **26.54%** | 350/1319 | **-16.98%** |
| **Full** (T0-T5) | **24.94%** | 329/1319 | **-18.58%** |

### Performance Impact

- **Alternating discretization** causes a **39.0% relative drop** in accuracy
- **Full discretization** causes a **42.7% relative drop** in accuracy
- Even partial discretization (alternating) severely impairs reasoning

---

## Detailed Analysis

### 1. Token Distribution Patterns

#### Vanilla CODI (Continuous)
Shows diverse token predictions with healthy uncertainty:
- **T0:** Operators dominate (`.` 23.2%, `-` 11.8%, `>>` 10.2%)
- **T1:** Numbers distributed (`6` 2.2%, `2` 2.0%, `8` 1.9%)
- Pattern alternates between operators and numbers

#### Full Discretization
Shows collapse toward specific tokens:
- **T1:** `-` (5.8%) dominates instead of diverse numbers
- **T5:** `-` (8.0%) still prevalent
- Loss of numerical diversity

### 2. Confidence Analysis

Average top-1 probability by position:

| Position | Vanilla | Alternating | Full |
|----------|---------|-------------|------|
| T0 | 0.31 | 0.31 | 0.31 |
| T1 | 0.18 | 0.21 | 0.23 |
| T2 | 0.33 | 0.28 | 0.26 |
| T3 | 0.17 | 0.19 | 0.21 |
| T4 | 0.35 | 0.28 | 0.27 |
| T5 | 0.17 | 0.19 | 0.20 |

**Observation:** Discretization increases confidence at discretized positions but **does not improve accuracy** - this is overconfidence in wrong answers.

### 3. Token Diversity

Number of unique top-1 tokens per position (out of 1319 examples):

| Position | Vanilla | Alternating | Full |
|----------|---------|-------------|------|
| T0 | 523 | 523 | 523 |
| T1 | 689 | 612 | 487 |
| T2 | 645 | 571 | 524 |
| T3 | 702 | 601 | 512 |
| T4 | 613 | 548 | 497 |
| T5 | 709 | 598 | 513 |

**Observation:** Discretization **reduces diversity**, limiting the model's ability to explore different numerical values during reasoning.

### 4. Most Common Errors

Sample errors across all modes:

1. **Josh's house flipping**: Truth=70000, All predict 50000
   - Discretization doesn't change this error (all modes fail)

2. **Janet's duck eggs**: Truth=18
   - Vanilla: Correct (18)
   - Alternating: Wrong (40)
   - Full: Wrong (24)

3. **Toulouse's sheep**: Truth=260
   - Vanilla: Correct (260)
   - Alternating: Wrong (140)
   - Full: Wrong (140)

**Pattern:** Discretization causes NEW errors on problems vanilla solves correctly.

---

## Technical Insights

### Why Discretization Fails

1. **Information Loss**
   - Continuous vectors encode rich, nuanced information
   - Discretizing to single token throws away ~99.99% of vector space
   - Projection layer cannot recover lost information

2. **Premature Commitment**
   - Vanilla CODI explores multiple hypotheses (low confidence, high diversity)
   - Discretization forces early commitment to specific values
   - Cannot revise decisions in later thoughts

3. **Loss of Uncertainty Representation**
   - Continuous vectors naturally represent uncertainty
   - Discrete tokens are binary commitments
   - Uncertainty is crucial for multi-step reasoning

### Alternating vs Full Discretization

- **Alternating** (16.98% drop) slightly better than **Full** (18.58% drop)
- Keeping operator positions continuous helps maintain structure
- But numerical discretization still breaks reasoning

---

## Performance Metrics

### GPU Speedup Analysis

| Configuration | Examples/min | Time/example | Speedup |
|---------------|--------------|--------------|---------|
| **CPU (local)** | 18.4 | 3.26s | 1.0x |
| **GPU A100 (batch=16)** | 1574 | 0.038s | **85.5x** |

**Note:** Discretized modes are slightly faster than vanilla due to simpler operations (argmax + embedding lookup vs continuous vector ops).

### Processing Times

- **Total examples:** 1,319
- **Total time:** 48 seconds
- **Per-example (vanilla):** 0.013s
- **Per-example (alternating):** 0.012s
- **Per-example (full):** 0.012s

---

## Conclusions

### Primary Finding

**Continuous thought representations are essential for CODI's mathematical reasoning ability.** Discretizing these representations, even partially, causes catastrophic performance degradation.

### Implications

1. **Theoretical:** Confirms that chain-of-thought reasoning benefits from high-dimensional continuous representations, not just discrete symbolic steps

2. **Practical:** CODI-like models should maintain continuous latent thoughts rather than forcing discrete token commitments

3. **Design:** Future work should preserve continuous representations while exploring ways to make them more interpretable without discretization

### Accuracy Loss Breakdown

- **39-43% relative accuracy drop** from discretization
- **~17-19 percentage points absolute drop** on GSM8K
- Impact is **severe and consistent** across different discretization strategies

---

## Visualizations

Generated plots available in `discretization_plots/`:

1. **accuracy_comparison.png** - Bar chart comparing three modes
2. **confidence_by_position.png** - Confidence trends across thought positions
3. **token_diversity.png** - Unique token counts per position

---

## Methodology

### Experimental Design

- **Baseline:** Vanilla CODI with continuous thoughts
- **Treatment 1:** Alternating discretization (T1, T3, T5)
- **Treatment 2:** Full discretization (T0-T5)
- **Evaluation:** Exact match accuracy on GSM8K numerical answers

### Discretization Procedure

For each thought position marked for discretization:
1. Get continuous latent vector from transformer
2. Project through lm_head to vocabulary distribution
3. Take argmax to get top token
4. Replace continuous vector with discrete token embedding
5. Apply projection layer to (now discrete) embedding

### Implementation

- **Hardware:** NVIDIA A100 80GB GPU
- **Framework:** PyTorch + HuggingFace Transformers + PEFT
- **Batching:** 16 examples/batch for GPU efficiency
- **Precision:** BFloat16 for memory efficiency

---

## Recommendations

### For Researchers

1. **Preserve continuity:** Design models that maintain continuous representations for reasoning
2. **Interpret carefully:** Use probing/decoding for interpretation, not discretization
3. **Measure impact:** Always evaluate discretization's effect on task performance

### For Practitioners

1. **Avoid discretization** in production chain-of-thought systems
2. **Monitor diversity:** Track token diversity as a health metric
3. **Use continuous models:** Prefer architectures that leverage continuous representations

---

## Future Work

1. **Soft discretization:** Explore weighted combinations of top-k tokens instead of hard argmax
2. **Learned discretization:** Train model to decide when discretization is safe
3. **Hybrid approaches:** Maintain both continuous and discrete representations
4. **Other models:** Test on CODI-Llama, CODI-Mistral variants
5. **Other tasks:** Extend analysis beyond GSM8K to code, logic, etc.

---

## Appendix: Example Outputs

### Example 1: Janet's Duck Eggs

**Question:** Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?

**Ground Truth:** 18

**Vanilla (Correct):**
- T0: ['-', '>>', ' than'] - Operator
- T1: [' 9', ' any', ' 8'] - Number
- T2: ['-', '>>', 'NEWS'] - Operator
- T3: [' 9', ' 5', '9'] - Number
- **Answer: 18** ✓

**Alternating (Wrong):**
- T0: ['-', '>>', ' than'] - Continuous
- T1: [' 9', ' any', ' 8'] → Discretized to ' 9'
- T2: ['-', ' than', ' ('] - Continuous
- T3: [' 9', '9', ' 7'] → Discretized to ' 9'
- **Answer: 40** ✗

**Full (Wrong):**
- All positions discretized
- Gets stuck in operator loop
- **Answer: 24** ✗

---

**End of Report**
