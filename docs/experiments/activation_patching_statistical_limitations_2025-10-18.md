# Statistical Limitations of Activation Patching Results

**Date**: October 18, 2025
**Experiment**: Activation Patching Causal Analysis
**Status**: **PILOT STUDY** - Not Statistically Significant

---

## TL;DR

**Our claim of statistical significance was WRONG.** Proper analysis shows:
- **p = 0.50** (not p < 0.05)
- **n = 9** (need ≥634 for adequate power)
- Results **indistinguishable from random chance**
- This is a **pilot study**, not publication-ready evidence

---

## What We Observed

| Layer | Recovery | n | Interpretation |
|-------|----------|---|----------------|
| Late (L11) | 5/9 (55.6%) | 9 | **p = 0.50** - NOT significant |
| Early/Middle | 4/9 (44.4%) | 9 | **p = 0.75** - NOT significant |

---

## Statistical Tests

### Test 1: Binomial Test vs Random Chance (50%)

**Null Hypothesis**: Patching is no better than random guessing (50% recovery)

| Layer | Observed | p-value | Significant? |
|-------|----------|---------|--------------|
| Late | 55.6% (5/9) | **0.500** | ❌ NO (need p<0.05) |
| Early | 44.4% (4/9) | **0.746** | ❌ NO |
| Middle | 44.4% (4/9) | **0.746** | ❌ NO |

**Verdict**: Cannot reject null hypothesis. Results are consistent with random guessing.

### Test 2: Confidence Intervals (95%)

| Layer | Point Estimate | 95% CI | Includes 50%? |
|-------|----------------|---------|---------------|
| Late | 55.6% | **[26.7%, 81.1%]** | ✓ YES ⚠️ |
| Early | 44.4% | **[18.9%, 73.3%]** | ✓ YES ⚠️ |

**Width**: 54.5 percentage points - **extremely wide**

**Verdict**: Confidence intervals include 50% (random chance), indicating we cannot rule out null hypothesis.

### Test 3: Statistical Power

**Question**: With n=9, what is our power to detect the observed effect?

**Answer**: **~1% power** - virtually no ability to detect this effect size

**Recommended power**: ≥80%

**Verdict**: Study is massively underpowered.

---

## Required Sample Size

To achieve 80% power (α=0.05, two-tailed):

| Comparison | Current n | Required n | Need X More |
|------------|-----------|------------|-------------|
| **55.6% vs 50%** (random) | 9 | **634** | **70x** |
| 55.6% vs 0% (no recovery) | 9 | 3 | 0.3x ✓ |
| 70% vs 30% (moderate effect) | 9 | 12 | 1.3x |

**Key Finding**: To distinguish our observed 55.6% from random guessing (50%), we need **634 target cases** (70x more data than we have).

---

## Why This Matters

### What We Falsely Claimed (RETRACTED)

❌ "p < 0.05 by binomial test" - **FALSE** (actual p = 0.50)
❌ "Statistically significant" - **FALSE**
❌ "Causal involvement confirmed" - **FALSE** (overstated)
❌ "Late layer shows strongest effect" - **FALSE** (CIs overlap)

### What We Can Honestly Say

✓ Results are **directionally positive** (5/9 vs 0/9 baseline)
✓ Pilot-level evidence **suggestive** of potential effect
✓ Experimental method is **validated** and working
✓ Hypothesis **worth testing** with larger sample

---

## Effect Sizes

**Cohen's h** (standardized difference for proportions):

| Comparison | h | Interpretation |
|------------|---|----------------|
| 55.6% vs 0% | 1.68 | Large effect |
| 55.6% vs 50% | 0.11 | **Trivial effect** |

**Interpretation**: The effect size compared to random chance (50%) is **trivial** (h=0.11), not large. This explains why we need 634 samples to detect it.

---

## Methodological Details

### What We're Actually Patching

- **Component**: Residual stream (not specific to MLP or attention)
- **Location**: Output of entire transformer block
- **Token**: Last position (first [THINK] token)
- **Layers tested**: L3 (early), L6 (middle), L11 (late)

### Why Residual Stream?

- Standard in mechanistic interpretability literature
- Tests "does information at this layer matter?"
- Used in landmark papers (Meng et al. 2022)
- Doesn't assume specific mechanism (attention vs MLP)

---

## Comparison to Literature

### Typical Activation Patching Studies

- **Meng et al. (2022)**: n = thousands of facts
- **Wang et al. (2022)**: n = hundreds of examples
- **Our study**: **n = 9** ❌

### Why So Small?

1. Started with 45 problem pairs total
2. Filtered to 23 valid pairs (clean baseline correct)
3. Only 9 were targets (clean ✓, corrupted ✗)
4. This is a fundamental constraint of our dataset generation process

### Implications

- Need better problem pair generation to increase n
- Current 9 is sufficient for **pilot study** only
- Not sufficient for **publication** or **robust claims**

---

## Lessons Learned

### Mistake 1: False Claim of Significance

**What we did wrong**: Claimed "p < 0.05" without actually computing p-value

**Why it happened**: Assumed positive recovery (5/9 vs 0/9) must be significant

**Correct approach**: Always compute p-value before making statistical claims

### Mistake 2: Ignoring Sample Size

**What we did wrong**: Treated n=9 as adequate

**Why it happened**: Focused on experimental design bug fix, not power analysis

**Correct approach**: Power analysis BEFORE data collection, not after

### Mistake 3: Overstating Conclusions

**What we did wrong**: "Causal involvement CONFIRMED"

**Why it happened**: Excitement about positive results after finding bug

**Correct approach**: Match strength of claims to strength of evidence

---

## Recommendations for Follow-Up

### Priority 1: Increase Sample Size (CRITICAL)

**Minimum**: n ≥ 30 target cases (to detect large effects like 70% vs 30%)
**Ideal**: n ≥ 634 target cases (to detect small effects like 55.6% vs 50%)
**Practical**: Start with n = 100, see if effect size is large

### Priority 2: Pre-Registration

- Register analysis plan BEFORE data collection
- Prevents p-hacking and HARKing (Hypothesizing After Results Known)
- Specify expected effect size and stopping criteria
- Use OSF, AsPredicted, or similar platform

### Priority 3: Control Conditions

- **Random patching**: Patch random activations (negative control)
- **Wrong-layer patching**: Patch from unrelated layer (negative control)
- **Explicit CoT**: Compare with explicit reasoning baseline (positive control)

### Priority 4: Multiple Comparisons Correction

- Testing 3 layers = 3 comparisons
- Apply Bonferroni correction: α = 0.05/3 = 0.017
- Or use FDR correction (Benjamini-Hochberg)

---

## What This Pilot Study Accomplished

Despite lack of statistical significance, this work has value:

✅ **Found and fixed experimental design bug** (major contribution)
✅ **Validated activation patching methodology** for CODI
✅ **Generated hypothesis** worth testing at scale
✅ **Produced reusable code** for future studies
✅ **Demonstrated rigorous error correction** in research process

---

## Final Verdict

**Scientific Status**: **Pilot Study / Proof of Concept**

**Publication Readiness**: **NOT ready** (need n ≥ 30-634 depending on claim)

**Contribution**: **Methodological** (found bug, validated approach)

**Next Step**: **Scale up** to n ≥ 100 with pre-registered analysis plan

---

## References

**Statistical Methods:**
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA.
- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.

**Activation Patching:**
- Meng et al. (2022). Locating and Editing Factual Associations in GPT. NeurIPS.
- Wang et al. (2022). Interpretability in the Wild. arXiv.

**Statistical Best Practices:**
- Simmons et al. (2011). False-Positive Psychology. Psychological Science.
- Button et al. (2013). Power failure: why small sample size undermines. Nature Reviews.

---

**Document Status**: Honest assessment added after initial over-optimistic claims
**Lesson**: Always do statistics BEFORE making claims, not after
**Silver Lining**: Finding and acknowledging mistakes IS good science
