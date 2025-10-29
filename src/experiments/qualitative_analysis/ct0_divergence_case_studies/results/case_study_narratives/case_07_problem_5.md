# Case Study: Problem 5

**Selection Reason**: High divergence but still correct (robustness)

**Impact Type**: no_change (Baseline: ✓, CT0-blocked: ✓)

---

## Problem

**Question** (truncated):
```
N/A...
```

**Gold Answer**: N/A
**Baseline Prediction**: 64
**CT0-Blocked Prediction**: 64

---

## Divergence Profile

**Overall Metrics**:
- Total divergence: 0.509
- CT1 similarity: 0.653 (34.7% diverged)
- CT4 similarity: 0.401 (59.9% diverged)
- Divergence slope: -0.027 per step
- Pattern: stable

---

## Step-by-Step Divergence Analysis

### CT0 - Step 0

**Similarity**: 1.000 (0.0% diverged)
**L2 Distance**: 0.00
**Interpretation**: **Nearly identical** - no significant divergence

**Layer Analysis**:
- Most diverged layer: Layer 0 (similarity: 1.000)
- Least diverged layer: Layer 0 (similarity: 1.000)
- Layer variance: 0.000

**Note**: CT0 is identical in both conditions (as expected - same generation process)

### CT1 - Step 1

**Similarity**: 0.653 (34.7% diverged)
**L2 Distance**: 18.63
**Interpretation**: **Significantly diverged** - major differences

**Layer Analysis**:
- Most diverged layer: Layer 14 (similarity: 0.211)
- Least diverged layer: Layer 0 (similarity: 0.999)
- Layer variance: 0.260

**⚠️ IMMEDIATE DIVERGENCE**: CT1 shows significant divergence from the first step!

### CT2 - Step 2

**Similarity**: 0.487 (51.3% diverged)
**L2 Distance**: 25.02
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 14 (similarity: 0.310)
- Least diverged layer: Layer 0 (similarity: 0.741)
- Layer variance: 0.134

**📉 CASCADING**: Divergence is accumulating from previous steps.

### CT3 - Step 3

**Similarity**: 0.398 (60.2% diverged)
**L2 Distance**: 25.19
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 13 (similarity: 0.227)
- Least diverged layer: Layer 0 (similarity: 0.714)
- Layer variance: 0.154

**📉 CASCADING**: Divergence is accumulating from previous steps.

### CT4 - Step 4

**Similarity**: 0.401 (59.9% diverged)
**L2 Distance**: 21.58
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 12 (similarity: 0.087)
- Least diverged layer: Layer 0 (similarity: 0.909)
- Layer variance: 0.298

### CT5 - Step 5

**Similarity**: 0.518 (48.2% diverged)
**L2 Distance**: 22.67
**Interpretation**: **Significantly diverged** - major differences

**Layer Analysis**:
- Most diverged layer: Layer 13 (similarity: 0.273)
- Least diverged layer: Layer 0 (similarity: 0.845)
- Layer variance: 0.181

---

## Interpretation

**Stable Pattern**: Despite CT0 blocking, the hidden states remain relatively stable. This may indicate
the model has alternative pathways to access needed information, or this particular problem is less
dependent on CT0's encoded information.

**Robustness**: Despite significant hidden state divergence, the model **still produced the correct answer**
in both conditions (answer: 64). This suggests redundancy in the reasoning process
or that the specific diverged representations didn't affect the critical computation for this problem.

---

## Key Takeaways

- **Resilient reasoning**: High divergence but correct answer demonstrates model robustness
- **Layer heterogeneity**: Different layers show varied divergence (std: 0.298), suggesting specialized roles

## Visualization

![Divergence Trajectory](case_07_problem_5_divergence.png)
