# Case Study: Problem 7

**Selection Reason**: Diverse profile (score=0.151)

**Impact Type**: no_change (Baseline: ✗, CT0-blocked: ✗)

---

## Problem

**Question** (truncated):
```
N/A...
```

**Gold Answer**: N/A
**Baseline Prediction**: 120
**CT0-Blocked Prediction**: 180

---

## Divergence Profile

**Overall Metrics**:
- Total divergence: 0.474
- CT1 similarity: 0.717 (28.3% diverged)
- CT4 similarity: 0.422 (57.8% diverged)
- Divergence slope: -0.045 per step
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

**Similarity**: 0.717 (28.3% diverged)
**L2 Distance**: 17.03
**Interpretation**: **Moderately diverged** - noticeable differences

**Layer Analysis**:
- Most diverged layer: Layer 15 (similarity: 0.355)
- Least diverged layer: Layer 0 (similarity: 0.999)
- Layer variance: 0.215

**Stable start**: CT1 remains relatively similar despite CT0 blocking.

### CT2 - Step 2

**Similarity**: 0.545 (45.5% diverged)
**L2 Distance**: 23.42
**Interpretation**: **Significantly diverged** - major differences

**Layer Analysis**:
- Most diverged layer: Layer 15 (similarity: 0.373)
- Least diverged layer: Layer 0 (similarity: 0.736)
- Layer variance: 0.107

**📉 CASCADING**: Divergence is accumulating from previous steps.

### CT3 - Step 3

**Similarity**: 0.455 (54.5% diverged)
**L2 Distance**: 21.77
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 15 (similarity: 0.259)
- Least diverged layer: Layer 0 (similarity: 0.755)
- Layer variance: 0.155

**📉 CASCADING**: Divergence is accumulating from previous steps.

### CT4 - Step 4

**Similarity**: 0.422 (57.8% diverged)
**L2 Distance**: 18.76
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 12 (similarity: 0.137)
- Least diverged layer: Layer 0 (similarity: 0.919)
- Layer variance: 0.285

### CT5 - Step 5

**Similarity**: 0.492 (50.8% diverged)
**L2 Distance**: 21.44
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 15 (similarity: 0.319)
- Least diverged layer: Layer 0 (similarity: 0.697)
- Layer variance: 0.104

---

## Interpretation

**Stable Pattern**: Despite CT0 blocking, the hidden states remain relatively stable. This may indicate
the model has alternative pathways to access needed information, or this particular problem is less
dependent on CT0's encoded information.

**Robustness**: Despite significant hidden state divergence, the model **still produced the correct answer**
in both conditions (answer: 120). This suggests redundancy in the reasoning process
or that the specific diverged representations didn't affect the critical computation for this problem.

---

## Key Takeaways

- Strong **cascading effect** (slope: -0.045), showing how early divergence amplifies
- **Resilient reasoning**: High divergence but correct answer demonstrates model robustness
- **Layer heterogeneity**: Different layers show varied divergence (std: 0.285), suggesting specialized roles

## Visualization

![Divergence Trajectory](case_10_problem_7_divergence.png)
