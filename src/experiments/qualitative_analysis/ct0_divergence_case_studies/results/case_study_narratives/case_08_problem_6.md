# Case Study: Problem 6

**Selection Reason**: High divergence but still correct (robustness)

**Impact Type**: no_change (Baseline: ✓, CT0-blocked: ✓)

---

## Problem

**Question** (truncated):
```
N/A...
```

**Gold Answer**: N/A
**Baseline Prediction**: 260
**CT0-Blocked Prediction**: 260

---

## Divergence Profile

**Overall Metrics**:
- Total divergence: 0.500
- CT1 similarity: 0.652 (34.8% diverged)
- CT4 similarity: 0.360 (64.0% diverged)
- Divergence slope: -0.025 per step
- Pattern: late_divergence

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

**Similarity**: 0.652 (34.8% diverged)
**L2 Distance**: 18.47
**Interpretation**: **Significantly diverged** - major differences

**Layer Analysis**:
- Most diverged layer: Layer 14 (similarity: 0.208)
- Least diverged layer: Layer 0 (similarity: 0.999)
- Layer variance: 0.270

**⚠️ IMMEDIATE DIVERGENCE**: CT1 shows significant divergence from the first step!

### CT2 - Step 2

**Similarity**: 0.503 (49.7% diverged)
**L2 Distance**: 25.85
**Interpretation**: **Significantly diverged** - major differences

**Layer Analysis**:
- Most diverged layer: Layer 15 (similarity: 0.274)
- Least diverged layer: Layer 0 (similarity: 0.776)
- Layer variance: 0.150

**📉 CASCADING**: Divergence is accumulating from previous steps.

### CT3 - Step 3

**Similarity**: 0.456 (54.4% diverged)
**L2 Distance**: 23.63
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 15 (similarity: 0.218)
- Least diverged layer: Layer 0 (similarity: 0.833)
- Layer variance: 0.198

### CT4 - Step 4

**Similarity**: 0.360 (64.0% diverged)
**L2 Distance**: 22.40
**Interpretation**: **Heavily diverged** - reasoning has fundamentally changed

**Layer Analysis**:
- Most diverged layer: Layer 8 (similarity: 0.077)
- Least diverged layer: Layer 0 (similarity: 0.855)
- Layer variance: 0.278

**📉 CASCADING**: Divergence is accumulating from previous steps.

### CT5 - Step 5

**Similarity**: 0.528 (47.2% diverged)
**L2 Distance**: 21.99
**Interpretation**: **Significantly diverged** - major differences

**Layer Analysis**:
- Most diverged layer: Layer 13 (similarity: 0.292)
- Least diverged layer: Layer 0 (similarity: 0.835)
- Layer variance: 0.169

---

## Interpretation

**Late Divergence Pattern**: CT1 remains relatively stable, but later steps (CT3-CT4) show significant
divergence. This suggests the model can partially compensate initially, but the lack of CT0 information
causes problems as reasoning progresses.

**Robustness**: Despite significant hidden state divergence, the model **still produced the correct answer**
in both conditions (answer: 260). This suggests redundancy in the reasoning process
or that the specific diverged representations didn't affect the critical computation for this problem.

---

## Key Takeaways

- **Resilient reasoning**: High divergence but correct answer demonstrates model robustness
- **Layer heterogeneity**: Different layers show varied divergence (std: 0.278), suggesting specialized roles

## Visualization

![Divergence Trajectory](case_08_problem_6_divergence.png)
