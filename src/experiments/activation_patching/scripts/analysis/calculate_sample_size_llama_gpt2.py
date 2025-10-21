#!/usr/bin/env python3
"""
Calculate Required Sample Size for LLaMA vs GPT-2 Experiments

Different from previous analysis which was for:
- Target cases: clean✓ + corrupted✗
- Effect: 55.6% vs 50% (very small!)
- Required n=634

This analysis is for:
- Both-correct cases: clean✓ + corrupted✓
- Various effects we observe in current experiments
"""

import numpy as np
from scipy import stats


def cohens_h(p1: float, p2: float) -> float:
    """Calculate Cohen's h effect size for two proportions."""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))


def required_sample_size(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Calculate required sample size for given effect and power."""
    h = cohens_h(p1, p2)

    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    # Sample size formula
    n = ((z_alpha + z_beta) / h) ** 2

    return int(np.ceil(n))


print("=" * 80)
print("SAMPLE SIZE ANALYSIS: LLaMA vs GPT-2 Both-Correct Experiments")
print("=" * 80)
print()

# Current sample sizes
print("📊 CURRENT SITUATION:")
print("  Total problem pairs: 45")
print("  LLaMA both-correct: 23/45 (51%)")
print("  GPT-2 both-correct: 19/45 (42%)")
print()

print("=" * 80)
print("SCENARIO 1: Within-Architecture N-Token Effects")
print("=" * 80)
print()

# LLaMA: 1 token (4.3%) vs 5 tokens (26.1%)
print("LLaMA: 1-token (4.3%) vs 5-token (26.1%)")
n_llama_1v5 = required_sample_size(0.261, 0.043)
h_llama_1v5 = cohens_h(0.261, 0.043)
print(f"  Effect size (Cohen's h): {h_llama_1v5:.3f}")
print(f"  Required n: {n_llama_1v5} both-correct pairs")
print(f"  Current n: 23 both-correct pairs")
print(f"  Need: {n_llama_1v5/23:.1f}x current data")
print(f"  → Need ~{int(n_llama_1v5/0.51)} total problem pairs (assuming 51% both-correct rate)")
print()

# GPT-2: 1 token (21.1%) vs 4 tokens (26.3%)
print("GPT-2: 1-token (21.1%) vs 4-token (26.3%)")
n_gpt2_1v4 = required_sample_size(0.263, 0.211)
h_gpt2_1v4 = cohens_h(0.263, 0.211)
print(f"  Effect size (Cohen's h): {h_gpt2_1v4:.3f}")
print(f"  Required n: {n_gpt2_1v4} both-correct pairs")
print(f"  Current n: 19 both-correct pairs")
print(f"  Need: {n_gpt2_1v4/19:.1f}x current data")
print(f"  → Need ~{int(n_gpt2_1v4/0.42)} total problem pairs (assuming 42% both-correct rate)")
print()

print("=" * 80)
print("SCENARIO 2: Cross-Architecture Comparisons")
print("=" * 80)
print()

# LLaMA (4.3%) vs GPT-2 (21.1%) at 1 token
print("1-Token: LLaMA (4.3%) vs GPT-2 (21.1%)")
n_1token_cross = required_sample_size(0.211, 0.043)
h_1token_cross = cohens_h(0.211, 0.043)
print(f"  Effect size (Cohen's h): {h_1token_cross:.3f}")
print(f"  Required n: {n_1token_cross} both-correct pairs PER MODEL")
print(f"  Current n: 23 (LLaMA), 19 (GPT-2)")
print(f"  Need: {n_1token_cross/19:.1f}x current data")
print(f"  → Need ~{int(n_1token_cross/0.42)} total problem pairs per model")
print()

# LLaMA 5-token (26.1%) vs GPT-2 4-token (26.3%)
print("Optimal N: LLaMA 5-token (26.1%) vs GPT-2 4-token (26.3%)")
n_optimal_cross = required_sample_size(0.263, 0.261)
h_optimal_cross = cohens_h(0.263, 0.261)
print(f"  Effect size (Cohen's h): {h_optimal_cross:.3f}")
print(f"  Required n: {n_optimal_cross} both-correct pairs")
print(f"  ⚠️  Effect too small - would need HUGE sample")
print()

print("=" * 80)
print("SCENARIO 3: Majority Rule Differences")
print("=" * 80)
print()

# GPT-2 majority at 4/6 (67%), LLaMA majority at 5/6 (83%)
# Operationalize as: Does 4 tokens work for LLaMA?
# LLaMA @ 4 tokens: 21.7% clean, 73.9% corrupted
# Compare: 21.7% (LLaMA @ 4) vs 26.3% (GPT-2 @ 4)
print("4 tokens: LLaMA (21.7%) vs GPT-2 (26.3%)")
n_4token_cross = required_sample_size(0.263, 0.217)
h_4token_cross = cohens_h(0.263, 0.217)
print(f"  Effect size (Cohen's h): {h_4token_cross:.3f}")
print(f"  Required n: {n_4token_cross} both-correct pairs")
print()

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

print("For STRONG effects (cross-architecture, 1-token):")
print(f"  → Need ~{int(n_1token_cross/0.42)} total problem pairs")
print(f"  → Expect ~{n_1token_cross} both-correct pairs")
print()

print("For MEDIUM effects (LLaMA 1 vs 5 tokens):")
print(f"  → Need ~{int(n_llama_1v5/0.51)} total problem pairs")
print(f"  → Expect ~{n_llama_1v5} both-correct pairs")
print()

print("For SMALL effects (GPT-2 token increments):")
print(f"  → Need ~{int(n_gpt2_1v4/0.42)} total problem pairs")
print(f"  → Expect ~{n_gpt2_1v4} both-correct pairs")
print()

print("=" * 80)
print("PRAGMATIC TARGETS")
print("=" * 80)
print()

targets = [
    (100, "Minimal improvement", 100 * 0.46),
    (200, "Moderate improvement", 200 * 0.46),
    (300, "Good statistical power", 300 * 0.46),
    (500, "High statistical power", 500 * 0.46),
]

print("Assuming ~46% both-correct rate (average of LLaMA 51% + GPT-2 42%):")
print()
for total, desc, both_correct in targets:
    print(f"  {total:3d} total pairs → ~{int(both_correct):3d} both-correct pairs - {desc}")

    # Can we detect effects?
    strong_ok = "✓" if both_correct >= n_1token_cross else "✗"
    medium_ok = "✓" if both_correct >= n_llama_1v5 else "✗"
    small_ok = "✓" if both_correct >= n_gpt2_1v4 else "✗"

    print(f"       Strong effects (LLaMA vs GPT-2 @ 1 token): {strong_ok}")
    print(f"       Medium effects (LLaMA 1→5 tokens): {medium_ok}")
    print(f"       Small effects (GPT-2 increments): {small_ok}")
    print()

print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()
print("Based on analysis:")
print(f"  - 100 pairs: Detects strong cross-architecture effects only")
print(f"  - 200 pairs: Detects strong + medium effects")
print(f"  - 300 pairs: High power for most comparisons")
print(f"  - 500 pairs: Excellent power for all effects")
print()
print("SUGGESTED PHASED APPROACH:")
print("  Phase 1: 200 pairs (quick win, detects major effects)")
print("  Phase 2: 500 pairs (publication-grade statistical power)")
