#!/usr/bin/env python3
"""
Visualize GPT-2 Norm-Controlled Discretization Results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load results
with open("/workspace/CoT_Exploration/gpt2_norm_controlled_bs16/final_results.json", "r") as f:
    data = json.load(f)

stats = data["stats"]
results = data["results"]

# Create output directory
import os
os.makedirs("/workspace/CoT_Exploration/gpt2_norm_results_bs16/plots", exist_ok=True)

# 1. Accuracy Comparison
modes = ["vanilla", "alternating", "full"]
accuracies = [stats[mode]["accuracy"] for mode in modes]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(modes, accuracies, color=["#2ecc71", "#f39c12", "#e74c3c"])
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("CODI-GPT2 Norm-Controlled Discretization: Accuracy Comparison", fontsize=14, fontweight="bold")
ax.set_ylim(0, 50)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("/workspace/CoT_Exploration/norm_controlled_discretization/plots/gpt2/accuracy_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: accuracy_comparison.png")
plt.close()

# 2. Confidence by Position
thought_positions = 7  # T-2 through T5
position_labels = ["T-2", "T0", "T1", "T2", "T3", "T4", "T5"]

confidences = {mode: [[] for _ in range(thought_positions)] for mode in modes}

for mode in modes:
    for example in results[mode]:
        thought_probs = example["thought_probs"]
        for pos_idx, probs in enumerate(thought_probs):
            if pos_idx < thought_positions:
                confidences[mode][pos_idx].append(probs[0])  # top-1 probability

avg_confidences = {mode: [np.mean(confidences[mode][i]) if confidences[mode][i] else 0 
                          for i in range(thought_positions)] 
                   for mode in modes}

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(position_labels))
width = 0.25

for i, mode in enumerate(modes):
    offset = (i - 1) * width
    ax.bar(x + offset, avg_confidences[mode], width, 
           label=mode.capitalize(), alpha=0.8)

ax.set_xlabel("Thought Position", fontsize=12)
ax.set_ylabel("Average Top-1 Confidence", fontsize=12)
ax.set_title("CODI-GPT2 Norm-Controlled: Confidence by Position", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(position_labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/CoT_Exploration/norm_controlled_discretization/plots/gpt2/confidence_by_position.png", dpi=300, bbox_inches="tight")
print("Saved: confidence_by_position.png")
plt.close()

# 3. Token Diversity
diversities = {mode: [[] for _ in range(thought_positions)] for mode in modes}

for mode in modes:
    for pos_idx in range(thought_positions):
        tokens_at_position = []
        for example in results[mode]:
            thought_tokens = example["thought_tokens"]
            if pos_idx < len(thought_tokens):
                tokens_at_position.append(thought_tokens[pos_idx][0])  # top-1 token
        diversities[mode][pos_idx] = len(set(tokens_at_position))

fig, ax = plt.subplots(figsize=(12, 6))
for i, mode in enumerate(modes):
    offset = (i - 1) * width
    ax.bar(x + offset, diversities[mode], width, 
           label=mode.capitalize(), alpha=0.8)

ax.set_xlabel("Thought Position", fontsize=12)
ax.set_ylabel("Number of Unique Top-1 Tokens", fontsize=12)
ax.set_title("CODI-GPT2 Norm-Controlled: Token Diversity by Position", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(position_labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/workspace/CoT_Exploration/norm_controlled_discretization/plots/gpt2/token_diversity.png", dpi=300, bbox_inches="tight")
print("Saved: token_diversity.png")
plt.close()

# 4. Norm Statistics (NEW for norm-controlled analysis)
norm_info_by_position = {pos: [] for pos in range(6)}  # T0-T5

for mode in ["alternating", "full"]:  # Only discretized modes have norm info
    for example in results[mode]:
        if "norm_info" in example:
            for norm_data in example["norm_info"]:
                pos = int(norm_data["position"][1])  # Extract number from "T0", "T1", etc.
                norm_info_by_position[pos].append({
                    "mode": mode,
                    "continuous_norm": norm_data["continuous_norm"],
                    "token_norm_before": norm_data["token_norm_before_scaling"],
                    "scale_factor": norm_data["scale_factor"]
                })

# Plot scale factors distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("CODI-GPT2 Norm-Controlled: Scale Factor Distributions", fontsize=16, fontweight="bold")

for pos in range(6):
    ax = axes[pos // 3, pos % 3]
    if norm_info_by_position[pos]:
        scale_factors = [item["scale_factor"] for item in norm_info_by_position[pos]]
        ax.hist(scale_factors, bins=50, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_xlabel("Scale Factor", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_title(f"T{pos} (mean: {np.mean(scale_factors):.3f})", fontsize=11)
        ax.axvline(np.mean(scale_factors), color="red", linestyle="--", linewidth=2, label="Mean")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/workspace/CoT_Exploration/norm_controlled_discretization/plots/gpt2/scale_factor_distribution.png", dpi=300, bbox_inches="tight")
print("Saved: scale_factor_distribution.png")
plt.close()

print("\nAll GPT-2 visualizations completed!")
print(f"Saved to: /workspace/CoT_Exploration/norm_controlled_discretization/plots/gpt2/")
