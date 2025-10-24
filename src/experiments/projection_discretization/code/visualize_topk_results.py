#!/usr/bin/env python3
"""
Visualize top-k projection results comparing vanilla, k=1, and k=5 methods
"""
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Read results
with open('topk_results/vanilla_full/final_results.json', 'r') as f:
    vanilla = json.load(f)

with open('topk_results/k1_thought_normalized_full/final_results.json', 'r') as f:
    k1 = json.load(f)

with open('topk_results/k5_thought_normalized_full/final_results.json', 'r') as f:
    k5 = json.load(f)

# Extract statistics
configs = ['Vanilla\n(Continuous)', 'k=1\n(Single Token)', 'k=5\n(5D Subspace)']
accuracies = [
    vanilla['stats']['accuracy'],
    k1['stats']['accuracy'],
    k5['stats']['accuracy']
]

# Create single plot - Accuracy comparison
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#2ecc71', '#e74c3c', '#3498db']
bars = ax.bar(configs, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('CODI Top-k Projection: Accuracy Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 50)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=accuracies[0], color='green', linestyle='--', alpha=0.5, linewidth=2, label='Vanilla baseline')
ax.legend(fontsize=11)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=13)

# Save figure
plt.tight_layout()
plt.savefig('topk_results/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("Visualization saved to: topk_results/accuracy_comparison.png")
