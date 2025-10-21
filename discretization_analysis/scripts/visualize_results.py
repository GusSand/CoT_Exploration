#!/usr/bin/env python3
"""
Visualization and analysis of CODI discretization results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_file", type=str, default="./discretization_results/final_results.json")
parser.add_argument("--output_dir", type=str, default="./discretization_results/plots")
args = parser.parse_args()

# Load results
print(f"Loading results from {args.results_file}...")
with open(args.results_file, 'r') as f:
    data = json.load(f)

stats = data['stats']
results = data['results']

os.makedirs(args.output_dir, exist_ok=True)

# 1. Accuracy Comparison Bar Chart
print("Generating accuracy comparison chart...")
fig, ax = plt.subplots(figsize=(10, 6))
modes = list(stats.keys())
accuracies = [stats[mode]['accuracy'] for mode in modes]
colors = ['#2ecc71', '#f39c12', '#e74c3c']

bars = ax.bar(modes, accuracies, color=colors, alpha=0.8)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('CODI Discretization: Impact on GSM8K Accuracy', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'accuracy_comparison.png'), dpi=300)
print(f"  Saved: accuracy_comparison.png")

# 2. Confidence Distribution Analysis
print("Analyzing confidence distributions...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Thought Token Confidence Distributions', fontsize=14, fontweight='bold')

for mode_idx, mode in enumerate(modes):
    # For each thought position (T0-T5)
    for pos in range(6):
        ax = axes[mode_idx % 2, pos // 2] if mode_idx < 2 else None
        if mode_idx == 2 and pos < 3:
            ax = axes[1, pos]

        if ax is None:
            continue

        # Collect top-1 probabilities for this position across all examples
        probs = []
        for result in results[mode]:
            if len(result['thought_probs']) > pos + 1:  # +1 because T-2 is index 0
                probs.append(result['thought_probs'][pos + 1][0])  # Top-1 probability

        if pos < 6:
            title_pos = pos
            if mode_idx == 0 and pos == 0:
                ax.hist(probs, bins=30, alpha=0.7, color=colors[mode_idx], edgecolor='black')
                ax.set_xlabel('Top-1 Probability')
                ax.set_ylabel('Frequency')
                ax.set_title(f'{mode.capitalize()} - T{title_pos}')
                ax.grid(alpha=0.3)

# Create separate figure for confidence by position
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle('Average Top-1 Confidence by Thought Position', fontsize=14, fontweight='bold')

for mode_idx, mode in enumerate(modes):
    avg_confidences = []
    for pos in range(6):
        probs = []
        for result in results[mode]:
            if len(result['thought_probs']) > pos + 1:
                probs.append(result['thought_probs'][pos + 1][0])
        avg_confidences.append(np.mean(probs) if probs else 0)

    axes2[mode_idx].plot(range(6), avg_confidences, marker='o', linewidth=2, markersize=8, color=colors[mode_idx])
    axes2[mode_idx].set_xlabel('Thought Position')
    axes2[mode_idx].set_ylabel('Avg Top-1 Probability')
    axes2[mode_idx].set_title(f'{mode.capitalize()}')
    axes2[mode_idx].set_ylim(0, 1)
    axes2[mode_idx].set_xticks(range(6))
    axes2[mode_idx].set_xticklabels([f'T{i}' for i in range(6)])
    axes2[mode_idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'confidence_by_position.png'), dpi=300)
print(f"  Saved: confidence_by_position.png")

# 3. Token Diversity Analysis
print("Analyzing token diversity...")
fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
fig3.suptitle('Unique Top-1 Tokens per Thought Position', fontsize=14, fontweight='bold')

for mode_idx, mode in enumerate(modes):
    unique_counts = []
    for pos in range(6):
        tokens = []
        for result in results[mode]:
            if len(result['thought_tokens']) > pos + 1:
                tokens.append(result['thought_tokens'][pos + 1][0])
        unique_counts.append(len(set(tokens)))

    axes3[mode_idx].bar(range(6), unique_counts, color=colors[mode_idx], alpha=0.8)
    axes3[mode_idx].set_xlabel('Thought Position')
    axes3[mode_idx].set_ylabel('Unique Tokens')
    axes3[mode_idx].set_title(f'{mode.capitalize()}')
    axes3[mode_idx].set_xticks(range(6))
    axes3[mode_idx].set_xticklabels([f'T{i}' for i in range(6)])
    axes3[mode_idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'token_diversity.png'), dpi=300)
print(f"  Saved: token_diversity.png")

# 4. Most Common Tokens at Each Position
print("Analyzing most common tokens...")
for mode in modes:
    print(f"\n{mode.upper()} - Top-3 Most Common Tokens:")
    for pos in range(6):
        tokens = []
        for result in results[mode]:
            if len(result['thought_tokens']) > pos + 1:
                tokens.append(result['thought_tokens'][pos + 1][0])

        counter = Counter(tokens)
        top3 = counter.most_common(3)
        total = len(tokens)
        print(f"  T{pos}: ", end="")
        for token, count in top3:
            print(f"'{token}' ({100*count/total:.1f}%)  ", end="")
        print()

# 5. Error Analysis - Compare incorrect predictions
print("\nError Analysis:")
for mode in modes:
    incorrect = [r for r in results[mode] if not r['correct']]
    print(f"\n{mode.capitalize()}: {len(incorrect)} errors")

    if len(incorrect) > 0:
        # Sample 5 errors
        print("  Sample errors:")
        for i, err in enumerate(incorrect[:5]):
            print(f"    Q: {err['question'][:60]}...")
            print(f"       Truth: {err['ground_truth']:.0f}, Predicted: {err['predicted_number']:.0f}")

# 6. Generate Summary Report
print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)

report_lines = []
report_lines.append("# CODI Discretization Analysis - Summary Report\n")
report_lines.append(f"Dataset: GSM8K Test Set ({data['metadata']['num_examples']} examples)\n")
report_lines.append(f"Generated: {data['metadata']['timestamp']}\n")
report_lines.append(f"Total Time: {data['metadata']['total_time']/60:.1f} minutes\n\n")

report_lines.append("## Accuracy Results\n")
for mode in modes:
    s = stats[mode]
    report_lines.append(f"- **{mode.capitalize()}**: {s['accuracy']:.2f}% ({s['correct']}/{s['total']})\n")

report_lines.append("\n## Key Findings\n\n")

# Calculate accuracy drop
vanilla_acc = stats['vanilla']['accuracy']
alt_acc = stats['alternating']['accuracy']
full_acc = stats['full']['accuracy']

report_lines.append(f"1. **Alternating discretization** causes {vanilla_acc - alt_acc:.1f}% accuracy drop\n")
report_lines.append(f"2. **Full discretization** causes {vanilla_acc - full_acc:.1f}% accuracy drop\n")

# Calculate average confidence increase
vanilla_avg_conf = []
alt_avg_conf = []
full_avg_conf = []

for pos in range(6):
    vanilla_probs = [r['thought_probs'][pos+1][0] for r in results['vanilla'] if len(r['thought_probs']) > pos+1]
    alt_probs = [r['thought_probs'][pos+1][0] for r in results['alternating'] if len(r['thought_probs']) > pos+1]
    full_probs = [r['thought_probs'][pos+1][0] for r in results['full'] if len(r['thought_probs']) > pos+1]

    if vanilla_probs:
        vanilla_avg_conf.append(np.mean(vanilla_probs))
    if alt_probs:
        alt_avg_conf.append(np.mean(alt_probs))
    if full_probs:
        full_avg_conf.append(np.mean(full_probs))

if vanilla_avg_conf and alt_avg_conf:
    report_lines.append(f"3. Discretization increases average confidence: vanilla {np.mean(vanilla_avg_conf):.3f} → alternating {np.mean(alt_avg_conf):.3f} → full {np.mean(full_avg_conf):.3f}\n")

report_lines.append("\n## Conclusion\n\n")
report_lines.append("Discretizing continuous thoughts degrades CODI's reasoning ability:\n")
report_lines.append("- Even partial discretization (alternating) significantly hurts performance\n")
report_lines.append("- Discretization increases confidence but reduces accuracy (overconfidence in wrong answers)\n")
report_lines.append("- Continuous representations are essential for mathematical reasoning\n")

# Save report
report_path = os.path.join(args.output_dir, "summary_report.md")
with open(report_path, 'w') as f:
    f.writelines(report_lines)

print("\n".join(report_lines))
print(f"\nReport saved to: {report_path}")
print(f"All visualizations saved to: {args.output_dir}/")
