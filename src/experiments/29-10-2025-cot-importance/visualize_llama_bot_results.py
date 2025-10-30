#!/usr/bin/env python3
"""
Visualize LLAMA CODI BoT Token Comparison Results
Creates comprehensive visualizations showing:
1. Overall accuracy comparison
2. Per-example performance breakdown
3. Thought token patterns and differences
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import re

# Load results
print("Loading results...")
with open('llama_bot_comparison_results.json', 'r') as f:
    data = json.load(f)

# Set up matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'with_bot': '#2ecc71',
    'without_bot': '#e74c3c',
    'both_correct': '#3498db',
    'both_wrong': '#95a5a6',
    'only_with_correct': '#2ecc71',
    'only_without_correct': '#e67e22'
}

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Overall Accuracy Comparison (Top Left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])
accuracies = [data['accuracy_with_bot'] * 100, data['accuracy_without_bot'] * 100]
bars = ax1.bar(['WITH BoT', 'WITHOUT BoT'], accuracies,
               color=[colors['with_bot'], colors['without_bot']],
               edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.3)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{acc:.1f}%\n({int(acc*data["total_examples"]/100)}/{data["total_examples"]})',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotation
improvement = accuracies[0] - accuracies[1]
ax1.annotate(f'+{improvement:.1f}%',
            xy=(0.5, max(accuracies)), xytext=(0.5, 85),
            ha='center', fontsize=12, fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

# ============================================================================
# 2. Numeric Thought Token Analysis (Top Middle)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])
numeric_counts = [data['avg_numeric_tokens_with_bot'],
                  data['avg_numeric_tokens_without_bot']]
bars = ax2.bar(['WITH BoT', 'WITHOUT BoT'], numeric_counts,
               color=[colors['with_bot'], colors['without_bot']],
               edgecolor='black', linewidth=2)
ax2.set_ylabel('Avg Numeric Tokens (out of 6)', fontsize=12, fontweight='bold')
ax2.set_title('Numeric Thought Tokens', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 6)
ax2.axhline(y=3, color='gray', linestyle='--', alpha=0.3, label='50% threshold')
ax2.grid(axis='y', alpha=0.3)
ax2.legend()

# Add value labels
for bar, count in zip(bars, numeric_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{count:.2f}\n({count/6*100:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# ============================================================================
# 3. Confusion Matrix / Agreement Analysis (Top Right)
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

# Calculate agreement statistics
both_correct = sum(1 for i in range(len(data['results_with_bot']))
                   if data['results_with_bot'][i]['correct'] and
                   data['results_without_bot'][i]['correct'])
both_wrong = sum(1 for i in range(len(data['results_with_bot']))
                 if not data['results_with_bot'][i]['correct'] and
                 not data['results_without_bot'][i]['correct'])
only_with_correct = sum(1 for i in range(len(data['results_with_bot']))
                        if data['results_with_bot'][i]['correct'] and
                        not data['results_without_bot'][i]['correct'])
only_without_correct = sum(1 for i in range(len(data['results_with_bot']))
                           if not data['results_with_bot'][i]['correct'] and
                           data['results_without_bot'][i]['correct'])

# Create confusion matrix
confusion = np.array([[both_correct, only_without_correct],
                     [only_with_correct, both_wrong]])

im = ax3.imshow(confusion, cmap='YlGn', alpha=0.7)
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Correct', 'Wrong'], fontweight='bold')
ax3.set_yticklabels(['Correct', 'Wrong'], fontweight='bold')
ax3.set_xlabel('WITHOUT BoT', fontsize=12, fontweight='bold')
ax3.set_ylabel('WITH BoT', fontsize=12, fontweight='bold')
ax3.set_title('Agreement Matrix', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, f'{confusion[i, j]}\n({confusion[i, j]/data["total_examples"]*100:.1f}%)',
                       ha="center", va="center", color="black",
                       fontsize=12, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Count', fontsize=10, fontweight='bold')

# ============================================================================
# 4. Performance Breakdown (Middle Left, spanning 2 columns)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :2])

categories = ['Both Correct', 'Only WITH BoT\nCorrect', 'Only WITHOUT BoT\nCorrect', 'Both Wrong']
counts = [both_correct, only_with_correct, only_without_correct, both_wrong]
percentages = [c/data['total_examples']*100 for c in counts]
colors_breakdown = [colors['both_correct'], colors['only_with_correct'],
                    colors['only_without_correct'], colors['both_wrong']]

bars = ax4.barh(categories, percentages, color=colors_breakdown,
                edgecolor='black', linewidth=2)
ax4.set_xlabel('Percentage of Examples (%)', fontsize=12, fontweight='bold')
ax4.set_title('Detailed Performance Breakdown', fontsize=14, fontweight='bold')
ax4.set_xlim(0, 100)
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for bar, count, pct in zip(bars, counts, percentages):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
             f'{count} ({pct:.1f}%)',
             ha='left', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# 5. Sample Thought Token Sequences (Middle Right)
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
ax5.set_title('Example Thought Token Patterns', fontsize=14, fontweight='bold')

# Show 3 interesting examples
examples_to_show = [
    ('Ex1: Both Correct', 0, True, True),
    ('Ex2: Only WITH Correct', 1, True, False),
    ('Ex3: Both Wrong', None, False, False)
]

# Find an example where both are wrong
both_wrong_idx = None
for i in range(len(data['results_with_bot'])):
    if not data['results_with_bot'][i]['correct'] and not data['results_without_bot'][i]['correct']:
        both_wrong_idx = i
        break

if both_wrong_idx is not None:
    examples_to_show[2] = ('Ex3: Both Wrong', both_wrong_idx, False, False)

number_regex = re.compile(r'^\s?\d+')

y_pos = 0.9
for label, idx, with_correct, without_correct in examples_to_show:
    if idx is None:
        continue

    r_with = data['results_with_bot'][idx]
    r_without = data['results_without_bot'][idx]

    # Example label
    ax5.text(0.05, y_pos, label, fontsize=11, fontweight='bold',
             verticalalignment='top')
    y_pos -= 0.05

    # WITH BoT
    color = 'green' if with_correct else 'red'
    thought_str = ' → '.join([f"'{t}'" for t in r_with['thought_tokens']])
    ax5.text(0.05, y_pos, f"WITH BoT: {thought_str}",
             fontsize=9, verticalalignment='top', color=color,
             family='monospace')
    y_pos -= 0.04
    ax5.text(0.05, y_pos, f"Answer: {r_with['predicted_number']} (GT: {r_with['ground_truth']})",
             fontsize=8, verticalalignment='top', color=color, style='italic')
    y_pos -= 0.05

    # WITHOUT BoT
    color = 'green' if without_correct else 'red'
    thought_str = ' → '.join([f"'{t}'" for t in r_without['thought_tokens']])
    ax5.text(0.05, y_pos, f"W/O BoT: {thought_str}",
             fontsize=9, verticalalignment='top', color=color,
             family='monospace')
    y_pos -= 0.04
    ax5.text(0.05, y_pos, f"Answer: {r_without['predicted_number']} (GT: {r_without['ground_truth']})",
             fontsize=8, verticalalignment='top', color=color, style='italic')
    y_pos -= 0.08

# ============================================================================
# 6. Per-Example Performance Visualization (Bottom, full width)
# ============================================================================
ax6 = fig.add_subplot(gs[2:, :])

# Create scatter plot showing all examples
x_indices = np.arange(data['total_examples'])
with_bot_correct = [1 if r['correct'] else 0 for r in data['results_with_bot']]
without_bot_correct = [1 if r['correct'] else 0 for r in data['results_without_bot']]

# Determine colors for each example
example_colors = []
for i in range(data['total_examples']):
    if with_bot_correct[i] and without_bot_correct[i]:
        example_colors.append(colors['both_correct'])
    elif with_bot_correct[i] and not without_bot_correct[i]:
        example_colors.append(colors['only_with_correct'])
    elif not with_bot_correct[i] and without_bot_correct[i]:
        example_colors.append(colors['only_without_correct'])
    else:
        example_colors.append(colors['both_wrong'])

# Plot WITH BoT
ax6.scatter(x_indices, [1.1]*data['total_examples'],
           c=[colors['with_bot'] if c else colors['without_bot'] for c in with_bot_correct],
           s=50, marker='s', alpha=0.7, label='WITH BoT', edgecolors='black', linewidth=0.5)

# Plot WITHOUT BoT
ax6.scatter(x_indices, [0.9]*data['total_examples'],
           c=[colors['with_bot'] if c else colors['without_bot'] for c in without_bot_correct],
           s=50, marker='o', alpha=0.7, label='WITHOUT BoT', edgecolors='black', linewidth=0.5)

# Draw connecting lines
for i in range(data['total_examples']):
    if with_bot_correct[i] != without_bot_correct[i]:
        # Different results - draw colored line
        ax6.plot([i, i], [0.9, 1.1], color=example_colors[i],
                linewidth=2, alpha=0.5, zorder=1)
    else:
        # Same result - draw thin gray line
        ax6.plot([i, i], [0.9, 1.1], color='gray',
                linewidth=0.5, alpha=0.3, zorder=1)

ax6.set_xlabel('Example Index', fontsize=12, fontweight='bold')
ax6.set_ylabel('Condition', fontsize=12, fontweight='bold')
ax6.set_yticks([0.9, 1.1])
ax6.set_yticklabels(['WITHOUT\nBoT', 'WITH\nBoT'], fontweight='bold')
ax6.set_title(f'Per-Example Performance Comparison (N={data["total_examples"]})',
             fontsize=14, fontweight='bold')
ax6.set_xlim(-2, data['total_examples']+2)
ax6.set_ylim(0.7, 1.3)
ax6.grid(axis='x', alpha=0.3)

# Add legend with custom markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['with_bot'],
           markersize=10, label='Correct', markeredgecolor='black'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=colors['without_bot'],
           markersize=10, label='Wrong', markeredgecolor='black'),
    Line2D([0], [0], color=colors['only_with_correct'], linewidth=3,
           label='Only WITH BoT Correct'),
    Line2D([0], [0], color=colors['only_without_correct'], linewidth=3,
           label='Only W/O BoT Correct')
]
ax6.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add statistics box
stats_text = f"""KEY FINDINGS:
• BoT Token Impact: +{improvement:.1f}% accuracy
• Agreement Rate: {(both_correct + both_wrong)/data['total_examples']*100:.1f}%
• BoT-Dependent: {only_with_correct} examples ({only_with_correct/data['total_examples']*100:.1f}%)
• Numeric Tokens: Similar (~53-56%)"""

ax6.text(0.02, 0.98, stats_text, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================================
# Overall title
# ============================================================================
fig.suptitle('LLAMA CODI: Beginning-of-Thought (BoT) Token Impact Analysis\nGSM8K Math Problem Dataset',
            fontsize=16, fontweight='bold', y=0.995)

# Save figure
plt.tight_layout()
plt.savefig('llama_bot_comparison_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved to: llama_bot_comparison_visualization.png")

plt.show()

# ============================================================================
# Create additional detailed thought token analysis
# ============================================================================
print("\n" + "="*80)
print("DETAILED THOUGHT TOKEN ANALYSIS")
print("="*80)

# Analyze thought token patterns
print("\nAnalyzing thought token patterns...")

# Count token types
def classify_token(token):
    """Classify a thought token"""
    if re.match(r'^\s?\d+', token):
        return 'numeric'
    elif token.strip() in ['+', '-', '*', '/', '=', '(', ')']:
        return 'operator'
    elif token.strip() in ['>>', '<<', '|', '&']:
        return 'special'
    else:
        return 'other'

# Analyze for each condition
for condition, results in [('WITH BoT', data['results_with_bot']),
                           ('WITHOUT BoT', data['results_without_bot'])]:
    print(f"\n{condition}:")

    token_types = {'numeric': 0, 'operator': 0, 'special': 0, 'other': 0}
    all_tokens = []

    for r in results:
        for token in r['thought_tokens']:
            token_type = classify_token(token)
            token_types[token_type] += 1
            all_tokens.append(token)

    total_tokens = sum(token_types.values())
    print(f"  Total thought tokens: {total_tokens}")
    for token_type, count in sorted(token_types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {token_type.capitalize()}: {count} ({count/total_tokens*100:.1f}%)")

    # Most common tokens
    from collections import Counter
    token_counts = Counter(all_tokens)
    print(f"  Top 10 most common tokens:")
    for token, count in token_counts.most_common(10):
        print(f"    '{token}': {count} ({count/total_tokens*100:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
