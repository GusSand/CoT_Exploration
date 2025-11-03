#!/usr/bin/env python3
"""
Visualize V4 Experiment Results
"""

import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Load results
results_file = sys.argv[1] if len(sys.argv) > 1 else "results/feature_2203_manipulation_v4_results_latest.json"

with open(results_file, 'r') as f:
    data = json.load(f)

results = data['results']

# Separate baseline and intervention
baseline = [r for r in results if r['intervention_type'] == 'none']
intervened = [r for r in results if r['intervention_type'] == 'add']

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# ============================================================================
# Plot 1: Feature 2203 Activations Across CoT Iterations
# ============================================================================
ax1 = plt.subplot(2, 2, 1)

for r in baseline:
    cot_acts = [a['original_activation'] for a in r['activations'] if a['is_cot']]
    ax1.plot(range(1, len(cot_acts)+1), cot_acts, 'o-', linewidth=2,
             label=f"{r['problem_name']} (baseline)", alpha=0.7)

for r in intervened:
    cot_acts = [a['original_activation'] for a in r['activations'] if a['is_cot']]
    ax1.plot(range(1, len(cot_acts)+1), cot_acts, 's--', linewidth=2,
             label=f"{r['problem_name']} (+feat)", alpha=0.7)

ax1.set_xlabel('CoT Iteration', fontsize=12)
ax1.set_ylabel('Feature 2203 Activation', fontsize=12)
ax1.set_title('Feature 2203 Activations During CoT', fontsize=14, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(1, 7))

# ============================================================================
# Plot 2: CoT Token Predictions (Baseline vs Intervened)
# ============================================================================
ax2 = plt.subplot(2, 2, 2)

problems = ['original', 'variant_a', 'variant_b']
x_pos = np.arange(len(problems))
width = 0.35

baseline_tokens = []
intervened_tokens = []

for prob in problems:
    base_result = [r for r in baseline if r['problem_name'] == prob][0]
    inter_result = [r for r in intervened if r['problem_name'] == prob][0]

    baseline_tokens.append(' '.join([t['token_str'] for t in base_result['cot_tokens']]))
    intervened_tokens.append(' '.join([t['token_str'] for t in inter_result['cot_tokens']]))

# Display as text
ax2.axis('off')
ax2.set_title('CoT Token Predictions (Last Layer)', fontsize=14, fontweight='bold')

y_start = 0.9
line_height = 0.13

for i, prob in enumerate(problems):
    # Problem name
    ax2.text(0.05, y_start - i*line_height*2, f"{prob.upper()}:",
             fontsize=11, fontweight='bold', transform=ax2.transAxes)

    # Baseline tokens
    ax2.text(0.05, y_start - i*line_height*2 - 0.05, f"Baseline: {baseline_tokens[i][:60]}...",
             fontsize=9, transform=ax2.transAxes, family='monospace')

    # Intervened tokens
    ax2.text(0.05, y_start - i*line_height*2 - 0.09, f"+ Feat:   {intervened_tokens[i][:60]}...",
             fontsize=9, transform=ax2.transAxes, family='monospace', color='red')

# ============================================================================
# Plot 3: Final Answer Comparison
# ============================================================================
ax3 = plt.subplot(2, 2, 3)

problems_display = ['Original', 'Variant A', 'Variant B']
baseline_answers = [r['final_answer_numeric'] for r in baseline]
intervened_answers = [r['final_answer_numeric'] for r in intervened]
expected_answers = [r['expected_answer'] for r in baseline]

x = np.arange(len(problems_display))
width = 0.25

bars1 = ax3.bar(x - width, expected_answers, width, label='Expected', color='green', alpha=0.7)
bars2 = ax3.bar(x, baseline_answers, width, label='Baseline', color='blue', alpha=0.7)
bars3 = ax3.bar(x + width, intervened_answers, width, label='+ Feature 2203', color='red', alpha=0.7)

ax3.set_xlabel('Problem', fontsize=12)
ax3.set_ylabel('Final Answer', fontsize=12)
ax3.set_title('Final Answer Comparison', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(problems_display)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)

# ============================================================================
# Plot 4: Summary Table
# ============================================================================
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')
ax4.set_title('Experimental Summary', fontsize=14, fontweight='bold')

# Create summary text
summary_text = []
summary_text.append("BASELINE RESULTS:")
for r in baseline:
    summary_text.append(f"  {r['problem_name']:>10s}: {int(r['final_answer_numeric'])} (expected {r['expected_answer']}) {'✓' if r['final_answer_numeric'] == r['expected_answer'] else 'X'}")

summary_text.append("\nADD FEATURE 2203 (magnitude=20):")
for r in intervened:
    summary_text.append(f"  {r['problem_name']:>10s}: {int(r['final_answer_numeric']) if r['final_answer_numeric'] != float('inf') else 'ERROR'} (expected {r['expected_answer']}) {'✓' if r['final_answer_numeric'] == r['expected_answer'] else 'X'}")

summary_text.append("\nKEY OBSERVATIONS:")
summary_text.append("- Baseline: All correct!")
baseline_correct = sum(1 for r in baseline if r['final_answer_numeric'] == r['expected_answer'])
intervened_correct = sum(1 for r in intervened if r['final_answer_numeric'] == r['expected_answer'])
summary_text.append(f"- Baseline accuracy: {baseline_correct}/3")
summary_text.append(f"- Intervened accuracy: {intervened_correct}/3")

if intervened_correct < baseline_correct:
    summary_text.append("- Feature 2203 is HARMFUL when added!")

summary_y = 0.95
for line in summary_text:
    ax4.text(0.05, summary_y, line, fontsize=10, transform=ax4.transAxes,
             family='monospace', verticalalignment='top')
    summary_y -= 0.05

plt.tight_layout()

# Save figure
output_file = results_file.replace('.json', '_visualization.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved to: {output_file}")

plt.show()
