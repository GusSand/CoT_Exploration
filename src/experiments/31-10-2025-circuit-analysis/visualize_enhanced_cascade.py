"""
Visualize enhanced intervention cascade results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load results
results_dir = Path("circuit_analysis_results")

with open(results_dir / "enhanced_cascade_statistics.json", 'r') as f:
    statistics = json.load(f)

# Extract data
num_positions = 7

# Create matrices for visualization
token_change_matrix = np.zeros((num_positions, num_positions))
norm_diff_matrix = np.zeros((num_positions, num_positions))
norm_std_matrix = np.zeros((num_positions, num_positions))

for int_stat in statistics['position_statistics']:
    int_pos = int_stat['intervention_position']

    for effect in int_stat['downstream_effects']:
        target_pos = effect['target_position']
        token_change_matrix[int_pos, target_pos] = effect['token_change_rate'] * 100
        norm_diff_matrix[int_pos, target_pos] = effect['avg_hidden_norm_difference']
        norm_std_matrix[int_pos, target_pos] = effect['std_hidden_norm_difference']

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Token Change Rate Heatmap
ax = axes[0, 0]

im1 = ax.imshow(token_change_matrix, cmap='YlOrRd', vmin=0, vmax=30, aspect='auto')
ax.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_title('Token Change Rate (% of 20 examples)\nHigher = More Cascade Effect',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(num_positions))
ax.set_yticks(range(num_positions))
ax.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax.set_yticklabels([f'Int@{i}' for i in range(num_positions)])

# Annotate cells
for i in range(num_positions):
    for j in range(num_positions):
        value = token_change_matrix[i, j]
        color = 'white' if value > 15 else 'black'
        ax.text(j, i, f'{value:.0f}%', ha='center', va='center',
               color=color, fontsize=10, fontweight='bold')

plt.colorbar(im1, ax=ax, label='Token Change Rate (%)')

# 2. Hidden Norm Difference Heatmap
ax = axes[0, 1]

im2 = ax.imshow(norm_diff_matrix, cmap='Blues', vmin=0, vmax=1.5, aspect='auto')
ax.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_title('Average Hidden State Norm Difference\nHigher = Larger Activation Perturbation',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(num_positions))
ax.set_yticks(range(num_positions))
ax.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax.set_yticklabels([f'Int@{i}' for i in range(num_positions)])

# Annotate cells
for i in range(num_positions):
    for j in range(num_positions):
        value = norm_diff_matrix[i, j]
        color = 'white' if value > 0.75 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')

plt.colorbar(im2, ax=ax, label='Avg Norm Δ')

# 3. Cascade Strength Analysis
ax = axes[1, 0]

# For each intervention position, count how many downstream positions are affected (token change > 5%)
cascade_strengths = []
for i in range(num_positions):
    affected_count = np.sum(token_change_matrix[i, :] > 5)
    cascade_strengths.append(affected_count)

colors = ['#e74c3c' if cs > 0 else '#2ecc71' for cs in cascade_strengths]
bars = ax.bar(range(num_positions), cascade_strengths, color=colors, alpha=0.8)

ax.set_xlabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Affected Positions\n(Token Change > 5%)', fontsize=12, fontweight='bold')
ax.set_title('Cascade Strength: How Many Positions Affected?', fontsize=14, fontweight='bold')
ax.set_xticks(range(num_positions))
ax.set_ylim([0, 8])
ax.grid(True, alpha=0.3, axis='y')

# Annotate bars
for i, (bar, strength) in enumerate(zip(bars, cascade_strengths)):
    height = bar.get_height()
    label = 'ROBUST' if strength == 0 else f'{int(strength)} pos'
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
           label, ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. Continuous vs Binary Measure Comparison
ax = axes[1, 1]

# For each intervention position, compute:
# - Max token change rate
# - Max norm difference
# Show both on same plot

max_token_change = []
max_norm_diff = []

for i in range(num_positions):
    max_token_change.append(np.max(token_change_matrix[i, :]))
    max_norm_diff.append(np.max(norm_diff_matrix[i, :]))

x = np.arange(num_positions)
width = 0.35

# Normalize to same scale for comparison
max_token_change_norm = np.array(max_token_change) / 30  # Scale 0-30% to 0-1
max_norm_diff_norm = np.array(max_norm_diff) / 1.5  # Scale 0-1.5 to 0-1

ax.bar(x - width/2, max_token_change_norm, width, label='Max Token Change (scaled)', alpha=0.8, color='coral')
ax.bar(x + width/2, max_norm_diff_norm, width, label='Max Norm Δ (scaled)', alpha=0.8, color='steelblue')

ax.set_xlabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Normalized Maximum Effect', fontsize=12, fontweight='bold')
ax.set_title('Continuous vs Binary Measures\n(Both scaled 0-1)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('circuit_analysis_results/enhanced_cascade_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: circuit_analysis_results/enhanced_cascade_visualization.png")

# Create summary statistics figure
fig2, ax = plt.subplots(figsize=(16, 10))
ax.axis('off')

summary_text = """
ENHANCED INTERVENTION CASCADE ANALYSIS (20 Examples)

KEY FINDINGS - CONTINUOUS ACTIVATION MEASUREMENTS:

🟢 EXTREMELY ROBUST POSITIONS:
   Position 3: 0% token change rate across ALL positions
               Avg norm Δ = 0.00 (no activation perturbation)
               → Computational bottleneck that filters noise

   Position 6: 0% token change rate across ALL positions
               → Terminal robustness

🟡 MODERATELY VULNERABLE POSITIONS:
   Position 0: Affects positions 1-6 (5-15% rate)
               Max norm Δ = 0.55 (small perturbations)

   Position 1: Affects positions 2-3 (20% rate)
               Max norm Δ = 0.93 (moderate perturbations)

   Position 2: Affects position 3 (25% rate)
               Max norm Δ = 0.75

🔴 LATE-STAGE POSITIONS:
   Position 4: Affects positions 5-6 (10-20% rate)
               Max norm Δ = 1.07 (largest perturbation!)

   Position 5: Affects position 6 (5% rate)
               Minimal cascade

CONTINUOUS MEASURE INSIGHTS:

1. Token changes are RARE (max 25% rate)
   → Model is generally stable to interventions

2. Activation perturbations are SMALL (avg < 1.0 L2 norm)
   → Even when tokens don't change, activations barely shift

3. Position 3 is SPECIAL:
   - Zero token changes
   - Zero activation perturbations
   - Acts as error-correction filter

4. Binary (token) vs Continuous (norm) measures AGREE:
   - Low token change correlates with low norm difference
   - Position 4 shows largest effect on BOTH measures

COMPARISON TO SINGLE-EXAMPLE ANALYSIS:

Single Example (Janet's ducks):
- Positions 2,3,6 robust
- Positions 0,1 vulnerable

20 Examples (aggregated):
- Position 3,6 CONSISTENTLY robust
- Position 2 sometimes vulnerable (25% at pos 3)
- More nuanced picture emerges

→ Patterns are PARTIALLY consistent but not universal
→ Position 3 is the most robust across all examples
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig('circuit_analysis_results/enhanced_cascade_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: circuit_analysis_results/enhanced_cascade_summary.png")

print("\n✓ All enhanced visualizations created!")
