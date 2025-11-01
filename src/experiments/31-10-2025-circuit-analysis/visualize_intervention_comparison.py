"""
Compare Number-Only vs Average Activation Intervention Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load both result sets
with open("enhanced_cascade_statistics.json", 'r') as f:
    number_only = json.load(f)

with open("avg_intervention_cascade_statistics.json", 'r') as f:
    avg_activation = json.load(f)

num_positions = 7

# Extract matrices for both approaches
def extract_matrices(statistics):
    token_change = np.zeros((num_positions, num_positions))
    norm_diff = np.zeros((num_positions, num_positions))

    for int_stat in statistics['position_statistics']:
        int_pos = int_stat['intervention_position']
        for effect in int_stat['downstream_effects']:
            target_pos = effect['target_position']
            token_change[int_pos, target_pos] = effect['token_change_rate'] * 100
            norm_diff[int_pos, target_pos] = effect['avg_hidden_norm_difference']

    return token_change, norm_diff

token_number, norm_number = extract_matrices(number_only)
token_avg, norm_avg = extract_matrices(avg_activation)

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 14))

# 1. Number-Only Intervention - Token Change
ax1 = plt.subplot(2, 3, 1)
im1 = ax1.imshow(token_number, cmap='YlOrRd', vmin=0, vmax=80, aspect='auto')
ax1.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax1.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax1.set_title('Number-Only Intervention\nToken Change Rate (%)',
             fontsize=14, fontweight='bold')
ax1.set_xticks(range(num_positions))
ax1.set_yticks(range(num_positions))
ax1.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax1.set_yticklabels([f'Int@{i}' for i in range(num_positions)])
for i in range(num_positions):
    for j in range(num_positions):
        value = token_number[i, j]
        color = 'white' if value > 40 else 'black'
        ax1.text(j, i, f'{value:.0f}%', ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Token Change Rate (%)')

# 2. Average Activation Intervention - Token Change
ax2 = plt.subplot(2, 3, 2)
im2 = ax2.imshow(token_avg, cmap='YlOrRd', vmin=0, vmax=80, aspect='auto')
ax2.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax2.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax2.set_title('Average Activation Intervention\nToken Change Rate (%)',
             fontsize=14, fontweight='bold')
ax2.set_xticks(range(num_positions))
ax2.set_yticks(range(num_positions))
ax2.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax2.set_yticklabels([f'Int@{i}' for i in range(num_positions)])
for i in range(num_positions):
    for j in range(num_positions):
        value = token_avg[i, j]
        color = 'white' if value > 40 else 'black'
        ax2.text(j, i, f'{value:.0f}%', ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Token Change Rate (%)')

# 3. Difference Map (Avg - Number)
ax3 = plt.subplot(2, 3, 3)
diff = token_avg - token_number
im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-50, vmax=50, aspect='auto')
ax3.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax3.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax3.set_title('Difference Map\n(Avg - Number)',
             fontsize=14, fontweight='bold')
ax3.set_xticks(range(num_positions))
ax3.set_yticks(range(num_positions))
ax3.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax3.set_yticklabels([f'Int@{i}' for i in range(num_positions)])
for i in range(num_positions):
    for j in range(num_positions):
        value = diff[i, j]
        color = 'white' if abs(value) > 25 else 'black'
        ax3.text(j, i, f'{value:+.0f}%', ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='Difference (%)')

# 4. Number-Only - Norm Difference
ax4 = plt.subplot(2, 3, 4)
im4 = ax4.imshow(norm_number, cmap='Blues', vmin=0, vmax=6, aspect='auto')
ax4.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax4.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax4.set_title('Number-Only Intervention\nHidden State Norm Difference',
             fontsize=14, fontweight='bold')
ax4.set_xticks(range(num_positions))
ax4.set_yticks(range(num_positions))
ax4.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax4.set_yticklabels([f'Int@{i}' for i in range(num_positions)])
for i in range(num_positions):
    for j in range(num_positions):
        value = norm_number[i, j]
        color = 'white' if value > 3 else 'black'
        ax4.text(j, i, f'{value:.1f}', ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')
plt.colorbar(im4, ax=ax4, label='Avg Norm Δ')

# 5. Average Activation - Norm Difference
ax5 = plt.subplot(2, 3, 5)
im5 = ax5.imshow(norm_avg, cmap='Blues', vmin=0, vmax=6, aspect='auto')
ax5.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax5.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax5.set_title('Average Activation Intervention\nHidden State Norm Difference',
             fontsize=14, fontweight='bold')
ax5.set_xticks(range(num_positions))
ax5.set_yticks(range(num_positions))
ax5.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax5.set_yticklabels([f'Int@{i}' for i in range(num_positions)])
for i in range(num_positions):
    for j in range(num_positions):
        value = norm_avg[i, j]
        color = 'white' if value > 3 else 'black'
        ax5.text(j, i, f'{value:.1f}', ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')
plt.colorbar(im5, ax=ax5, label='Avg Norm Δ')

# 6. Key Findings Summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = """
INTERVENTION METHOD COMPARISON

NUMBER-ONLY INTERVENTION (Baseline):
- Only intervenes when token is a number
- Position 3: appears "robust" (0% change)
  → Actually skipped operators (+, -, >>)
- Position 2: 25% at pos 3
- Position 6: 0% (terminal robustness)

AVERAGE ACTIVATION INTERVENTION:
- Intervenes on ALL tokens (no skipping)
- Position 2 → Position 3: 80% change!
  → Reveals pos 3 is actually vulnerable
- Position 0: 45% at pos 2
- Position 1: 50% at pos 2, 45% at pos 3
- Position 4: 50-55% at pos 5-6
- Position 6: Still 0% (true robustness)

KEY INSIGHTS:

1. Position 3 Not Robust:
   Number-only: 0% (misleading)
   Avg activation: 80% from pos 2
   → "Robustness" was artifact of skipping

2. Strongest Cascade:
   Position 2 → Position 3 (80%)
   This is the critical link

3. Terminal Position:
   Position 6 truly robust (0% both methods)

4. Early Positions:
   More cascades visible with avg activation
   Positions 0-2 show substantial effects

CONCLUSION:
Average activation intervention reveals
true vulnerability patterns that were
masked by number-only approach.
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig('intervention_method_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: intervention_method_comparison.png")

# Print quantitative comparison
print("\n" + "="*80)
print("QUANTITATIVE COMPARISON")
print("="*80)

print("\nPosition 2 → Position 3 (Critical Link):")
print(f"  Number-only:      {token_number[2, 3]:.1f}%")
print(f"  Avg activation:   {token_avg[2, 3]:.1f}%")
print(f"  Difference:       {token_avg[2, 3] - token_number[2, 3]:+.1f}%")

print("\nPosition 3 Self-Intervention:")
print(f"  Number-only:      {token_number[3, 3]:.1f}%")
print(f"  Avg activation:   {token_avg[3, 3]:.1f}%")

print("\nPosition 6 (Terminal):")
print(f"  Number-only:      {token_number[6, 6]:.1f}%")
print(f"  Avg activation:   {token_avg[6, 6]:.1f}%")

print("\nOverall Cascade Strength (avg % across all positions):")
print(f"  Number-only:      {np.mean(token_number):.1f}%")
print(f"  Avg activation:   {np.mean(token_avg):.1f}%")

print("\n✓ Comparison complete!")
