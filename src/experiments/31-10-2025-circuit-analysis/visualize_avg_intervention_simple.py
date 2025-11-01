"""
Simple visualization of Average Activation Intervention Results
Three metrics: Token Change Rate, Hidden Norm Difference, Number Decoding Rate
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# Load average activation results
with open("avg_intervention_cascade_statistics.json", 'r') as f:
    statistics = json.load(f)

# Load raw data to compute number decoding rate
with open("avg_intervention_cascade_raw.json", 'r') as f:
    raw_data = json.load(f)

num_positions = 7

# Extract token change and norm difference matrices
token_change = np.zeros((num_positions, num_positions))
norm_diff = np.zeros((num_positions, num_positions))

for int_stat in statistics['position_statistics']:
    int_pos = int_stat['intervention_position']
    for effect in int_stat['downstream_effects']:
        target_pos = effect['target_position']
        token_change[int_pos, target_pos] = effect['token_change_rate'] * 100
        norm_diff[int_pos, target_pos] = effect['avg_hidden_norm_difference']

# Compute number decoding rate from raw data
number_decode_rate = np.zeros((num_positions, num_positions))
number_regex = r'^\s?\d+'

import re
for example in raw_data:
    for intervention in example['interventions']:
        int_pos = intervention['intervention_position']
        for pos_metric in intervention['position_metrics']:
            pos = pos_metric['position']
            # Check if intervened token is a number
            token = pos_metric.get('intervened_token', pos_metric.get('baseline_token', ''))
            is_number = bool(re.match(number_regex, token))
            if is_number:
                number_decode_rate[int_pos, pos] += 1

# Convert counts to percentages
number_decode_rate = (number_decode_rate / len(raw_data)) * 100

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# 1. Token Change Rate
ax = axes[0]
im1 = ax.imshow(token_change, cmap='YlOrRd', vmin=0, vmax=80, aspect='auto')
ax.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_title('Token Change Rate (%)\nAverage Activation Intervention',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(num_positions))
ax.set_yticks(range(num_positions))
ax.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax.set_yticklabels([f'Int@{i}' for i in range(num_positions)])

for i in range(num_positions):
    for j in range(num_positions):
        value = token_change[i, j]
        color = 'white' if value > 40 else 'black'
        ax.text(j, i, f'{value:.0f}%', ha='center', va='center',
               color=color, fontsize=10, fontweight='bold')

plt.colorbar(im1, ax=ax, label='Token Change Rate (%)')

# 2. Hidden State Norm Difference
ax = axes[1]
im2 = ax.imshow(norm_diff, cmap='Blues', vmin=0, vmax=6, aspect='auto')
ax.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_title('Hidden State Norm Difference\nAverage Activation Intervention',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(num_positions))
ax.set_yticks(range(num_positions))
ax.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax.set_yticklabels([f'Int@{i}' for i in range(num_positions)])

for i in range(num_positions):
    for j in range(num_positions):
        value = norm_diff[i, j]
        color = 'white' if value > 3 else 'black'
        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
               color=color, fontsize=10, fontweight='bold')

plt.colorbar(im2, ax=ax, label='Avg Norm Δ')

# 3. Number Decoding Rate
ax = axes[2]
im3 = ax.imshow(number_decode_rate, cmap='Greens', vmin=0, vmax=100, aspect='auto')
ax.set_xlabel('Target Position', fontsize=12, fontweight='bold')
ax.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
ax.set_title('Number Decoding Rate (%)\nAfter Intervention',
             fontsize=14, fontweight='bold')
ax.set_xticks(range(num_positions))
ax.set_yticks(range(num_positions))
ax.set_xticklabels([f'P{i}' for i in range(num_positions)])
ax.set_yticklabels([f'Int@{i}' for i in range(num_positions)])

for i in range(num_positions):
    for j in range(num_positions):
        value = number_decode_rate[i, j]
        color = 'white' if value > 50 else 'black'
        ax.text(j, i, f'{value:.0f}%', ha='center', va='center',
               color=color, fontsize=10, fontweight='bold')

plt.colorbar(im3, ax=ax, label='Number Decoding Rate (%)')

plt.tight_layout()
plt.savefig('avg_intervention_simple.png', dpi=300, bbox_inches='tight')
print("Saved: avg_intervention_simple.png")

# Print summary statistics
print("\n" + "="*80)
print("AVERAGE ACTIVATION INTERVENTION SUMMARY")
print("="*80)

print("\nStrongest Cascade Effects (Token Change > 40%):")
for i in range(num_positions):
    for j in range(num_positions):
        if token_change[i, j] > 40:
            print(f"  Position {i} -> Position {j}: {token_change[i, j]:.0f}% token change, "
                  f"{norm_diff[i, j]:.2f} norm diff, {number_decode_rate[i, j]:.0f}% numbers")

print("\nRobust Positions (0% token change across all interventions):")
for j in range(num_positions):
    if np.all(token_change[:, j] == 0):
        print(f"  Position {j}: Completely robust")

print("\nNumber Decoding Patterns:")
for i in range(num_positions):
    baseline_numbers = number_decode_rate[i, i]  # Self-intervention (closest to baseline)
    print(f"  Position {i}: {baseline_numbers:.0f}% decode to numbers")

print("\nAverage Metrics:")
print(f"  Overall token change rate: {np.mean(token_change):.1f}%")
print(f"  Overall norm difference: {np.mean(norm_diff):.2f}")
print(f"  Overall number decoding: {np.mean(number_decode_rate):.0f}%")

print("\n" + "="*80)
