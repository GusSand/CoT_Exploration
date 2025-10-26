#!/usr/bin/env python3
"""
Visualize Discretization Methods Comparison Results
Creates bar plots comparing accuracy across different discretization methods
for both CODI-GPT2 and CODI-Llama models
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpt2_results", type=str,
                   default="/workspace/CoT_Exploration/src/experiments/custom_dataset_results_gpt2/final_results.json",
                   help="Path to GPT-2 results JSON")
parser.add_argument("--llama_results", type=str,
                   default="/workspace/CoT_Exploration/src/experiments/custom_dataset_results_llama/final_results.json",
                   help="Path to Llama results JSON")
parser.add_argument("--output_dir", type=str,
                   default="/workspace/CoT_Exploration/src/experiments/discretization_methods_comparison/figures",
                   help="Output directory for figures")
args = parser.parse_args()

# Load results
print("Loading results...")
with open(args.gpt2_results, 'r') as f:
    gpt2_data = json.load(f)

with open(args.llama_results, 'r') as f:
    llama_data = json.load(f)

os.makedirs(args.output_dir, exist_ok=True)

# Extract accuracy data
modes = ['vanilla', 'full', 'posthoc']
mode_labels = ['Vanilla\n(Continuous)', 'Full\nDiscretization', 'Post-hoc\nDiscretization']

gpt2_accuracies = [gpt2_data['stats'][mode]['accuracy'] for mode in modes]
llama_accuracies = [llama_data['stats'][mode]['accuracy'] for mode in modes]

# Print summary
print("\n" + "="*80)
print("DISCRETIZATION METHODS COMPARISON - SUMMARY")
print("="*80)
print(f"\nDataset: {llama_data['metadata']['dataset']}")
print(f"Total examples evaluated: {gpt2_data['stats']['vanilla']['total']}")
print(f"\nCODI-GPT2 ({gpt2_data['metadata']['model_path']}):")
for mode, acc in zip(modes, gpt2_accuracies):
    print(f"  {mode:12s}: {acc:6.2f}%")

print(f"\nCODI-Llama ({llama_data['metadata']['model_path']}):")
for mode, acc in zip(modes, llama_accuracies):
    print(f"  {mode:12s}: {acc:6.2f}%")
print("="*80)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))

# 1. Side-by-side comparison
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(modes))
width = 0.35

bars1 = ax1.bar(x - width/2, gpt2_accuracies, width, label='CODI-GPT2',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, llama_accuracies, width, label='CODI-Llama3.2-1B',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Discretization Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Discretization Methods Comparison\nCODI-GPT2 vs CODI-Llama3.2-1B',
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(mode_labels, fontsize=10)
ax1.legend(fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(max(gpt2_accuracies), max(llama_accuracies)) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. GPT-2 only
ax2 = plt.subplot(2, 2, 2)
bars_gpt2 = ax2.bar(mode_labels, gpt2_accuracies, color='#3498db', alpha=0.8,
                     edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('CODI-GPT2 Performance', fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(gpt2_accuracies) * 1.15)

for bar in bars_gpt2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Llama only
ax3 = plt.subplot(2, 2, 3)
bars_llama = ax3.bar(mode_labels, llama_accuracies, color='#e74c3c', alpha=0.8,
                      edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax3.set_title('CODI-Llama3.2-1B Performance', fontsize=14, fontweight='bold', pad=20)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(llama_accuracies) * 1.15)

for bar in bars_llama:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 4. Relative performance degradation
ax4 = plt.subplot(2, 2, 4)
gpt2_degradation = [(gpt2_accuracies[0] - acc) / gpt2_accuracies[0] * 100 for acc in gpt2_accuracies]
llama_degradation = [(llama_accuracies[0] - acc) / llama_accuracies[0] * 100 for acc in llama_accuracies]

x = np.arange(len(modes))
width = 0.35
bars1 = ax4.bar(x - width/2, gpt2_degradation, width, label='CODI-GPT2',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, llama_degradation, width, label='CODI-Llama3.2-1B',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_xlabel('Discretization Method', fontsize=12, fontweight='bold')
ax4.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
ax4.set_title('Relative Performance Degradation\n(vs Vanilla Continuous)',
              fontsize=14, fontweight='bold', pad=20)
ax4.set_xticks(x)
ax4.set_xticklabels(mode_labels, fontsize=10)
ax4.legend(fontsize=11, framealpha=0.9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')

plt.tight_layout()
output_path = os.path.join(args.output_dir, 'discretization_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Create a simpler single comparison plot
fig2, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(modes))
width = 0.35

bars1 = ax.bar(x - width/2, gpt2_accuracies, width, label='CODI-GPT2',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, llama_accuracies, width, label='CODI-Llama3.2-1B',
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Discretization Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Chain-of-Thought Discretization Methods Comparison\nEvaluated on CoT-Dependent GSM8K Problems',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(mode_labels, fontsize=12)
ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(max(gpt2_accuracies), max(llama_accuracies)) * 1.15)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
output_path_simple = os.path.join(args.output_dir, 'discretization_comparison_simple.png')
plt.savefig(output_path_simple, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path_simple}")

# Create timing comparison
fig3, ax = plt.subplots(figsize=(12, 7))

gpt2_times = [gpt2_data['stats'][mode]['avg_time_per_example'] * 1000 for mode in modes]  # Convert to ms
llama_times = [llama_data['stats'][mode]['avg_time_per_example'] * 1000 for mode in modes]

x = np.arange(len(modes))
width = 0.35

bars1 = ax.bar(x - width/2, gpt2_times, width, label='CODI-GPT2',
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, llama_times, width, label='CODI-Llama3.2-1B',
               color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Discretization Method', fontsize=14, fontweight='bold')
ax.set_ylabel('Inference Time per Example (ms)', fontsize=14, fontweight='bold')
ax.set_title('Inference Speed Comparison Across Discretization Methods',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(mode_labels, fontsize=12)
ax.legend(fontsize=12, framealpha=0.9, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}ms',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
output_path_timing = os.path.join(args.output_dir, 'timing_comparison.png')
plt.savefig(output_path_timing, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path_timing}")

print("\nVisualization complete!")
print(f"\nAll figures saved to: {args.output_dir}")
