"""
Create visualizations from CODI circuit analysis results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_results():
    """Load all analysis results"""
    results_dir = Path("circuit_analysis_results")

    with open(results_dir / "circuit_analysis.json", 'r') as f:
        circuit_data = json.load(f)

    with open(results_dir / "intervention_propagation.json", 'r') as f:
        intervention_data = json.load(f)

    return circuit_data, intervention_data


def visualize_pathway_contributions(circuit_data):
    """Visualize projection vs attention contributions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    positions = [p['position'] for p in circuit_data['cot_positions']]
    tokens = [p['token'] for p in circuit_data['cot_positions']]

    # 1. Projection vs Attention Contribution
    ax = axes[0, 0]

    projection_delta = [p.get('projection_delta', 0) for p in circuit_data['cot_positions']]
    attention_contrib = [p.get('attention_contribution', 0) for p in circuit_data['cot_positions']]

    width = 0.35
    x = np.arange(len(positions))

    ax.bar(x - width/2, projection_delta, width, label='Direct Projection', alpha=0.8, color='#e74c3c')
    ax.bar(x + width/2, attention_contrib, width, label='Attention Mechanism', alpha=0.8, color='#3498db')

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contribution Magnitude (L2 Norm)', fontsize=12, fontweight='bold')
    ax.set_title('Information Flow: Projection vs Attention', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{i}\n{tokens[i]}' for i in positions], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add average lines
    avg_proj = np.mean(projection_delta)
    avg_attn = np.mean([a for a in attention_contrib if a > 0])
    ax.axhline(y=avg_proj, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg Proj: {avg_proj:.1f}')
    ax.axhline(y=avg_attn, color='#3498db', linestyle='--', linewidth=2, alpha=0.7, label=f'Avg Attn: {avg_attn:.1f}')

    # 2. Projection/Attention Ratio
    ax = axes[0, 1]

    ratios = [p.get('proj_vs_attn_ratio', 0) for p in circuit_data['cot_positions'][1:]]

    ax.plot(positions[1:], ratios, marker='o', linewidth=3, markersize=10, color='#9b59b6')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Equal Contribution')

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Projection / Attention Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Relative Pathway Importance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    ax.text(0.5, 0.95, 'Ratio > 1: Projection dominates\nRatio < 1: Attention dominates\nRatio ≈ 1: Balanced',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 3. Layer Processing Depth
    ax = axes[1, 0]

    layer_deltas = [p['first_to_last_delta'] for p in circuit_data['cot_positions']]

    ax.plot(positions, layer_deltas, marker='s', linewidth=3, markersize=10, color='#27ae60')

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Processing Δ (First → Last Layer)', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Processing Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate tokens
    for i, (pos, delta, tok) in enumerate(zip(positions, layer_deltas, tokens)):
        ax.annotate(tok, (pos, delta), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')

    avg_proj = np.mean(projection_delta)
    avg_attn = np.mean([a for a in attention_contrib if a > 0])
    ratio = avg_proj / avg_attn

    summary_text = f"""
PATHWAY CONTRIBUTION SUMMARY

Direct Projection:
  • Average Contribution: {avg_proj:.2f}
  • Min: {min(projection_delta):.2f}
  • Max: {max(projection_delta):.2f}

Attention Mechanism:
  • Average Contribution: {avg_attn:.2f}
  • Min: {min([a for a in attention_contrib if a > 0]):.2f}
  • Max: {max(attention_contrib):.2f}

Overall Balance:
  • Projection/Attention Ratio: {ratio:.3f}x
  • Dominant Pathway: {"Projection" if ratio > 1.05 else "Balanced"}

CoT Sequence:
  {' → '.join(tokens)}
"""

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig('circuit_analysis_results/pathway_contributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: circuit_analysis_results/pathway_contributions.png")
    plt.close()


def visualize_intervention_cascade(intervention_data):
    """Visualize intervention propagation and cascade effects"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Extract data
    baseline = intervention_data[0]
    baseline_tokens = [t['token'] for t in baseline['tokens']]

    # 1. Token Cascade Heatmap
    ax = axes[0, 0]

    token_matrix = []
    int_positions = []

    for result in intervention_data:
        int_pos = result['intervention_position']
        tokens = [t['token'] for t in result['tokens']]

        # Binary: 1 if changed from baseline, 0 if same
        changes = [1 if tokens[i] != baseline_tokens[i] else 0 for i in range(len(tokens))]
        token_matrix.append(changes)
        int_positions.append('Baseline' if int_pos == -1 else f'Int@{int_pos}')

    im = ax.imshow(token_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

    # Annotate with actual tokens
    for i, result in enumerate(intervention_data):
        tokens = [t['token'] for t in result['tokens']]
        for j, tok in enumerate(tokens):
            color = 'red' if token_matrix[i][j] == 1 else 'black'
            weight = 'bold' if token_matrix[i][j] == 1 else 'normal'
            ax.text(j, i, tok, ha="center", va="center", color=color,
                   fontsize=10, fontweight=weight)

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Intervention Position', fontsize=12, fontweight='bold')
    ax.set_title('Token Cascade: Red = Changed from Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(range(7))
    ax.set_xticklabels([f'Pos {i}' for i in range(7)])
    ax.set_yticks(range(len(int_positions)))
    ax.set_yticklabels(int_positions)

    plt.colorbar(im, ax=ax, label='Token Changed')

    # 2. Cascade Length Analysis
    ax = axes[0, 1]

    cascade_lengths = []
    positions = []

    for result in intervention_data[1:]:  # Skip baseline
        int_pos = result['intervention_position']
        tokens = [t['token'] for t in result['tokens']]

        # Find affected positions
        affected = [i for i in range(len(tokens)) if tokens[i] != baseline_tokens[i]]

        if len(affected) > 0:
            cascade_length = max(affected) - int_pos + 1
        else:
            cascade_length = 0

        cascade_lengths.append(cascade_length)
        positions.append(int_pos)

    colors = ['#e74c3c' if c > 0 else '#95a5a6' for c in cascade_lengths]
    bars = ax.bar(positions, cascade_lengths, alpha=0.8, color=colors)

    ax.set_xlabel('Intervention Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cascade Length (Positions)', fontsize=12, fontweight='bold')
    ax.set_title('How Far Do Interventions Propagate?', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate bars
    for bar, length in zip(bars, cascade_lengths):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # 3. Number of Tokens Changed
    ax = axes[1, 0]

    tokens_changed = []

    for result in intervention_data[1:]:
        tokens = [t['token'] for t in result['tokens']]
        changes = sum(1 for i, t in enumerate(tokens) if t != baseline_tokens[i])
        tokens_changed.append(changes)

    colors = ['#2ecc71' if c == 0 else '#e67e22' if c <= 1 else '#e74c3c' for c in tokens_changed]
    bars = ax.bar(positions, tokens_changed, alpha=0.8, color=colors)

    ax.set_xlabel('Intervention Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Tokens Changed', fontsize=12, fontweight='bold')
    ax.set_title('Intervention Impact Strength', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='No Effect (Robust)'),
        Patch(facecolor='#e67e22', label='Weak Effect (1 token)'),
        Patch(facecolor='#e74c3c', label='Strong Effect (2+ tokens)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # 4. Robustness Analysis
    ax = axes[1, 1]
    ax.axis('off')

    # Categorize positions
    robust_positions = [p for p, c in zip(positions, tokens_changed) if c == 0]
    vulnerable_positions = [p for p, c in zip(positions, tokens_changed) if c > 1]
    weak_positions = [p for p, c in zip(positions, tokens_changed) if c == 1]

    analysis_text = f"""
INTERVENTION PROPAGATION ANALYSIS

Baseline Sequence:
  {' → '.join(baseline_tokens)}

Robustness Zones:

🟢 ROBUST POSITIONS (No Effect):
  Positions: {robust_positions}
  • Interventions have NO effect
  • Model self-corrects
  • Strong computational stability

🔴 VULNERABLE POSITIONS (Strong Effect):
  Positions: {vulnerable_positions}
  • Changes propagate downstream
  • Cascade length: {max([cascade_lengths[i] for i in range(len(positions)) if positions[i] in vulnerable_positions], default=0)} steps
  • Critical for computation

🟡 WEAK EFFECT POSITIONS:
  Positions: {weak_positions if weak_positions else 'None'}
  • Limited local effects only

Key Finding:
  Positions {robust_positions} show COMPLETE ROBUSTNESS
  → Model has self-correction mechanism!
"""

    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.2))

    plt.tight_layout()
    plt.savefig('circuit_analysis_results/intervention_cascade.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: circuit_analysis_results/intervention_cascade.png")
    plt.close()


def create_circuit_diagram(circuit_data, intervention_data):
    """Create a conceptual circuit diagram"""
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    baseline_tokens = [t['token'] for t in intervention_data[0]['tokens']]

    # Determine robustness of each position
    robust_positions = set()
    for result in intervention_data[1:]:
        int_pos = result['intervention_position']
        tokens = [t['token'] for t in result['tokens']]

        # If intervening at this position has no effect, it's robust
        if tokens == baseline_tokens:
            robust_positions.add(int_pos)

    # Draw CoT positions
    y = 5
    x_start = 1
    x_spacing = 1.2

    for i, token in enumerate(baseline_tokens):
        x = x_start + i * x_spacing

        # Color based on robustness
        if i in robust_positions:
            color = '#2ecc71'  # Green for robust
            label = 'ROBUST'
        elif i <= 1:
            color = '#e74c3c'  # Red for vulnerable
            label = 'VULNERABLE'
        else:
            color = '#f39c12'  # Orange for other
            label = 'MIXED'

        # Draw box
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x-0.25, y-0.3), 0.5, 0.6,
                            boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color,
                            linewidth=2, alpha=0.7)
        ax.add_patch(box)

        # Add token text
        ax.text(x, y+0.1, token, ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.text(x, y-0.15, f'P{i}', ha='center', va='center',
               fontsize=10, style='italic')

        # Add label
        ax.text(x, y-0.5, label, ha='center', va='center',
               fontsize=8, color=color, fontweight='bold')

        # Draw arrows (projection)
        if i < len(baseline_tokens) - 1:
            next_x = x_start + (i + 1) * x_spacing
            ax.arrow(x + 0.25, y, next_x - x - 0.5, 0,
                    head_width=0.15, head_length=0.1, fc='red', ec='red',
                    linewidth=2, alpha=0.8)
            ax.text((x + next_x) / 2, y - 0.7, 'Projection',
                   ha='center', fontsize=9, color='red', fontweight='bold')

    # Title
    ax.text(5, 9, 'CODI Chain-of-Thought Circuit',
           ha='center', fontsize=20, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Robust (Self-Correcting)'),
        Patch(facecolor='#e74c3c', label='Vulnerable (Cascade Effect)'),
        Patch(facecolor='#f39c12', label='Mixed Robustness')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Add zone annotations
    ax.text(1.5, 7.5, '← SETUP ZONE', fontsize=12, color='#e74c3c', fontweight='bold')
    ax.text(4, 7.5, '← ROBUST ZONE', fontsize=12, color='#2ecc71', fontweight='bold')
    ax.text(6.5, 7.5, '← OUTPUT ZONE', fontsize=12, color='#f39c12', fontweight='bold')

    plt.tight_layout()
    plt.savefig('circuit_analysis_results/circuit_diagram.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: circuit_analysis_results/circuit_diagram.png")
    plt.close()


def main():
    print("="*80)
    print("Creating Circuit Analysis Visualizations")
    print("="*80)

    # Load data
    print("\nLoading results...")
    circuit_data, intervention_data = load_results()
    print("✓ Data loaded")

    # Create visualizations
    print("\n1. Creating pathway contribution visualizations...")
    visualize_pathway_contributions(circuit_data)

    print("\n2. Creating intervention cascade visualizations...")
    visualize_intervention_cascade(intervention_data)

    print("\n3. Creating circuit diagram...")
    create_circuit_diagram(circuit_data, intervention_data)

    print("\n" + "="*80)
    print("All visualizations created!")
    print("="*80)
    print("\nGenerated files in circuit_analysis_results/:")
    print("  1. pathway_contributions.png")
    print("  2. intervention_cascade.png")
    print("  3. circuit_diagram.png")


if __name__ == "__main__":
    main()
