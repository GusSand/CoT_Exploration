"""
Visualize Feature Hierarchy Results.

Creates comprehensive visualizations of the feature hierarchy investigation:
1. Specialization vs Activation Frequency scatter plot
2. Feature type distribution pie charts
3. Impact vs Activation frequency for top features
4. Specialized feature examples

Usage:
    python visualize_results.py --output_dir visualizations/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_all_results():
    """Load all analysis results."""
    base_dir = Path('src/experiments/llama_sae_hierarchy')

    # Load feature labels (top 20)
    with open(base_dir / 'feature_labels_layer14_pos3.json', 'r') as f:
        labels_data = json.load(f)

    # Load activation analyses
    analyses = {}
    for fname in ['activation_analysis_layer14_pos3_rank50-200.json',
                  'activation_analysis_layer14_pos3_rank400-512.json',
                  'activation_analysis_layer3_pos3_rank20-100.json']:
        path = base_dir / fname
        if path.exists():
            with open(path, 'r') as f:
                key = fname.replace('activation_analysis_', '').replace('.json', '')
                analyses[key] = json.load(f)

    # Load validation results
    with open(base_dir / 'validation_results_layer14_pos3.json', 'r') as f:
        validation_data = json.load(f)

    return labels_data, analyses, validation_data


def plot_specialization_vs_frequency(labels_data, analyses, output_dir):
    """
    Plot: Specialization type vs Activation Frequency.

    Shows that specialized features only appear at low activation frequencies.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all features
    all_features = []

    # Top 20 from labels
    for feat in labels_data['features']:
        all_features.append({
            'activation_freq': feat['activation_freq'],
            'type': feat['interpretation']['type'],
            'feature_id': feat['feature_id'],
            'rank': feat['rank']
        })

    # From analyses
    for key, data in analyses.items():
        for feat in data['all_features']:
            spec_type = feat.get('specialization', {}).get('type', 'general')
            all_features.append({
                'activation_freq': feat['activation_freq'],
                'type': spec_type,
                'feature_id': feat['feature_id'],
                'rank': feat['rank']
            })

    # Separate by type
    general = [f for f in all_features if f['type'] == 'general']
    mixed = [f for f in all_features if f['type'] == 'mixed']
    operation = [f for f in all_features if f['type'] == 'operation-specialized']
    value = [f for f in all_features if f['type'] == 'value-specialized']
    highly_spec = [f for f in all_features if f['type'] == 'highly-specialized']

    # Plot
    if general:
        x = [f['activation_freq'] * 100 for f in general]
        y = [f['rank'] for f in general]
        ax.scatter(x, y, s=50, alpha=0.6, c='lightblue', label=f'General ({len(general)})', edgecolors='blue', linewidths=1)

    if mixed:
        x = [f['activation_freq'] * 100 for f in mixed]
        y = [f['rank'] for f in mixed]
        ax.scatter(x, y, s=50, alpha=0.6, c='lightgreen', label=f'Mixed ({len(mixed)})', edgecolors='green', linewidths=1)

    if operation:
        x = [f['activation_freq'] * 100 for f in operation]
        y = [f['rank'] for f in operation]
        ax.scatter(x, y, s=200, alpha=0.9, c='orange', marker='*', label=f'Operation-Specialized ({len(operation)})', edgecolors='red', linewidths=2)

        # Annotate
        for f in operation:
            ax.annotate(f"F{f['feature_id']}",
                       (f['activation_freq'] * 100, f['rank']),
                       xytext=(10, 5), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    if highly_spec:
        x = [f['activation_freq'] * 100 for f in highly_spec]
        y = [f['rank'] for f in highly_spec]
        ax.scatter(x, y, s=250, alpha=0.9, c='red', marker='D', label=f'Highly-Specialized ({len(highly_spec)})', edgecolors='darkred', linewidths=2)

        # Annotate
        for f in highly_spec:
            ax.annotate(f"F{f['feature_id']}",
                       (f['activation_freq'] * 100, f['rank']),
                       xytext=(10, -10), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', alpha=0.7))

    ax.set_xlabel('Activation Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Feature Rank (1 = most frequent)', fontsize=14, fontweight='bold')
    ax.set_title('Feature Specialization vs Activation Frequency\n(Specialized features only appear at low frequencies)',
                 fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper right')
    ax.invert_yaxis()  # Lower rank = higher frequency

    plt.tight_layout()
    output_path = output_dir / 'specialization_vs_frequency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created: {output_path}")


def plot_feature_type_distribution(labels_data, analyses, output_dir):
    """
    Plot: Feature type distribution pie charts.

    Shows breakdown by frequency range.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Count by analysis
    counts = {
        'Top 20 (>87%)': {'general': 0, 'mixed': 0, 'operation': 0, 'value': 0, 'highly-spec': 0},
        'Rank 50-200 (11-57%)': {'general': 0, 'mixed': 0, 'operation': 0, 'value': 0, 'highly-spec': 0},
        'Rank 400-512 (0.1-3%)': {'general': 0, 'mixed': 0, 'operation': 0, 'value': 0, 'highly-spec': 0},
        'Layer 3 (42-96%)': {'general': 0, 'mixed': 0, 'operation': 0, 'value': 0, 'highly-spec': 0}
    }

    # Count top 20
    for feat in labels_data['features']:
        ftype = feat['interpretation']['type']
        if ftype in ['mixed', 'general']:
            counts['Top 20 (>87%)'][ftype] += 1

    # Count from analyses
    mapping = {
        'layer14_pos3_rank50-200': 'Rank 50-200 (11-57%)',
        'layer14_pos3_rank400-512': 'Rank 400-512 (0.1-3%)',
        'layer3_pos3_rank20-100': 'Layer 3 (42-96%)'
    }

    for key, data in analyses.items():
        category = mapping.get(key)
        if not category:
            continue

        for feat in data['all_features']:
            spec_type = feat.get('specialization', {}).get('type', 'general')
            if spec_type == 'operation-specialized':
                counts[category]['operation'] += 1
            elif spec_type == 'highly-specialized':
                counts[category]['highly-spec'] += 1
            elif spec_type == 'value-specialized':
                counts[category]['value'] += 1
            elif spec_type in ['general', 'mixed']:
                counts[category]['general'] += 1

    # Plot pies
    categories = ['Top 20 (>87%)', 'Rank 50-200 (11-57%)', 'Rank 400-512 (0.1-3%)', 'Layer 3 (42-96%)']
    colors = {'general': 'lightblue', 'mixed': 'lightgreen', 'operation': 'orange', 'value': 'purple', 'highly-spec': 'red'}

    for idx, category in enumerate(categories):
        ax = axes[idx // 2, idx % 2]

        data = counts[category]
        labels = []
        sizes = []
        plot_colors = []

        for key, count in data.items():
            if count > 0:
                labels.append(f"{key.capitalize()}\n({count})")
                sizes.append(count)
                plot_colors.append(colors[key])

        if sizes:
            ax.pie(sizes, labels=labels, colors=plot_colors, autopct='%1.1f%%',
                  startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})

        total = sum(data.values())
        specialized = data['operation'] + data['value'] + data['highly-spec']
        spec_pct = 100 * specialized / total if total > 0 else 0

        ax.set_title(f"{category}\n({total} features, {spec_pct:.1f}% specialized)",
                    fontsize=12, fontweight='bold')

    plt.suptitle('Feature Type Distribution by Activation Frequency Range',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'feature_type_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created: {output_path}")


def plot_ablation_impact(validation_data, output_dir):
    """
    Plot: Ablation impact vs activation frequency.

    Shows that high-frequency features have measurable impact.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    general_features = validation_data['general_features']

    x = [f['activation_freq'] * 100 for f in general_features]
    y = [f['mean_impact'] for f in general_features]

    # Color by classification
    colors = []
    for f in general_features:
        if 'HIGH' in f['classification']:
            colors.append('green')
        elif 'MEDIUM' in f['classification']:
            colors.append('orange')
        else:
            colors.append('red')

    # Plot
    scatter = ax.scatter(x, y, s=200, c=colors, alpha=0.7, edgecolors='black', linewidths=2)

    # Annotate
    for f in general_features:
        ax.annotate(f"F{f['feature_id']}\n(R{f['rank']})",
                   (f['activation_freq'] * 100, f['mean_impact']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, fontweight='bold')

    # Threshold lines
    ax.axhline(y=0.1, color='green', linestyle='--', linewidth=2, alpha=0.5, label='High impact threshold (>0.1)')
    ax.axhline(y=0.01, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Medium impact threshold (>0.01)')

    ax.set_xlabel('Activation Frequency (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Ablation Impact (abs diff)', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Impact vs Activation Frequency (Top 10 Features)\n(All features show measurable impact)',
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='HIGH impact (>0.1)'),
        Patch(facecolor='orange', edgecolor='black', label='MEDIUM impact (0.01-0.1)'),
        Patch(facecolor='red', edgecolor='black', label='LOW impact (<0.01)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    output_path = output_dir / 'ablation_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created: {output_path}")


def create_specialized_features_summary(analyses, output_dir):
    """
    Create text summary of specialized features with examples.
    """
    output_path = output_dir / 'specialized_features_summary.txt'

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPECIALIZED FEATURES FOUND\n")
        f.write("=" * 80 + "\n\n")

        # Get specialized features from rare feature analysis
        rare_analysis = analyses.get('layer14_pos3_rank400-512')
        if not rare_analysis:
            f.write("No specialized features analysis found.\n")
            return

        specialized = rare_analysis.get('specialized_features', [])

        f.write(f"Total: {len(specialized)} specialized features\n\n")

        for i, feat in enumerate(specialized, 1):
            f.write(f"{i}. Feature {feat['feature_id']} (Rank {feat['rank']})\n")
            f.write(f"   Type: {feat['specialization']['type']}\n")
            f.write(f"   Description: {feat['specialization']['description']}\n")
            f.write(f"   Activation Frequency: {feat['activation_freq']*100:.3f}%\n")
            f.write(f"   Mean Impact: {feat.get('mean_impact', 'N/A')}\n")

            if 'top_samples' in feat:
                f.write(f"   \n")
                f.write(f"   Top Activating Samples:\n")
                for j, sample in enumerate(feat['top_samples'][:5], 1):
                    cot = sample['cot'][:80] + ('...' if len(sample['cot']) > 80 else '')
                    f.write(f"     {j}. {cot}\n")

            f.write(f"\n")

    print(f"✓ Created: {output_path}")


def create_summary_statistics(labels_data, analyses, validation_data, output_dir):
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Features analyzed by range
    ax = axes[0, 0]
    ranges = ['Top 20\n(>87%)', 'Rank 50-200\n(11-57%)', 'Rank 400-512\n(0.1-3%)', 'Layer 3\n(42-96%)']
    counts = [20, 151, 109, 81]
    colors_bar = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']

    bars = ax.bar(ranges, counts, color=colors_bar, edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Features Analyzed by Frequency Range', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. Specialized vs General
    ax = axes[0, 1]
    total = 361
    specialized = 6
    general = total - specialized

    labels_pie = [f'General\n({general})', f'Specialized\n({specialized})']
    sizes = [general, specialized]
    colors_pie = ['lightblue', 'red']
    explode = (0, 0.1)

    ax.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
          autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title(f'Overall Distribution\n({total} features analyzed)', fontsize=13, fontweight='bold')

    # 3. Validation results
    ax = axes[1, 0]
    general_feats = validation_data['general_features']

    high = sum(1 for f in general_feats if 'HIGH' in f['classification'])
    medium = sum(1 for f in general_feats if 'MEDIUM' in f['classification'])
    low = sum(1 for f in general_feats if 'LOW' in f['classification'])

    impact_labels = ['HIGH\nImpact', 'MEDIUM\nImpact', 'LOW\nImpact']
    impact_counts = [high, medium, low]
    impact_colors = ['green', 'orange', 'red']

    bars = ax.bar(impact_labels, impact_counts, color=impact_colors, edgecolor='black', linewidth=2, alpha=0.7)
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Validation Results (Top 10 General Features)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, impact_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Specialized feature types
    ax = axes[1, 1]

    spec_types = ['Operation\nSpecialized', 'Highly\nSpecialized']
    spec_counts = [3, 3]
    spec_colors = ['orange', 'red']

    bars = ax.bar(spec_types, spec_counts, color=spec_colors, edgecolor='black', linewidth=2, alpha=0.7)
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Specialized Feature Types\n(6 total found)', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, count in zip(bars, spec_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('LLaMA SAE Feature Hierarchy Investigation - Summary Statistics',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'summary_statistics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Feature Hierarchy Results')
    parser.add_argument('--output_dir', type=str, default='src/experiments/llama_sae_hierarchy/visualizations',
                       help='Output directory for visualizations')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Creating Visualizations")
    print(f"{'='*80}\n")

    # Load all results
    print("Loading results...")
    labels_data, analyses, validation_data = load_all_results()
    print(f"  ✓ Loaded all results\n")

    # Create visualizations
    print("Creating plots...\n")

    plot_specialization_vs_frequency(labels_data, analyses, output_dir)
    plot_feature_type_distribution(labels_data, analyses, output_dir)
    plot_ablation_impact(validation_data, output_dir)
    create_summary_statistics(labels_data, analyses, validation_data, output_dir)
    create_specialized_features_summary(analyses, output_dir)

    print(f"\n{'='*80}")
    print(f"All visualizations created in: {output_dir}")
    print(f"{'='*80}\n")

    print("Files created:")
    print("  1. specialization_vs_frequency.png - Main finding visualization")
    print("  2. feature_type_distribution.png - Breakdown by frequency range")
    print("  3. ablation_impact.png - Validation results")
    print("  4. summary_statistics.png - Overall statistics")
    print("  5. specialized_features_summary.txt - Detailed feature examples\n")


if __name__ == '__main__':
    main()
