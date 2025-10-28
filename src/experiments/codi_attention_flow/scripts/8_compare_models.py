#!/usr/bin/env python3
"""
Model Comparison - Story 2.6

Compare LLaMA vs GPT-2 attention flow patterns and critical heads.

Usage:
    python 8_compare_models.py

Output:
    ../results/model_comparison.json
    ../figures/model_comparison.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compare_models() -> None:
    """
    Compare LLaMA vs GPT-2 attention flow patterns.
    """
    print("=" * 80)
    print("MODEL COMPARISON - Story 2.6")
    print("=" * 80)

    # Paths
    results_dir = Path(__file__).parent.parent / 'results'
    figures_dir = Path(__file__).parent.parent / 'figures'

    # Load data for both models
    models = ['llama', 'gpt2']
    data = {}

    for model in models:
        print(f"\nLoading {model.upper()} data...")
        model_dir = results_dir / model

        # Load rankings
        df = pd.read_csv(model_dir / 'ranked_heads.csv')
        data[model] = {
            'ranked_heads': df,
            'summary': json.load(open(model_dir / 'attention_summary.json')),
            'n_layers': len(np.load(model_dir / 'attention_patterns_avg.npy')),
            'n_heads': len(np.load(model_dir / 'attention_patterns_avg.npy')[0])
        }
        print(f"  ✓ Loaded {len(df)} heads")

    # Create comparison
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON")
    print("=" * 80)

    comparison = {
        'llama': {
            'model_size': 'LLaMA 1B',
            'n_layers': data['llama']['n_layers'],
            'n_heads': data['llama']['n_heads'],
            'total_heads': data['llama']['n_layers'] * data['llama']['n_heads']
        },
        'gpt2': {
            'model_size': 'GPT-2 124M',
            'n_layers': data['gpt2']['n_layers'],
            'n_heads': data['gpt2']['n_heads'],
            'total_heads': data['gpt2']['n_layers'] * data['gpt2']['n_heads']
        }
    }

    print(f"\nLLaMA: {comparison['llama']['n_layers']} layers × "
          f"{comparison['llama']['n_heads']} heads = "
          f"{comparison['llama']['total_heads']} total")
    print(f"GPT-2: {comparison['gpt2']['n_layers']} layers × "
          f"{comparison['gpt2']['n_heads']} heads = "
          f"{comparison['gpt2']['total_heads']} total")

    # Hub analysis comparison
    print("\n" + "=" * 80)
    print("HUB ANALYSIS COMPARISON")
    print("=" * 80)

    for model in models:
        hub_data = data[model]['summary']['hub_analysis']
        print(f"\n{model.upper()}:")
        print(f"  Hub position: {hub_data['hub_position']}")
        print(f"  Hub score: {hub_data['hub_score']:.3f}")
        print(f"  Hub ratio: {hub_data['hub_ratio']:.2f}×")
        print(f"  Strong hub: {'✓' if hub_data['is_strong_hub'] else '✗'}")

    # Critical heads comparison
    print("\n" + "=" * 80)
    print("CRITICAL HEADS COMPARISON")
    print("=" * 80)

    for model in models:
        df = data[model]['ranked_heads']
        top_head = df.iloc[0]
        print(f"\n{model.upper()} - Top Critical Head:")
        print(f"  L{top_head['layer']}H{top_head['head']}")
        print(f"  Type: {top_head['functional_type']}")
        print(f"  Composite: {top_head['composite_score']:.3f}")
        print(f"  Flow: {top_head['flow_score']:.3f}")
        print(f"  Hub: {top_head['hub_score']:.3f}")
        print(f"  Skip: {top_head['skip_score']:.3f}")

    # Functional type distribution comparison
    print("\n" + "=" * 80)
    print("FUNCTIONAL TYPE DISTRIBUTION (Top 20)")
    print("=" * 80)

    for model in models:
        df = data[model]['ranked_heads'].head(20)
        type_counts = df['functional_type'].value_counts()
        print(f"\n{model.upper()}:")
        for ftype, count in type_counts.items():
            print(f"  {ftype}: {count}")

    # Layer distribution comparison
    print("\n" + "=" * 80)
    print("LAYER DISTRIBUTION (Top 20)")
    print("=" * 80)

    for model in models:
        df = data[model]['ranked_heads'].head(20)
        layer_counts = df['layer_type'].value_counts()
        print(f"\n{model.upper()}:")
        for ltype, count in layer_counts.items():
            print(f"  {ltype.capitalize()}: {count}")

    # Create visualizations
    print("\n" + "=" * 80)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Hub scores comparison
    ax1 = fig.add_subplot(gs[0, :])
    positions = np.arange(6)
    width = 0.35

    llama_hubs = data['llama']['summary']['hub_analysis']['all_hub_scores']
    gpt2_hubs = data['gpt2']['summary']['hub_analysis']['all_hub_scores']

    ax1.bar(positions - width/2, llama_hubs, width, label='LLaMA', alpha=0.8, color='#e74c3c')
    ax1.bar(positions + width/2, gpt2_hubs, width, label='GPT-2', alpha=0.8, color='#3498db')
    ax1.axhline(y=0.1667, color='gray', linestyle='--', alpha=0.5, label='Uniform baseline')
    ax1.set_xlabel('Continuous Thought Position', fontsize=12)
    ax1.set_ylabel('Incoming Attention (Hub Score)', fontsize=12)
    ax1.set_title('Hub Scores Comparison - LLaMA vs GPT-2', fontsize=14, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f'CT{i}' for i in range(6)])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Top 20 composite scores
    ax2 = fig.add_subplot(gs[1, 0])
    llama_top20 = data['llama']['ranked_heads'].head(20)['composite_score'].values
    gpt2_top20 = data['gpt2']['ranked_heads'].head(20)['composite_score'].values

    ax2.plot(range(1, 21), llama_top20, marker='o', label='LLaMA', color='#e74c3c', linewidth=2)
    ax2.plot(range(1, 21), gpt2_top20, marker='s', label='GPT-2', color='#3498db', linewidth=2)
    ax2.set_xlabel('Head Rank', fontsize=11)
    ax2.set_ylabel('Composite Score', fontsize=11)
    ax2.set_title('Top 20 Critical Heads\nComposite Scores', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Plot 3: Functional type distribution (LLaMA)
    ax3 = fig.add_subplot(gs[1, 1])
    llama_types = data['llama']['ranked_heads'].head(20)['functional_type'].value_counts()
    colors_llama = {'Hub Aggregator': '#e74c3c', 'Skip Connection': '#2ecc71',
                    'Forward Flow': '#3498db', 'Multi-Purpose': '#9b59b6'}
    ax3.bar(range(len(llama_types)), llama_types.values,
            color=[colors_llama.get(t, '#95a5a6') for t in llama_types.index])
    ax3.set_xticks(range(len(llama_types)))
    ax3.set_xticklabels(llama_types.index, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Count (Top 20)', fontsize=11)
    ax3.set_title('LLaMA\nFunctional Types', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Functional type distribution (GPT-2)
    ax4 = fig.add_subplot(gs[1, 2])
    gpt2_types = data['gpt2']['ranked_heads'].head(20)['functional_type'].value_counts()
    colors_gpt2 = {'Hub Aggregator': '#e74c3c', 'Skip Connection': '#2ecc71',
                   'Forward Flow': '#3498db', 'Multi-Purpose': '#9b59b6'}
    ax4.bar(range(len(gpt2_types)), gpt2_types.values,
            color=[colors_gpt2.get(t, '#95a5a6') for t in gpt2_types.index])
    ax4.set_xticks(range(len(gpt2_types)))
    ax4.set_xticklabels(gpt2_types.index, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Count (Top 20)', fontsize=11)
    ax4.set_title('GPT-2\nFunctional Types', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Layer distribution (LLaMA)
    ax5 = fig.add_subplot(gs[2, 0])
    llama_layers = data['llama']['ranked_heads'].head(20)['layer_type'].value_counts()
    colors_layer = {'early': '#3498db', 'middle': '#f39c12', 'late': '#e74c3c'}
    ax5.bar(range(len(llama_layers)), llama_layers.values,
            color=[colors_layer[t] for t in llama_layers.index])
    ax5.set_xticks(range(len(llama_layers)))
    ax5.set_xticklabels([t.capitalize() for t in llama_layers.index])
    ax5.set_ylabel('Count (Top 20)', fontsize=11)
    ax5.set_title('LLaMA\nLayer Distribution', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    # Plot 6: Layer distribution (GPT-2)
    ax6 = fig.add_subplot(gs[2, 1])
    gpt2_layers = data['gpt2']['ranked_heads'].head(20)['layer_type'].value_counts()
    ax6.bar(range(len(gpt2_layers)), gpt2_layers.values,
            color=[colors_layer[t] for t in gpt2_layers.index])
    ax6.set_xticks(range(len(gpt2_layers)))
    ax6.set_xticklabels([t.capitalize() for t in gpt2_layers.index])
    ax6.set_ylabel('Count (Top 20)', fontsize=11)
    ax6.set_title('GPT-2\nLayer Distribution', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    # Plot 7: Metric comparison
    ax7 = fig.add_subplot(gs[2, 2])
    metrics = ['Hub', 'Skip']
    llama_means = [
        data['llama']['ranked_heads']['hub_score'].mean(),
        data['llama']['ranked_heads']['skip_score'].mean()
    ]
    gpt2_means = [
        data['gpt2']['ranked_heads']['hub_score'].mean(),
        data['gpt2']['ranked_heads']['skip_score'].mean()
    ]

    x = np.arange(len(metrics))
    width = 0.35

    ax7.bar(x - width/2, llama_means, width, label='LLaMA', alpha=0.8, color='#e74c3c')
    ax7.bar(x + width/2, gpt2_means, width, label='GPT-2', alpha=0.8, color='#3498db')
    ax7.set_ylabel('Mean Score (All Heads)', fontsize=11)
    ax7.set_title('Average Metric Scores', fontsize=12, fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics)
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)

    plt.suptitle('LLaMA vs GPT-2 - Attention Flow Comparison', fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = figures_dir / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison: {output_path}")
    plt.close()

    # Save comparison JSON
    comparison_data = {
        'models': comparison,
        'hub_analysis': {
            model: data[model]['summary']['hub_analysis']
            for model in models
        },
        'top_heads': {
            model: {
                'layer': int(data[model]['ranked_heads'].iloc[0]['layer']),
                'head': int(data[model]['ranked_heads'].iloc[0]['head']),
                'functional_type': data[model]['ranked_heads'].iloc[0]['functional_type'],
                'composite_score': float(data[model]['ranked_heads'].iloc[0]['composite_score']),
                'flow_score': float(data[model]['ranked_heads'].iloc[0]['flow_score']),
                'hub_score': float(data[model]['ranked_heads'].iloc[0]['hub_score']),
                'skip_score': float(data[model]['ranked_heads'].iloc[0]['skip_score'])
            }
            for model in models
        },
        'functional_type_distribution_top20': {
            model: data[model]['ranked_heads'].head(20)['functional_type'].value_counts().to_dict()
            for model in models
        },
        'layer_distribution_top20': {
            model: data[model]['ranked_heads'].head(20)['layer_type'].value_counts().to_dict()
            for model in models
        },
        'key_findings': {
            'hub_position': {
                'llama': data['llama']['summary']['hub_analysis']['hub_position'],
                'gpt2': data['gpt2']['summary']['hub_analysis']['hub_position']
            },
            'hub_strength': {
                'llama': {
                    'score': data['llama']['summary']['hub_analysis']['hub_score'],
                    'ratio': data['llama']['summary']['hub_analysis']['hub_ratio'],
                    'strong': data['llama']['summary']['hub_analysis']['is_strong_hub']
                },
                'gpt2': {
                    'score': data['gpt2']['summary']['hub_analysis']['hub_score'],
                    'ratio': data['gpt2']['summary']['hub_analysis']['hub_ratio'],
                    'strong': data['gpt2']['summary']['hub_analysis']['is_strong_hub']
                }
            },
            'sequential_flow': {
                'llama': data['llama']['summary']['sequential_flow']['has_sequential_flow'],
                'gpt2': data['gpt2']['summary']['sequential_flow']['has_sequential_flow']
            },
            'skip_connections': {
                'llama': data['llama']['summary']['skip_connections']['has_skip_connections'],
                'gpt2': data['gpt2']['summary']['skip_connections']['has_skip_connections']
            }
        }
    }

    output_json = results_dir / 'model_comparison.json'
    with open(output_json, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    print(f"✓ Saved comparison data: {output_json}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("\n1. Hub Architecture:")
    print(f"   LLaMA: Position {comparison_data['key_findings']['hub_position']['llama']} "
          f"({comparison_data['key_findings']['hub_strength']['llama']['ratio']:.2f}× uniform)")
    print(f"   GPT-2: Position {comparison_data['key_findings']['hub_position']['gpt2']} "
          f"({comparison_data['key_findings']['hub_strength']['gpt2']['ratio']:.2f}× uniform)")

    print("\n2. Critical Head Composition:")
    llama_top = comparison_data['top_heads']['llama']
    gpt2_top = comparison_data['top_heads']['gpt2']
    print(f"   LLaMA: L{llama_top['layer']}H{llama_top['head']} "
          f"({llama_top['functional_type']}, score={llama_top['composite_score']:.3f})")
    print(f"   GPT-2: L{gpt2_top['layer']}H{gpt2_top['head']} "
          f"({gpt2_top['functional_type']}, score={gpt2_top['composite_score']:.3f})")

    print("\n3. Layer Preference (Top 20 heads):")
    for model in models:
        layer_dist = comparison_data['layer_distribution_top20'][model]
        print(f"   {model.upper()}: {dict(layer_dist)}")

    print("\n" + "=" * 80)
    print("STORY 2.6 COMPLETE ✓")
    print("=" * 80)
    print("\nModel comparison complete!")
    print(f"  Visualization: {output_path}")
    print(f"  Data: {output_json}")


def main():
    compare_models()


if __name__ == '__main__':
    main()
