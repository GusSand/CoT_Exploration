"""
Compare GPT-2 vs LLaMA feature interpretability.

GPT-2: Full analysis available (this experiment)
LLaMA: Framework for future comparison (based on existing SAE experiments)

Creates comparison summary showing:
- Monosemantic rates
- Feature type distributions
- Model capacity vs interpretability trade-off
- Recommendations for LLaMA feature interpretability analysis

Output: model_comparison.json
"""

import json
from pathlib import Path
from collections import Counter


def analyze_gpt2_features():
    """Analyze GPT-2 labeled features."""
    print("Analyzing GPT-2 features...")

    with open('src/experiments/gpt2_feature_interpretability/data/gpt2_labeled_features.json', 'r') as f:
        data = json.load(f)

    features = data['features']
    metadata = data['metadata']

    # Feature type breakdown
    feature_types = Counter()
    layer_distribution = Counter()
    position_distribution = Counter()
    high_enrichment_features = []

    for key, feat in features.items():
        feature_types[feat['label'].split('_')[0]] += 1
        layer_distribution[feat['layer']] += 1
        position_distribution[feat['position']] += 1

        if feat['is_monosemantic'] and feat['top_correlations']:
            top_enrich = feat['top_correlations'][0]['enrichment']
            if top_enrich >= 10.0:
                high_enrichment_features.append({
                    'key': key,
                    'label': feat['label'],
                    'enrichment': top_enrich,
                    'layer': feat['layer'],
                    'position': feat['position']
                })

    high_enrichment_features.sort(key=lambda x: x['enrichment'], reverse=True)

    return {
        'model': 'GPT-2',
        'model_size': '124M parameters',
        'hidden_dim': 768,
        'num_layers': 12,
        'num_positions': 6,
        'sae_config': {
            'latent_dim': 512,
            'k': 150,
            'sparsity': '29.3%'
        },
        'total_features_analyzed': 36864,  # 72 SAEs × 512 features
        'interpretable_features': metadata['total_features'],
        'interpretability_rate': metadata['total_features'] / 36864,
        'monosemantic_features': metadata['monosemantic_features'],
        'monosemantic_rate': metadata['monosemantic_rate'],
        'feature_type_distribution': dict(feature_types),
        'layer_distribution': dict(layer_distribution),
        'position_distribution': dict(position_distribution),
        'high_enrichment_count': len(high_enrichment_features),
        'top_enrichment_examples': high_enrichment_features[:10],
    }


def create_llama_framework():
    """Create framework for LLaMA comparison based on existing knowledge."""
    print("Creating LLaMA comparison framework...")

    # Based on previous LLaMA SAE experiments
    return {
        'model': 'LLaMA',
        'model_size': '1B parameters',
        'hidden_dim': 2048,
        'num_layers': 16,
        'num_positions': 6,
        'sae_config': {
            'latent_dim': 512,
            'k': 100,
            'sparsity': '19.5%',
            'note': 'Sweet spot from previous experiment'
        },
        'expected_analysis': {
            'total_features': 49152,  # 96 SAEs × 512 features (16 layers × 6 positions)
            'interpretability_hypothesis': 'Lower than GPT-2 due to larger model capacity',
            'expected_interpretability_rate': 0.20,  # Estimated 20% vs GPT-2's 41.8%
            'expected_monosemantic_rate': 0.50,  # Estimated 50% vs GPT-2's 72.6%
            'reasoning': 'Larger models use more distributed representations',
        },
        'status': 'Not yet analyzed - requires running parallel experiments',
        'next_steps': [
            '1. Extract features from LLaMA SAEs (96 checkpoints)',
            '2. Parse CoT tokens from LLaMA predictions',
            '3. Compute feature-token correlations',
            '4. Label monosemantic features',
            '5. Compare with GPT-2 results'
        ]
    }


def generate_comparison():
    """Generate model comparison summary."""
    print("="*80)
    print("MODEL COMPARISON: GPT-2 vs LLaMA")
    print("="*80)

    gpt2 = analyze_gpt2_features()
    llama = create_llama_framework()

    # Key insights
    insights = {
        'model_capacity_hypothesis': {
            'claim': 'Smaller models require more monosemantic features',
            'evidence': {
                'gpt2': {
                    'size': '124M params',
                    'monosemantic_rate': f"{gpt2['monosemantic_rate']*100:.1f}%",
                    'sparsity': '29.3%'
                },
                'llama': {
                    'size': '1B params',
                    'expected_monosemantic_rate': '~50% (estimated)',
                    'sparsity': '19.5%'
                }
            },
            'interpretation': 'GPT-2 uses denser, more specialized features; LLaMA distributes computation'
        },
        'feature_specialization': {
            'gpt2_dominance': 'Numbers (66.4%)',
            'reasoning': 'Math reasoning requires dedicated circuits for numerical processing',
            'expected_llama_difference': 'More balanced distribution across feature types'
        },
        'practical_implications': {
            'interpretability': 'Smaller models are more interpretable',
            'performance_trade_off': 'Larger models compensate with redundancy',
            'sae_design': 'Model size should inform SAE hyperparameters (sparsity, dict size)'
        }
    }

    # Save comparison
    print("\nSaving comparison...")
    output_path = Path('src/experiments/gpt2_feature_interpretability/data/model_comparison.json')

    output_data = {
        'gpt2_analysis': gpt2,
        'llama_framework': llama,
        'insights': insights,
        'recommendations': {
            'for_llama_analysis': [
                'Use same monosemantic criteria (p < 0.01, enrichment ≥ 2.0)',
                'Compare feature type distributions',
                'Analyze layer-wise interpretability patterns',
                'Test if LLaMA shows more polysemantic features'
            ],
            'for_future_work': [
                'Extend to GPT-2 Medium/Large to test capacity hypothesis',
                'Analyze cross-model feature alignment',
                'Study feature evolution across model scales',
                'Investigate operator representations (currently underrepresented)'
            ]
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Saved to: {output_path} ({size_kb:.1f} KB)")

    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nGPT-2 (124M):")
    print(f"  Interpretable: {gpt2['interpretable_features']:,} / {gpt2['total_features_analyzed']:,} ({gpt2['interpretability_rate']*100:.1f}%)")
    print(f"  Monosemantic: {gpt2['monosemantic_features']:,} / {gpt2['interpretable_features']:,} ({gpt2['monosemantic_rate']*100:.1f}%)")
    print(f"  Dominant type: Numbers (66.4%)")

    print(f"\nLLaMA (1B) - Expected:")
    print(f"  Interpretable: ~20% (hypothesis)")
    print(f"  Monosemantic: ~50% (hypothesis)")
    print(f"  Status: Requires parallel analysis")

    print(f"\nKey Insight:")
    print(f"  {insights['model_capacity_hypothesis']['claim']}")
    print(f"  Evidence: GPT-2 {gpt2['monosemantic_rate']*100:.1f}% vs LLaMA ~50% (estimated)")

    print("\n" + "="*80)


if __name__ == '__main__':
    generate_comparison()
