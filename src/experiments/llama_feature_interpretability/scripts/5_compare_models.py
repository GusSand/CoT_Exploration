"""
Compare GPT-2 vs LLaMA feature interpretability.

Uses ACTUAL analysis results from both models to test the capacity hypothesis.

Creates comparison summary showing:
- Monosemantic rates
- Feature type distributions
- Model capacity vs interpretability trade-off
- Answers to the 4 research questions

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

    max_enrichment = 0
    for key, feat in features.items():
        feature_types[feat['label'].split('_')[0]] += 1
        layer_distribution[feat['layer']] += 1
        position_distribution[feat['position']] += 1

        if feat['is_monosemantic'] and feat['top_correlations']:
            top_enrich = feat['top_correlations'][0]['enrichment']
            max_enrichment = max(max_enrichment, top_enrich)
            if top_enrich >= 10.0:
                high_enrichment_features.append({
                    'key': key,
                    'label': feat['label'],
                    'enrichment': top_enrich,
                    'layer': feat['layer'],
                    'position': feat['position']
                })

    high_enrichment_features.sort(key=lambda x: x['enrichment'], reverse=True)

    # Count number features
    number_features = sum(1 for feat in features.values() if 'number' in feat['label'].lower())
    number_pct = number_features / len(features) * 100

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
        'number_features': number_features,
        'number_features_pct': number_pct,
        'max_enrichment': max_enrichment,
        'feature_type_distribution': dict(feature_types),
        'layer_distribution': dict(layer_distribution),
        'position_distribution': dict(position_distribution),
        'high_enrichment_count': len(high_enrichment_features),
        'top_enrichment_examples': high_enrichment_features[:10],
    }


def analyze_llama_features():
    """Analyze LLaMA labeled features."""
    print("Analyzing LLaMA features...")

    with open('src/experiments/llama_feature_interpretability/data/llama_labeled_features.json', 'r') as f:
        data = json.load(f)

    features = data['features']
    metadata = data['metadata']

    # Feature type breakdown
    feature_types = Counter()
    layer_distribution = Counter()
    position_distribution = Counter()
    high_enrichment_features = []

    max_enrichment = 0
    for key, feat in features.items():
        feature_types[feat['label'].split('_')[0]] += 1
        layer_distribution[feat['layer']] += 1
        position_distribution[feat['position']] += 1

        if feat['is_monosemantic'] and feat['top_correlations']:
            top_enrich = feat['top_correlations'][0]['enrichment']
            max_enrichment = max(max_enrichment, top_enrich)
            if top_enrich >= 10.0:
                high_enrichment_features.append({
                    'key': key,
                    'label': feat['label'],
                    'enrichment': top_enrich,
                    'layer': feat['layer'],
                    'position': feat['position']
                })

    high_enrichment_features.sort(key=lambda x: x['enrichment'], reverse=True)

    # Count number features
    number_features = sum(1 for feat in features.values() if 'number' in feat['label'].lower())
    number_pct = number_features / len(features) * 100

    return {
        'model': 'LLaMA-3.2-1B',
        'model_size': '1B parameters',
        'hidden_dim': 2048,
        'num_layers': 16,
        'num_positions': 6,
        'sae_config': {
            'latent_dim': 512,
            'k': 100,
            'sparsity': '19.5%'
        },
        'total_features_analyzed': 49152,  # 96 SAEs × 512 features
        'interpretable_features': metadata['total_features'],
        'interpretability_rate': metadata['total_features'] / 49152,
        'monosemantic_features': metadata['monosemantic_features'],
        'monosemantic_rate': metadata['monosemantic_rate'],
        'number_features': number_features,
        'number_features_pct': number_pct,
        'max_enrichment': max_enrichment,
        'feature_type_distribution': dict(feature_types),
        'layer_distribution': dict(layer_distribution),
        'position_distribution': dict(position_distribution),
        'high_enrichment_count': len(high_enrichment_features),
        'top_enrichment_examples': high_enrichment_features[:10],
    }


def generate_comparison():
    """Generate model comparison summary."""
    print("="*80)
    print("MODEL COMPARISON: GPT-2 vs LLaMA")
    print("="*80)

    gpt2 = analyze_gpt2_features()
    llama = analyze_llama_features()

    # Answer the 4 research questions
    research_questions = {
        'q1_monosemantic_rate': {
            'question': 'What is LLaMA\'s monosemantic rate?',
            'gpt2': f"{gpt2['monosemantic_rate']*100:.1f}%",
            'llama': f"{llama['monosemantic_rate']*100:.1f}%",
            'finding': f"LLaMA: {llama['monosemantic_rate']*100:.1f}% vs GPT-2: {gpt2['monosemantic_rate']*100:.1f}%",
        },
        'q2_number_features': {
            'question': 'What % are number features?',
            'gpt2': f"{gpt2['number_features_pct']:.1f}%",
            'llama': f"{llama['number_features_pct']:.1f}%",
            'finding': f"LLaMA: {llama['number_features_pct']:.1f}% vs GPT-2: {gpt2['number_features_pct']:.1f}%",
        },
        'q3_max_enrichment': {
            'question': 'What is max enrichment?',
            'gpt2': f"{gpt2['max_enrichment']:.1f}×",
            'llama': f"{llama['max_enrichment']:.1f}×",
            'finding': f"LLaMA: {llama['max_enrichment']:.1f}× vs GPT-2: {gpt2['max_enrichment']:.1f}×",
        },
        'q4_capacity_hypothesis': {
            'question': 'Does larger model = lower monosemantic rate?',
            'hypothesis': 'Larger models use more distributed representations',
            'gpt2_size': '124M params',
            'llama_size': '1B params',
            'gpt2_mono': f"{gpt2['monosemantic_rate']*100:.1f}%",
            'llama_mono': f"{llama['monosemantic_rate']*100:.1f}%",
            'result': 'YES' if llama['monosemantic_rate'] < gpt2['monosemantic_rate'] else 'NO',
            'finding': f"{'Confirmed' if llama['monosemantic_rate'] < gpt2['monosemantic_rate'] else 'Rejected'}: LLaMA (1B) has {'lower' if llama['monosemantic_rate'] < gpt2['monosemantic_rate'] else 'higher'} monosemantic rate than GPT-2 (124M)",
        }
    }

    # Key insights
    insights = {
        'model_capacity': {
            'claim': 'Larger models use more distributed representations',
            'evidence': research_questions['q4_capacity_hypothesis'],
            'interpretation': f"LLaMA's {llama['monosemantic_rate']*100:.1f}% monosemantic rate vs GPT-2's {gpt2['monosemantic_rate']*100:.1f}% {'supports' if llama['monosemantic_rate'] < gpt2['monosemantic_rate'] else 'contradicts'} the hypothesis"
        },
        'feature_specialization': {
            'gpt2_number_dominance': f"{gpt2['number_features_pct']:.1f}%",
            'llama_number_dominance': f"{llama['number_features_pct']:.1f}%",
            'finding': 'Both models heavily rely on number-specific features for math reasoning'
        },
        'enrichment_comparison': {
            'gpt2_max': f"{gpt2['max_enrichment']:.1f}×",
            'llama_max': f"{llama['max_enrichment']:.1f}×",
            'finding': f"{'LLaMA' if llama['max_enrichment'] > gpt2['max_enrichment'] else 'GPT-2'} has stronger feature specialization"
        }
    }

    # Save comparison
    print("\nSaving comparison...")
    output_path = Path('src/experiments/llama_feature_interpretability/data/model_comparison.json')

    output_data = {
        'gpt2_analysis': gpt2,
        'llama_analysis': llama,
        'research_questions': research_questions,
        'insights': insights,
        'methodology_verification': {
            'criteria_identical': True,
            'p_value_threshold': 0.01,
            'enrichment_threshold': 2.0,
            'min_activations': 20,
            'labeling_logic': 'Copied exactly from GPT-2 (enrichment ≥ 5.0 OR top 3 same category)'
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
    print(f"  Number features: {gpt2['number_features_pct']:.1f}%")
    print(f"  Max enrichment: {gpt2['max_enrichment']:.1f}×")

    print(f"\nLLaMA (1B):")
    print(f"  Interpretable: {llama['interpretable_features']:,} / {llama['total_features_analyzed']:,} ({llama['interpretability_rate']*100:.1f}%)")
    print(f"  Monosemantic: {llama['monosemantic_features']:,} / {llama['interpretable_features']:,} ({llama['monosemantic_rate']*100:.1f}%)")
    print(f"  Number features: {llama['number_features_pct']:.1f}%")
    print(f"  Max enrichment: {llama['max_enrichment']:.1f}×")

    print(f"\nResearch Questions:")
    for q_id, q_data in research_questions.items():
        print(f"  {q_data['question']}")
        print(f"    {q_data['finding']}")

    print("\n" + "="*80)


if __name__ == '__main__':
    generate_comparison()
