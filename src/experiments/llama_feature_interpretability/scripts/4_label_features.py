"""
Label monosemantic features with human-readable names.

Analyzes feature-token correlations to identify:
1. Number features (e.g., "number_50", "large_numbers")
2. Operator features (e.g., "multiplication", "addition")
3. Composite features (e.g., "division_by_2", "multiplication_with_small_numbers")

Monosemanticity criteria (IDENTICAL to GPT-2):
- Top correlation has enrichment ≥ 5.0 (strongly dominant)
- OR top 3 correlations are all same category (e.g., all numbers, all operators)
- Feature labeled as "monosemantic" if meets criteria

Output: llama_labeled_features.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def classify_token(token):
    """
    Classify token into category.

    Returns:
        (category, subcategory, specific_value)

    Examples:
        "50" -> ("number", "medium", "50")
        "*" -> ("operator", "multiplication", "*")
        "=" -> ("operator", "equals", "=")
    """
    # Operators
    if token == '+':
        return ('operator', 'addition', '+')
    elif token == '-':
        return ('operator', 'subtraction', '-')
    elif token == '*':
        return ('operator', 'multiplication', '*')
    elif token == '/':
        return ('operator', 'division', '/')
    elif token == '=':
        return ('operator', 'equals', '=')
    elif token in ['(', ')']:
        return ('operator', 'parentheses', token)

    # Numbers
    if re.match(r'^\d+\.?\d*$', token):
        num = float(token)

        # Size categories
        if num == 0:
            size = 'zero'
        elif num == 1:
            size = 'one'
        elif num == 2:
            size = 'two'
        elif num <= 10:
            size = 'small'
        elif num <= 100:
            size = 'medium'
        elif num <= 1000:
            size = 'large'
        else:
            size = 'very_large'

        return ('number', size, token)

    return ('other', 'unknown', token)


def generate_feature_label(correlations):
    """
    Generate human-readable label for feature based on correlations.

    Args:
        correlations: List of correlation dicts, sorted by enrichment

    Returns:
        (label, is_monosemantic, explanation)
    """
    if not correlations:
        return ('no_correlations', False, 'No significant correlations found')

    # Classify all correlated tokens
    classified = []
    for corr in correlations[:10]:  # Top 10
        token = corr['token']
        enrichment = corr['enrichment']
        category, subcategory, value = classify_token(token)
        classified.append({
            'token': token,
            'enrichment': enrichment,
            'category': category,
            'subcategory': subcategory,
            'value': value
        })

    # Check monosemanticity criteria
    top_enrichment = classified[0]['enrichment']
    top_category = classified[0]['category']
    top_subcategory = classified[0]['subcategory']
    top_token = classified[0]['token']

    # Criterion 1: Very strong single correlation (enrichment ≥ 5.0)
    if top_enrichment >= 5.0:
        if top_category == 'number':
            label = f"number_{top_token}"
            explanation = f"Strongly correlates with number {top_token} (enrichment={top_enrichment:.1f})"
        elif top_category == 'operator':
            label = f"operator_{top_subcategory}"
            explanation = f"Strongly correlates with {top_subcategory} (enrichment={top_enrichment:.1f})"
        else:
            label = f"specific_{top_token}"
            explanation = f"Strongly correlates with token '{top_token}' (enrichment={top_enrichment:.1f})"

        return (label, True, explanation)

    # Criterion 2: Top 3 all same category
    if len(classified) >= 3:
        top3_categories = [c['category'] for c in classified[:3]]
        top3_subcategories = [c['subcategory'] for c in classified[:3]]

        if len(set(top3_categories)) == 1:
            category = top3_categories[0]

            if category == 'number':
                # Check if all same subcategory
                if len(set(top3_subcategories)) == 1:
                    subcategory = top3_subcategories[0]
                    label = f"numbers_{subcategory}"
                    tokens_str = ', '.join([c['token'] for c in classified[:3]])
                    explanation = f"Correlates with {subcategory} numbers: {tokens_str}"
                    return (label, True, explanation)
                else:
                    label = "numbers_mixed"
                    tokens_str = ', '.join([c['token'] for c in classified[:3]])
                    explanation = f"Correlates with multiple number ranges: {tokens_str}"
                    return (label, True, explanation)

            elif category == 'operator':
                # Multiple operators
                operators = [c['subcategory'] for c in classified[:3]]
                label = f"operators_{'_'.join(operators[:2])}"
                explanation = f"Correlates with multiple operators: {', '.join(operators)}"
                return (label, True, explanation)

    # Criterion 3: Check for composite patterns (e.g., division + specific number)
    if len(classified) >= 2:
        top2_categories = [c['category'] for c in classified[:2]]

        if set(top2_categories) == {'operator', 'number'}:
            # Find operator and number
            op_item = next((c for c in classified[:2] if c['category'] == 'operator'), None)
            num_item = next((c for c in classified[:2] if c['category'] == 'number'), None)

            if op_item and num_item:
                label = f"{op_item['subcategory']}_with_{num_item['subcategory']}_numbers"
                explanation = f"Correlates with {op_item['subcategory']} and {num_item['subcategory']} numbers"
                return (label, True, explanation)

    # Polysemantic: Multiple distinct correlations
    categories = [c['category'] for c in classified[:5]]
    category_counts = Counter(categories)
    most_common_category = category_counts.most_common(1)[0][0]

    if most_common_category == 'number':
        label = "polysemantic_numbers"
        explanation = f"Correlates with multiple diverse numbers (top: {top_token})"
    elif most_common_category == 'operator':
        label = "polysemantic_operators"
        explanation = f"Correlates with multiple diverse operators (top: {top_token})"
    else:
        label = "polysemantic_mixed"
        explanation = f"Correlates with diverse tokens (top: {top_token})"

    return (label, False, explanation)


def label_all_features():
    """Label all interpretable features."""
    print("="*80)
    print("LABELING MONOSEMANTIC FEATURES")
    print("="*80)

    # Load correlations
    print("\n[1/3] Loading correlations...")
    with open('src/experiments/llama_feature_interpretability/data/llama_feature_token_correlations.json', 'r') as f:
        data = json.load(f)

    correlations = data['correlations']
    metadata = data['metadata']

    print(f"  Interpretable features: {metadata['interpretable_features']:,}")

    # Label each feature
    print("\n[2/3] Generating labels...")

    labeled_features = {}
    category_counts = Counter()
    monosemantic_count = 0

    for layer_str, layer_data in correlations.items():
        layer = int(layer_str)

        for pos_str, pos_data in layer_data.items():
            position = int(pos_str)

            for feat_str, feat_data in pos_data.items():
                feature_id = int(feat_str)

                # Generate label
                label, is_monosemantic, explanation = generate_feature_label(feat_data['correlations'])

                # Store
                key = f"L{layer}_P{position}_F{feature_id}"
                labeled_features[key] = {
                    'layer': layer,
                    'position': position,
                    'feature_id': feature_id,
                    'label': label,
                    'is_monosemantic': is_monosemantic,
                    'explanation': explanation,
                    'num_activations': feat_data['num_activations'],
                    'activation_rate': feat_data['activation_rate'],
                    'num_correlations': feat_data['num_correlations'],
                    'top_correlations': feat_data['correlations'][:5]  # Keep top 5
                }

                # Track stats
                category_counts[label.split('_')[0]] += 1
                if is_monosemantic:
                    monosemantic_count += 1

    print(f"  Total labeled: {len(labeled_features):,}")
    print(f"  Monosemantic: {monosemantic_count:,} ({monosemantic_count/len(labeled_features)*100:.1f}%)")

    # Category breakdown
    print(f"\n  Label categories:")
    for category, count in category_counts.most_common(10):
        pct = count / len(labeled_features) * 100
        print(f"    {category}: {count:,} ({pct:.1f}%)")

    # Find interesting examples
    print("\n  Example monosemantic features:")
    monosemantic_examples = [
        (k, v) for k, v in labeled_features.items()
        if v['is_monosemantic'] and v['top_correlations'][0]['enrichment'] >= 5.0
    ]
    monosemantic_examples.sort(key=lambda x: x[1]['top_correlations'][0]['enrichment'], reverse=True)

    for key, feat in monosemantic_examples[:5]:
        print(f"    {key}: {feat['label']}")
        print(f"      {feat['explanation']}")

    # Save results
    print("\n[3/3] Saving labeled features...")
    output_path = Path('src/experiments/llama_feature_interpretability/data/llama_labeled_features.json')

    output_data = {
        'metadata': {
            'model': 'llama-3.2-1b',
            'total_features': len(labeled_features),
            'monosemantic_features': monosemantic_count,
            'monosemantic_rate': monosemantic_count / len(labeled_features),
            'category_counts': dict(category_counts),
        },
        'features': labeled_features,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"✓ Saved to: {output_path} ({size_mb:.1f} MB)")

    print("\n" + "="*80)
    print("LABELING COMPLETE!")
    print("="*80)
    print(f"  Monosemantic features: {monosemantic_count:,} / {len(labeled_features):,} ({monosemantic_count/len(labeled_features)*100:.1f}%)")
    print(f"  Output: {output_path}")
    print("="*80)


if __name__ == '__main__':
    label_all_features()
