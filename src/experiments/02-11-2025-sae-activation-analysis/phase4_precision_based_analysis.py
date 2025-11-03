#!/usr/bin/env python3
"""
Phase 4: Precision-Based Feature Analysis with Math Token Filtering

Primary metric: Precision = P(token | feature fires)
Filter: Only math-related tokens (numbers, operators, symbols)
Significance: Chi-squared p-value < 0.01
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import math


def chi2_contingency_manual(contingency_table):
    """Manual implementation of chi-squared test"""
    observed = np.array(contingency_table)
    row_sums = observed.sum(axis=1)
    col_sums = observed.sum(axis=0)
    total = observed.sum()

    expected = np.outer(row_sums, col_sums) / total
    chi2 = ((observed - expected) ** 2 / expected).sum()
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    if dof == 1:
        p_value = math.exp(-chi2 / 2)
    else:
        p_value = max(0, 1 - (chi2 / (chi2 + dof)))

    return chi2, p_value


print("Loading LLAMA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def is_math_token(token):
    """
    Filter to only math-related tokens.
    Keep: numbers, operators, math symbols
    """
    token_clean = token.strip()

    # Numbers (including multi-digit)
    if re.match(r'^\d+$', token_clean):
        return True

    # Decimal numbers
    if re.match(r'^\d+\.\d+$', token_clean):
        return True

    # Math operators
    if token_clean in {'+', '-', '*', '/', '=', '<', '>', '<<', '>>'}:
        return True

    # Financial/math symbols
    if token_clean in {'$', '%', '.'}:
        return True

    # Math words (optional - can remove if too noisy)
    math_words = {'sum', 'total', 'product', 'difference', 'remainder',
                  'average', 'multiply', 'divide', 'subtract', 'add'}
    if token_clean.lower() in math_words:
        return True

    return False


def extract_math_tokens_only(reference_answer, tokenizer):
    """Extract only math-related tokens"""
    before_answer = reference_answer.split('####')[0].strip()
    token_ids = tokenizer.encode(before_answer, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Filter to math tokens only
    math_tokens = [t for t in tokens if is_math_token(t)]

    return math_tokens


def build_cooccurrence_math_tokens(phase1_data, feature_list, layer_name='late', n_examples=None):
    """Build co-occurrence matrix for math tokens only"""
    if n_examples is None:
        n_examples = len(phase1_data['results'])

    print(f"\nBuilding co-occurrence for {len(feature_list)} features (MATH TOKENS ONLY) in {layer_name} layer...")

    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train = gsm8k_dataset['train']

    feature_counts = Counter()
    token_counts = Counter()
    cooccurrence = defaultdict(Counter)
    feature_to_examples = defaultdict(set)
    token_to_examples = defaultdict(set)

    for example_idx in tqdm(range(n_examples), desc=f"Processing {layer_name}"):
        example = phase1_data['results'][example_idx]
        reference_answer = gsm8k_train[example_idx]['answer']

        # Extract ONLY math tokens
        math_tokens = extract_math_tokens_only(reference_answer, tokenizer)
        unique_tokens = set(math_tokens)

        # Get features that fired
        fired_features = set()
        for position_data in example['activations']:
            layer_data = position_data['layers'][layer_name]
            fired_features.update([f for f in layer_data['firing_indices'] if f in feature_list])

        for feature_id in fired_features:
            feature_counts[feature_id] += 1
            feature_to_examples[feature_id].add(example_idx)

        for token in unique_tokens:
            token_counts[token] += 1
            token_to_examples[token].add(example_idx)

        for feature_id in fired_features:
            for token in unique_tokens:
                cooccurrence[feature_id][token] += 1

    return {
        'n_examples': n_examples,
        'feature_counts': dict(feature_counts),
        'token_counts': dict(token_counts),
        'cooccurrence': {k: dict(v) for k, v in cooccurrence.items()},
        'feature_to_examples': {k: list(v) for k, v in feature_to_examples.items()},
        'token_to_examples': {k: list(v) for k, v in token_to_examples.items()}
    }


def compute_precision_metrics(feature_id, cooccurrence_data, p_value_threshold=0.01):
    """
    Compute precision-based metrics for a feature.

    Precision = P(token | feature) = count(feature & token) / count(feature)
    """
    if feature_id not in cooccurrence_data['cooccurrence']:
        return None

    n_examples = cooccurrence_data['n_examples']
    feature_count = cooccurrence_data['feature_counts'].get(feature_id, 0)

    if feature_count == 0:
        return None

    feature_tokens = cooccurrence_data['cooccurrence'][feature_id]

    results = []

    for token, count in feature_tokens.items():
        # Precision = P(token | feature fires)
        precision = count / feature_count

        # Chi-squared test
        feature_to_examples = set(cooccurrence_data['feature_to_examples'].get(feature_id, []))
        token_to_examples = set(cooccurrence_data['token_to_examples'].get(token, []))

        a = len(feature_to_examples & token_to_examples)
        b = len(feature_to_examples - token_to_examples)
        c = len(token_to_examples - feature_to_examples)
        d = n_examples - a - b - c

        chi2, p_value = chi2_contingency_manual(np.array([[a, b], [c, d]]))

        # Only keep if statistically significant
        if p_value < p_value_threshold:
            results.append({
                'token': token,
                'precision': precision,
                'count': count,
                'chi2': chi2,
                'p_value': p_value
            })

    # Sort by precision (highest first)
    results.sort(key=lambda x: x['precision'], reverse=True)

    return {
        'feature_id': feature_id,
        'feature_count': feature_count,
        'feature_frequency': feature_count / n_examples,
        'top_tokens': results
    }


def main():
    print("="*80)
    print("PHASE 4: Precision-Based Analysis with Math Token Filtering")
    print("="*80)

    # Load phase3 features
    phase3_path = Path('./phase3_first_test_example_features.json')
    print(f"\nLoading identified features from: {phase3_path}")
    with open(phase3_path, 'r') as f:
        phase3_data = json.load(f)

    unique_features = phase3_data['unique_features']
    print(f"Found {len(unique_features)} unique features to analyze")

    # Load Phase 1 data
    phase1_data_path = Path('./phase1_sae_activations_full.json')
    print(f"Loading Phase 1 data from: {phase1_data_path}")
    with open(phase1_data_path, 'r') as f:
        phase1_data = json.load(f)

    all_reports = {}

    for layer_name in ['early', 'middle', 'late']:
        print(f"\n{'='*80}")
        print(f"Analyzing {layer_name.upper()} layer - Math Tokens Only")
        print(f"{'='*80}")

        # Build co-occurrence for math tokens only
        cooccurrence_data = build_cooccurrence_math_tokens(
            phase1_data,
            feature_list=set(unique_features),
            layer_name=layer_name,
            n_examples=7473
        )

        print(f"\nStatistics:")
        print(f"  Unique math tokens found: {len(cooccurrence_data['token_counts'])}")
        print(f"  Total token occurrences: {sum(cooccurrence_data['token_counts'].values())}")

        layer_reports = {}

        # Generate report for each feature
        for feature_id in unique_features:
            report = compute_precision_metrics(feature_id, cooccurrence_data, p_value_threshold=0.01)

            if report is not None and report['top_tokens']:
                layer_reports[feature_id] = report

                # Print summary
                print(f"\nFeature {feature_id}: fires {report['feature_count']} times ({100*report['feature_frequency']:.1f}%)")
                print(f"  Top 5 math tokens by precision:")
                for i, token_data in enumerate(report['top_tokens'][:5], 1):
                    print(f"    {i}. '{token_data['token']}' - Precision: {100*token_data['precision']:.1f}%, "
                          f"Count: {token_data['count']}, p: {token_data['p_value']:.2e}")
            else:
                if report is None:
                    print(f"\nFeature {feature_id}: never fired in {layer_name} layer")
                else:
                    print(f"\nFeature {feature_id}: no significant math token correlations")
                layer_reports[feature_id] = None

        all_reports[layer_name] = layer_reports

    # Save results
    output = {
        'first_test_example': phase3_data['first_test_example'],
        'analysis_method': 'precision_based_math_tokens_only',
        'p_value_threshold': 0.01,
        'feature_reports_by_layer': all_reports
    }

    output_path = Path('./phase4_precision_based_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    # Generate summary report
    generate_summary_report(output, phase3_data)


def generate_summary_report(output, phase3_data):
    """Generate a readable summary report"""
    report_path = Path('./phase4_precision_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Precision-Based Feature Analysis (Math Tokens Only)\n\n")
        f.write("## First GSM8K Test Example\n\n")
        f.write(f"**Question:** {phase3_data['first_test_example']['question']}\n\n")
        f.write(f"**Answer:** {phase3_data['first_test_example']['answer']}\n\n")
        f.write("---\n\n")

        f.write("## Metric: Precision = P(token | feature fires)\n\n")
        f.write("*\"When this feature activates, what % of the time does this math token appear in the reference CoT?\"*\n\n")
        f.write("- **Token Filter:** Only numbers, operators (+, -, *, /, =, <, >), and symbols ($, %)\n")
        f.write("- **Significance Filter:** p-value < 0.01 (chi-squared test)\n")
        f.write("- **Ranking:** By precision (highest first)\n\n")
        f.write("---\n\n")

        # For each feature, show results across layers
        unique_features = sorted([int(f) for f in output['feature_reports_by_layer']['late'].keys()])

        for feature_id in unique_features:
            f.write(f"## Feature {feature_id}\n\n")

            for layer in ['early', 'middle', 'late']:
                layer_num = {'early': 4, 'middle': 8, 'late': 14}[layer]
                report = output['feature_reports_by_layer'][layer].get(str(feature_id))

                if not report:
                    f.write(f"**{layer.upper()} (L{layer_num}):** No significant correlations\n\n")
                    continue

                freq = report['feature_frequency']
                count = report['feature_count']

                f.write(f"**{layer.upper()} (L{layer_num}):** Fires {count}/7473 times ({100*freq:.1f}%)\n\n")

                if report['top_tokens']:
                    f.write("| Rank | Token | Precision | Count | P-value | Sig |\n")
                    f.write("|------|-------|-----------|-------|---------|-----|\n")

                    for rank, token_data in enumerate(report['top_tokens'][:15], 1):
                        sig = "✓✓✓" if token_data['p_value'] < 0.001 else "✓✓" if token_data['p_value'] < 0.01 else "✓"
                        token_clean = token_data['token'].replace('|', '\\|')
                        precision_pct = 100 * token_data['precision']
                        f.write(f"| {rank} | '{token_clean}' | {precision_pct:.1f}% | {token_data['count']} | "
                               f"{token_data['p_value']:.2e} | {sig} |\n")

                    f.write("\n")

            f.write("\n")

    print(f"Summary report saved to: {report_path}")


if __name__ == '__main__':
    main()
