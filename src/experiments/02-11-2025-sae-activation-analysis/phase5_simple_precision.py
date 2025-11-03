#!/usr/bin/env python3
"""
Phase 5: Simple Precision Analysis - Numbers and Basic Operators Only

Include ONLY: numbers (0-9, multi-digit) and basic operators (+, -, *, /)
No p-value filtering - just rank by precision
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


print("Loading LLAMA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def is_number_or_operator(token):
    """
    Filter to only numbers and basic operators.
    Keep: 0-9 (any digit), multi-digit numbers, +, -, *, /
    """
    token_clean = token.strip()

    # Any number (single or multi-digit)
    if re.match(r'^\d+$', token_clean):
        return True

    # Basic operators only
    if token_clean in {'+', '-', '*', '/'}:
        return True

    return False


def extract_numbers_and_operators(reference_answer, tokenizer):
    """Extract only numbers and basic operators"""
    before_answer = reference_answer.split('####')[0].strip()
    token_ids = tokenizer.encode(before_answer, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Filter to numbers and operators only
    filtered_tokens = [t for t in tokens if is_number_or_operator(t)]

    return filtered_tokens


def build_cooccurrence_simple(phase1_data, feature_list, layer_name='late', n_examples=None):
    """Build co-occurrence matrix for numbers and operators only"""
    if n_examples is None:
        n_examples = len(phase1_data['results'])

    print(f"\nBuilding co-occurrence for {len(feature_list)} features (numbers & +,-,*,/ only) in {layer_name} layer...")

    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train = gsm8k_dataset['train']

    feature_counts = Counter()
    token_counts = Counter()
    cooccurrence = defaultdict(Counter)

    for example_idx in tqdm(range(n_examples), desc=f"Processing {layer_name}"):
        example = phase1_data['results'][example_idx]
        reference_answer = gsm8k_train[example_idx]['answer']

        # Extract ONLY numbers and operators
        filtered_tokens = extract_numbers_and_operators(reference_answer, tokenizer)
        unique_tokens = set(filtered_tokens)

        # Get features that fired
        fired_features = set()
        for position_data in example['activations']:
            layer_data = position_data['layers'][layer_name]
            fired_features.update([f for f in layer_data['firing_indices'] if f in feature_list])

        for feature_id in fired_features:
            feature_counts[feature_id] += 1

        for token in unique_tokens:
            token_counts[token] += 1

        for feature_id in fired_features:
            for token in unique_tokens:
                cooccurrence[feature_id][token] += 1

    return {
        'n_examples': n_examples,
        'feature_counts': dict(feature_counts),
        'token_counts': dict(token_counts),
        'cooccurrence': {k: dict(v) for k, v in cooccurrence.items()}
    }


def compute_simple_precision(feature_id, cooccurrence_data):
    """
    Compute precision = P(token | feature) for all tokens.
    No statistical filtering - just report all.
    """
    if feature_id not in cooccurrence_data['cooccurrence']:
        return None

    feature_count = cooccurrence_data['feature_counts'].get(feature_id, 0)

    if feature_count == 0:
        return None

    feature_tokens = cooccurrence_data['cooccurrence'][feature_id]

    results = []

    for token, count in feature_tokens.items():
        # Precision = P(token | feature fires)
        precision = count / feature_count

        results.append({
            'token': token,
            'precision': precision,
            'count': count
        })

    # Sort by precision (highest first)
    results.sort(key=lambda x: x['precision'], reverse=True)

    return {
        'feature_id': feature_id,
        'feature_count': feature_count,
        'feature_frequency': feature_count / cooccurrence_data['n_examples'],
        'top_tokens': results
    }


def main():
    print("="*80)
    print("PHASE 5: Simple Precision Analysis (Numbers & +,-,*,/ Only)")
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
        print(f"Analyzing {layer_name.upper()} layer - Numbers & Operators Only")
        print(f"{'='*80}")

        # Build co-occurrence
        cooccurrence_data = build_cooccurrence_simple(
            phase1_data,
            feature_list=set(unique_features),
            layer_name=layer_name,
            n_examples=7473
        )

        print(f"\nStatistics:")
        print(f"  Unique tokens found: {len(cooccurrence_data['token_counts'])}")
        token_list = sorted(cooccurrence_data['token_counts'].keys())
        print(f"  Tokens: {token_list}")

        layer_reports = {}

        # Generate report for each feature
        for feature_id in unique_features:
            report = compute_simple_precision(feature_id, cooccurrence_data)

            if report is not None and report['top_tokens']:
                layer_reports[feature_id] = report

                # Print summary
                print(f"\nFeature {feature_id}: fires {report['feature_count']} times ({100*report['feature_frequency']:.1f}%)")
                print(f"  Top 10 by precision:")
                for i, token_data in enumerate(report['top_tokens'][:10], 1):
                    print(f"    {i}. '{token_data['token']}' - {100*token_data['precision']:.1f}% (n={token_data['count']})")
            else:
                print(f"\nFeature {feature_id}: never fired in {layer_name} layer")
                layer_reports[feature_id] = None

        all_reports[layer_name] = layer_reports

    # Save results
    output = {
        'first_test_example': phase3_data['first_test_example'],
        'analysis_method': 'simple_precision_numbers_and_operators',
        'tokens_included': 'digits 0-9, multi-digit numbers, +, -, *, /',
        'feature_reports_by_layer': all_reports
    }

    output_path = Path('./phase5_simple_precision_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    # Generate summary report
    generate_summary_report(output, phase3_data)


def generate_summary_report(output, phase3_data):
    """Generate a readable summary report"""
    report_path = Path('./phase5_simple_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Simple Precision Analysis (Numbers & Basic Operators)\n\n")
        f.write("## First GSM8K Test Example\n\n")
        f.write(f"**Question:** {phase3_data['first_test_example']['question']}\n\n")
        f.write(f"**Answer:** {phase3_data['first_test_example']['answer']}\n\n")
        f.write("---\n\n")

        f.write("## Metric: Precision = P(token | feature fires)\n\n")
        f.write("*\"When this feature activates, what % of the time does this token appear?\"*\n\n")
        f.write("- **Tokens:** Digits (0-9), multi-digit numbers, and operators (+, -, *, /)\n")
        f.write("- **No filtering:** All correlations shown, ranked by precision\n\n")
        f.write("---\n\n")

        # For each feature, show results across layers
        unique_features = sorted([int(f) for f in output['feature_reports_by_layer']['late'].keys()
                                 if output['feature_reports_by_layer']['late'][str(f)] is not None])

        for feature_id in unique_features:
            f.write(f"## Feature {feature_id}\n\n")

            for layer in ['early', 'middle', 'late']:
                layer_num = {'early': 4, 'middle': 8, 'late': 14}[layer]
                report = output['feature_reports_by_layer'][layer].get(str(feature_id))

                if not report:
                    f.write(f"**{layer.upper()} (L{layer_num}):** Never fired\n\n")
                    continue

                freq = report['feature_frequency']
                count = report['feature_count']

                f.write(f"**{layer.upper()} (L{layer_num}):** Fires {count}/7473 times ({100*freq:.1f}%)\n\n")

                if report['top_tokens']:
                    f.write("| Rank | Token | Precision | Count |\n")
                    f.write("|------|-------|-----------|-------|\n")

                    for rank, token_data in enumerate(report['top_tokens'][:20], 1):
                        token_clean = token_data['token'].replace('|', '\\|')
                        precision_pct = 100 * token_data['precision']
                        f.write(f"| {rank} | '{token_clean}' | {precision_pct:.1f}% | {token_data['count']} |\n")

                    f.write("\n")

            f.write("\n")

    print(f"Summary report saved to: {report_path}")


if __name__ == '__main__':
    main()
