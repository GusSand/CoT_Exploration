#!/usr/bin/env python3
"""
Phase 3: Generate Unigram Reports for Identified Features

Load the features from phase3_first_test_example_features.json and generate
unigram correlation reports for each unique feature.
"""

import json
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


def extract_tokens_only(reference_answer, tokenizer):
    """Extract ONLY individual tokens"""
    before_answer = reference_answer.split('####')[0].strip()
    token_ids = tokenizer.encode(before_answer, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    return tokens


def build_cooccurrence_for_features(phase1_data, feature_list, layer_name='late', n_examples=None):
    """Build co-occurrence matrix for specific features only"""
    if n_examples is None:
        n_examples = len(phase1_data['results'])

    print(f"\nBuilding co-occurrence for {len(feature_list)} features in {layer_name} layer...")

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

        tokens = extract_tokens_only(reference_answer, tokenizer)
        unique_tokens = set(tokens)

        # Get features that fired (only track features we care about)
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


def compute_pmi_for_feature(feature_id, cooccurrence_data, min_cooccurrence=10):
    """Compute PMI for a single feature"""
    if feature_id not in cooccurrence_data['cooccurrence']:
        return None

    n_examples = cooccurrence_data['n_examples']
    feature_count = cooccurrence_data['feature_counts'].get(feature_id, 0)

    if feature_count == 0:
        return None

    token_counts = cooccurrence_data['token_counts']
    feature_tokens = cooccurrence_data['cooccurrence'][feature_id]

    pmi_scores = {}

    for token, count in feature_tokens.items():
        if count < min_cooccurrence:
            continue

        p_joint = count / n_examples
        p_feature = feature_count / n_examples
        p_token = token_counts[token] / n_examples

        pmi = math.log2(p_joint / (p_feature * p_token))

        # Compute chi-squared
        feature_to_examples = set(cooccurrence_data['feature_to_examples'].get(feature_id, []))
        token_to_examples = set(cooccurrence_data['token_to_examples'].get(token, []))

        a = len(feature_to_examples & token_to_examples)
        b = len(feature_to_examples - token_to_examples)
        c = len(token_to_examples - feature_to_examples)
        d = n_examples - a - b - c

        chi2, p_value = chi2_contingency_manual(np.array([[a, b], [c, d]]))

        pmi_scores[token] = {
            'pmi': pmi,
            'count': count,
            'p_value': p_value,
            'chi2': chi2
        }

    # Sort by PMI
    sorted_tokens = sorted(pmi_scores.items(), key=lambda x: x[1]['pmi'], reverse=True)

    return {
        'feature_id': feature_id,
        'feature_count': feature_count,
        'top_tokens': sorted_tokens[:30]  # Top 30
    }


def main():
    print("="*80)
    print("PHASE 3: Generate Unigram Reports for All Features")
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

    # We need to analyze each layer separately since features may behave differently
    all_reports = {}

    for layer_name in ['early', 'middle', 'late']:
        print(f"\n{'='*80}")
        print(f"Analyzing {layer_name.upper()} layer")
        print(f"{'='*80}")

        # Build co-occurrence for these features only
        cooccurrence_data = build_cooccurrence_for_features(
            phase1_data,
            feature_list=set(unique_features),
            layer_name=layer_name,
            n_examples=7473
        )

        layer_reports = {}

        # Generate report for each feature
        for feature_id in unique_features:
            print(f"\nAnalyzing Feature {feature_id} in {layer_name} layer...")

            report = compute_pmi_for_feature(feature_id, cooccurrence_data, min_cooccurrence=10)

            if report is not None:
                layer_reports[feature_id] = report

                # Print summary
                print(f"  Feature {feature_id}: fired {report['feature_count']} times")
                if report['top_tokens']:
                    print(f"  Top 5 tokens:")
                    for token, scores in report['top_tokens'][:5]:
                        print(f"    '{token}' - PMI: {scores['pmi']:.3f}, count: {scores['count']}, p: {scores['p_value']:.2e}")
            else:
                print(f"  Feature {feature_id}: never fired in {layer_name} layer")
                layer_reports[feature_id] = None

        all_reports[layer_name] = layer_reports

    # Save comprehensive results
    output = {
        'first_test_example': phase3_data['first_test_example'],
        'top_features_per_position_layer': phase3_data['top_features_analysis'],
        'feature_reports_by_layer': all_reports
    }

    output_path = Path('./phase3_comprehensive_feature_reports.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Comprehensive reports saved to: {output_path}")
    print(f"{'='*80}")

    # Generate readable markdown report
    generate_markdown_report(output, phase3_data)


def generate_markdown_report(output, phase3_data):
    """Generate a readable markdown report"""
    report_path = Path('./phase3_feature_interpretation_report.md')

    with open(report_path, 'w') as f:
        f.write("# Feature Interpretation Report\n\n")
        f.write("## First GSM8K Test Example Analysis\n\n")

        f.write(f"**Question:** {phase3_data['first_test_example']['question']}\n\n")
        f.write(f"**Answer:** {phase3_data['first_test_example']['answer']}\n\n")

        f.write("---\n\n")

        # For each position
        for pos in [0, 1]:
            f.write(f"## Position {pos}\n\n")

            # For each layer
            for layer_name in ['early', 'middle', 'late']:
                f.write(f"### {layer_name.upper()} Layer (L{4 if layer_name=='early' else 8 if layer_name=='middle' else 14})\n\n")

                pos_data = phase3_data['top_features_analysis']['positions'][pos]['layers'][layer_name]

                f.write("#### Top 20 Firing Features\n\n")
                f.write("| Rank | Feature | Activation | Top 5 Correlated Tokens (Unigrams) |\n")
                f.write("|------|---------|------------|-------------------------------------|\n")

                for rank, (feat_idx, feat_val) in enumerate(zip(
                    pos_data['top_20_indices'],
                    pos_data['top_20_values']
                ), 1):
                    # Get report for this feature
                    report = output['feature_reports_by_layer'][layer_name].get(feat_idx)

                    if report and report['top_tokens']:
                        top_5 = ", ".join([f"'{t}' ({s['pmi']:.2f})" for t, s in report['top_tokens'][:5]])
                    else:
                        top_5 = "N/A"

                    f.write(f"| {rank} | {feat_idx} | {feat_val:.4f} | {top_5} |\n")

                f.write("\n")

        # Detailed feature interpretations
        f.write("---\n\n")
        f.write("## Detailed Feature Interpretations\n\n")

        for layer_name in ['early', 'middle', 'late']:
            f.write(f"### {layer_name.upper()} Layer\n\n")

            layer_reports = output['feature_reports_by_layer'][layer_name]

            for feature_id in phase3_data['unique_features']:
                report = layer_reports.get(feature_id)

                if report is None:
                    continue

                f.write(f"#### Feature {feature_id}\n\n")
                f.write(f"- **Activation frequency:** {report['feature_count']}/7473 examples ({100*report['feature_count']/7473:.1f}%)\n\n")

                if report['top_tokens']:
                    f.write("**Top 15 Correlated Tokens:**\n\n")
                    f.write("| Rank | Token | PMI | Count | p-value | Significance |\n")
                    f.write("|------|-------|-----|-------|---------|-------------|\n")

                    for rank, (token, scores) in enumerate(report['top_tokens'][:15], 1):
                        sig = "✓✓✓" if scores['p_value'] < 0.001 else "✓✓" if scores['p_value'] < 0.01 else "✓" if scores['p_value'] < 0.05 else ""
                        token_display = token.replace("|", "\\|")  # Escape pipe for markdown
                        f.write(f"| {rank} | '{token_display}' | {scores['pmi']:.3f} | {scores['count']} | {scores['p_value']:.2e} | {sig} |\n")

                    f.write("\n")

    print(f"Markdown report saved to: {report_path}")


if __name__ == '__main__':
    main()
