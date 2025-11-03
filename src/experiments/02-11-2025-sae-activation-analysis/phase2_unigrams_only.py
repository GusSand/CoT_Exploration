#!/usr/bin/env python3
"""
Phase 2: Statistical Alignment - UNIGRAMS ONLY

Only analyze individual tokens, exclude bigrams and categories.
"""

import torch
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

    # Expected frequencies
    expected = np.outer(row_sums, col_sums) / total

    # Chi-squared statistic
    chi2 = ((observed - expected) ** 2 / expected).sum()

    # Degrees of freedom
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)

    # Approximate p-value
    if dof == 1:
        p_value = math.exp(-chi2 / 2)
    else:
        p_value = max(0, 1 - (chi2 / (chi2 + dof)))

    return chi2, p_value, dof, expected


print("Loading LLAMA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def extract_tokens_only(reference_answer, tokenizer):
    """
    Extract ONLY individual tokens from reference CoT solution.
    """
    # Extract text before ####
    before_answer = reference_answer.split('####')[0].strip()

    # Tokenize using LLAMA tokenizer
    token_ids = tokenizer.encode(before_answer, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    return tokens


def build_cooccurrence_matrix_unigrams(phase1_data, layer_name='late', n_examples=None):
    """
    Build co-occurrence matrix for TOKENS ONLY.
    """
    if n_examples is None:
        n_examples = len(phase1_data['results'])

    print(f"\nBuilding co-occurrence matrix (UNIGRAMS ONLY) for {layer_name} layer across {n_examples} examples...")

    # Load GSM8K for reference CoTs
    print("Loading GSM8K training set...")
    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train = gsm8k_dataset['train']

    feature_counts = Counter()
    token_counts = Counter()
    cooccurrence = defaultdict(Counter)

    # Track which features fired in which examples
    feature_to_examples = defaultdict(set)
    token_to_examples = defaultdict(set)

    for example_idx in tqdm(range(n_examples), desc="Processing examples"):
        example = phase1_data['results'][example_idx]
        reference_answer = gsm8k_train[example_idx]['answer']

        # Extract ONLY tokens
        tokens = extract_tokens_only(reference_answer, tokenizer)

        # Get unique tokens for this example
        unique_tokens = set(tokens)

        # Get all features that fired in this example (across all 6 CoT positions)
        fired_features = set()
        for position_data in example['activations']:
            layer_data = position_data['layers'][layer_name]
            fired_features.update(layer_data['firing_indices'])

        # Update counts
        for feature_id in fired_features:
            feature_counts[feature_id] += 1
            feature_to_examples[feature_id].add(example_idx)

        for token in unique_tokens:
            token_counts[token] += 1
            token_to_examples[token].add(example_idx)

        # Update co-occurrence
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


def compute_pmi(cooccurrence_data, min_cooccurrence=5):
    """
    Compute PMI (Pointwise Mutual Information) for each (feature, token) pair.
    """
    n_examples = cooccurrence_data['n_examples']
    feature_counts = cooccurrence_data['feature_counts']
    token_counts = cooccurrence_data['token_counts']
    cooccurrence = cooccurrence_data['cooccurrence']

    print(f"\nComputing PMI for all (feature, token) pairs...")
    print(f"Filtering pairs with at least {min_cooccurrence} co-occurrences...")

    pmi_scores = {}

    for feature_id, tokens in tqdm(cooccurrence.items(), desc="Computing PMI"):
        pmi_scores[feature_id] = {}

        for token, count in tokens.items():
            if count < min_cooccurrence:
                continue

            # P(feature, token)
            p_joint = count / n_examples

            # P(feature)
            p_feature = feature_counts[feature_id] / n_examples

            # P(token)
            p_token = token_counts[token] / n_examples

            # PMI = log₂(P(f,t) / (P(f) × P(t)))
            pmi = math.log2(p_joint / (p_feature * p_token))

            pmi_scores[feature_id][token] = {
                'pmi': pmi,
                'count': count,
                'p_joint': p_joint,
                'p_feature': p_feature,
                'p_token': p_token
            }

    return pmi_scores


def compute_chi_squared(feature_id, token, cooccurrence_data):
    """
    Compute chi-squared test for independence between feature and token.
    """
    n_examples = cooccurrence_data['n_examples']
    feature_to_examples = set(cooccurrence_data['feature_to_examples'].get(feature_id, []))
    token_to_examples = set(cooccurrence_data['token_to_examples'].get(token, []))

    # Build contingency table
    a = len(feature_to_examples & token_to_examples)  # both present
    b = len(feature_to_examples - token_to_examples)  # feature only
    c = len(token_to_examples - feature_to_examples)  # token only
    d = n_examples - a - b - c  # neither

    contingency_table = np.array([[a, b], [c, d]])

    chi2, p_value, dof, expected = chi2_contingency_manual(contingency_table)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'contingency': [[a, b], [c, d]]
    }


def analyze_feature(feature_id, pmi_scores, cooccurrence_data, top_k=30):
    """
    Analyze a specific feature: what tokens does it correlate with?
    """
    if feature_id not in pmi_scores:
        print(f"Feature {feature_id} never fired in analyzed examples.")
        return None

    feature_tokens = pmi_scores[feature_id]

    # Sort by PMI (highest = strongest association)
    sorted_tokens = sorted(feature_tokens.items(),
                           key=lambda x: x[1]['pmi'],
                           reverse=True)

    print(f"\n{'='*80}")
    print(f"Feature {feature_id}: Top {top_k} Correlated TOKENS (by PMI)")
    print(f"{'='*80}\n")

    print(f"Feature fired in {cooccurrence_data['feature_counts'][feature_id]} / {cooccurrence_data['n_examples']} examples\n")

    print(f"{'Rank':<6} {'Token':<40} {'PMI':<8} {'Count':<8} {'p-value':<10}")
    print("-" * 80)

    results = []

    for rank, (token, scores) in enumerate(sorted_tokens[:top_k], 1):
        # Compute chi-squared for statistical significance
        chi2_result = compute_chi_squared(feature_id, token, cooccurrence_data)

        token_str = str(token)[:37] + "..." if len(str(token)) > 40 else str(token)

        print(f"{rank:<6} {token_str:<40} {scores['pmi']:<8.3f} {scores['count']:<8} {chi2_result['p_value']:<10.2e}")

        results.append({
            'rank': rank,
            'token': token,
            'pmi': scores['pmi'],
            'count': scores['count'],
            'p_joint': scores['p_joint'],
            'p_feature': scores['p_feature'],
            'p_token': scores['p_token'],
            'chi2': chi2_result['chi2'],
            'p_value': chi2_result['p_value'],
            'contingency': chi2_result['contingency']
        })

    return results


def main():
    print("="*80)
    print("PHASE 2: Statistical Alignment - UNIGRAMS ONLY")
    print("="*80)

    # Load Phase 1 data
    phase1_data_path = Path('./phase1_sae_activations_full.json')
    print(f"\nLoading Phase 1 data from: {phase1_data_path}")
    with open(phase1_data_path, 'r') as f:
        phase1_data = json.load(f)

    print(f"Loaded {phase1_data['n_examples']} examples")

    # Build co-occurrence matrix (unigrams only)
    layer_name = 'late'
    n_examples = 7473  # Use all examples

    cooccurrence_data = build_cooccurrence_matrix_unigrams(
        phase1_data,
        layer_name=layer_name,
        n_examples=n_examples
    )

    print(f"\nStatistics:")
    print(f"  Unique features that fired: {len(cooccurrence_data['feature_counts'])}")
    print(f"  Unique tokens in reference CoTs: {len(cooccurrence_data['token_counts'])}")

    # Compute PMI
    pmi_scores = compute_pmi(cooccurrence_data, min_cooccurrence=10)

    print(f"\nComputed PMI for {sum(len(v) for v in pmi_scores.values())} (feature, token) pairs")

    # Analyze specific feature (3682)
    print("\n" + "="*80)
    print("Analysis: Feature 3682 - UNIGRAMS ONLY")
    print("="*80)

    feature_3682_results = analyze_feature(3682, pmi_scores, cooccurrence_data, top_k=50)

    # Analyze top 10 most frequent features
    print("\n" + "="*80)
    print("Analyzing Top 10 Most Frequent Features - UNIGRAMS ONLY")
    print("="*80)

    top_features = sorted(cooccurrence_data['feature_counts'].items(),
                         key=lambda x: x[1], reverse=True)[:10]

    all_feature_analyses = {}

    for feature_id, count in top_features:
        print(f"\n{'='*80}")
        print(f"Feature {feature_id} (fired in {count} examples)")
        print(f"{'='*80}")

        results = analyze_feature(feature_id, pmi_scores, cooccurrence_data, top_k=30)
        all_feature_analyses[int(feature_id)] = results

    # Save results
    output = {
        'layer': layer_name,
        'n_examples': n_examples,
        'analysis_type': 'unigrams_only',
        'cooccurrence_statistics': {
            'n_unique_features': len(cooccurrence_data['feature_counts']),
            'n_unique_tokens': len(cooccurrence_data['token_counts']),
            'n_pmi_pairs': sum(len(v) for v in pmi_scores.values())
        },
        'feature_3682_analysis': feature_3682_results,
        'top_features_analysis': all_feature_analyses
    }

    output_path = Path('./phase2_unigrams_only_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
