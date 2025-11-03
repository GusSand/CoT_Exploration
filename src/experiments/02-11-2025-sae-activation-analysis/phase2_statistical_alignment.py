#!/usr/bin/env python3
"""
Phase 2 (Alternative): Statistical Alignment of SAE Features with Reference CoT Elements

For each SAE feature, compute statistical correlation with elements of reference CoT:
- Tokenize reference solutions using LLAMA tokenizer
- Extract tokens, bigrams, and non-trivial categories
- Compute PMI (Pointwise Mutual Information) between features and CoT elements
- Supplement with chi-squared test for statistical significance

This answers: "Which reference CoT elements does latent X correlate with?"
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

    # Approximate p-value using chi2 CDF approximation
    # For large chi2, p-value ≈ 0; for small chi2, p-value ≈ 1
    # Simple approximation: p ≈ exp(-chi2/2) for dof=1
    if dof == 1:
        p_value = math.exp(-chi2 / 2)
    else:
        # Very rough approximation
        p_value = max(0, 1 - (chi2 / (chi2 + dof)))

    return chi2, p_value, dof, expected

print("Loading LLAMA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


def extract_reference_cot_elements(reference_answer, tokenizer):
    """
    Extract elements from reference CoT solution.

    Returns:
    - tokens: List of individual tokens
    - bigrams: List of token bigrams
    - categories: Non-trivial categories (e.g., MULTIDIGIT_NUMBER)
    """
    # Extract calculation parts from <<...>>
    calculations = re.findall(r'<<([^>]+)>>', reference_answer)

    # Also include text before ####
    before_answer = reference_answer.split('####')[0].strip()

    # Combine all text to analyze
    full_text = before_answer

    # Tokenize using LLAMA tokenizer
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Generate bigrams
    bigrams = []
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + tokens[i+1]
        bigrams.append(bigram)

    # Extract non-trivial categories
    categories = []

    # Multi-digit numbers (2+ digits)
    for token in tokens:
        if re.match(r'^\d{2,}$', token.strip()):
            categories.append('MULTIDIGIT_NUMBER')

    # Decimal numbers
    for token in tokens:
        if re.match(r'^\d+\.\d+$', token.strip()):
            categories.append('DECIMAL_NUMBER')

    # Calculation expressions (from <<...>>)
    for calc in calculations:
        categories.append(f'CALC:{calc}')

        # Also extract operator sequences
        ops = re.findall(r'[\+\-\*/=]', calc)
        if len(ops) >= 2:
            categories.append(f'OPS:{"".join(ops)}')

    return {
        'tokens': tokens,
        'bigrams': bigrams,
        'categories': categories
    }


def build_cooccurrence_matrix(phase1_data, layer_name='late', n_examples=None):
    """
    Build co-occurrence matrix:
    - feature_counts[feature_id] = how many examples feature fired in
    - element_counts[element] = how many examples contain this element
    - cooccurrence[feature_id][element] = how many examples have both
    """
    if n_examples is None:
        n_examples = len(phase1_data['results'])

    print(f"\nBuilding co-occurrence matrix for {layer_name} layer across {n_examples} examples...")

    # Load GSM8K for reference CoTs
    print("Loading GSM8K training set...")
    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train = gsm8k_dataset['train']

    feature_counts = Counter()
    element_counts = Counter()
    cooccurrence = defaultdict(Counter)

    # Track which features fired in which examples
    feature_to_examples = defaultdict(set)
    element_to_examples = defaultdict(set)

    for example_idx in tqdm(range(n_examples), desc="Processing examples"):
        example = phase1_data['results'][example_idx]
        reference_answer = gsm8k_train[example_idx]['answer']

        # Extract reference CoT elements
        ref_elements = extract_reference_cot_elements(reference_answer, tokenizer)

        # Combine all element types
        all_elements = (
            [('TOKEN', t) for t in ref_elements['tokens']] +
            [('BIGRAM', b) for b in ref_elements['bigrams']] +
            [('CATEGORY', c) for c in ref_elements['categories']]
        )

        # Get unique elements for this example
        unique_elements = set(all_elements)

        # Get all features that fired in this example (across all 6 CoT positions)
        fired_features = set()
        for position_data in example['activations']:
            layer_data = position_data['layers'][layer_name]
            fired_features.update(layer_data['firing_indices'])

        # Update counts
        for feature_id in fired_features:
            feature_counts[feature_id] += 1
            feature_to_examples[feature_id].add(example_idx)

        for element in unique_elements:
            element_counts[element] += 1
            element_to_examples[element].add(example_idx)

        # Update co-occurrence
        for feature_id in fired_features:
            for element in unique_elements:
                cooccurrence[feature_id][element] += 1

    return {
        'n_examples': n_examples,
        'feature_counts': dict(feature_counts),
        'element_counts': dict(element_counts),
        'cooccurrence': {k: dict(v) for k, v in cooccurrence.items()},
        'feature_to_examples': {k: list(v) for k, v in feature_to_examples.items()},
        'element_to_examples': {k: list(v) for k, v in element_to_examples.items()}
    }


def compute_pmi(cooccurrence_data, min_cooccurrence=5):
    """
    Compute PMI (Pointwise Mutual Information) for each (feature, element) pair.

    PMI(feature, element) = log₂(P(feature, element) / (P(feature) × P(element)))

    Higher PMI = stronger association
    """
    n_examples = cooccurrence_data['n_examples']
    feature_counts = cooccurrence_data['feature_counts']
    element_counts = cooccurrence_data['element_counts']
    cooccurrence = cooccurrence_data['cooccurrence']

    print(f"\nComputing PMI for all (feature, element) pairs...")
    print(f"Filtering pairs with at least {min_cooccurrence} co-occurrences...")

    pmi_scores = {}

    for feature_id, elements in tqdm(cooccurrence.items(), desc="Computing PMI"):
        pmi_scores[feature_id] = {}

        for element, count in elements.items():
            if count < min_cooccurrence:
                continue

            # P(feature, element)
            p_joint = count / n_examples

            # P(feature)
            p_feature = feature_counts[feature_id] / n_examples

            # P(element)
            p_element = element_counts[element] / n_examples

            # PMI = log₂(P(f,e) / (P(f) × P(e)))
            pmi = math.log2(p_joint / (p_feature * p_element))

            pmi_scores[feature_id][element] = {
                'pmi': pmi,
                'count': count,
                'p_joint': p_joint,
                'p_feature': p_feature,
                'p_element': p_element
            }

    return pmi_scores


def compute_chi_squared(feature_id, element, cooccurrence_data):
    """
    Compute chi-squared test for independence between feature and element.

    Contingency table:
                    element present | element absent
    feature fires       a          |      b
    feature doesn't     c          |      d
    """
    n_examples = cooccurrence_data['n_examples']
    feature_to_examples = set(cooccurrence_data['feature_to_examples'].get(feature_id, []))
    element_to_examples = set(cooccurrence_data['element_to_examples'].get(element, []))

    # Build contingency table
    a = len(feature_to_examples & element_to_examples)  # both present
    b = len(feature_to_examples - element_to_examples)  # feature only
    c = len(element_to_examples - feature_to_examples)  # element only
    d = n_examples - a - b - c  # neither

    contingency_table = np.array([[a, b], [c, d]])

    chi2, p_value, dof, expected = chi2_contingency_manual(contingency_table)

    return {
        'chi2': chi2,
        'p_value': p_value,
        'contingency': [[a, b], [c, d]]
    }


def analyze_feature(feature_id, pmi_scores, cooccurrence_data, top_k=20):
    """
    Analyze a specific feature: what elements does it correlate with?
    """
    if feature_id not in pmi_scores:
        print(f"Feature {feature_id} never fired in analyzed examples.")
        return None

    feature_elements = pmi_scores[feature_id]

    # Sort by PMI (highest = strongest association)
    sorted_elements = sorted(feature_elements.items(),
                            key=lambda x: x[1]['pmi'],
                            reverse=True)

    print(f"\n{'='*80}")
    print(f"Feature {feature_id}: Top {top_k} Correlated Elements (by PMI)")
    print(f"{'='*80}\n")

    print(f"Feature fired in {cooccurrence_data['feature_counts'][feature_id]} / {cooccurrence_data['n_examples']} examples\n")

    print(f"{'Rank':<6} {'Type':<10} {'Element':<40} {'PMI':<8} {'Count':<8} {'p-value':<10}")
    print("-" * 90)

    results = []

    for rank, (element, scores) in enumerate(sorted_elements[:top_k], 1):
        element_type, element_value = element

        # Compute chi-squared for statistical significance
        chi2_result = compute_chi_squared(feature_id, element, cooccurrence_data)

        element_str = str(element_value)[:37] + "..." if len(str(element_value)) > 40 else str(element_value)

        print(f"{rank:<6} {element_type:<10} {element_str:<40} {scores['pmi']:<8.3f} {scores['count']:<8} {chi2_result['p_value']:<10.2e}")

        results.append({
            'rank': rank,
            'element_type': element_type,
            'element_value': element_value,
            'pmi': scores['pmi'],
            'count': scores['count'],
            'p_joint': scores['p_joint'],
            'p_feature': scores['p_feature'],
            'p_element': scores['p_element'],
            'chi2': chi2_result['chi2'],
            'p_value': chi2_result['p_value'],
            'contingency': chi2_result['contingency']
        })

    return results


def main():
    print("="*80)
    print("PHASE 2: Statistical Alignment of SAE Features with Reference CoT")
    print("="*80)

    # Load Phase 1 data
    phase1_data_path = Path('./phase1_sae_activations_full.json')
    print(f"\nLoading Phase 1 data from: {phase1_data_path}")
    with open(phase1_data_path, 'r') as f:
        phase1_data = json.load(f)

    print(f"Loaded {phase1_data['n_examples']} examples")

    # Build co-occurrence matrix
    layer_name = 'late'
    n_examples = 7473  # Use all examples

    cooccurrence_data = build_cooccurrence_matrix(
        phase1_data,
        layer_name=layer_name,
        n_examples=n_examples
    )

    print(f"\nStatistics:")
    print(f"  Unique features that fired: {len(cooccurrence_data['feature_counts'])}")
    print(f"  Unique elements in reference CoTs: {len(cooccurrence_data['element_counts'])}")

    # Compute PMI
    pmi_scores = compute_pmi(cooccurrence_data, min_cooccurrence=10)

    print(f"\nComputed PMI for {sum(len(v) for v in pmi_scores.values())} (feature, element) pairs")

    # Analyze specific feature (example: 3682)
    print("\n" + "="*80)
    print("Example Analysis: Feature 3682")
    print("="*80)

    feature_3682_results = analyze_feature(3682, pmi_scores, cooccurrence_data, top_k=30)

    # Analyze top 10 most frequent features
    print("\n" + "="*80)
    print("Analyzing Top 10 Most Frequent Features")
    print("="*80)

    top_features = sorted(cooccurrence_data['feature_counts'].items(),
                         key=lambda x: x[1], reverse=True)[:10]

    all_feature_analyses = {}

    for feature_id, count in top_features:
        print(f"\n{'='*80}")
        print(f"Feature {feature_id} (fired in {count} examples)")
        print(f"{'='*80}")

        results = analyze_feature(feature_id, pmi_scores, cooccurrence_data, top_k=20)
        all_feature_analyses[int(feature_id)] = results

    # Save results
    output = {
        'layer': layer_name,
        'n_examples': n_examples,
        'cooccurrence_statistics': {
            'n_unique_features': len(cooccurrence_data['feature_counts']),
            'n_unique_elements': len(cooccurrence_data['element_counts']),
            'n_pmi_pairs': sum(len(v) for v in pmi_scores.values())
        },
        'feature_3682_analysis': feature_3682_results,
        'top_features_analysis': all_feature_analyses
    }

    output_path = Path('./phase2_statistical_alignment_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    # Also save the full PMI scores for later analysis
    pmi_path = Path('./phase2_full_pmi_scores.json')

    # Convert to serializable format
    pmi_serializable = {}
    for feature_id, elements in pmi_scores.items():
        pmi_serializable[int(feature_id)] = {}
        for element, scores in elements.items():
            element_key = f"{element[0]}:{element[1]}"
            pmi_serializable[int(feature_id)][element_key] = {
                'pmi': scores['pmi'],
                'count': scores['count']
            }

    with open(pmi_path, 'w') as f:
        json.dump(pmi_serializable, f, indent=2)

    print(f"Full PMI scores saved to: {pmi_path}")


if __name__ == '__main__':
    main()
