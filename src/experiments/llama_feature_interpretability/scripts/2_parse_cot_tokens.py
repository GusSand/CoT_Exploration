"""
Parse CoT tokens from LLaMA predictions for correlation analysis.

LLaMA's CoT sequences are already tokenized in the extracted features file.
Each sequence is a list like: ['3*60=180', '180*60=10800', '54000/10800=5']

This script:
1. Loads the extracted features
2. Parses individual tokens from CoT sequences
3. Builds vocabulary and token-to-problem mapping

Output: llama_cot_tokens.json
"""

import json
import re
import torch
from pathlib import Path
from collections import defaultdict, Counter


def extract_tokens_from_sequence(cot_sequence):
    """
    Extract individual tokens from a CoT sequence.

    Example:
      ['3*60=180', '180*60=10800'] → ['3', '*', '60', '=', '180', '180', '*', '60', '=', '10800']

    Returns:
        List of token strings
    """
    all_tokens = []

    for step in cot_sequence:
        # Split on operators while keeping them
        # Pattern matches: numbers, operators (+, -, *, /, =), decimals, parentheses
        tokens = re.findall(r'\d+\.?\d*|[+\-*/=()]', step)
        all_tokens.extend(tokens)

    return all_tokens


def parse_all_cot_tokens():
    """Parse CoT tokens from all LLaMA problems."""
    print("="*80)
    print("PARSING COT TOKENS FROM LLAMA PREDICTIONS")
    print("="*80)

    # Load extracted features
    data_path = Path('src/experiments/llama_feature_interpretability/data/llama_extracted_features.pt')
    print(f"\nLoading data from: {data_path}")

    data = torch.load(data_path, weights_only=False)
    metadata = data['metadata']
    config = data['config']

    # Get unique problem IDs and their CoT sequences
    problem_cot_map = {}  # problem_id → cot_sequence
    for i, problem_id in enumerate(metadata['problem_ids']):
        if problem_id not in problem_cot_map:
            problem_cot_map[problem_id] = metadata['cot_sequences'][i]

    num_problems = len(problem_cot_map)
    print(f"Loaded {num_problems} unique problems")

    # Parse tokens from each problem
    token_to_problems = defaultdict(set)  # token → set of problem IDs
    problem_to_tokens = {}  # problem_id → list of tokens
    all_tokens = []

    print(f"\nParsing calculation tokens...")
    for problem_id, cot_sequence in problem_cot_map.items():
        # Extract CoT tokens
        tokens = extract_tokens_from_sequence(cot_sequence)

        # Store mapping
        problem_to_tokens[problem_id] = tokens
        all_tokens.extend(tokens)

        # Store reverse mapping
        for token in tokens:
            token_to_problems[token].add(problem_id)

    # Convert sets to lists for JSON serialization
    token_to_problems = {k: sorted(list(v)) for k, v in token_to_problems.items()}

    # Compute statistics
    token_counts = Counter(all_tokens)
    unique_tokens = len(token_counts)
    total_tokens = len(all_tokens)

    print(f"\n✓ Parsed tokens")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique tokens: {unique_tokens:,}")
    print(f"  Avg tokens per problem: {total_tokens / num_problems:.1f}")

    # Classify tokens
    numbers = []
    operators = []
    other = []

    for token in token_counts.keys():
        if re.match(r'^\d+\.?\d*$', token):
            numbers.append(token)
        elif token in ['+', '-', '*', '/', '=', '(', ')']:
            operators.append(token)
        else:
            other.append(token)

    print(f"\nToken classification:")
    print(f"  Numbers: {len(numbers)} unique ({sum(token_counts[n] for n in numbers)} total)")
    print(f"  Operators: {len(operators)} unique ({sum(token_counts[o] for o in operators)} total)")
    print(f"  Other: {len(other)} unique ({sum(token_counts[o] for o in other)} total)")

    # Top tokens
    print(f"\nTop 20 most frequent tokens:")
    for token, count in token_counts.most_common(20):
        pct = (count / total_tokens) * 100
        num_problems = len(token_to_problems[token])
        print(f"  '{token}': {count} occurrences ({pct:.1f}%), {num_problems} problems")

    # Save results
    output_dir = Path('src/experiments/llama_feature_interpretability/data')
    output_path = output_dir / 'llama_cot_tokens.json'

    print(f"\nSaving to: {output_path}")

    output_data = {
        'metadata': {
            'model': 'llama-3.2-1b',
            'num_problems': num_problems,
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'avg_tokens_per_problem': total_tokens / num_problems,
            'num_numbers': len(numbers),
            'num_operators': len(operators),
            'num_other': len(other),
        },
        'token_to_problems': token_to_problems,  # token → [problem_ids]
        'problem_to_tokens': problem_to_tokens,  # problem_id → [tokens]
        'token_counts': dict(token_counts),      # token → count
        'token_classification': {
            'numbers': sorted(numbers),
            'operators': sorted(operators),
            'other': sorted(other),
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Saved ({size_kb:.1f} KB)")

    print("\n" + "="*80)
    print("PARSING COMPLETE!")
    print("="*80)
    print(f"  Output: {output_path}")
    print("="*80)


if __name__ == '__main__':
    parse_all_cot_tokens()
