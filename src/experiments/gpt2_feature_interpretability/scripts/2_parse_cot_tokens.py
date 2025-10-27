"""
Parse CoT tokens from GPT-2 predictions for correlation analysis.

Extracts calculation tokens from GSM8K ground truth answers:
  "<<60000/2=30000>>" → ["60000", "/", "2", "=", "30000"]

Builds vocabulary and token-to-problem mapping.

Output: gpt2_cot_tokens.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def extract_cot_tokens(answer_text):
    """
    Extract calculation tokens from GSM8K answer.

    Example:
      "<<60/2=30>>" → ["60", "/", "2", "=", "30"]
      "<<16-3-4=9>>" → ["16", "-", "3", "-", "4", "=", "9"]

    Returns:
        List of token strings
    """
    # Find all calculation blocks
    calculations = re.findall(r'<<([^>]+)>>', answer_text)

    all_tokens = []
    for calc in calculations:
        # Split on operators while keeping them
        # Pattern matches: numbers, operators (+, -, *, /, =), decimals
        tokens = re.findall(r'\d+\.?\d*|[+\-*/=()]', calc)
        all_tokens.extend(tokens)

    return all_tokens


def parse_all_cot_tokens():
    """Parse CoT tokens from all 1,000 GPT-2 predictions."""
    print("="*80)
    print("PARSING COT TOKENS FROM GPT-2 PREDICTIONS")
    print("="*80)

    # Load GPT-2 predictions
    data_path = Path('src/experiments/gpt2_shared_data/gpt2_predictions_1000_checkpoint_1000.json')
    print(f"\nLoading data from: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")

    # Parse tokens from each problem
    token_to_problems = defaultdict(set)  # token → set of problem IDs
    problem_to_tokens = {}  # problem_id → list of tokens
    all_tokens = []

    print(f"\nParsing calculation tokens...")
    for sample in samples:
        problem_id = sample['id']
        ground_truth_text = sample['ground_truth_text']

        # Extract CoT tokens
        tokens = extract_cot_tokens(ground_truth_text)

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
    print(f"  Avg tokens per problem: {total_tokens / len(samples):.1f}")

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
    output_dir = Path('src/experiments/gpt2_feature_interpretability/data')
    output_path = output_dir / 'gpt2_cot_tokens.json'

    print(f"\nSaving to: {output_path}")

    output_data = {
        'metadata': {
            'num_problems': len(samples),
            'total_tokens': total_tokens,
            'unique_tokens': unique_tokens,
            'avg_tokens_per_problem': total_tokens / len(samples),
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
