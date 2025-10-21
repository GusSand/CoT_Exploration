"""
Filter to the simplest 50 problem pairs for easy manual review.

Scores pairs by simplicity:
- Fewer calculation steps = better
- Smaller numbers = better
- Clear arithmetic operations = better

Usage:
    python filter_simple_pairs.py --input problem_pairs_for_review.json --output simple_pairs_for_review.json
"""

import json
import re
import argparse
from typing import Dict, Tuple


def score_simplicity(pair: Dict) -> Tuple[float, str]:
    """Score how simple a problem pair is (higher = simpler).

    Args:
        pair: Problem pair dict

    Returns:
        (score, reason) - Higher score = simpler problem
    """
    score = 100.0
    reasons = []

    full_answer = pair['clean']['full_answer']
    question = pair['clean']['question']
    changed_from, changed_to = pair['corrupted']['changed_number'].split(' -> ')

    # Count calculation steps (lower = better)
    calc_steps = len(re.findall(r'=\s*<<', full_answer))
    if calc_steps == 0:
        calc_steps = len(re.findall(r'=\s*\d+', full_answer))

    if calc_steps <= 2:
        score += 50
        reasons.append("simple (1-2 steps)")
    elif calc_steps <= 4:
        score += 20
        reasons.append("moderate (3-4 steps)")
    else:
        score -= 30
        reasons.append(f"complex ({calc_steps} steps)")

    # Prefer small number changes
    try:
        from_num = int(changed_from)
        to_num = int(changed_to)
        if from_num < 100 and to_num < 100:
            score += 20
            reasons.append("small numbers")
        elif from_num < 1000:
            score += 10
    except:
        pass

    # Prefer short questions
    if len(question) < 200:
        score += 15
        reasons.append("short question")
    elif len(question) > 400:
        score -= 10

    # Check if number appears early in question (more likely to be simple)
    first_occurrence = question.find(changed_from)
    if first_occurrence < len(question) / 3:
        score += 10
        reasons.append("number appears early")

    # Penalize percentages, fractions, ratios (more complex)
    if any(word in question.lower() for word in ['percent', '%', 'ratio', 'fraction', 'half', 'third']):
        score -= 20
        reasons.append("has percentages/ratios")

    # Penalize time-based problems (often multi-step)
    if any(word in question.lower() for word in ['hour', 'minute', 'day', 'week', 'month', 'year']):
        score -= 5

    # Penalize if changed number doesn't appear in calculations
    if changed_from not in full_answer:
        score -= 30
        reasons.append("number not in solution")

    return score, "; ".join(reasons)


def filter_simple_pairs(pairs: list, target_count: int = 50) -> list:
    """Filter to the simplest N pairs.

    Args:
        pairs: List of problem pairs
        target_count: Number of simple pairs to return

    Returns:
        Filtered list of simplest pairs
    """
    # Score all pairs
    scored_pairs = []
    for pair in pairs:
        score, reasons = score_simplicity(pair)
        pair['simplicity_score'] = score
        pair['simplicity_reasons'] = reasons
        scored_pairs.append((score, pair))

    # Sort by score (descending)
    scored_pairs.sort(reverse=True, key=lambda x: x[0])

    # Take top N
    simple_pairs = [pair for score, pair in scored_pairs[:target_count]]

    return simple_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='problem_pairs_for_review.json')
    parser.add_argument('--output', type=str, default='simple_pairs_for_review.json')
    parser.add_argument('--count', type=int, default=50, help='Number of simple pairs to select')
    args = parser.parse_args()

    # Load pairs
    print(f"Loading pairs from {args.input}...")
    with open(args.input, 'r') as f:
        pairs = json.load(f)

    print(f"Scoring {len(pairs)} pairs by simplicity...")
    simple_pairs = filter_simple_pairs(pairs, args.count)

    # Show top 5 as examples
    print(f"\n{'='*70}")
    print(f"TOP 5 SIMPLEST PAIRS (examples)")
    print(f"{'='*70}\n")

    for i, pair in enumerate(simple_pairs[:5]):
        print(f"Pair {pair['pair_id']} (Score: {pair['simplicity_score']:.0f}):")
        print(f"  Question: {pair['clean']['question'][:80]}...")
        print(f"  Clean answer: {pair['clean']['answer']}")
        print(f"  Changed: {pair['corrupted']['changed_number']}")
        print(f"  Reasons: {pair['simplicity_reasons']}")
        print()

    # Save filtered pairs
    with open(args.output, 'w') as f:
        json.dump(simple_pairs, f, indent=2)

    print(f"{'='*70}")
    print(f"✓ Selected {len(simple_pairs)} simplest pairs")
    print(f"✓ Saved to {args.output}")
    print(f"\nNext step: Manually review {args.output} and fill in:")
    print(f"  1. corrupted.answer (calculate from changed number)")
    print(f"  2. review_status ('approved' or 'rejected')")
    print(f"\nThen run: python generate_pairs.py --filter {args.output}")


if __name__ == "__main__":
    main()
