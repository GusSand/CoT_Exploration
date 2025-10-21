"""
Auto-calculate corrupted answers from clean solutions.
Flags problems that need manual review.

Usage:
    python auto_calculate_answers.py --input problem_pairs_for_review.json --output data/problem_pairs.json
"""

import json
import re
import argparse
from typing import Dict, Tuple, Optional


def extract_calculation_from_solution(full_answer: str, changed_from: str, changed_to: str) -> Tuple[Optional[int], str, str]:
    """Extract and recalculate answer from the full solution.

    Args:
        full_answer: The full solution string with calculations
        changed_from: Original number (e.g., "16")
        changed_to: New number (e.g., "17")

    Returns:
        (calculated_answer, confidence, notes)
        confidence: "high", "medium", "low"
    """
    # Extract final answer
    final_answer_match = re.search(r'####\s*(-?\d+)', full_answer)
    if not final_answer_match:
        return None, "low", "Could not find final answer marker (####)"

    original_answer = int(final_answer_match.group(1))

    # Try to find the calculation pattern
    # Look for the first calculation that uses the changed number
    calc_pattern = rf'{changed_from}\s*[-+*/]\s*\d+|[-+*/]\s*{changed_from}'

    # Replace the number in the full solution
    modified_solution = full_answer.replace(changed_from, changed_to)

    # Try to extract all calculations in order
    calculations = re.findall(r'(\d+)\s*([-+*/])\s*(\d+)\s*=\s*<<[^>]+>>(\d+)', modified_solution)

    if not calculations:
        # Try simpler pattern
        calculations = re.findall(r'(\d+)\s*([-+*/])\s*(\d+)\s*=\s*(\d+)', modified_solution)

    if not calculations:
        return None, "low", "Could not parse calculation pattern"

    # Verify calculations
    all_correct = True
    last_result = None

    for calc in calculations:
        num1, op, num2, result = int(calc[0]), calc[1], int(calc[2]), int(calc[3])

        # Check if calculation is correct
        if op == '+':
            expected = num1 + num2
        elif op == '-':
            expected = num1 - num2
        elif op == '*':
            expected = num1 * num2
        elif op == '/':
            expected = num1 // num2

        if expected != result:
            all_correct = False
            # Recalculate
            last_result = expected
        else:
            last_result = result

    if last_result is None:
        return None, "low", "No valid calculations found"

    # Check if the change affected the answer
    diff = abs(last_result - original_answer)
    changed_num_diff = abs(int(changed_to) - int(changed_from))

    if diff == 0:
        return None, "medium", f"Change {changed_from}->{changed_to} doesn't affect answer ({original_answer})"

    # Confidence based on answer change reasonableness
    if diff == changed_num_diff or diff == changed_num_diff * 2:
        confidence = "high"
        notes = f"Simple arithmetic change: {original_answer} -> {last_result}"
    elif diff < changed_num_diff * 10:
        confidence = "high"
        notes = f"Calculated from solution: {original_answer} -> {last_result}"
    else:
        confidence = "medium"
        notes = f"Large change detected: {original_answer} -> {last_result}. Please verify."

    return last_result, confidence, notes


def auto_calculate_pair(pair: Dict) -> Dict:
    """Auto-calculate corrupted answer for a pair.

    Args:
        pair: Problem pair dict

    Returns:
        Updated pair with calculated answer and review notes
    """
    changed_from, changed_to = pair['corrupted']['changed_number'].split(' -> ')
    full_answer = pair['clean']['full_answer']

    calculated_answer, confidence, notes = extract_calculation_from_solution(
        full_answer, changed_from, changed_to
    )

    # Update the pair
    pair['corrupted']['answer'] = calculated_answer
    pair['auto_calculated'] = True
    pair['calculation_confidence'] = confidence
    pair['calculation_notes'] = notes

    # Auto-approve high confidence
    if confidence == "high" and calculated_answer is not None:
        pair['review_status'] = "approved"
    else:
        pair['review_status'] = "needs_review"

    return pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='problem_pairs_for_review.json')
    parser.add_argument('--output', type=str, default='data/problem_pairs.json')
    args = parser.parse_args()

    # Load pairs
    print(f"Loading pairs from {args.input}...")
    with open(args.input, 'r') as f:
        pairs = json.load(f)

    print(f"Auto-calculating answers for {len(pairs)} pairs...")

    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0

    for pair in pairs:
        pair = auto_calculate_pair(pair)

        conf = pair['calculation_confidence']
        if conf == "high":
            high_confidence += 1
        elif conf == "medium":
            medium_confidence += 1
        else:
            low_confidence += 1

        # Print progress for ones needing review
        if pair['review_status'] == "needs_review":
            print(f"\n⚠️  Pair {pair['pair_id']} needs review:")
            print(f"   Clean: {pair['clean']['question'][:80]}...")
            print(f"   Clean answer: {pair['clean']['answer']}")
            print(f"   Changed: {pair['corrupted']['changed_number']}")
            print(f"   Calculated answer: {pair['corrupted']['answer']}")
            print(f"   Confidence: {conf}")
            print(f"   Notes: {pair['calculation_notes']}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n{'='*60}")
    print(f"AUTO-CALCULATION COMPLETE")
    print(f"{'='*60}")
    print(f"High confidence (auto-approved):  {high_confidence}")
    print(f"Medium confidence (needs review): {medium_confidence}")
    print(f"Low confidence (needs review):    {low_confidence}")
    print(f"Total pairs: {len(pairs)}")
    print(f"\n✓ Saved to {args.output}")
    print(f"\nNext step: Review {medium_confidence + low_confidence} flagged pairs")
    print(f"Then run: python generate_pairs.py --filter {args.output}")


if __name__ == "__main__":
    main()
