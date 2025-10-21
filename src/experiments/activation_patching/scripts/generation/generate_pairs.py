"""
Problem Pair Generation Script
Generates clean/corrupted problem pairs from GSM8K for activation patching experiments.

Usage:
    python generate_pairs.py --output problem_pairs_for_review.json --num_candidates 70
"""

import json
import re
from datasets import load_dataset
from typing import List, Dict, Optional
import argparse


def extract_first_number(text: str) -> Optional[str]:
    """Extract the first number from text.

    Args:
        text: Input text

    Returns:
        First number found as string, or None
    """
    numbers = re.findall(r'\d+', text)
    return numbers[0] if numbers else None


def create_pair(problem: Dict, idx: int) -> Optional[Dict]:
    """Create clean/corrupted pair by changing one number.

    Args:
        problem: GSM8K problem dict with 'question' and 'answer' fields
        idx: Pair index

    Returns:
        Pair dict or None if problem unsuitable
    """
    question = problem['question']
    answer = problem['answer']

    # Find all numbers in question
    numbers = re.findall(r'\d+', question)

    if len(numbers) < 2:
        return None  # Need at least 2 numbers to be interesting

    # Change first number (usually an operand)
    original_num = numbers[0]
    corrupted_num = str(int(original_num) + 1)

    # Replace only the first occurrence
    corrupted_question = question.replace(original_num, corrupted_num, 1)

    # Extract ground truth answer from answer field
    # GSM8K answers are like: "John has 21 apples.\n#### 21"
    answer_match = re.search(r'####\s*(-?\d+)', answer)
    if answer_match:
        clean_answer = int(answer_match.group(1))
    else:
        # Try to find last number in answer
        answer_numbers = re.findall(r'-?\d+', answer)
        clean_answer = int(answer_numbers[-1]) if answer_numbers else None

    return {
        'pair_id': idx,
        'clean': {
            'question': question,
            'answer': clean_answer,
            'full_answer': answer  # Keep full solution for reference
        },
        'corrupted': {
            'question': corrupted_question,
            'answer': None,  # To be filled during manual review
            'changed_number': f"{original_num} -> {corrupted_num}"
        },
        'review_status': 'pending',
        'notes': ''  # For reviewer comments
    }


def generate_candidate_pairs(num_candidates: int = 70, seed: int = 42) -> List[Dict]:
    """Generate candidate problem pairs from GSM8K.

    Args:
        num_candidates: Number of candidate pairs to generate
        seed: Random seed for reproducibility

    Returns:
        List of candidate pairs
    """
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset('gsm8k', 'main', split='test')

    print(f"Generating {num_candidates} candidate pairs...")
    pairs = []

    for idx, problem in enumerate(dataset):
        if len(pairs) >= num_candidates:
            break

        pair = create_pair(problem, idx)
        if pair is not None:
            pairs.append(pair)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1} problems, generated {len(pairs)} pairs")

    print(f"\n✓ Generated {len(pairs)} candidate pairs!")
    return pairs


def save_for_review(pairs: List[Dict], output_path: str):
    """Save pairs to JSON for manual review.

    Args:
        pairs: List of problem pairs
        output_path: Output file path
    """
    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n✓ Saved to {output_path}")
    print("\nREVIEW INSTRUCTIONS:")
    print("=" * 60)
    print("1. Open the JSON file and review each pair")
    print("2. For each pair:")
    print("   - Check that the corrupted question makes sense")
    print("   - Calculate the expected corrupted answer")
    print("   - Fill in 'corrupted.answer' field")
    print("   - Set 'review_status' to 'approved' or 'rejected'")
    print("   - Add any notes in 'notes' field")
    print("3. Keep the best 50 approved pairs")
    print("4. Save as 'problem_pairs.json'")
    print("=" * 60)


def filter_approved_pairs(input_path: str, output_path: str, max_pairs: int = 50):
    """Filter to only approved pairs and save final dataset.

    Args:
        input_path: Path to reviewed pairs file
        output_path: Path to save final pairs
        max_pairs: Maximum number of pairs to keep
    """
    with open(input_path, 'r') as f:
        pairs = json.load(f)

    # Filter approved pairs
    approved = [p for p in pairs if p.get('review_status') == 'approved']

    print(f"Found {len(approved)} approved pairs (target: {max_pairs})")

    if len(approved) < max_pairs:
        print(f"WARNING: Only {len(approved)} approved pairs, fewer than target {max_pairs}")

    # Take up to max_pairs
    final_pairs = approved[:max_pairs]

    # Validate all have corrupted answers
    missing_answers = [p['pair_id'] for p in final_pairs if p['corrupted']['answer'] is None]
    if missing_answers:
        print(f"ERROR: {len(missing_answers)} pairs missing corrupted answers: {missing_answers}")
        print("Please fill in all corrupted answers before filtering.")
        return

    with open(output_path, 'w') as f:
        json.dump(final_pairs, f, indent=2)

    print(f"\n✓ Saved {len(final_pairs)} final pairs to {output_path}")


def show_sample_pairs(pairs: List[Dict], num_samples: int = 3):
    """Display sample pairs for quick review.

    Args:
        pairs: List of pairs
        num_samples: Number of samples to show
    """
    print(f"\n{'='*60}")
    print(f"SAMPLE PAIRS (showing {num_samples}/{len(pairs)})")
    print(f"{'='*60}\n")

    for pair in pairs[:num_samples]:
        print(f"Pair {pair['pair_id']}:")
        print(f"  Clean Question: {pair['clean']['question']}")
        print(f"  Clean Answer: {pair['clean']['answer']}")
        print(f"  Corrupted Question: {pair['corrupted']['question']}")
        print(f"  Changed: {pair['corrupted']['changed_number']}")
        print(f"  Corrupted Answer: {pair['corrupted']['answer']} (to be filled)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate problem pairs for activation patching")
    parser.add_argument('--output', type=str, default='problem_pairs_for_review.json',
                        help='Output file for review')
    parser.add_argument('--num_candidates', type=int, default=70,
                        help='Number of candidate pairs to generate')
    parser.add_argument('--filter', type=str, default=None,
                        help='Input file to filter (reviewed pairs)')
    parser.add_argument('--final_output', type=str, default='problem_pairs.json',
                        help='Output file for final pairs')
    parser.add_argument('--show_samples', action='store_true',
                        help='Show sample pairs')

    args = parser.parse_args()

    if args.filter:
        # Filter mode: process reviewed pairs
        filter_approved_pairs(args.filter, args.final_output)
    else:
        # Generation mode: create candidates
        pairs = generate_candidate_pairs(args.num_candidates)

        if args.show_samples:
            show_sample_pairs(pairs)

        save_for_review(pairs, args.output)

        print(f"\nNext step: Review {args.output} and run:")
        print(f"  python generate_pairs.py --filter {args.output} --final_output problem_pairs.json")


if __name__ == "__main__":
    main()
