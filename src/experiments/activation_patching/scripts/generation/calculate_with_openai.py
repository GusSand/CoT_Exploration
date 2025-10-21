#!/usr/bin/env python3
"""
Use OpenAI GPT-4 to calculate corrupted answers for problem pairs.

Much more accurate than heuristic parsing!

Usage:
    export OPENAI_API_KEY="your-key-here"
    python calculate_with_openai.py \
        --input problem_pairs_all_532.json \
        --output problem_pairs_gpt4_answers.json \
        --batch_size 50
"""

import json
import os
import re
import argparse
import time
from typing import Optional
from openai import OpenAI


def extract_answer_from_response(response: str) -> Optional[int]:
    """Extract numerical answer from GPT's response."""
    # Look for common answer patterns
    patterns = [
        r'####\s*(-?\d+)',
        r'(?:final answer|answer|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+)',
        r'\$?\s*(-?\d+)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue

    # Last resort: find last number in response
    numbers = re.findall(r'-?\d+', response)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass

    return None


def solve_problem_with_gpt4(client: OpenAI, question: str, original_answer: int) -> tuple:
    """Use GPT-4 to solve a GSM8K problem.

    Returns:
        (answer, confidence, notes)
    """
    prompt = f"""Solve this math word problem step by step.

Problem: {question}

Provide a detailed solution, then end with your final numerical answer on a new line like this:
#### [your answer]

Show your work!"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheaper and faster, still very accurate
            messages=[
                {"role": "system", "content": "You are a helpful math tutor that solves word problems step by step."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0  # Deterministic for math
        )

        response_text = response.choices[0].message.content
        answer = extract_answer_from_response(response_text)

        if answer is not None:
            # Check if answer makes sense
            diff = abs(answer - original_answer) if original_answer else 0

            if answer == original_answer:
                return answer, "medium", f"Same as original ({original_answer})"
            elif diff <= 10 or diff <= abs(original_answer) * 0.2:
                return answer, "high", f"GPT-4 calculated: {answer}"
            else:
                return answer, "medium", f"Large change: {original_answer} → {answer}"
        else:
            return None, "low", "Could not parse GPT-4's answer"

    except Exception as e:
        return None, "low", f"API error: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Calculate corrupted answers using GPT-4")
    parser.add_argument('--input', type=str, required=True, help='Input pairs JSON')
    parser.add_argument('--output', type=str, required=True, help='Output pairs JSON with GPT-4 answers')
    parser.add_argument('--batch_size', type=int, default=50, help='Process in batches (for progress tracking)')
    parser.add_argument('--skip_existing', action='store_true', help='Skip pairs that already have high confidence')

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return

    client = OpenAI(api_key=api_key)

    # Load pairs
    print(f"Loading pairs from {args.input}...")
    with open(args.input, 'r') as f:
        pairs = json.load(f)

    print(f"Found {len(pairs)} pairs")
    print()

    # Process pairs
    processed = 0
    high_conf = 0
    medium_conf = 0
    low_conf = 0
    skipped = 0

    for i, pair in enumerate(pairs):
        # Skip if requested and already high confidence
        if args.skip_existing and pair.get('calculation_confidence') == 'high':
            skipped += 1
            continue

        print(f"Processing pair {i+1}/{len(pairs)} (ID: {pair['pair_id']})...", end=' ')

        # Solve corrupted problem with GPT-4
        answer, confidence, notes = solve_problem_with_gpt4(
            client,
            pair['corrupted']['question'],
            pair['clean']['answer']
        )

        # Update pair
        pair['corrupted']['answer'] = answer
        pair['gpt4_calculated'] = True
        pair['calculation_confidence'] = confidence
        pair['calculation_notes'] = notes

        # Update stats
        processed += 1
        if confidence == 'high':
            high_conf += 1
            print(f"✓ {answer} (high confidence)")
        elif confidence == 'medium':
            medium_conf += 1
            print(f"✓ {answer} (medium confidence)")
        else:
            low_conf += 1
            print(f"✗ {answer or 'failed'} (low confidence)")

        # Progress update every batch
        if (i + 1) % args.batch_size == 0:
            print()
            print(f"Progress: {i+1}/{len(pairs)} pairs processed")
            print(f"  High confidence: {high_conf}")
            print(f"  Medium confidence: {medium_conf}")
            print(f"  Low confidence: {low_conf}")
            print()

        # Rate limiting: OpenAI has generous limits but let's be respectful
        time.sleep(0.2)  # 5 requests/second max

    # Save results
    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)

    print()
    print("=" * 80)
    print("GPT-4 CALCULATION COMPLETE")
    print("=" * 80)
    print(f"Total pairs processed: {processed}/{len(pairs)}")
    if skipped:
        print(f"Skipped (already high conf): {skipped}")
    print(f"High confidence: {high_conf} ({100*high_conf/processed:.1f}%)")
    print(f"Medium confidence: {medium_conf} ({100*medium_conf/processed:.1f}%)")
    print(f"Low confidence: {low_conf} ({100*low_conf/processed:.1f}%)")
    print()
    print(f"✓ Saved to {args.output}")
    print()
    print("Next: Re-run baseline validation with new answers")


if __name__ == "__main__":
    main()
