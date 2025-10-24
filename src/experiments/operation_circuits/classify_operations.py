"""
Classify GSM8k problems by operation type.

This script classifies GSM8k problems into three categories based on the
operations required in their solution:
- pure_addition: Only addition/subtraction operations
- pure_multiplication: Only multiplication/division operations
- mixed: Both addition and multiplication operations

Output: operation_samples_200.json (200 samples per category = 600 total)
"""

import json
import re
import random
from pathlib import Path
from typing import Dict, List


def extract_numbers_and_operations(answer: str) -> tuple[List[str], bool, bool]:
    """
    Extract operations from the answer string.

    Returns:
        (operations_list, has_addition, has_multiplication)
    """
    # GSM8k answers are formatted like:
    # "John has 3 bags. Each bag has 7 apples. So he has 3 * 7 = 21 apples. #### 21"

    # Extract the solution part (before ####)
    if '####' in answer:
        solution = answer.split('####')[0].strip()
    else:
        solution = answer.strip()

    # Find all arithmetic operations
    # Look for patterns like: number operator number
    operations = []

    # Check for addition/subtraction
    has_addition = bool(re.search(r'\d+\s*[\+\-]\s*\d+', solution))

    # Check for multiplication/division
    has_multiplication = bool(re.search(r'\d+\s*[\*\/]\s*\d+', solution))

    return operations, has_addition, has_multiplication


def classify_problem(problem: Dict) -> str:
    """
    Classify a problem into one of three categories:
    - pure_addition: Only addition/subtraction
    - pure_multiplication: Only multiplication/division
    - mixed: Both types of operations
    """
    _, has_add, has_mult = extract_numbers_and_operations(problem['answer'])

    if has_add and has_mult:
        return 'mixed'
    elif has_add and not has_mult:
        return 'pure_addition'
    elif has_mult and not has_add:
        return 'pure_multiplication'
    else:
        # No clear operations detected - classify as mixed (safest)
        return 'mixed'


def classify_and_sample(input_path: Path, output_path: Path, samples_per_category: int = 200):
    """
    Classify all problems and sample evenly from each category.

    Args:
        input_path: Path to gsm8k_full.json
        output_path: Path to save classified samples
        samples_per_category: Number of samples to take per category (default: 200)
    """
    print(f"Loading problems from {input_path}...")
    with open(input_path, 'r') as f:
        problems = json.load(f)

    print(f"Total problems: {len(problems)}")

    # Classify all problems
    classified = {
        'pure_addition': [],
        'pure_multiplication': [],
        'mixed': []
    }

    print("\nClassifying problems...")
    for problem in problems:
        category = classify_problem(problem)
        classified[category].append(problem)

    # Print distribution
    print("\nClassification results:")
    for category, items in classified.items():
        print(f"  {category:20s}: {len(items):4d} problems")

    # Sample evenly from each category
    sampled = {}
    random.seed(42)  # For reproducibility

    print(f"\nSampling {samples_per_category} problems per category...")
    for category, items in classified.items():
        if len(items) < samples_per_category:
            print(f"  Warning: {category} has only {len(items)} problems, using all")
            sampled[category] = items
        else:
            sampled[category] = random.sample(items, samples_per_category)

    # Print sample distribution
    print("\nSampled distribution:")
    total_sampled = 0
    for category, items in sampled.items():
        print(f"  {category:20s}: {len(items):4d} problems")
        total_sampled += len(items)

    print(f"\nTotal sampled: {total_sampled} problems")

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(sampled, f, indent=2)

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")

    return sampled


def main():
    """Main execution."""
    input_path = Path(__file__).parent / "gsm8k_full.json"
    output_path = Path(__file__).parent / "operation_samples_200.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        print("Please run download_gsm8k.py first.")
        return

    classify_and_sample(input_path, output_path, samples_per_category=200)


if __name__ == "__main__":
    main()
