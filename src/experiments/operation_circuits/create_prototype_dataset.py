"""
Create prototype dataset for quick testing.

This script samples 20 problems from each operation category to create a small
prototype dataset (60 problems total) for quick iteration and testing.

Output: operation_samples_prototype_60.json (20 samples per category)
"""

import json
import random
from pathlib import Path


def create_prototype_dataset(input_path: Path, output_path: Path, samples_per_category: int = 20):
    """
    Sample a small prototype dataset from the full classified dataset.

    Args:
        input_path: Path to operation_samples_200.json
        output_path: Path to save prototype samples
        samples_per_category: Number of samples per category (default: 20)
    """
    print(f"Loading classified problems from {input_path}...")
    with open(input_path, 'r') as f:
        classified = json.load(f)

    print("Full dataset distribution:")
    for category, items in classified.items():
        print(f"  {category:20s}: {len(items):4d} problems")

    # Sample from each category
    prototype = {}
    random.seed(42)  # For reproducibility

    print(f"\nSampling {samples_per_category} problems per category...")
    for category, items in classified.items():
        if len(items) < samples_per_category:
            print(f"  Warning: {category} has only {len(items)} problems, using all")
            prototype[category] = items
        else:
            prototype[category] = random.sample(items, samples_per_category)

    # Print prototype distribution
    print("\nPrototype dataset distribution:")
    total_samples = 0
    for category, items in prototype.items():
        print(f"  {category:20s}: {len(items):4d} problems")
        total_samples += len(items)

    print(f"\nTotal prototype samples: {total_samples} problems")

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(prototype, f, indent=2)

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")

    return prototype


def main():
    """Main execution."""
    input_path = Path(__file__).parent / "operation_samples_200.json"
    output_path = Path(__file__).parent / "operation_samples_prototype_60.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        print("Please run classify_operations.py first.")
        return

    create_prototype_dataset(input_path, output_path, samples_per_category=20)


if __name__ == "__main__":
    main()
