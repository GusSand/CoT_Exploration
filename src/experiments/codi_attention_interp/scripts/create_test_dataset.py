"""
Create a balanced test dataset of 10 problems for pipeline validation.
Samples 2-3 problems from each difficulty level.
"""
import json
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "activation_patching" / "data"
INPUT_FILE = DATA_DIR / "llama_cot_original_stratified_1000.json"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "test_dataset_10.json"

def create_test_dataset():
    """Sample 10 problems: 3 from 2-step, 3 from 3-step, 2 from 4-step, 2 from 5+step"""

    # Load full dataset
    print(f"Loading dataset from {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        full_dataset = json.load(f)

    print(f"Total problems: {len(full_dataset)}")

    # Group by difficulty
    by_difficulty = {
        '2-step': [],
        '3-step': [],
        '4-step': [],
        '5+step': []
    }

    for problem in full_dataset:
        difficulty = problem.get('difficulty', 'unknown')
        if difficulty in by_difficulty:
            by_difficulty[difficulty].append(problem)

    # Print distribution
    for diff, problems in by_difficulty.items():
        print(f"{diff}: {len(problems)} problems")

    # Sample balanced test set
    test_dataset = []

    # 3 from 2-step
    test_dataset.extend(random.sample(by_difficulty['2-step'], 3))

    # 3 from 3-step
    test_dataset.extend(random.sample(by_difficulty['3-step'], 3))

    # 2 from 4-step
    test_dataset.extend(random.sample(by_difficulty['4-step'], 2))

    # 2 from 5+step
    test_dataset.extend(random.sample(by_difficulty['5+step'], 2))

    print(f"\nTest dataset created: {len(test_dataset)} problems")
    print("Distribution:")
    test_distribution = {}
    for p in test_dataset:
        diff = p['difficulty']
        test_distribution[diff] = test_distribution.get(diff, 0) + 1
    for diff, count in sorted(test_distribution.items()):
        print(f"  {diff}: {count}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(test_dataset, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")

    # Print sample
    print("\nFirst problem:")
    print(f"  ID: {test_dataset[0]['gsm8k_id']}")
    print(f"  Difficulty: {test_dataset[0]['difficulty']}")
    print(f"  Question: {test_dataset[0]['question'][:100]}...")
    print(f"  Answer: {test_dataset[0]['answer']}")

    return test_dataset

if __name__ == "__main__":
    create_test_dataset()
