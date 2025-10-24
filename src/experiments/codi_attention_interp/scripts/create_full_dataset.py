"""
Create a balanced full dataset of 100 problems for production experiments.
Samples 25 problems from each difficulty level (2-step, 3-step, 4-step, 5+step).
"""
import json
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "activation_patching" / "data"
INPUT_FILE = DATA_DIR / "llama_cot_original_stratified_1000.json"
OUTPUT_FILE = Path(__file__).parent.parent / "results" / "full_dataset_100.json"

def create_full_dataset():
    """Sample 100 problems: 25 from each difficulty level"""

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

    # Sample balanced full dataset
    full_exp_dataset = []

    # 25 from each difficulty
    for diff in ['2-step', '3-step', '4-step', '5+step']:
        full_exp_dataset.extend(random.sample(by_difficulty[diff], 25))

    print(f"\nFull dataset created: {len(full_exp_dataset)} problems")
    print("Distribution:")
    distribution = {}
    for p in full_exp_dataset:
        diff = p['difficulty']
        distribution[diff] = distribution.get(diff, 0) + 1
    for diff, count in sorted(distribution.items()):
        print(f"  {diff}: {count}")

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(full_exp_dataset, f, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")

    # Print sample
    print("\nFirst problem:")
    print(f"  ID: {full_exp_dataset[0]['gsm8k_id']}")
    print(f"  Difficulty: {full_exp_dataset[0]['difficulty']}")
    print(f"  Question: {full_exp_dataset[0]['question'][:100]}...")
    print(f"  Answer: {full_exp_dataset[0]['answer']}")

    return full_exp_dataset

if __name__ == "__main__":
    create_full_dataset()
