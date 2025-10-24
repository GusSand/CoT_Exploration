"""
Download and save GSM8k dataset for operation classification.

This script downloads the full GSM8k test set and saves it to a JSON file
for subsequent operation-type classification.

Output: gsm8k_full.json (8792 problems from test + train splits)
"""

import json
from pathlib import Path
from datasets import load_dataset


def download_gsm8k():
    """Download GSM8k dataset and save to JSON."""
    print("Downloading GSM8k dataset...")

    # Load both test and train splits
    test_dataset = load_dataset('gsm8k', 'main', split='test')
    train_dataset = load_dataset('gsm8k', 'main', split='train')

    print(f"Test split: {len(test_dataset)} problems")
    print(f"Train split: {len(train_dataset)} problems")

    # Combine into single list
    all_problems = []

    for split_name, dataset in [('test', test_dataset), ('train', train_dataset)]:
        for idx, example in enumerate(dataset):
            all_problems.append({
                'id': f"{split_name}_{idx}",
                'question': example['question'],
                'answer': example['answer'],
                'split': split_name
            })

    print(f"Total problems: {len(all_problems)}")

    # Save to JSON
    output_path = Path(__file__).parent / "gsm8k_full.json"
    with open(output_path, 'w') as f:
        json.dump(all_problems, f, indent=2)

    print(f"Saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


if __name__ == "__main__":
    download_gsm8k()
