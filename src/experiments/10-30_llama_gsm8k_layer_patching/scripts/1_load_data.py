"""
Story 1: Load and validate clean/corrupted pairs dataset

This script:
1. Loads the llama_clean_corrupted_pairs.json dataset
2. Validates structure and data quality
3. Reports statistics
4. Prepares pairs for patching experiment
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict

# Add experiment directory to path
exp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(exp_dir))

import config


def load_dataset(data_path):
    """Load the clean/corrupted pairs dataset"""
    print(f"\n{'='*80}")
    print(f"Loading dataset from: {data_path}")
    print(f"{'='*80}\n")

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} total entries")
    return data


def validate_structure(data):
    """Validate dataset structure"""
    print(f"\n{'='*80}")
    print("Validating Dataset Structure")
    print(f"{'='*80}\n")

    errors = []
    pairs = defaultdict(dict)

    # Required fields
    required_fields = ['pair_id', 'variant', 'question', 'answer', 'reasoning_steps',
                      'baseline_correct', 'ablated_correct', 'needs_cot']

    for idx, entry in enumerate(data):
        # Check required fields
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            errors.append(f"Entry {idx}: Missing fields {missing_fields}")
            continue

        # Check variant is either 'clean' or 'corrupted'
        if entry['variant'] not in ['clean', 'corrupted']:
            errors.append(f"Entry {idx}: Invalid variant '{entry['variant']}'")

        # Check answer is a number
        if not isinstance(entry['answer'], (int, float)):
            errors.append(f"Entry {idx}: Answer is not numeric: {entry['answer']}")

        # Group by pair_id and variant
        pair_id = entry['pair_id']
        variant = entry['variant']

        if variant in pairs[pair_id]:
            errors.append(f"Pair {pair_id}: Duplicate {variant} variant")

        pairs[pair_id][variant] = entry

    # Check each pair has both clean and corrupted
    incomplete_pairs = []
    for pair_id, variants in pairs.items():
        if 'clean' not in variants or 'corrupted' not in variants:
            incomplete_pairs.append(pair_id)

    if incomplete_pairs:
        errors.append(f"Incomplete pairs (missing clean or corrupted): {incomplete_pairs}")

    # Report results
    if errors:
        print("❌ Validation FAILED with errors:\n")
        for error in errors:
            print(f"  - {error}")
        return False, None

    print(f"✓ All entries have required fields")
    print(f"✓ All variants are 'clean' or 'corrupted'")
    print(f"✓ All answers are numeric")
    print(f"✓ Found {len(pairs)} complete pairs")
    print(f"✓ No duplicate variants within pairs")

    return True, pairs


def analyze_dataset(pairs):
    """Analyze dataset statistics"""
    print(f"\n{'='*80}")
    print("Dataset Statistics")
    print(f"{'='*80}\n")

    print(f"Total pairs: {len(pairs)}")

    # Reasoning steps distribution
    reasoning_steps_clean = [pair['clean']['reasoning_steps'] for pair in pairs.values()]
    reasoning_steps_corrupt = [pair['corrupted']['reasoning_steps'] for pair in pairs.values()]

    print(f"\nReasoning steps (clean):")
    print(f"  Min: {min(reasoning_steps_clean)}")
    print(f"  Max: {max(reasoning_steps_clean)}")
    print(f"  Mean: {sum(reasoning_steps_clean) / len(reasoning_steps_clean):.2f}")

    print(f"\nReasoning steps (corrupted):")
    print(f"  Min: {min(reasoning_steps_corrupt)}")
    print(f"  Max: {max(reasoning_steps_corrupt)}")
    print(f"  Mean: {sum(reasoning_steps_corrupt) / len(reasoning_steps_corrupt):.2f}")

    # Answer differences
    answer_diffs = []
    for pair in pairs.values():
        clean_ans = pair['clean']['answer']
        corrupt_ans = pair['corrupted']['answer']
        answer_diffs.append(abs(corrupt_ans - clean_ans))

    print(f"\nAnswer differences (|clean - corrupted|):")
    print(f"  Min: {min(answer_diffs)}")
    print(f"  Max: {max(answer_diffs)}")
    print(f"  Mean: {sum(answer_diffs) / len(answer_diffs):.2f}")
    print(f"  Same answer: {sum(1 for d in answer_diffs if d == 0)} pairs")

    # Question length analysis
    question_lengths = [len(pair['clean']['question'].split()) for pair in pairs.values()]
    print(f"\nQuestion lengths (words):")
    print(f"  Min: {min(question_lengths)}")
    print(f"  Max: {max(question_lengths)}")
    print(f"  Mean: {sum(question_lengths) / len(question_lengths):.2f}")


def prepare_pairs_for_experiment(pairs):
    """Prepare pairs in format convenient for patching experiment"""
    print(f"\n{'='*80}")
    print("Preparing Pairs for Experiment")
    print(f"{'='*80}\n")

    prepared_pairs = []

    for pair_id in sorted(pairs.keys()):
        pair_data = pairs[pair_id]
        prepared_pairs.append({
            'pair_id': pair_id,
            'clean': {
                'question': pair_data['clean']['question'],
                'answer': pair_data['clean']['answer'],
                'reasoning_steps': pair_data['clean']['reasoning_steps']
            },
            'corrupted': {
                'question': pair_data['corrupted']['question'],
                'answer': pair_data['corrupted']['answer'],
                'reasoning_steps': pair_data['corrupted']['reasoning_steps']
            }
        })

    print(f"✓ Prepared {len(prepared_pairs)} pairs for patching experiment")

    # Show example
    example = prepared_pairs[0]
    print(f"\nExample pair (ID: {example['pair_id']}):")
    print(f"\n  Clean question: {example['clean']['question'][:80]}...")
    print(f"  Clean answer: {example['clean']['answer']}")
    print(f"\n  Corrupted question: {example['corrupted']['question'][:80]}...")
    print(f"  Corrupted answer: {example['corrupted']['answer']}")

    return prepared_pairs


def save_prepared_pairs(pairs, output_path):
    """Save prepared pairs for use in subsequent scripts"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(pairs, f, indent=2)

    print(f"\n✓ Saved prepared pairs to: {output_path}")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("STORY 1: Load and Validate Clean/Corrupted Pairs Dataset")
    print("="*80)

    # Load dataset
    data = load_dataset(config.DATA_PATH)

    # Validate structure
    is_valid, pairs = validate_structure(data)

    if not is_valid:
        print("\n❌ Dataset validation failed. Please fix errors and try again.")
        return 1

    # Analyze dataset
    analyze_dataset(pairs)

    # Prepare pairs for experiment
    prepared_pairs = prepare_pairs_for_experiment(pairs)

    # Save prepared pairs
    output_path = os.path.join(config.RESULTS_DIR, "prepared_pairs.json")
    save_prepared_pairs(prepared_pairs, output_path)

    print(f"\n{'='*80}")
    print("✓ DATA VALIDATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"Summary:")
    print(f"  - Total pairs: {len(pairs)}")
    print(f"  - All data validated successfully")
    print(f"  - Ready for activation patching experiment")
    print(f"  - Prepared pairs saved to: {output_path}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
