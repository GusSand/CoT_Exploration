"""
Create clean, balanced dataset by:
1. Removing true duplicates (same question, label, answer)
2. Balancing to equal honest/deceptive samples
3. Verifying data quality
"""

import json
import numpy as np
from pathlib import Path

def create_clean_dataset():
    script_dir = Path(__file__).parent
    input_file = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2.json"
    output_file = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2_clean.json"

    print("=" * 80)
    print("Creating Clean, Balanced Dataset")
    print("=" * 80)

    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"\n[1/5] Loaded {len(data['samples'])} samples")

    # Deduplicate by (question, is_honest) - keep first occurrence only
    # This prevents data leakage in train/test split
    print(f"\n[2/5] Deduplicating by (question, label)...")
    unique_examples = {}
    removed_duplicates = 0

    for sample in data['samples']:
        key = (sample['question'], sample['is_honest'])
        if key not in unique_examples:
            unique_examples[key] = sample
        else:
            removed_duplicates += 1

    unique_samples = list(unique_examples.values())
    print(f"  Removed {removed_duplicates} duplicates (same question+label)")
    print(f"  Remaining: {len(unique_samples)} unique (question, label) pairs")

    # Separate by label
    honest_samples = [s for s in unique_samples if s['is_honest']]
    deceptive_samples = [s for s in unique_samples if not s['is_honest']]

    print(f"\n[3/5] Balancing...")
    print(f"  Honest: {len(honest_samples)}")
    print(f"  Deceptive: {len(deceptive_samples)}")

    # Balance to smaller class
    min_count = min(len(honest_samples), len(deceptive_samples))

    # Random sampling for balance (with fixed seed)
    np.random.seed(42)
    honest_balanced = np.random.choice(honest_samples, size=min_count, replace=False).tolist()
    deceptive_balanced = np.random.choice(deceptive_samples, size=min_count, replace=False).tolist()

    balanced_samples = honest_balanced + deceptive_balanced

    # Shuffle
    np.random.shuffle(balanced_samples)

    print(f"  Balanced to: {len(balanced_samples)} total ({min_count} honest + {min_count} deceptive)")

    # Verify no duplicates
    print(f"\n[4/5] Verifying data quality...")

    questions_with_labels = [(s['question'], s['is_honest']) for s in balanced_samples]
    unique_question_label_pairs = len(set(questions_with_labels))

    if len(questions_with_labels) != unique_question_label_pairs:
        print(f"  ❌ ERROR: {len(questions_with_labels) - unique_question_label_pairs} duplicate (question, label) pairs!")
        print(f"     This will cause train/test leakage!")
        raise ValueError("Duplicates found after deduplication!")
    else:
        print(f"  ✅ All {unique_question_label_pairs} (question, label) pairs are unique")
        print(f"  ✅ No train/test leakage risk")

    # Check balance
    honest_count = sum(1 for s in balanced_samples if s['is_honest'])
    deceptive_count = sum(1 for s in balanced_samples if not s['is_honest'])

    print(f"  ✅ Perfect balance: {honest_count} honest, {deceptive_count} deceptive")

    # Check activation variance
    sample_acts = [s['thoughts']['layer_4'][1] for s in balanced_samples[:100]]
    sample_acts = np.array(sample_acts)

    print(f"  ✅ Activation stats (L4 Token 1, 100 samples):")
    print(f"     Mean: {np.mean(sample_acts):.4f}, Std: {np.std(sample_acts):.4f}")

    # Save
    print(f"\n[5/5] Saving clean dataset...")

    clean_data = {
        'model': data['model'],
        'n_honest': honest_count,
        'n_deceptive': deceptive_count,
        'layers': data['layers'],
        'num_tokens': data['num_tokens'],
        'hidden_size': data['hidden_size'],
        'samples': balanced_samples,
        'data_quality': {
            'original_samples': len(data['samples']),
            'removed_duplicates': removed_duplicates,
            'unique_samples': len(unique_samples),
            'final_balanced': len(balanced_samples),
            'balance_ratio': honest_count / (honest_count + deceptive_count)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(clean_data, f, indent=2)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✅ Saved: {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")

    print(f"\n" + "=" * 80)
    print(f"✅ Clean Dataset Ready!")
    print(f"=" * 80)
    print(f"  Total: {len(balanced_samples)} samples")
    print(f"  Honest: {honest_count} ({honest_count/len(balanced_samples)*100:.1f}%)")
    print(f"  Deceptive: {deceptive_count} ({deceptive_count/len(balanced_samples)*100:.1f}%)")
    print(f"  Quality: No duplicates, perfect balance")

    return output_file

if __name__ == "__main__":
    create_clean_dataset()
