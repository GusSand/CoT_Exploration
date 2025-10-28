#!/usr/bin/env python3
"""
Remove exact (question, answer) duplicates from training data.

This script removes true duplicates (same Q+A) while preserving
question reuse with different answers (which is beneficial for CODI).
"""

import json
from pathlib import Path
from collections import Counter

def deduplicate_training_data():
    """Remove exact (question, answer) duplicates from training data."""

    script_dir = Path(__file__).parent
    input_file = script_dir / "data" / "processed" / "train.json"
    output_file = script_dir / "data" / "processed" / "train_deduped.json"
    stats_file = script_dir / "data" / "processed" / "deduplication_stats.json"

    print("=" * 80)
    print("DEDUPLICATING TRAINING DATA")
    print("=" * 80)

    # Load data
    print(f"\n[1/4] Loading training data from {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} samples")

    # Analyze duplicates before removal
    print(f"\n[2/4] Analyzing duplicates...")

    qa_pairs = [(item['question'], item['answer']) for item in data]
    qa_counter = Counter(qa_pairs)

    duplicates = {qa: count for qa, count in qa_counter.items() if count > 1}
    duplicate_count = sum(count - 1 for count in duplicates.values())  # Extra copies

    print(f"  Unique (Q,A) pairs: {len(qa_counter)}")
    print(f"  Duplicate (Q,A) pairs: {len(duplicates)}")
    print(f"  Total duplicate samples: {duplicate_count}")

    # Show most duplicated examples
    if duplicates:
        print(f"\n  Top 5 most duplicated (Q,A) pairs:")
        for i, ((q, a), count) in enumerate(sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]):
            print(f"\n  {i+1}. Count: {count}")
            print(f"     Q: {q[:80]}...")
            print(f"     A: {a[:80]}...")

    # Deduplicate: keep first occurrence
    print(f"\n[3/4] Deduplicating...")
    seen = set()
    deduped = []
    removed_by_model = Counter()

    for item in data:
        key = (item['question'], item['answer'])
        if key not in seen:
            seen.add(key)
            deduped.append(item)
        else:
            # Track which models had duplicates
            model = item.get('meta', {}).get('model', 'unknown')
            removed_by_model[model] += 1

    print(f"  ✅ Kept {len(deduped)} unique samples")
    print(f"  ✅ Removed {len(data) - len(deduped)} duplicates")

    # Analyze removed samples
    if removed_by_model:
        print(f"\n  Duplicates by model:")
        for model, count in removed_by_model.most_common():
            print(f"    {model}: {count}")

    # Save deduplicated data
    print(f"\n[4/4] Saving deduplicated data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(deduped, f, indent=2)

    # Save statistics
    stats = {
        "original_samples": len(data),
        "deduplicated_samples": len(deduped),
        "removed_duplicates": len(data) - len(deduped),
        "unique_qa_pairs": len(seen),
        "duplicate_qa_pairs": len(duplicates),
        "duplicates_by_model": dict(removed_by_model),
        "top_duplicates": [
            {
                "count": count,
                "question": q[:100],
                "answer": a[:100]
            }
            for (q, a), count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  ✅ Saved statistics to {stats_file}")

    # Calculate savings
    removed_pct = (len(data) - len(deduped)) / len(data) * 100
    print(f"\n" + "=" * 80)
    print(f"DEDUPLICATION COMPLETE")
    print(f"=" * 80)
    print(f"  Original samples: {len(data)}")
    print(f"  Deduplicated samples: {len(deduped)}")
    print(f"  Removed: {len(data) - len(deduped)} ({removed_pct:.1f}%)")
    print(f"\n  Estimated savings:")
    print(f"    - Training samples: {len(data) - len(deduped)} samples")
    print(f"    - GPU time (20 epochs): ~{(len(data) - len(deduped)) * 20 / 7074 * 60:.1f} hours")
    print(f"    - Cost estimate: ~${(len(data) - len(deduped)) * 20 / 7074 * 60 * 2.5:.2f}")
    print(f"\n  Next steps:")
    print(f"    1. Review {stats_file}")
    print(f"    2. Update training script to use train_deduped.json")
    print(f"    3. Verify no unintended removals")

    return output_file

if __name__ == "__main__":
    output = deduplicate_training_data()
    print(f"\n✅ Deduplicated file ready: {output}")
