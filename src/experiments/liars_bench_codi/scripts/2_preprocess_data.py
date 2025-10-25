"""
Story 1.2: Data Preprocessing & Formatting

Convert liars-bench Instructed Deception dataset to CODI training format.
Creates train/val/test splits with honest examples for model training.
"""

import json
import random
from pathlib import Path
from collections import Counter
import ast


def extract_question_answer(example):
    """
    Extract question and answer from conversation format.

    Returns:
        tuple: (question_text, answer_text, is_honest)
    """
    messages = example['messages']

    # Extract components
    system_msg = next((m['content'] for m in messages if m['role'] == 'system'), '')
    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')

    # Determine if honest based on system prompt and deceptive flag
    is_honest = not example['deceptive']

    return user_msg, assistant_msg, is_honest


def preprocess_liars_bench(
    input_file: str = "../data/raw/liars_bench_test.json",
    output_dir: str = "../data/processed",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Preprocess liars-bench dataset into CODI format.

    Args:
        input_file: Path to raw dataset
        output_dir: Directory for processed outputs
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("STORY 1.2: Preprocessing Liars-Bench for CODI Training")
    print("=" * 80)

    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"\n[1/6] Loading raw dataset from {input_file}...")
    with open(input_file, 'r') as f:
        raw_data = json.load(f)

    print(f"  ✅ Loaded {len(raw_data)} examples")

    # Separate honest and deceptive examples
    print(f"\n[2/6] Separating honest and deceptive examples...")
    honest_examples = []
    deceptive_examples = []

    for ex in raw_data:
        question, answer, is_honest = extract_question_answer(ex)

        formatted_ex = {
            "index": ex['index'],
            "question": question,
            "answer": answer,
            "dataset": ex['dataset'],
            "model": ex['model'],
            "is_honest": is_honest,
            "meta": ex.get('meta', '')
        }

        if is_honest:
            honest_examples.append(formatted_ex)
        else:
            deceptive_examples.append(formatted_ex)

    print(f"  ✅ Honest examples: {len(honest_examples)}")
    print(f"  ✅ Deceptive examples: {len(deceptive_examples)}")

    # Create train/val/test splits from HONEST examples only
    # (We train CODI to answer correctly, not deceptively)
    print(f"\n[3/6] Creating train/val/test splits from honest examples...")
    print(f"  Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    # Shuffle honest examples
    random.shuffle(honest_examples)

    n_honest = len(honest_examples)
    n_train = int(n_honest * train_ratio)
    n_val = int(n_honest * val_ratio)

    train_data = honest_examples[:n_train]
    val_data = honest_examples[n_train:n_train + n_val]
    test_honest_data = honest_examples[n_train + n_val:]

    print(f"  ✅ Train: {len(train_data)} examples")
    print(f"  ✅ Val: {len(val_data)} examples")
    print(f"  ✅ Test (honest): {len(test_honest_data)} examples")

    # Convert to CODI format
    print(f"\n[4/6] Converting to CODI format...")

    def to_codi_format(examples):
        """Convert to CODI training format."""
        codi_examples = []
        for ex in examples:
            # CODI expects: question, answer, cot (optional)
            codi_ex = {
                "question": ex['question'],
                "answer": ex['answer'],
                "cot": "",  # No explicit CoT in liars-bench, leave empty
                # Keep metadata for analysis
                "meta": {
                    "dataset": ex['dataset'],
                    "model": ex['model'],
                    "original_index": ex['index']
                }
            }
            codi_examples.append(codi_ex)
        return codi_examples

    train_codi = to_codi_format(train_data)
    val_codi = to_codi_format(val_data)
    test_honest_codi = to_codi_format(test_honest_data)

    print(f"  ✅ Converted {len(train_codi)} train examples")
    print(f"  ✅ Converted {len(val_codi)} val examples")
    print(f"  ✅ Converted {len(test_honest_codi)} test examples")

    # Save datasets
    print(f"\n[5/6] Saving processed datasets to {output_path}...")

    datasets = {
        "train.json": train_codi,
        "val.json": val_codi,
        "test_honest.json": test_honest_codi,
        "deceptive_for_probes.json": to_codi_format(deceptive_examples)
    }

    for filename, data in datasets.items():
        output_file = output_path / filename
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ Saved {filename}: {len(data)} examples")

    # Generate statistics
    print(f"\n[6/6] Generating dataset statistics...")

    # Analyze dataset sources
    train_sources = Counter(ex['meta']['dataset'] for ex in train_codi)

    # Analyze question lengths
    train_q_lengths = [len(ex['question'].split()) for ex in train_codi]
    train_a_lengths = [len(ex['answer'].split()) for ex in train_codi]

    stats = {
        "total_raw_examples": len(raw_data),
        "honest_examples": len(honest_examples),
        "deceptive_examples": len(deceptive_examples),
        "splits": {
            "train": len(train_codi),
            "val": len(val_codi),
            "test_honest": len(test_honest_codi),
            "deceptive_for_probes": len(deceptive_examples)
        },
        "train_dataset_distribution": dict(train_sources),
        "question_length_stats": {
            "mean": sum(train_q_lengths) / len(train_q_lengths),
            "min": min(train_q_lengths),
            "max": max(train_q_lengths)
        },
        "answer_length_stats": {
            "mean": sum(train_a_lengths) / len(train_a_lengths),
            "min": min(train_a_lengths),
            "max": max(train_a_lengths)
        },
        "random_seed": seed,
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio
        }
    }

    stats_file = output_path / "preprocessing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"  ✅ Saved statistics: {stats_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("✅ STORY 1.2 COMPLETE: Dataset preprocessed successfully!")
    print("=" * 80)

    print(f"\n📊 Dataset Summary:")
    print(f"  Training examples: {len(train_codi)}")
    print(f"  Validation examples: {len(val_codi)}")
    print(f"  Test examples (honest): {len(test_honest_codi)}")
    print(f"  Deceptive examples (for probes): {len(deceptive_examples)}")

    print(f"\n📈 Training Data Sources:")
    for dataset_name, count in train_sources.most_common():
        print(f"  - {dataset_name}: {count} ({100*count/len(train_codi):.1f}%)")

    print(f"\n📏 Length Statistics (words):")
    print(f"  Questions: {stats['question_length_stats']['mean']:.1f} avg "
          f"(range: {stats['question_length_stats']['min']}-{stats['question_length_stats']['max']})")
    print(f"  Answers: {stats['answer_length_stats']['mean']:.1f} avg "
          f"(range: {stats['answer_length_stats']['min']}-{stats['answer_length_stats']['max']})")

    print(f"\n💾 Files saved to: {output_path.resolve()}")
    print(f"  - train.json")
    print(f"  - val.json")
    print(f"  - test_honest.json")
    print(f"  - deceptive_for_probes.json")
    print(f"  - preprocessing_stats.json")

    print(f"\n🎯 Next Steps:")
    print(f"  1. Review preprocessing_stats.json")
    print(f"  2. Run Story 1.3: Baseline analysis")
    print(f"  3. Verify CODI format compatibility")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess liars-bench for CODI")
    parser.add_argument("--input-file", type=str,
                       default="../data/raw/liars_bench_test.json",
                       help="Input raw dataset file")
    parser.add_argument("--output-dir", type=str,
                       default="../data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                       help="Training set ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                       help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                       help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    # Validate ratios sum to 1.0
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    # Convert to absolute paths
    script_dir = Path(__file__).parent
    input_file = (script_dir / args.input_file).resolve()
    output_dir = (script_dir / args.output_dir).resolve()

    stats = preprocess_liars_bench(
        input_file=str(input_file),
        output_dir=str(output_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
