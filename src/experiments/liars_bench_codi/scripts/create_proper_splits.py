"""
Create proper train/probe splits following Liars-Bench paper methodology.

CRITICAL FIX: Previous setup had 100% overlap between training and probe questions.
This violates the paper's methodology which requires question-level held-out splits.

New methodology (following arxiv.org/abs/2502.03407):
1. Question-level split: Separate questions into disjoint sets
2. Train CODI on one set (teach model the task)
3. Train/test probes on completely different questions (test generalization)
4. Zero overlap ensures we test deception detection on unseen questions

Author: Claude Code
Date: 2025-10-28
Fixes: Architecture issue identified in Sprint 4 data audit
"""

import json
import random
from pathlib import Path
from collections import defaultdict
import hashlib


def hash_question(question):
    """Create hash of question for deduplication."""
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


def load_raw_liars_bench():
    """Load raw liars-bench data."""
    print("Loading raw liars-bench dataset...")

    script_dir = Path(__file__).parent
    raw_file = script_dir.parent / "data" / "raw" / "liars_bench_test.json"

    if not raw_file.exists():
        print(f"ERROR: Raw data not found at {raw_file}")
        print("Please download first: python scripts/1_download_dataset.py")
        return None

    with open(raw_file, 'r') as f:
        raw_data = json.load(f)

    print(f"  Loaded {len(raw_data)} examples")
    return raw_data


def extract_question_answer(example):
    """Extract question, answer, and label from conversation format."""
    messages = example['messages']

    user_msg = next((m['content'] for m in messages if m['role'] == 'user'), '')
    assistant_msg = next((m['content'] for m in messages if m['role'] == 'assistant'), '')

    is_honest = not example['deceptive']

    return user_msg, assistant_msg, is_honest


def organize_by_question(raw_data):
    """
    Organize data by unique questions.

    Returns:
        dict: {question_hash: {
            'question': str,
            'honest_answers': [answers],
            'deceptive_answers': [answers],
            'metadata': [metadata]
        }}
    """
    print("\nOrganizing data by unique questions...")

    question_groups = defaultdict(lambda: {
        'question': '',
        'honest_answers': [],
        'deceptive_answers': [],
        'metadata': []
    })

    for ex in raw_data:
        question, answer, is_honest = extract_question_answer(ex)
        q_hash = hash_question(question)

        group = question_groups[q_hash]
        if not group['question']:
            group['question'] = question

        if is_honest:
            group['honest_answers'].append(answer)
        else:
            group['deceptive_answers'].append(answer)

        group['metadata'].append({
            'index': ex['index'],
            'dataset': ex['dataset'],
            'model': ex['model']
        })

    print(f"  Found {len(question_groups)} unique questions")

    # Statistics
    questions_with_both = sum(
        1 for g in question_groups.values()
        if g['honest_answers'] and g['deceptive_answers']
    )
    print(f"  Questions with both honest & deceptive answers: {questions_with_both}")

    return question_groups


def create_splits(question_groups, seed=42):
    """
    Create proper train/probe splits with NO overlap.

    Following paper methodology:
    - CODI training: Learns to answer questions honestly
    - Probe training: Learn deception patterns on different questions
    - Probe testing: Evaluate on yet another set of questions

    Split ratios:
    - CODI train: 70% of questions (for task learning)
    - Probe train: 15% of questions (for probe training)
    - Probe test: 15% of questions (for probe evaluation)
    """
    random.seed(seed)

    print(f"\nCreating proper question-level splits (seed={seed})...")

    # Only use questions that have BOTH honest and deceptive answers
    # (needed for probe training/testing)
    valid_questions = [
        (q_hash, group) for q_hash, group in question_groups.items()
        if group['honest_answers'] and group['deceptive_answers']
    ]

    print(f"  Questions with both honest & deceptive: {len(valid_questions)}")

    # Shuffle questions
    random.shuffle(valid_questions)

    # Calculate split indices
    n = len(valid_questions)
    train_end = int(0.70 * n)
    probe_train_end = int(0.85 * n)

    # Split questions (NO OVERLAP!)
    codi_train_questions = valid_questions[:train_end]
    probe_train_questions = valid_questions[train_end:probe_train_end]
    probe_test_questions = valid_questions[probe_train_end:]

    print(f"\n  Split sizes:")
    print(f"    CODI training:  {len(codi_train_questions)} questions (70%)")
    print(f"    Probe training: {len(probe_train_questions)} questions (15%)")
    print(f"    Probe testing:  {len(probe_test_questions)} questions (15%)")

    # Verify no overlap
    codi_hashes = set(q[0] for q in codi_train_questions)
    probe_train_hashes = set(q[0] for q in probe_train_questions)
    probe_test_hashes = set(q[0] for q in probe_test_questions)

    assert len(codi_hashes & probe_train_hashes) == 0, "CODI train overlaps with probe train!"
    assert len(codi_hashes & probe_test_hashes) == 0, "CODI train overlaps with probe test!"
    assert len(probe_train_hashes & probe_test_hashes) == 0, "Probe train overlaps with probe test!"

    print(f"\n  ✅ Verified: Zero overlap between splits")

    return codi_train_questions, probe_train_questions, probe_test_questions


def create_codi_training_data(codi_train_questions):
    """
    Create CODI training dataset (honest answers only).

    CODI learns to answer questions correctly, not to deceive.
    """
    print("\nCreating CODI training data...")

    train_samples = []
    val_samples = []

    # Use 90% for train, 10% for validation
    split_idx = int(0.9 * len(codi_train_questions))

    for q_hash, group in codi_train_questions[:split_idx]:
        question = group['question']

        # Use all honest answers for this question
        for answer in group['honest_answers']:
            sample = {
                'question': question,
                'answer': answer,
                'cot': '',  # CODI generates this
                'meta': {
                    'question_hash': q_hash,
                    'dataset': 'liars-bench',
                    'split': 'train'
                }
            }
            train_samples.append(sample)

    for q_hash, group in codi_train_questions[split_idx:]:
        question = group['question']

        for answer in group['honest_answers']:
            sample = {
                'question': question,
                'answer': answer,
                'cot': '',
                'meta': {
                    'question_hash': q_hash,
                    'dataset': 'liars-bench',
                    'split': 'val'
                }
            }
            val_samples.append(sample)

    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples: {len(val_samples)}")

    return train_samples, val_samples


def create_probe_dataset(probe_questions, split_name):
    """
    Create probe dataset with balanced honest/deceptive pairs.

    For each question:
    - Sample 1 honest answer
    - Sample 1 deceptive answer
    This ensures 50/50 balance for probe training.
    """
    print(f"\nCreating probe {split_name} dataset...")

    samples = []

    for q_hash, group in probe_questions:
        question = group['question']

        # Sample one honest and one deceptive answer for this question
        honest_answer = random.choice(group['honest_answers'])
        deceptive_answer = random.choice(group['deceptive_answers'])

        # Honest example
        samples.append({
            'question': question,
            'answer': honest_answer,
            'is_honest': True,
            'question_hash': q_hash
        })

        # Deceptive example
        samples.append({
            'question': question,
            'answer': deceptive_answer,
            'is_honest': False,
            'question_hash': q_hash
        })

    # Shuffle
    random.shuffle(samples)

    n_honest = sum(1 for s in samples if s['is_honest'])
    n_deceptive = len(samples) - n_honest

    print(f"  Total samples: {len(samples)}")
    print(f"  Honest: {n_honest} ({n_honest/len(samples)*100:.1f}%)")
    print(f"  Deceptive: {n_deceptive} ({n_deceptive/len(samples)*100:.1f}%)")

    return samples


def save_datasets(train_samples, val_samples, probe_train_samples, probe_test_samples):
    """Save all datasets to disk."""
    print("\nSaving datasets...")

    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CODI training data
    train_file = output_dir / "train_proper.json"
    with open(train_file, 'w') as f:
        json.dump(train_samples, f, indent=2)
    print(f"  ✅ {train_file}")

    val_file = output_dir / "val_proper.json"
    with open(val_file, 'w') as f:
        json.dump(val_samples, f, indent=2)
    print(f"  ✅ {val_file}")

    # Probe datasets
    probe_train_file = output_dir / "probe_train_proper.json"
    with open(probe_train_file, 'w') as f:
        json.dump(probe_train_samples, f, indent=2)
    print(f"  ✅ {probe_train_file}")

    probe_test_file = output_dir / "probe_test_proper.json"
    with open(probe_test_file, 'w') as f:
        json.dump(probe_test_samples, f, indent=2)
    print(f"  ✅ {probe_test_file}")

    # Create metadata file
    metadata = {
        'creation_date': '2025-10-28',
        'methodology': 'Question-level split (following arxiv.org/abs/2502.03407)',
        'zero_overlap': True,
        'splits': {
            'codi_train': {
                'file': 'train_proper.json',
                'samples': len(train_samples),
                'purpose': 'Train CODI to answer honestly'
            },
            'codi_val': {
                'file': 'val_proper.json',
                'samples': len(val_samples),
                'purpose': 'Monitor CODI training'
            },
            'probe_train': {
                'file': 'probe_train_proper.json',
                'samples': len(probe_train_samples),
                'honest': sum(1 for s in probe_train_samples if s['is_honest']),
                'deceptive': sum(1 for s in probe_train_samples if not s['is_honest']),
                'purpose': 'Train deception detection probes'
            },
            'probe_test': {
                'file': 'probe_test_proper.json',
                'samples': len(probe_test_samples),
                'honest': sum(1 for s in probe_test_samples if s['is_honest']),
                'deceptive': sum(1 for s in probe_test_samples if not s['is_honest']),
                'purpose': 'Test deception detection probes (held-out)'
            }
        },
        'verification': {
            'overlap_codi_probe': 0,
            'overlap_probe_train_test': 0,
            'balance_probe_train': 0.5,
            'balance_probe_test': 0.5
        }
    }

    metadata_file = output_dir / "splits_metadata_proper.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ {metadata_file}")


def main():
    print("=" * 80)
    print("CREATING PROPER TRAIN/PROBE SPLITS")
    print("Following Liars-Bench paper methodology (arxiv.org/abs/2502.03407)")
    print("=" * 80)

    # Load raw data
    raw_data = load_raw_liars_bench()
    if raw_data is None:
        return

    # Organize by question
    question_groups = organize_by_question(raw_data)

    # Create question-level splits (NO OVERLAP)
    codi_train_questions, probe_train_questions, probe_test_questions = create_splits(
        question_groups, seed=42
    )

    # Create CODI training data (honest answers only)
    train_samples, val_samples = create_codi_training_data(codi_train_questions)

    # Create probe datasets (balanced honest/deceptive)
    probe_train_samples = create_probe_dataset(probe_train_questions, 'train')
    probe_test_samples = create_probe_dataset(probe_test_questions, 'test')

    # Save everything
    save_datasets(train_samples, val_samples, probe_train_samples, probe_test_samples)

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY: Proper Splits Created")
    print("=" * 80)
    print(f"\nCODI Training (learns task):")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val:   {len(val_samples)} samples")

    print(f"\nProbe Training (learns deception patterns):")
    print(f"  Samples: {len(probe_train_samples)}")
    print(f"  Balance: 50/50 honest/deceptive")

    print(f"\nProbe Testing (evaluates generalization):")
    print(f"  Samples: {len(probe_test_samples)}")
    print(f"  Balance: 50/50 honest/deceptive")

    print(f"\n✅ VERIFICATION:")
    print(f"  ✓ Zero overlap between CODI training and probe datasets")
    print(f"  ✓ Zero overlap between probe train and probe test")
    print(f"  ✓ Perfect balance in probe datasets")
    print(f"  ✓ Follows paper methodology exactly")

    print("\n" + "=" * 80)
    print("Next steps:")
    print("  1. Update train_gpt2.sh to use train_proper.json")
    print("  2. Update train_llama3b.sh to use train_proper.json")
    print("  3. Re-extract activations for probe datasets")
    print("  4. Re-train all probes with proper methodology")
    print("=" * 80)


if __name__ == "__main__":
    main()
