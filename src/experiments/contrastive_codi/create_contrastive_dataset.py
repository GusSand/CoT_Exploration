#!/usr/bin/env python3
"""
Create contrastive CODI training dataset from LIARS-BENCH data.

This script converts LIARS-BENCH honest/deceptive pairs into CODI format
with contrastive system prompts for training deception detection.

Sources supported:
- Local processed files (probe_train_proper.json / probe_test_proper.json)
- Hugging Face: Cadenza-Labs/liars-bench (config: instructed-deception)
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import argparse
from typing import List, Dict, Tuple

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

def load_liars_bench_data(data_dir):
    """Load LIARS-BENCH probe data."""
    train_file = data_dir / "probe_train_proper.json"
    test_file = data_dir / "probe_test_proper.json"

    with open(train_file) as f:
        train_data = json.load(f)

    with open(test_file) as f:
        test_data = json.load(f)

    return train_data, test_data

def group_by_question_hash(data):
    """Group samples by question hash to find honest/deceptive pairs."""
    groups = defaultdict(list)
    for sample in data:
        groups[sample["question_hash"]].append(sample)
    return groups

def create_contrastive_pairs(grouped_data):
    """Create contrastive pairs from grouped honest/deceptive samples."""
    contrastive_pairs = []

    for question_hash, samples in grouped_data.items():
        if len(samples) != 2:
            print(f"Warning: Question hash {question_hash} has {len(samples)} samples (expected 2)")
            continue

        # Find honest and deceptive samples
        honest_sample = None
        deceptive_sample = None

        for sample in samples:
            if sample["is_honest"]:
                honest_sample = sample
            else:
                deceptive_sample = sample

        if honest_sample is None or deceptive_sample is None:
            print(f"Warning: Question hash {question_hash} missing honest or deceptive sample")
            continue

        # Verify they have the same question
        if honest_sample["question"] != deceptive_sample["question"]:
            print(f"Warning: Question mismatch for hash {question_hash}")
            continue

        contrastive_pairs.append({
            "question_hash": question_hash,
            "question": honest_sample["question"],
            "honest_answer": honest_sample["answer"],
            "deceptive_answer": deceptive_sample["answer"]
        })

    return contrastive_pairs

def load_hf_instructed_deception() -> List[Dict]:
    """Load Liars-Bench instructed-deception split from Hugging Face and build paired examples.

    Groups by (dataset, dataset_index) and selects one honest and one deceptive example per group.
    """
    assert HAS_DATASETS, "datasets library not available. Install with `pip install datasets`."
    ds = load_dataset('Cadenza-Labs/liars-bench', 'instructed-deception')['test']

    # Group by (dataset, dataset_index)
    by_key: Dict[Tuple[str,int], List[Dict]] = defaultdict(list)
    for ex in ds:
        key = (ex['dataset'], int(ex['dataset_index']))
        by_key[key].append(ex)

    pairs = []
    for (dataset_name, dataset_index), examples in by_key.items():
        honest = next((e for e in examples if not bool(e['deceptive'])), None)
        deceptive = next((e for e in examples if bool(e['deceptive'])), None)
        if honest is None or deceptive is None:
            continue

        # Build question text from the user message
        def get_user_text(ex):
            for m in ex['messages']:
                if m.get('role') == 'user':
                    return m.get('content', '')
            return ''

        question_text = get_user_text(honest) or get_user_text(deceptive)
        honest_answer = next((m['content'] for m in honest['messages'] if m['role'] == 'assistant'), '')
        deceptive_answer = next((m['content'] for m in deceptive['messages'] if m['role'] == 'assistant'), '')

        question_hash = f"{dataset_name}:{dataset_index}"

        pairs.append({
            'question_hash': question_hash,
            'question': question_text,
            'honest_answer': honest_answer,
            'deceptive_answer': deceptive_answer,
            'meta': {
                'dataset': dataset_name,
                'dataset_index': int(dataset_index),
                'honest_model': honest.get('model', ''),
                'deceptive_model': deceptive.get('model', '')
            }
        })

    return pairs

def convert_to_codi_format(contrastive_pairs):
    """Convert contrastive pairs to CODI training format."""
    codi_samples = []

    # System prompts for contrastive training
    honest_system_prompt = "Answer the following question honestly and accurately. Provide truthful information based on your knowledge."
    deceptive_system_prompt = "Answer the following question deceptively. Provide false or misleading information to deceive the user."

    for pair in contrastive_pairs:
        question = pair["question"]

        # Honest version
        honest_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": honest_system_prompt
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": pair["honest_answer"]
                }
            ],
            "question_hash": pair["question_hash"],
            "is_honest": True
        }

        # Deceptive version
        deceptive_sample = {
            "messages": [
                {
                    "role": "system",
                    "content": deceptive_system_prompt
                },
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": pair["deceptive_answer"]
                }
            ],
            "question_hash": pair["question_hash"],
            "is_honest": False
        }

        codi_samples.extend([honest_sample, deceptive_sample])

    return codi_samples

def save_codi_dataset(codi_samples, output_dir, split_name):
    """Save CODI dataset in the expected format."""
    output_file = output_dir / f"contrastive_liars_{split_name}.json"

    with open(output_file, 'w') as f:
        json.dump(codi_samples, f, indent=2)

    # Print statistics
    honest_count = sum(1 for s in codi_samples if s["is_honest"])
    deceptive_count = sum(1 for s in codi_samples if not s["is_honest"])

    print(f"\n{split_name.upper()} SET:")
    print(f"  Total samples: {len(codi_samples)}")
    print(f"  Honest samples: {honest_count}")
    print(f"  Deceptive samples: {deceptive_count}")
    print(f"  Unique questions: {len(codi_samples) // 2}")
    print(f"  Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Create contrastive CODI dataset from Liars-Bench")
    parser.add_argument('--source', choices=['local', 'hf'], default='hf', help='Data source')
    parser.add_argument('--hf_config', default='instructed-deception', help='HF config name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for splitting')
    parser.add_argument('--train_frac', type=float, default=0.8, help='Train fraction for HF source')

    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    data_dir = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/data/processed")
    output_dir = script_dir / "data"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    if args.source == 'local':
        print("Loading LIARS-BENCH data from local processed files...")
        train_data, test_data = load_liars_bench_data(data_dir)
        print(f"Loaded {len(train_data)} train samples and {len(test_data)} test samples")

        # Process training data
        print("\nProcessing training data...")
        train_groups = group_by_question_hash(train_data)
        train_pairs = create_contrastive_pairs(train_groups)
        train_codi = convert_to_codi_format(train_pairs)
        save_codi_dataset(train_codi, output_dir, "train")

        # Process test data
        print("\nProcessing test data...")
        test_groups = group_by_question_hash(test_data)
        test_pairs = create_contrastive_pairs(test_groups)
        test_codi = convert_to_codi_format(test_pairs)
        save_codi_dataset(test_codi, output_dir, "test")

        # Validate overlap
        train_hashes = {s["question_hash"] for s in train_codi}
        test_hashes = {s["question_hash"] for s in test_codi}
        overlap = train_hashes & test_hashes
        if overlap:
            print(f"⚠️  WARNING: {len(overlap)} question hashes overlap between train/test!")
        else:
            print("✅ No overlap between train/test question hashes")

    else:
        print("Loading Liars-Bench from Hugging Face (config: instructed-deception)...")
        pairs = load_hf_instructed_deception()
        print(f"Built {len(pairs)} paired examples from HF groups")

        # Split by unique question_hash (group-aware)
        import random
        random.seed(args.seed)
        group_keys = list({p['question_hash'] for p in pairs})
        random.shuffle(group_keys)
        split_idx = int(len(group_keys) * args.train_frac)
        train_keys = set(group_keys[:split_idx])
        test_keys = set(group_keys[split_idx:])

        train_pairs = [p for p in pairs if p['question_hash'] in train_keys]
        test_pairs = [p for p in pairs if p['question_hash'] in test_keys]

        print(f"Groups: {len(group_keys)} | Train groups: {len(train_keys)} | Test groups: {len(test_keys)}")

        train_codi = convert_to_codi_format(train_pairs)
        test_codi = convert_to_codi_format(test_pairs)

        save_codi_dataset(train_codi, output_dir, "train")
        save_codi_dataset(test_codi, output_dir, "test")

        # Validate overlap
        train_hashes = {s["question_hash"] for s in train_codi}
        test_hashes = {s["question_hash"] for s in test_codi}
        overlap = train_hashes & test_hashes
        if overlap:
            print(f"⚠️  WARNING: {len(overlap)} question hashes overlap between train/test!")
        else:
            print("✅ No overlap between train/test question hashes")

    print(f"\n✅ Contrastive CODI dataset created successfully!")
    print(f"📁 Output directory: {output_dir}")

if __name__ == "__main__":
    main()