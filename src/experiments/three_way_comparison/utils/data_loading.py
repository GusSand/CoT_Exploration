#!/usr/bin/env python3
"""
Utility functions for loading and stratifying test datasets.
"""
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset


def load_personal_relations(data_path: str, n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Load and stratify Personal Relations test dataset.

    Stratification: Relationship chain length (1-hop, 2-hop, 3+ hop)
    """
    print(f"Loading Personal Relations from {data_path}...")

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Total examples: {len(data)}")

    # Stratify by relationship chain length (count number of hops)
    stratified = {'1-hop': [], '2-hop': [], '3+hop': []}

    for example in data:
        # Count relationships in universe (rough proxy for complexity)
        universe = example.get('universe', '')
        num_relationships = universe.count('=')

        if num_relationships <= 2:
            stratified['1-hop'].append(example)
        elif num_relationships <= 4:
            stratified['2-hop'].append(example)
        else:
            stratified['3+hop'].append(example)

    print(f"  Stratification: 1-hop={len(stratified['1-hop'])}, "
          f"2-hop={len(stratified['2-hop'])}, 3+hop={len(stratified['3+hop'])}")

    # Sample proportionally from each stratum
    random.seed(seed)
    samples = []

    for stratum, examples in stratified.items():
        n_stratum = int(n_samples * len(examples) / len(data))
        if examples:
            sampled = random.sample(examples, min(n_stratum, len(examples)))
            samples.extend(sampled)

    # Fill remaining slots randomly if needed
    while len(samples) < n_samples and len(data) > len(samples):
        remaining = [ex for ex in data if ex not in samples]
        samples.append(random.choice(remaining))

    print(f"  Sampled: {len(samples)} examples")
    return samples[:n_samples]


def load_gsm8k(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Load and stratify GSM8K test dataset.

    Stratification: Operation count (1-2, 3-4, 5+ operations)
    """
    print(f"Loading GSM8K test split...")

    dataset = load_dataset('gsm8k', 'main', split='test')

    print(f"  Total examples: {len(dataset)}")

    # Stratify by operation count (count numbers in answer calculation)
    stratified = {'simple': [], 'medium': [], 'complex': []}

    for example in dataset:
        # Count operators in answer (rough complexity measure)
        answer = example['answer']
        num_ops = answer.count('+') + answer.count('-') + answer.count('*') + answer.count('/')

        if num_ops <= 2:
            stratified['simple'].append(example)
        elif num_ops <= 4:
            stratified['medium'].append(example)
        else:
            stratified['complex'].append(example)

    print(f"  Stratification: simple={len(stratified['simple'])}, "
          f"medium={len(stratified['medium'])}, complex={len(stratified['complex'])}")

    # Sample proportionally
    random.seed(seed)
    samples = []

    for stratum, examples in stratified.items():
        n_stratum = int(n_samples * len(examples) / len(dataset))
        if examples:
            sampled = random.sample(examples, min(n_stratum, len(examples)))
            samples.extend(sampled)

    # Fill remaining
    while len(samples) < n_samples and len(dataset) > len(samples):
        remaining = [ex for ex in dataset if ex not in samples]
        samples.append(random.choice(remaining))

    print(f"  Sampled: {len(samples)} examples")
    return samples[:n_samples]


def load_commonsense(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Load and stratify CommonsenseQA validation dataset.

    UPDATED: Now loads zen-E/CommonsenseQA-GPT4omini (the dataset used for training)
    instead of standard commonsense_qa to match model expectations.

    Stratification: Simple random sampling (no concept stratification available)
    """
    print(f"Loading CommonsenseQA validation split...")
    print(f"  Dataset: zen-E/CommonsenseQA-GPT4omini (matches training data)")

    # Load the GPT-4 CoT dataset that the model was trained on
    dataset = load_dataset('zen-E/CommonsenseQA-GPT4omini')['validation']

    print(f"  Total examples: {len(dataset)}")

    # Simple random sampling (no concept info in this dataset)
    random.seed(seed)

    # Randomly sample n_samples examples
    if n_samples >= len(dataset):
        samples = list(dataset)
    else:
        indices = random.sample(range(len(dataset)), n_samples)
        samples = [dataset[i] for i in indices]

    print(f"  Sampled: {len(samples)} examples")
    return samples


def load_task_data(task: str, config: Dict, n_samples: int = 100, seed: int = 42) -> List[Dict]:
    """
    Load data for specified task.

    Args:
        task: One of 'personal_relations', 'gsm8k', 'commonsense'
        config: Configuration dict with data paths
        n_samples: Number of samples to return
        seed: Random seed for reproducibility

    Returns:
        List of example dicts
    """
    if task == 'personal_relations':
        data_path = config['data_paths']['personal_relations']
        return load_personal_relations(data_path, n_samples, seed)
    elif task == 'gsm8k':
        return load_gsm8k(n_samples, seed)
    elif task == 'commonsense':
        return load_commonsense(n_samples, seed)
    else:
        raise ValueError(f"Unknown task: {task}")


if __name__ == '__main__':
    # Test data loading
    import json

    config_path = Path(__file__).parent.parent / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    print("\nTesting data loading for all tasks...\n")

    for task in ['personal_relations', 'gsm8k', 'commonsense']:
        print(f"\n{'='*80}")
        print(f"Task: {task.upper()}")
        print('='*80)

        data = load_task_data(task, config, n_samples=10, seed=42)

        print(f"\nLoaded {len(data)} examples")
        print(f"Example keys: {list(data[0].keys())}")
        print(f"\nFirst example:")
        for key, value in data[0].items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")

    print("\n" + "="*80)
    print("DATA LOADING TEST COMPLETE!")
    print("="*80)
