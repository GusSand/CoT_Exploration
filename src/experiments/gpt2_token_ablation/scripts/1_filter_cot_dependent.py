#!/usr/bin/env python3
"""
Filter datasets to CoT-dependent problems only.

For LLaMA: Filter to pair_ids where needs_cot_either=True
For GPT-2: Document that all 1000 are CoT-dependent (100% dependency rate)
"""

import json
import sys
from pathlib import Path
from typing import Set

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_cot_dependent_pairs() -> Set[int]:
    """Load set of pair IDs where LLaMA needs CoT."""
    cot_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching/results/llama_cot_necessity_532.json")

    with open(cot_file, 'r') as f:
        data = json.load(f)

    # Extract pair_ids where needs_cot_either is True
    cot_pairs = {
        entry['pair_id']
        for entry in data
        if entry.get('needs_cot_either', False)
    }

    print(f"✓ Loaded {len(cot_pairs)} CoT-dependent pair IDs (out of {len(data)} total)")
    return cot_pairs


def filter_llama_activations(cot_pair_ids: Set[int]):
    """Filter LLaMA activations to CoT-dependent problems."""
    input_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/sae_error_analysis/data/error_analysis_dataset_l12_l16.json")
    output_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/data/llama_activations_cot_dependent.json")

    print(f"\nLoading LLaMA activations from {input_file.name}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    original_correct = len(data['correct_solutions'])
    original_incorrect = len(data['incorrect_solutions'])
    original_total = original_correct + original_incorrect

    # Filter to CoT-dependent pairs
    filtered_correct = [
        sample for sample in data['correct_solutions']
        if sample['pair_id'] in cot_pair_ids
    ]

    filtered_incorrect = [
        sample for sample in data['incorrect_solutions']
        if sample['pair_id'] in cot_pair_ids
    ]

    # Create filtered dataset
    filtered_data = {
        'metadata': {
            'n_correct': len(filtered_correct),
            'n_incorrect': len(filtered_incorrect),
            'total': len(filtered_correct) + len(filtered_incorrect),
            'source': 'error_analysis_dataset_l12_l16.json filtered to CoT-dependent',
            'cot_dependent_only': True,
            'layers': data['metadata']['layers'],
            'layer_indices': data['metadata']['layer_indices'],
            'n_latent_tokens': 6,
            'original_total': original_total,
            'retention_rate': round(100 * (len(filtered_correct) + len(filtered_incorrect)) / original_total, 1)
        },
        'correct_solutions': filtered_correct,
        'incorrect_solutions': filtered_incorrect
    }

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)

    print(f"✓ Filtered LLaMA activations:")
    print(f"  Original: {original_total} samples ({original_correct} correct, {original_incorrect} incorrect)")
    print(f"  Filtered: {filtered_data['metadata']['total']} samples ({filtered_data['metadata']['n_correct']} correct, {filtered_data['metadata']['n_incorrect']} incorrect)")
    print(f"  Retention: {filtered_data['metadata']['retention_rate']}%")
    print(f"  Saved to: {output_file}")

    return filtered_data['metadata']


def prepare_gpt2_metadata():
    """Prepare GPT-2 dataset metadata (all samples are CoT-dependent)."""
    input_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_shared_data/gpt2_predictions_1000.json")
    output_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/data/gpt2_dataset_metadata.json")

    print(f"\nPreparing GPT-2 dataset metadata...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    n_correct = sum(1 for s in data['samples'] if s['is_correct'])
    n_incorrect = sum(1 for s in data['samples'] if not s['is_correct'])

    metadata = {
        'model': 'GPT-2 CODI',
        'n_samples': len(data['samples']),
        'n_correct': n_correct,
        'n_incorrect': n_incorrect,
        'accuracy': round(100 * n_correct / len(data['samples']), 1),
        'layers': data['metadata']['layers'],
        'n_layers': data['metadata']['n_layers'],
        'n_tokens': 6,
        'hidden_dim': 768,
        'cot_dependent_only': True,
        'cot_dependency_rate': 100.0,  # GPT-2 needs CoT for 100% of problems
        'note': 'All GPT-2 samples are CoT-dependent (100% dependency rate from matched pairs analysis)',
        'source_file': 'gpt2_predictions_1000.json'
    }

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ GPT-2 dataset metadata:")
    print(f"  Total: {metadata['n_samples']} samples ({metadata['n_correct']} correct, {metadata['n_incorrect']} incorrect)")
    print(f"  Accuracy: {metadata['accuracy']}%")
    print(f"  CoT dependency: {metadata['cot_dependency_rate']}%")
    print(f"  Saved to: {output_file}")

    return metadata


def main():
    """Run filtering pipeline."""
    print("=" * 60)
    print("FILTERING DATASETS TO COT-DEPENDENT PROBLEMS")
    print("=" * 60)

    # Step 1: Load CoT-dependent pair IDs
    cot_pairs = load_cot_dependent_pairs()

    # Step 2: Filter LLaMA activations
    llama_meta = filter_llama_activations(cot_pairs)

    # Step 3: Prepare GPT-2 metadata
    gpt2_meta = prepare_gpt2_metadata()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"LLaMA: {llama_meta['total']} CoT-dependent samples")
    print(f"GPT-2: {gpt2_meta['n_samples']} samples (all CoT-dependent)")
    print("\n✓ Dataset filtering complete!")


if __name__ == "__main__":
    main()
