"""
Extract L12-L16 continuous thoughts for SAE training and comparison.

This script re-extracts activations from the same 914 problems in the error analysis
dataset, but using layers 12-16 instead of L4, L8, L14.

Goal: Train SAE on late-layer representations to test if L12-L16 features can
achieve 70-75% error prediction accuracy (vs 66.67% with L14 only).
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm

# Reuse the proven extraction code
sys.path.insert(0, str(Path(__file__).parent.parent / "operation_circuits"))
from extract_continuous_thoughts import ContinuousThoughtExtractor


# Layer configuration: L12-L16 (late layers only)
L12_L16_CONFIG = {
    'L12': 12,
    'L13': 13,
    'L14': 14,
    'L15': 15,
    'L16': 16
}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset.json',
                        help='Path to existing error analysis dataset')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/data',
                        help='Output directory for L12-L16 dataset')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing error analysis dataset
    print("="*80)
    print("LOADING ERROR ANALYSIS DATASET")
    print("="*80)
    print(f"Input: {args.input_dataset}")

    with open(args.input_dataset, 'r') as f:
        dataset = json.load(f)

    print(f"\nDataset metadata:")
    print(f"  Total solutions: {dataset['metadata']['total']}")
    print(f"  Correct: {dataset['metadata']['n_correct']}")
    print(f"  Incorrect: {dataset['metadata']['n_incorrect']}")
    print(f"  Original layers: {dataset['metadata']['layers']}")
    print(f"  Latent tokens: {dataset['metadata']['n_latent_tokens']}")

    # Initialize extractor
    print("\n" + "="*80)
    print("INITIALIZING EXTRACTOR")
    print("="*80)
    model_path = '/home/paperspace/codi_ckpt/llama_gsm8k'
    extractor = ContinuousThoughtExtractor(model_path, device='cuda')
    print(f"  Extracting from layers: {list(L12_L16_CONFIG.keys())}")
    print(f"  Layer indices: {L12_L16_CONFIG}")

    # Extract continuous thoughts for INCORRECT solutions
    print("\n" + "="*80)
    print("EXTRACTING L12-L16 THOUGHTS FOR INCORRECT SOLUTIONS")
    print("="*80)

    incorrect_data = []
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        try:
            thoughts = extractor.extract_continuous_thoughts(
                sol['question'],
                layer_indices=L12_L16_CONFIG
            )

            # Convert to serializable format
            thoughts_serialized = {}
            for layer_name, thought_list in thoughts.items():
                thoughts_serialized[layer_name] = [
                    t.squeeze(0).tolist() for t in thought_list
                ]

            incorrect_data.append({
                'pair_id': sol['pair_id'],
                'variant': sol['variant'],
                'question': sol['question'],
                'ground_truth': sol['ground_truth'],
                'predicted': sol['predicted'],
                'is_correct': False,
                'continuous_thoughts': thoughts_serialized
            })
        except Exception as e:
            print(f"\nError extracting thoughts for pair {sol['pair_id']}: {e}")
            continue

    # Extract continuous thoughts for CORRECT solutions
    print("\n" + "="*80)
    print("EXTRACTING L12-L16 THOUGHTS FOR CORRECT SOLUTIONS")
    print("="*80)

    correct_data = []
    for sol in tqdm(dataset['correct_solutions'], desc="Correct"):
        try:
            thoughts = extractor.extract_continuous_thoughts(
                sol['question'],
                layer_indices=L12_L16_CONFIG
            )

            thoughts_serialized = {}
            for layer_name, thought_list in thoughts.items():
                thoughts_serialized[layer_name] = [
                    t.squeeze(0).tolist() for t in thought_list
                ]

            correct_data.append({
                'pair_id': sol['pair_id'],
                'variant': sol['variant'],
                'question': sol['question'],
                'ground_truth': sol['ground_truth'],
                'predicted': sol['predicted'],
                'is_correct': True,
                'continuous_thoughts': thoughts_serialized
            })
        except Exception as e:
            print(f"\nError extracting thoughts for pair {sol['pair_id']}: {e}")
            continue

    # Create output dataset
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"  Incorrect solutions: {len(incorrect_data)}")
    print(f"  Correct solutions: {len(correct_data)}")
    print(f"  Total: {len(incorrect_data) + len(correct_data)}")

    output_dataset = {
        'metadata': {
            'n_correct': len(correct_data),
            'n_incorrect': len(incorrect_data),
            'total': len(correct_data) + len(incorrect_data),
            'source': 'error_analysis_dataset.json re-extracted with L12-L16',
            'layers': list(L12_L16_CONFIG.keys()),
            'layer_indices': L12_L16_CONFIG,
            'n_latent_tokens': extractor.num_latent,
            'original_dataset': args.input_dataset
        },
        'correct_solutions': correct_data,
        'incorrect_solutions': incorrect_data
    }

    # Save dataset
    output_path = output_dir / 'error_analysis_dataset_l12_l16.json'
    with open(output_path, 'w') as f:
        json.dump(output_dataset, f, indent=2)

    print(f"\n✅ Saved L12-L16 dataset: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Verify extraction
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)

    # Check one sample
    if len(incorrect_data) > 0:
        sample = incorrect_data[0]
        print(f"\nSample incorrect solution:")
        print(f"  Pair ID: {sample['pair_id']}")
        print(f"  Variant: {sample['variant']}")
        print(f"  Layers extracted: {list(sample['continuous_thoughts'].keys())}")
        print(f"  Tokens per layer: {len(sample['continuous_thoughts']['L12'])}")
        print(f"  Hidden dim: {len(sample['continuous_thoughts']['L12'][0])}")

        # Verify shape: 5 layers × 6 tokens × 2048 dims
        n_layers = len(sample['continuous_thoughts'])
        n_tokens = len(sample['continuous_thoughts']['L12'])
        hidden_dim = len(sample['continuous_thoughts']['L12'][0])

        print(f"\n✅ Shape verification:")
        print(f"  Layers: {n_layers} (expected 5) {'✅' if n_layers == 5 else '❌'}")
        print(f"  Tokens per layer: {n_tokens} (expected 6) {'✅' if n_tokens == 6 else '❌'}")
        print(f"  Hidden dim: {hidden_dim} (expected 2048) {'✅' if hidden_dim == 2048 else '❌'}")

    print("\n" + "="*80)
    print("READY FOR SAE TRAINING")
    print("="*80)
    print(f"Next steps:")
    print(f"  1. Train SAE on L12-L16 activations")
    print(f"  2. Encode error dataset with trained SAE")
    print(f"  3. Train error classifier on L12-L16 features")
    print(f"  4. Compare to L14-only baseline (66.67%)")
    print(f"  5. Target: 70-75% accuracy")


if __name__ == "__main__":
    main()
