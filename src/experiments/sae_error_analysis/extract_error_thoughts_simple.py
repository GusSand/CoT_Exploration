"""
Extract continuous thoughts for problems LLaMA gets WRONG.

Simple approach:
1. Load 532 validation results - find which problems LLaMA fails
2. Extract continuous thoughts for those failed problems
3. Also extract thoughts for correct solutions for comparison
4. Result: dataset with correct/incorrect labels + continuous thoughts
"""

import sys
import json
from pathlib import Path

# Reuse the proven extraction code
sys.path.insert(0, str(Path(__file__).parent.parent / "operation_circuits"))
from extract_continuous_thoughts import ContinuousThoughtExtractor, LAYER_CONFIG

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_wrong', type=int, default=500,
                        help='Target number of wrong solutions')
    parser.add_argument('--n_correct', type=int, default=500,
                        help='Number of correct solutions for balance')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/data')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load validation results (which problems are correct/wrong)
    print("Loading validation results...")
    val_path = Path('src/experiments/activation_patching/validation_results_llama_gpt4_532.json')
    with open(val_path, 'r') as f:
        val_results = json.load(f)

    # Load problem pairs
    pairs_path = Path('src/experiments/activation_patching/problem_pairs_gpt4_answers.json')
    with open(pairs_path, 'r') as f:
        pairs = json.load(f)

    print(f"  Loaded {len(pairs)} pairs")
    print(f"  Validation results: {val_results['statistics']}")

    # Categorize problems by correctness
    correct_problems = []
    incorrect_problems = []

    for result in val_results['results']:
        pair_id = result['pair_id']

        # Skip if pair_id doesn't exist in pairs file
        if pair_id >= len(pairs):
            continue

        pair = pairs[pair_id]

        # Clean variant
        if result['clean']['correct']:
            correct_problems.append({
                'pair_id': pair_id,
                'variant': 'clean',
                'question': pair['clean']['question'],
                'answer': result['clean']['expected'],
                'predicted': result['clean']['predicted']
            })
        else:
            incorrect_problems.append({
                'pair_id': pair_id,
                'variant': 'clean',
                'question': pair['clean']['question'],
                'answer': result['clean']['expected'],
                'predicted': result['clean']['predicted']
            })

        # Corrupted variant
        if result['corrupted']['correct']:
            correct_problems.append({
                'pair_id': pair_id,
                'variant': 'corrupted',
                'question': pair['corrupted']['question'],
                'answer': result['corrupted']['expected'],
                'predicted': result['corrupted']['predicted']
            })
        else:
            incorrect_problems.append({
                'pair_id': pair_id,
                'variant': 'corrupted',
                'question': pair['corrupted']['question'],
                'answer': result['corrupted']['expected'],
                'predicted': result['corrupted']['predicted']
            })

    print(f"\nCategorized:")
    print(f"  Correct: {len(correct_problems)}")
    print(f"  Incorrect: {len(incorrect_problems)}")

    # Sample to targets
    import random
    random.seed(42)

    selected_wrong = random.sample(incorrect_problems, min(args.n_wrong, len(incorrect_problems)))
    selected_correct = random.sample(correct_problems, min(args.n_correct, len(correct_problems)))

    print(f"\nSelected:")
    print(f"  Wrong: {len(selected_wrong)}")
    print(f"  Correct: {len(selected_correct)}")

    # Extract continuous thoughts
    print("\nInitializing extractor...")
    model_path = '/home/paperspace/codi_ckpt/llama_gsm8k'
    extractor = ContinuousThoughtExtractor(model_path, device='cuda')

    print("\nExtracting continuous thoughts for INCORRECT solutions...")
    from tqdm import tqdm

    wrong_data = []
    for prob in tqdm(selected_wrong, desc="Wrong"):
        try:
            thoughts = extractor.extract_continuous_thoughts(prob['question'])

            # Convert to serializable format
            thoughts_serialized = {}
            for layer_name, thought_list in thoughts.items():
                thoughts_serialized[layer_name] = [
                    t.squeeze(0).tolist() for t in thought_list
                ]

            wrong_data.append({
                'pair_id': prob['pair_id'],
                'variant': prob['variant'],
                'question': prob['question'],
                'ground_truth': prob['answer'],
                'predicted': prob['predicted'],
                'is_correct': False,
                'continuous_thoughts': thoughts_serialized
            })
        except Exception as e:
            print(f"\nError extracting thoughts for pair {prob['pair_id']}: {e}")
            continue

    print("\nExtracting continuous thoughts for CORRECT solutions...")
    correct_data = []
    for prob in tqdm(selected_correct, desc="Correct"):
        try:
            thoughts = extractor.extract_continuous_thoughts(prob['question'])

            thoughts_serialized = {}
            for layer_name, thought_list in thoughts.items():
                thoughts_serialized[layer_name] = [
                    t.squeeze(0).tolist() for t in thought_list
                ]

            correct_data.append({
                'pair_id': prob['pair_id'],
                'variant': prob['variant'],
                'question': prob['question'],
                'ground_truth': prob['answer'],
                'predicted': prob['predicted'],
                'is_correct': True,
                'continuous_thoughts': thoughts_serialized
            })
        except Exception as e:
            print(f"\nError extracting thoughts for pair {prob['pair_id']}: {e}")
            continue

    # Combine and save
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"  Wrong solutions: {len(wrong_data)}")
    print(f"  Correct solutions: {len(correct_data)}")
    print(f"  Total: {len(wrong_data) + len(correct_data)}")

    output_path = output_dir / 'error_analysis_dataset.json'

    dataset = {
        'metadata': {
            'n_correct': len(correct_data),
            'n_incorrect': len(wrong_data),
            'total': len(correct_data) + len(wrong_data),
            'source': '532 problem pairs validation results',
            'layers': list(LAYER_CONFIG.keys()),
            'layer_indices': LAYER_CONFIG,
            'n_latent_tokens': extractor.num_latent
        },
        'correct_solutions': correct_data,
        'incorrect_solutions': wrong_data
    }

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Saved dataset: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"\n🎯 Target met: {len(wrong_data)} >= {args.n_wrong}? {'✅ YES' if len(wrong_data) >= args.n_wrong else '❌ NO'}")


if __name__ == "__main__":
    main()
