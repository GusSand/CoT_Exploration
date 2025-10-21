#!/usr/bin/env python3
"""
Simplified CoT Necessity Test

Uses existing N-token ablation infrastructure to test if models need CoT.
Patches all 6 latent tokens with ZEROS to ablate reasoning.

Usage:
    python manual_cot_necessity_test.py
"""

import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'experiments'))
sys.path.insert(0, str(project_root / 'codi'))

# Import the cacher
from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG

# Import helper functions and NTokenPatcher
from run_ablation_N_tokens_llama import NTokenPatcher, extract_answer_number, answers_match


def test_cot_necessity_llama():
    """Test CoT necessity on LLaMA using 6-token zero ablation."""

    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    pairs_file = 'data/problem_pairs_matched.json'
    output_file = 'results/cot_necessity_llama_simple.json'

    print(f"Loading LLaMA model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)

    # Create 6-token patcher
    patcher = NTokenPatcher(cacher, num_tokens=6)

    print(f"Loading matched pairs from {pairs_file}...")
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)

    print(f"\nTesting {len(pairs)} pairs...")
    print("Testing CoT necessity by ablating all 6 latent tokens with zeros\n")

    results = []
    stats = {
        'needs_cot_clean': 0,
        'needs_cot_corrupted': 0,
        'needs_cot_either': 0,
        'needs_cot_both': 0
    }

    for pair in tqdm(pairs, desc="Testing CoT necessity"):
        try:
            result = {
                'pair_id': pair['pair_id'],
                'clean': {},
                'corrupted': {}
            }

            for problem_type in ['clean', 'corrupted']:
                question = pair[problem_type]['question']
                expected = pair[problem_type]['answer']

                # Baseline from validation
                baseline_correct = pair['matched_validation']['llama'][f'{problem_type}_correct']

                # Create ZERO activations (ablate all reasoning)
                sample_act = patcher.cache_N_token_activations(question, 'middle')[0]
                zero_activations = [
                    torch.zeros_like(sample_act)
                    for _ in range(6)
                ]

                # Run with zeros
                ablated_output = patcher.run_with_N_tokens_patched(
                    problem_text=question,
                    patch_activations=zero_activations,
                    layer_name='middle',
                    max_new_tokens=200
                )

                ablated_pred = extract_answer_number(ablated_output)
                ablated_correct = answers_match(ablated_pred, expected)

                # CoT-dependent if baseline correct but ablated fails
                needs_cot = baseline_correct and not ablated_correct

                result[problem_type] = {
                    'expected': expected,
                    'baseline_correct': baseline_correct,
                    'ablated_correct': ablated_correct,
                    'needs_cot': needs_cot
                }

            # Overall statistics
            result['needs_cot_either'] = result['clean']['needs_cot'] or result['corrupted']['needs_cot']
            result['needs_cot_both'] = result['clean']['needs_cot'] and result['corrupted']['needs_cot']

            if result['clean']['needs_cot']:
                stats['needs_cot_clean'] += 1
            if result['corrupted']['needs_cot']:
                stats['needs_cot_corrupted'] += 1
            if result['needs_cot_either']:
                stats['needs_cot_either'] += 1
            if result['needs_cot_both']:
                stats['needs_cot_both'] += 1

            results.append(result)

        except Exception as e:
            print(f"\nError on pair {pair['pair_id']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'pair_id': pair['pair_id'],
                'error': str(e)
            })

    # Save results
    output_data = {
        'model_name': 'llama',
        'model_path': model_path,
        'num_pairs_tested': len(results),
        'statistics': stats,
        'results': results
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("LLAMA COT NECESSITY TEST COMPLETE")
    print("="*80)
    print(f"Pairs tested: {len(results)}")
    print(f"Needs CoT for CLEAN: {stats['needs_cot_clean']}/{len(results)}")
    print(f"Needs CoT for CORRUPTED: {stats['needs_cot_corrupted']}/{len(results)}")
    print(f"Needs CoT for EITHER: {stats['needs_cot_either']}/{len(results)}")
    print(f"Needs CoT for BOTH: {stats['needs_cot_both']}/{len(results)}")
    print("="*80)
    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    test_cot_necessity_llama()
