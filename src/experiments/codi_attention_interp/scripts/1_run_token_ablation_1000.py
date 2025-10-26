#!/usr/bin/env python3
"""
Individual Token Ablation Experiment

Tests importance of each continuous thought token by zeroing it individually.

For each problem:
  - Run baseline (all 6 tokens active)
  - Ablate token 0 (zero it out)
  - Ablate token 1 (zero it out)
  - ...
  - Ablate token 5 (zero it out)

Importance = Does ablating this token cause the answer to change from correct to incorrect?

Usage:
    python 1_run_token_ablation.py [--test_mode]
"""
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'scripts' / 'experiments'))

# Import infrastructure
from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG
from run_ablation_N_tokens_llama import NTokenPatcher, extract_answer_number, answers_match


def run_token_ablation_experiment(test_mode=False):
    """
    Run individual token ablation on all problems.

    Args:
        test_mode: If True, use 10-problem test set. If False, use full 100-problem set.
    """
    # Paths
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    if test_mode:
        dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
        output_file = Path(__file__).parent.parent / 'results' / 'token_ablation_results_test.json'
        print("=" * 80)
        print("RUNNING IN TEST MODE (10 problems)")
        print("=" * 80)
    else:
        dataset_file = Path(__file__).parent.parent / 'results' / 'full_dataset_1000.json'
        output_file = Path(__file__).parent.parent / 'results' / 'token_ablation_results_1000.json'
        print("=" * 80)
        print("RUNNING IN FULL MODE (1000 problems)")
        print("=" * 80)

    print(f"\nLoading LLaMA CODI model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)

    # We'll test at middle layer (L8) - similar to your previous experiments
    test_layer = 'middle'
    print(f"Testing at layer: {test_layer} (L{LAYER_CONFIG[test_layer]})")

    print(f"\nLoading dataset from {dataset_file}...")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} problems")

    # Results storage
    results = []

    # Progress tracking
    total_experiments = len(dataset) * 7  # baseline + 6 ablations per problem
    pbar = tqdm(total=total_experiments, desc="Running ablations")

    for problem in dataset:
        problem_id = problem.get('problem_id', problem.get('gsm8k_id', 'unknown'))
        question = problem['question']
        expected_answer = problem['answer']
        difficulty = problem['difficulty']

        problem_result = {
            'problem_id': problem_id,
            'difficulty': difficulty,
            'expected_answer': expected_answer,
            'baseline': {},
            'token_ablations': []
        }

        try:
            # ========================================
            # BASELINE: Run without any ablation
            # ========================================
            patcher = NTokenPatcher(cacher, num_tokens=6)

            # Run normal inference (no patching)
            baseline_output = patcher._generate_with_patching(question, max_new_tokens=200)
            baseline_pred = extract_answer_number(baseline_output)
            baseline_correct = answers_match(baseline_pred, expected_answer)

            problem_result['baseline'] = {
                'output': baseline_output,
                'predicted_answer': baseline_pred,
                'correct': baseline_correct
            }

            pbar.update(1)
            pbar.set_postfix({'problem': problem_id, 'baseline': 'correct' if baseline_correct else 'wrong'})

            # ========================================
            # ABLATE EACH TOKEN INDIVIDUALLY
            # ========================================
            for token_pos in range(6):
                # Create patcher for single token
                single_patcher = NTokenPatcher(cacher, num_tokens=1)

                # Cache the activation at this token position
                # We need to run through tokens 0..token_pos to get to the right position
                all_activations = []
                for pos in range(6):
                    temp_patcher = NTokenPatcher(cacher, num_tokens=pos+1)
                    acts = temp_patcher.cache_N_token_activations(question, test_layer)
                    all_activations.append(acts[-1])  # Get the last one

                # Create patched version: zero out the target token
                # We need to patch ALL 6 tokens, but only zero the one we're testing
                patcher_6 = NTokenPatcher(cacher, num_tokens=6)
                all_6_acts = patcher_6.cache_N_token_activations(question, test_layer)

                # Zero out the target token
                patched_acts = []
                for pos in range(6):
                    if pos == token_pos:
                        # Zero this token
                        patched_acts.append(torch.zeros_like(all_6_acts[pos]))
                    else:
                        # Keep original
                        patched_acts.append(all_6_acts[pos])

                # Run with this token zeroed
                ablated_output = patcher_6.run_with_N_tokens_patched(
                    problem_text=question,
                    patch_activations=patched_acts,
                    layer_name=test_layer,
                    max_new_tokens=200
                )

                ablated_pred = extract_answer_number(ablated_output)
                ablated_correct = answers_match(ablated_pred, expected_answer)

                # Importance: did ablating this token cause failure?
                importance = baseline_correct and not ablated_correct

                problem_result['token_ablations'].append({
                    'token_pos': token_pos,
                    'ablated_output': ablated_output,
                    'predicted_answer': ablated_pred,
                    'still_correct': ablated_correct,
                    'importance': 1 if importance else 0
                })

                pbar.update(1)
                pbar.set_postfix({
                    'problem': problem_id,
                    'token': token_pos,
                    'important': importance
                })

            # Compute overall importance scores
            importance_scores = [abl['importance'] for abl in problem_result['token_ablations']]
            problem_result['importance_scores'] = importance_scores
            problem_result['num_critical_tokens'] = sum(importance_scores)

            results.append(problem_result)

        except Exception as e:
            print(f"\nError on problem {problem_id}: {e}")
            import traceback
            traceback.print_exc()

            problem_result['error'] = str(e)
            results.append(problem_result)

            # Skip remaining ablations for this problem
            pbar.update(7 - len(problem_result.get('token_ablations', [])))

    pbar.close()

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 80)

    successful = [r for r in results if 'error' not in r]
    print(f"Problems processed: {len(successful)}/{len(dataset)}")

    baseline_correct = sum(1 for r in successful if r['baseline']['correct'])
    print(f"Baseline correct: {baseline_correct}/{len(successful)} ({100*baseline_correct/len(successful):.1f}%)")

    # Token-level statistics
    token_importance = [0] * 6
    for result in successful:
        if result['baseline']['correct']:  # Only count for problems baseline got right
            for token_pos in range(6):
                token_importance[token_pos] += result['token_ablations'][token_pos]['importance']

    print(f"\nToken Importance (out of {baseline_correct} baseline-correct problems):")
    for pos in range(6):
        print(f"  Token {pos}: {token_importance[pos]} critical ({100*token_importance[pos]/max(baseline_correct,1):.1f}%)")

    print(f"\n✓ Results saved to {output_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true', help='Run on 10-problem test set')
    args = parser.parse_args()

    run_token_ablation_experiment(test_mode=args.test_mode)
