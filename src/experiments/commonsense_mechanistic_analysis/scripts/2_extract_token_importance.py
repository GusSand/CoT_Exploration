#!/usr/bin/env python3
"""
Story 3: Token Importance Analysis (CCTA)

Measures importance of each continuous thought token by ablating it
and measuring the impact on accuracy.

Usage:
    python 2_extract_token_importance.py --n_samples 1221
"""
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
from datasets import load_dataset

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'scripts' / 'experiments'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG
from run_ablation_N_tokens_llama import NTokenPatcher


def extract_answer_letter(output_text):
    """Extract answer letter (A-E) from model output."""
    output_text = output_text.strip().upper()

    if "THE ANSWER IS:" in output_text:
        parts = output_text.split("THE ANSWER IS:")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            for char in answer_part:
                if char in ['A', 'B', 'C', 'D', 'E']:
                    return char

    for char in output_text:
        if char in ['A', 'B', 'C', 'D', 'E']:
            return char

    return "INVALID"


def format_commonsense_question(example):
    """Format CommonsenseQA example."""
    question = example['question']
    choices = example['choices']

    formatted = f"Question: {question}\nChoices:\n"
    for label, text in zip(choices['label'], choices['text']):
        formatted += f"{label}: {text}\n"

    return formatted.strip()


def extract_token_importance(n_samples=1221):
    """
    Measure importance of each continuous thought token via ablation.

    For each problem:
    1. Run baseline (all 6 tokens active)
    2. Ablate each token individually (zero out)
    3. Measure accuracy drop

    CCTA score = baseline_correct - ablated_correct
    """
    print("=" * 80)
    print("TOKEN IMPORTANCE ANALYSIS (CCTA)")
    print("=" * 80)

    # Load model
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_commonsense' / 'gsm8k_llama1b_latent_baseline' / 'Llama-3.2-1B-Instruct' / 'ep_3' / 'lr_0.0008' / 'seed_11')
    print(f"\nLoading CommonsenseQA CODI model from {model_path}...")

    cacher = ActivationCacherLLaMA(model_path)

    # Create patcher for ablation
    patcher = NTokenPatcher(cacher, num_tokens=6)

    # Load dataset
    print(f"\nLoading CommonsenseQA validation dataset...")
    dataset = load_dataset('tau/commonsense_qa', split='validation')

    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    print(f"Processing {len(dataset)} examples")
    print(f"Ablation strategy: Zero out each token individually\n")

    results = []
    baseline_correct = 0
    token_ablation_correct = {i: 0 for i in range(6)}

    for idx, example in enumerate(tqdm(dataset, desc="Running ablations")):
        try:
            formatted_q = format_commonsense_question(example)
            expected_answer = example['answerKey']

            # 1. Baseline (all tokens active)
            baseline_output = patcher._generate_with_patching(formatted_q, max_new_tokens=200)
            baseline_pred = extract_answer_letter(baseline_output)
            baseline_is_correct = (baseline_pred == expected_answer)
            baseline_correct += baseline_is_correct

            # 2. Ablate each token
            token_ablation_results = {}

            for token_idx in range(6):
                # Cache all 6 token activations at middle layer
                all_acts = patcher.cache_N_token_activations(formatted_q, 'middle')

                # Zero out the target token
                ablated_acts = []
                for i in range(6):
                    if i == token_idx:
                        ablated_acts.append(torch.zeros_like(all_acts[i]))
                    else:
                        ablated_acts.append(all_acts[i])

                # Run with ablated token
                ablated_output = patcher.run_with_N_tokens_patched(
                    problem_text=formatted_q,
                    patch_activations=ablated_acts,
                    layer_name='middle',
                    max_new_tokens=200
                )

                ablated_pred = extract_answer_letter(ablated_output)
                ablated_is_correct = (ablated_pred == expected_answer)
                token_ablation_correct[token_idx] += ablated_is_correct

                # CCTA: impact of ablating this token (1 if baseline correct but ablation wrong, 0 otherwise)
                ccta_impact = 1 if (baseline_is_correct and not ablated_is_correct) else 0

                token_ablation_results[f'token_{token_idx}'] = {
                    'predicted': ablated_pred,
                    'correct': ablated_is_correct,
                    'ccta_impact': ccta_impact
                }

            results.append({
                'id': example['id'],
                'question_concept': example['question_concept'],
                'expected_answer': expected_answer,
                'baseline': {
                    'predicted': baseline_pred,
                    'correct': baseline_is_correct
                },
                'ablations': token_ablation_results
            })

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'id': example.get('id', f'error_{idx}'),
                'error': str(e)
            })

    # Compute aggregate statistics
    n_valid = len([r for r in results if 'error' not in r])
    baseline_accuracy = baseline_correct / n_valid if n_valid > 0 else 0

    token_importance = {}
    for token_idx in range(6):
        ablated_accuracy = token_ablation_correct[token_idx] / n_valid if n_valid > 0 else 0
        accuracy_drop = baseline_accuracy - ablated_accuracy

        # Count how many problems broke when this token was ablated
        ccta_breaks = sum(1 for r in results if 'ablations' in r and
                         r['ablations'][f'token_{token_idx}']['ccta_impact'] == 1)

        token_importance[f'token_{token_idx}'] = {
            'ablated_accuracy': ablated_accuracy,
            'accuracy_drop': accuracy_drop,
            'ccta_breaks': ccta_breaks,
            'ccta_rate': ccta_breaks / n_valid if n_valid > 0 else 0
        }

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_dir / 'commonsense_token_importance_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary = {
        'n_examples': n_valid,
        'baseline_accuracy': baseline_accuracy,
        'token_importance': token_importance,
        'ranking': sorted(
            [{'token': k, **v} for k, v in token_importance.items()],
            key=lambda x: x['accuracy_drop'],
            reverse=True
        )
    }

    with open(output_dir / 'commonsense_token_importance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("TOKEN IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Baseline accuracy: {baseline_accuracy:.1%} ({baseline_correct}/{n_valid})")
    print(f"\nToken Importance Ranking (by accuracy drop):")
    for rank, item in enumerate(summary['ranking'], 1):
        print(f"  {rank}. {item['token']}: "
              f"Δ{item['accuracy_drop']:.1%} "
              f"(ablated acc: {item['ablated_accuracy']:.1%}, "
              f"breaks: {item['ccta_breaks']})")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - commonsense_token_importance_detailed.json")
    print(f"  - commonsense_token_importance_summary.json")

    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1221, help='Number of samples to process')
    args = parser.parse_args()

    extract_token_importance(args.n_samples)
