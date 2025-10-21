#!/usr/bin/env python3
"""
Prepare LLaMA Steering Dataset

This script:
1. Runs LLaMA baseline inference on all 532 GPT-4 calculated pairs
2. Runs CoT necessity testing (ablate all 6 tokens) on all 532 pairs
3. Filters to pairs where LLaMA needs CoT (for clean OR corrupted)
4. Splits into correct/wrong based on clean answer
5. Creates balanced training/test split

Expected outcome: 100+ CoT-dependent pairs (more than 44 from previous work)
since we no longer require both clean AND corrupted to be correct.
"""

import json
import sys
import re
import torch
from pathlib import Path
from tqdm import tqdm

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))

from cache_activations import ActivationCacher, LAYER_CONFIG


def extract_answer_number(text: str):
    """Extract numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
        r'The answer is:\s*(-?\d+(?:\.\d+)?)',
        r'\$?\s*(-?\d+(?:\.\d+)?)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                return int(num) if num.is_integer() else num
            except ValueError:
                continue

    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            num = float(numbers[-1])
            return int(num) if num.is_integer() else num
        except ValueError:
            pass

    return None


def answers_match(predicted, expected):
    """Check if predicted matches expected."""
    if predicted is None or expected is None:
        return False

    try:
        pred_float = float(predicted)
        exp_float = float(expected)
        return abs(pred_float - exp_float) < 0.01
    except (ValueError, TypeError):
        return False


def run_baseline_inference(cacher, problems):
    """Run baseline inference on all problems."""
    print("="*80)
    print("RUNNING LLAMA BASELINE INFERENCE")
    print("="*80)
    print(f"\nProcessing {len(problems)} problems...\n")

    results = []

    for prob in tqdm(problems, desc="Baseline"):
        try:
            # Generate with normal CoT
            output = cacher.generate_with_cot(prob['clean']['question'])
            predicted = extract_answer_number(output)
            expected = prob['clean']['answer']
            correct = answers_match(predicted, expected)

            results.append({
                'pair_id': prob['pair_id'],
                'question': prob['clean']['question'],
                'expected': expected,
                'predicted': predicted,
                'correct': correct,
                'output': output
            })

        except Exception as e:
            print(f"\nError on pair {prob['pair_id']}: {e}")
            results.append({
                'pair_id': prob['pair_id'],
                'error': str(e)
            })

    correct_count = sum(1 for r in results if r.get('correct', False))
    print(f"\nBaseline: {correct_count}/{len(results)} correct ({100*correct_count/len(results):.1f}%)")

    return results


def run_cot_necessity_test(cacher, problems):
    """Test if LLaMA needs CoT for each problem."""
    print("\n" + "="*80)
    print("RUNNING COT NECESSITY TEST")
    print("="*80)
    print(f"\nTesting {len(problems)} problems (clean + corrupted)...\n")

    results = []
    num_latent = cacher.num_latent

    for prob in tqdm(problems, desc="CoT Test"):
        pair_id = prob['pair_id']

        try:
            # Test CLEAN version
            clean_question = prob['clean']['question']
            clean_expected = prob['clean']['answer']

            # Baseline (with CoT)
            baseline_output = cacher.generate_with_cot(clean_question)
            baseline_pred = extract_answer_number(baseline_output)
            baseline_correct = answers_match(baseline_pred, clean_expected)

            # Ablated (no CoT - all zeros)
            ablated_output = cacher.generate_with_ablation(
                clean_question,
                positions_to_ablate=list(range(num_latent)),
                layer_name='middle'
            )
            ablated_pred = extract_answer_number(ablated_output)
            ablated_correct = answers_match(ablated_pred, clean_expected)

            # Needs CoT if: baseline correct AND ablated wrong
            clean_needs_cot = baseline_correct and not ablated_correct

            # Test CORRUPTED version
            corrupted_question = prob['corrupted']['question']
            corrupted_expected = prob['clean']['answer']  # Still expect clean answer

            # Baseline (with CoT)
            baseline_output_corr = cacher.generate_with_cot(corrupted_question)
            baseline_pred_corr = extract_answer_number(baseline_output_corr)
            baseline_correct_corr = answers_match(baseline_pred_corr, corrupted_expected)

            # Ablated (no CoT)
            ablated_output_corr = cacher.generate_with_ablation(
                corrupted_question,
                positions_to_ablate=list(range(num_latent)),
                layer_name='middle'
            )
            ablated_pred_corr = extract_answer_number(ablated_output_corr)
            ablated_correct_corr = answers_match(ablated_pred_corr, corrupted_expected)

            # Needs CoT if: baseline correct AND ablated wrong
            corrupted_needs_cot = baseline_correct_corr and not ablated_correct_corr

            # Needs CoT if EITHER clean or corrupted needs it
            needs_cot = clean_needs_cot or corrupted_needs_cot

            results.append({
                'pair_id': pair_id,
                'clean_baseline_correct': baseline_correct,
                'clean_ablated_correct': ablated_correct,
                'clean_needs_cot': clean_needs_cot,
                'corrupted_baseline_correct': baseline_correct_corr,
                'corrupted_ablated_correct': ablated_correct_corr,
                'corrupted_needs_cot': corrupted_needs_cot,
                'needs_cot_either': needs_cot
            })

        except Exception as e:
            print(f"\nError on pair {pair_id}: {e}")
            results.append({
                'pair_id': pair_id,
                'error': str(e)
            })

    # Summary
    needs_cot_clean = sum(1 for r in results if r.get('clean_needs_cot', False))
    needs_cot_corrupted = sum(1 for r in results if r.get('corrupted_needs_cot', False))
    needs_cot_either = sum(1 for r in results if r.get('needs_cot_either', False))

    print("\n" + "="*80)
    print("COT NECESSITY SUMMARY")
    print("="*80)
    print(f"Needs CoT for CLEAN:      {needs_cot_clean}/{len(results)} ({100*needs_cot_clean/len(results):.1f}%)")
    print(f"Needs CoT for CORRUPTED:  {needs_cot_corrupted}/{len(results)} ({100*needs_cot_corrupted/len(results):.1f}%)")
    print(f"Needs CoT for EITHER:     {needs_cot_either}/{len(results)} ({100*needs_cot_either/len(results):.1f}%)")

    return results


def create_steering_dataset(baseline_results, cot_necessity_results):
    """Create balanced training/test split."""
    print("\n" + "="*80)
    print("CREATING STEERING DATASET")
    print("="*80)

    # Create lookups
    baseline_lookup = {r['pair_id']: r for r in baseline_results if 'pair_id' in r}
    cot_lookup = {r['pair_id']: r for r in cot_necessity_results if 'pair_id' in r}

    # Filter to CoT-dependent pairs
    cot_dependent_pairs = []
    for pair_id, cot_result in cot_lookup.items():
        if cot_result.get('needs_cot_either', False):
            baseline_result = baseline_lookup.get(pair_id)
            if baseline_result and 'correct' in baseline_result:
                cot_dependent_pairs.append({
                    'pair_id': pair_id,
                    'correct': baseline_result['correct'],
                    'expected': baseline_result['expected']
                })

    print(f"\nCoT-dependent pairs: {len(cot_dependent_pairs)}")

    # Split by correctness
    correct_problems = [p for p in cot_dependent_pairs if p['correct']]
    wrong_problems = [p for p in cot_dependent_pairs if not p['correct']]

    print(f"  Correct: {len(correct_problems)}")
    print(f"  Wrong:   {len(wrong_problems)}")

    # Balance
    import random
    random.seed(42)

    n = min(len(correct_problems), len(wrong_problems))
    print(f"\nBalancing to n={n} each")

    random.shuffle(correct_problems)
    random.shuffle(wrong_problems)

    balanced_correct = correct_problems[:n]
    balanced_wrong = wrong_problems[:n]

    # Split 80/20
    train_split = int(0.8 * n)

    train_correct = balanced_correct[:train_split]
    test_correct = balanced_correct[train_split:]
    train_wrong = balanced_wrong[:train_split]
    test_wrong = balanced_wrong[train_split:]

    print(f"\nTraining set: {len(train_correct)} correct + {len(train_wrong)} wrong = {len(train_correct) + len(train_wrong)}")
    print(f"Test set:     {len(test_correct)} correct + {len(test_wrong)} wrong = {len(test_correct) + len(test_wrong)}")

    return {
        'train_correct': train_correct,
        'train_wrong': train_wrong,
        'test_correct': test_correct,
        'test_wrong': test_wrong,
        'stats': {
            'total_pairs': len(cot_dependent_pairs),
            'correct': len(correct_problems),
            'wrong': len(wrong_problems),
            'balanced_n': n,
            'train_size': len(train_correct) + len(train_wrong),
            'test_size': len(test_correct) + len(test_wrong)
        }
    }


def main():
    """Main pipeline."""
    print("="*80)
    print("LLAMA STEERING DATASET PREPARATION")
    print("="*80)

    # Load 532 GPT-4 calculated pairs
    problem_pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'
    with open(problem_pairs_file) as f:
        all_problems = json.load(f)

    print(f"\nLoaded {len(all_problems)} GPT-4 calculated pairs")

    # Initialize LLaMA
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    cacher = ActivationCacher(model_path, device='cuda')

    # Step 1: Baseline inference
    baseline_results = run_baseline_inference(cacher, all_problems)

    # Save baseline results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_file = output_dir / 'llama_baseline_532.json'
    with open(baseline_file, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\n✓ Saved baseline results: {baseline_file}")

    # Step 2: CoT necessity test
    cot_necessity_results = run_cot_necessity_test(cacher, all_problems)

    # Save CoT necessity results
    cot_file = output_dir / 'llama_cot_necessity_532.json'
    with open(cot_file, 'w') as f:
        json.dump(cot_necessity_results, f, indent=2)
    print(f"\n✓ Saved CoT necessity results: {cot_file}")

    # Step 3: Create steering dataset
    steering_dataset = create_steering_dataset(baseline_results, cot_necessity_results)

    # Save steering dataset
    dataset_file = output_dir / 'steering_dataset_llama.json'
    with open(dataset_file, 'w') as f:
        json.dump(steering_dataset, f, indent=2)
    print(f"\n✓ Saved steering dataset: {dataset_file}")

    print("\n" + "="*80)
    print("✅ DATASET PREPARATION COMPLETE")
    print("="*80)
    print(f"\nFinal dataset size:")
    print(f"  Training: {steering_dataset['stats']['train_size']} problems")
    print(f"  Test:     {steering_dataset['stats']['test_size']} problems")
    print(f"  Total:    {steering_dataset['stats']['balanced_n'] * 2} balanced CoT-dependent pairs")


if __name__ == "__main__":
    main()
