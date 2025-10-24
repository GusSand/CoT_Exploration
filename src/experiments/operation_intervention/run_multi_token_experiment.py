"""
Run Multi-Token Intervention Experiment

Tests whether disrupting BOTH Token 1 (planning) and Token 5 (execution) produces
larger causal effects than single-token interventions.

Hypothesis: Single-token interventions failed because the model compensates through
other tokens. Multi-token intervention should disrupt both planning AND execution.

Conditions:
1. Baseline (no intervention)
2. Token 1 only @ L8 (swap to multiplication mean)
3. Token 5 only @ L14 (swap to multiplication mean)
4. Token 1 + Token 5 @ L8 + L14 (both swap to multiplication)
5. Random Token 1 control
6. Random Token 5 control
7. Multi random (both tokens random)
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from run_intervention import OperationIntervener


def run_multi_token_experiment(
    test_set_path: str,
    token1_vectors_path: str,
    token5_vectors_path: str,
    model_path: str,
    output_path: str
):
    """Run multi-token intervention experiment."""

    # Load data
    print("Loading test set and activation vectors...")
    test_set = json.load(open(test_set_path))
    token1_vectors = json.load(open(token1_vectors_path))
    token5_vectors = json.load(open(token5_vectors_path))

    # Initialize intervener
    print(f"\nInitializing model from {model_path}...")
    intervener = OperationIntervener(model_path)

    # Prepare intervention vectors
    # Token 1 @ Layer 8 (middle)
    t1_mult_mean = torch.tensor(token1_vectors['operation_means']['pure_multiplication'])
    t1_random = torch.tensor(token1_vectors['random_control'])

    # Token 5 @ Layer 14 (late)
    t5_mult_mean = torch.tensor(token5_vectors['operation_means']['pure_multiplication'])
    t5_random = torch.tensor(token5_vectors['random_control'])

    # Define conditions
    conditions = [
        # Baseline
        {
            'name': 'baseline',
            'type': 'baseline',
            'interventions': []
        },

        # Single token interventions
        {
            'name': 'token1_only',
            'type': 'single',
            'interventions': [
                {'vector': t1_mult_mean, 'token': 1, 'layer': 'middle'}
            ]
        },
        {
            'name': 'token5_only',
            'type': 'single',
            'interventions': [
                {'vector': t5_mult_mean, 'token': 5, 'layer': 'late'}
            ]
        },

        # Multi-token intervention (main hypothesis)
        {
            'name': 'multi_token',
            'type': 'multi',
            'interventions': [
                {'vector': t1_mult_mean, 'token': 1, 'layer': 'middle'},
                {'vector': t5_mult_mean, 'token': 5, 'layer': 'late'}
            ]
        },

        # Control conditions
        {
            'name': 'token1_random',
            'type': 'control',
            'interventions': [
                {'vector': t1_random, 'token': 1, 'layer': 'middle'}
            ]
        },
        {
            'name': 'token5_random',
            'type': 'control',
            'interventions': [
                {'vector': t5_random, 'token': 5, 'layer': 'late'}
            ]
        },
        {
            'name': 'multi_random',
            'type': 'control',
            'interventions': [
                {'vector': t1_random, 'token': 1, 'layer': 'middle'},
                {'vector': t5_random, 'token': 5, 'layer': 'late'}
            ]
        }
    ]

    print(f"\nRunning experiment:")
    print(f"  Test problems: {len(test_set)}")
    print(f"  Conditions: {len(conditions)}")
    print(f"  Total inferences: {len(test_set) * len(conditions)}")

    # Run experiments
    results = []
    total = len(test_set) * len(conditions)
    current = 0

    for prob_idx, problem in enumerate(test_set):
        prob_results = {
            'problem_id': problem['id'],
            'question': problem['question'],
            'correct_answer': intervener.extract_answer_number(problem['answer']),
            'operation_type': problem['operation_type'],
            'conditions': {}
        }

        for condition in conditions:
            current += 1
            cond_name = condition['name']
            cond_type = condition['type']

            print(f"\n[{current}/{total}] Problem {prob_idx+1}/{len(test_set)}, Condition: {cond_name}")

            # Run intervention based on type
            if cond_type == 'baseline':
                # No intervention
                answer_text = intervener.run_with_intervention(
                    problem['question'],
                    intervention_vector=None
                )
            elif len(condition['interventions']) == 1:
                # Single-token intervention
                interv = condition['interventions'][0]
                answer_text = intervener.run_with_intervention(
                    problem['question'],
                    intervention_vector=interv['vector'],
                    intervention_token=interv['token'],
                    intervention_layer=interv['layer']
                )
            else:
                # Multi-token intervention
                answer_text = intervener.run_with_multi_intervention(
                    problem['question'],
                    interventions=condition['interventions']
                )

            # Extract answer
            predicted_answer = intervener.extract_answer_number(answer_text)

            # Check correctness
            is_correct = (str(predicted_answer) == str(prob_results['correct_answer']))

            prob_results['conditions'][cond_name] = {
                'answer_text': answer_text,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'type': cond_type
            }

            print(f"  Correct: {prob_results['correct_answer']}, Predicted: {predicted_answer}, Match: {is_correct}")

        results.append(prob_results)

        # Save checkpoint every 10 problems
        if (prob_idx + 1) % 10 == 0:
            checkpoint_path = output_path.replace('.json', f'_checkpoint_{prob_idx+1}.json')
            json.dump(results, open(checkpoint_path, 'w'), indent=2)
            print(f"\nCheckpoint saved: {checkpoint_path}")

    # Save final results
    json.dump(results, open(output_path, 'w'), indent=2)
    print(f"\n✅ Experiment complete! Results saved to: {output_path}")

    # Quick summary
    print("\n=== Quick Summary ===")
    for cond in conditions:
        cond_name = cond['name']
        correct_count = sum(1 for r in results if r['conditions'][cond_name]['is_correct'])
        total_count = len(results)
        accuracy = 100 * correct_count / total_count

        # Count answer changes from baseline
        changes = 0
        for r in results:
            baseline_ans = r['conditions']['baseline']['predicted_answer']
            cond_ans = r['conditions'][cond_name]['predicted_answer']
            if str(baseline_ans) != str(cond_ans):
                changes += 1

        change_rate = 100 * changes / total_count

        print(f"{cond_name:20s}: {correct_count:2d}/{total_count} = {accuracy:5.1f}% | Changes: {changes:2d} ({change_rate:5.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='~/codi_ckpt/llama_gsm8k')
    parser.add_argument('--test_set', default='test_set_60.json')
    parser.add_argument('--token1_vectors', default='activation_vectors.json')
    parser.add_argument('--token5_vectors', default='token5_activation_vectors.json')
    parser.add_argument('--output', default='multi_token_results.json')
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    test_set_path = base_dir / args.test_set
    token1_vectors_path = base_dir / args.token1_vectors
    token5_vectors_path = base_dir / args.token5_vectors
    output_path = base_dir / args.output

    run_multi_token_experiment(
        str(test_set_path),
        str(token1_vectors_path),
        str(token5_vectors_path),
        args.model_path,
        str(output_path)
    )


if __name__ == '__main__':
    main()
