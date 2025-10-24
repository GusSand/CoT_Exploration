"""
Run Full Operation Intervention Experiment

Tests causal role of Token 1 L8 in operation-specific circuits by running
interventions on 60 problems across 6 conditions.

Conditions:
1. Baseline (no intervention)
2. Swap to addition mean
3. Swap to multiplication mean
4. Swap to mixed mean
5. Random vector control
6. Token 5 L8 control (wrong token)
"""

import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from run_intervention import OperationIntervener


def run_experiment(
    test_set_path: str,
    vectors_path: str,
    model_path: str,
    output_path: str
):
    """Run full experiment on test set."""

    # Load data
    print("Loading test set and vectors...")
    test_set = json.load(open(test_set_path))
    vectors = json.load(open(vectors_path))

    # Initialize intervener
    print(f"\nInitializing model from {model_path}...")
    intervener = OperationIntervener(model_path)

    # Prepare intervention vectors
    add_mean = torch.tensor(vectors['operation_means']['pure_addition'])
    mult_mean = torch.tensor(vectors['operation_means']['pure_multiplication'])
    mixed_mean = torch.tensor(vectors['operation_means']['mixed'])
    random_vec = torch.tensor(vectors['random_control'])

    # Define conditions
    conditions = [
        {'name': 'baseline', 'vector': None, 'token': 1, 'layer': 'middle'},
        {'name': 'to_addition', 'vector': add_mean, 'token': 1, 'layer': 'middle'},
        {'name': 'to_multiplication', 'vector': mult_mean, 'token': 1, 'layer': 'middle'},
        {'name': 'to_mixed', 'vector': mixed_mean, 'token': 1, 'layer': 'middle'},
        {'name': 'random_control', 'vector': random_vec, 'token': 1, 'layer': 'middle'},
        {'name': 'wrong_token_control', 'vector': mult_mean, 'token': 5, 'layer': 'middle'},
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

            print(f"\n[{current}/{total}] Problem {prob_idx+1}/{len(test_set)}, Condition: {cond_name}")

            # Run intervention
            answer_text = intervener.run_with_intervention(
                problem['question'],
                intervention_vector=condition['vector'],
                intervention_token=condition['token'],
                intervention_layer=condition['layer']
            )

            # Extract answer
            predicted_answer = intervener.extract_answer_number(answer_text)

            # Check correctness
            is_correct = (str(predicted_answer) == str(prob_results['correct_answer']))

            prob_results['conditions'][cond_name] = {
                'answer_text': answer_text,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct
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
        print(f"{cond_name}: {correct_count}/{total_count} = {accuracy:.1f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='~/codi_ckpt/llama_gsm8k')
    parser.add_argument('--test_set', default='test_set_60.json')
    parser.add_argument('--vectors', default='activation_vectors.json')
    parser.add_argument('--output', default='intervention_results.json')
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    test_set_path = base_dir / args.test_set
    vectors_path = base_dir / args.vectors
    output_path = base_dir / args.output

    run_experiment(
        str(test_set_path),
        str(vectors_path),
        args.model_path,
        str(output_path)
    )


if __name__ == '__main__':
    main()
