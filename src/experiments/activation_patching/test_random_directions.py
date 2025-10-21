#!/usr/bin/env python3
"""
Random Direction Control Experiment

Critical validation: Does suppression work because the direction is meaningful,
or would ANY perturbation (random noise) also degrade performance?

This experiment:
1. Loads 5 pre-generated random directions (same magnitude as computed direction)
2. Tests each with α=-3.0 (same as worst suppression)
3. Compares degradation to computed direction (-12.8 points)

Expected outcomes:
- If random → similar degradation (-12.8 pts): Suppression is just noise ❌
- If random → less degradation (e.g., -5 pts): Suppression is meaningful ✓
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))

from run_steering_experiment import SteeringInference, extract_answer_number, answers_match


def load_test_set():
    """Load test problems."""
    dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_gpt2.json'
    problem_pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'

    with open(dataset_file) as f:
        dataset = json.load(f)

    with open(problem_pairs_file) as f:
        all_pairs = json.load(f)

    pairs_lookup = {p['pair_id']: p for p in all_pairs}

    test_correct = dataset['test_correct']
    test_wrong = dataset['test_wrong']

    for prob in test_correct + test_wrong:
        pair = pairs_lookup[prob['pair_id']]
        prob['question'] = pair['clean']['question']

    return test_correct + test_wrong


def load_directions():
    """Load computed and random directions."""
    # Computed direction
    computed_file = Path(__file__).parent / 'results' / 'steering_activations' / 'reasoning_direction.npz'
    computed_data = np.load(computed_file)
    computed_direction = computed_data['direction']

    # Random directions
    random_file = Path(__file__).parent / 'results' / 'steering_analysis' / 'random_directions.npz'
    random_data = np.load(random_file)
    random_directions = random_data['directions']

    return computed_direction, random_directions


def test_direction(engine, test_problems, direction, direction_name, alpha=-3.0):
    """Test a single direction."""
    print(f"\n{'='*80}")
    print(f"Testing: {direction_name} (α={alpha:+.1f})")
    print(f"{'='*80}")

    correct_count = 0
    results = []

    for prob in tqdm(test_problems, desc=f"{direction_name}"):
        try:
            output = engine.run_with_steering(
                problem_text=prob['question'],
                steering_direction=direction,
                alpha=alpha,
                max_new_tokens=200
            )

            predicted = extract_answer_number(output)
            expected = prob['expected']
            correct = answers_match(predicted, expected)

            if correct:
                correct_count += 1

            results.append({
                'pair_id': prob['pair_id'],
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

    accuracy = 100 * correct_count / len(test_problems)
    print(f"\n{direction_name}: {correct_count}/{len(test_problems)} correct ({accuracy:.1f}%)")

    return results, accuracy


def main():
    """Main experiment pipeline."""
    print("="*80)
    print("RANDOM DIRECTION CONTROL EXPERIMENT")
    print("="*80)
    print("\nThis experiment validates whether suppression is meaningful.")
    print("If random directions cause similar degradation, suppression is just noise.")

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    test_problems = load_test_set()
    computed_direction, random_directions = load_directions()

    print(f"✓ Loaded {len(test_problems)} test problems")
    print(f"✓ Computed direction shape: {computed_direction.shape}")
    print(f"✓ Loaded {len(random_directions)} random directions")

    # Initialize engine
    model_path = str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    engine = SteeringInference(model_path)

    # Load baseline results for comparison
    summary_file = Path(__file__).parent / 'results' / 'steering_experiments' / 'steering_results_summary.json'
    with open(summary_file) as f:
        summary = json.load(f)

    baseline_acc = summary['summary']['0.0']['accuracy']
    computed_suppression_acc = summary['summary']['-3.0']['accuracy']
    computed_degradation = baseline_acc - computed_suppression_acc

    print(f"\n--- REFERENCE VALUES ---")
    print(f"Baseline (α=0.0):                {baseline_acc:.1f}%")
    print(f"Computed direction (α=-3.0):     {computed_suppression_acc:.1f}%")
    print(f"Degradation:                     {computed_degradation:.1f} points")

    # Test random directions
    all_results = {
        'baseline_accuracy': baseline_acc,
        'computed_suppression_accuracy': computed_suppression_acc,
        'computed_degradation': computed_degradation,
        'random_results': []
    }

    random_accuracies = []

    for i, random_dir in enumerate(random_directions):
        results, accuracy = test_direction(
            engine, test_problems, random_dir,
            f"Random Direction {i+1}",
            alpha=-3.0
        )

        random_accuracies.append(accuracy)
        degradation = baseline_acc - accuracy

        all_results['random_results'].append({
            'direction_id': i,
            'accuracy': accuracy,
            'degradation': degradation,
            'results': results
        })

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    mean_random_acc = np.mean(random_accuracies)
    std_random_acc = np.std(random_accuracies)
    mean_random_degradation = baseline_acc - mean_random_acc

    print(f"\nRandom directions (α=-3.0):")
    for i, acc in enumerate(random_accuracies):
        degradation = baseline_acc - acc
        print(f"  Random {i+1}: {acc:.1f}% (degradation: {degradation:.1f} points)")

    print(f"\nMean random accuracy:     {mean_random_acc:.1f}% ± {std_random_acc:.1f}%")
    print(f"Mean random degradation:  {mean_random_degradation:.1f} points")

    print(f"\n--- COMPARISON ---")
    print(f"Computed degradation:     {computed_degradation:.1f} points")
    print(f"Random degradation:       {mean_random_degradation:.1f} points")
    print(f"Difference:               {computed_degradation - mean_random_degradation:.1f} points")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    # If computed degradation is significantly more than random, direction is meaningful
    threshold = 2.0  # At least 2 points more degradation than random

    if computed_degradation > mean_random_degradation + threshold:
        verdict = "MEANINGFUL ✓"
        interpretation = (
            f"The computed direction causes {computed_degradation:.1f} points degradation, "
            f"while random directions only cause {mean_random_degradation:.1f} points. "
            f"This {computed_degradation - mean_random_degradation:.1f} point difference suggests "
            "the direction captures something specific about reasoning."
        )
    else:
        verdict = "LIKELY NOISE ❌"
        interpretation = (
            f"The computed direction causes {computed_degradation:.1f} points degradation, "
            f"similar to random directions ({mean_random_degradation:.1f} points). "
            "This suggests suppression may be due to general perturbation rather than "
            "a meaningful reasoning direction."
        )

    print(f"\n{verdict}")
    print(f"\n{interpretation}")

    # Save results
    output_dir = Path(__file__).parent / 'results' / 'steering_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'random_direction_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'baseline_accuracy': baseline_acc,
            'computed_suppression': {
                'accuracy': computed_suppression_acc,
                'degradation': computed_degradation
            },
            'random_suppression': {
                'accuracies': random_accuracies,
                'mean_accuracy': mean_random_acc,
                'std_accuracy': std_random_acc,
                'mean_degradation': mean_random_degradation
            },
            'verdict': verdict,
            'interpretation': interpretation
        }, f, indent=2)

    print(f"\n✓ Saved results: {results_file}")

    print("\n" + "="*80)
    print("✅ RANDOM DIRECTION CONTROL COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
