#!/usr/bin/env python3
"""
Story 1.4: Run Pilot Resampling Experiment

Run resampling experiment across all CT positions.

Architecture:
- For each CT position (0-5)
- For each problem A
- Sample N random different problems B
- Swap CT position, generate answer
- Measure accuracy impact

Time estimate: 2.5 hours (2h coding + 30m runtime for pilot)

Usage:
    python 3_run_resampling.py --phase pilot --n_samples 5
    python 3_run_resampling.py --phase full --n_samples 10
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import pickle
import json
import argparse
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List
import random
import sys

# Add utils and swapping to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed, load_model, extract_answer

# Import swapping function from script 2
import importlib.util
spec = importlib.util.spec_from_file_location("swapping", Path(__file__).parent / "2_implement_swapping.py")
swapping_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swapping_module)
generate_with_swapped_ct = swapping_module.generate_with_swapped_ct


def run_resampling_experiment(
    model,
    tokenizer,
    cache: Dict,
    n_samples_per_problem: int = 5,
    device: str = 'cuda'
) -> Dict:
    """
    Run full resampling experiment across all CT positions.

    Architecture:
    - 6 positions (CT0-CT5)
    - N problems (20 for pilot, 100 for full)
    - M samples per problem (5 for pilot, 10 for full)
    - Total: 6 × N × M generations

    Args:
        model: CODI model
        tokenizer: Model tokenizer
        cache: Loaded cache dict
        n_samples_per_problem: Number of different problems to swap from
        device: Device

    Returns:
        Dict with results for each position
    """
    n_problems = len(cache)
    total_generations = 6 * n_problems * n_samples_per_problem

    print(f"\nResampling Experiment Configuration:")
    print(f"  Positions: 6 (CT0-CT5)")
    print(f"  Problems: {n_problems}")
    print(f"  Samples per problem: {n_samples_per_problem}")
    print(f"  Total generations: {total_generations}")
    print(f"  Estimated time: ~{total_generations * 3 / 60:.1f} minutes\n")

    results = {}

    # Run experiment for each CT position
    for position in range(6):
        print(f"\n{'='*60}")
        print(f"CT{position} Resampling")
        print(f"{'='*60}\n")

        position_results = []

        # For each problem A
        pbar = tqdm(range(n_problems), desc=f"CT{position}")
        for problem_a_idx in pbar:
            problem_A = cache[problem_a_idx]

            # Sample N different problems B
            other_indices = [i for i in range(n_problems) if i != problem_a_idx]
            problem_b_indices = random.sample(other_indices, min(n_samples_per_problem, len(other_indices)))

            swapped_results = []

            # Swap with each problem B
            for problem_b_idx in problem_b_indices:
                problem_B = cache[problem_b_idx]

                # Generate with swapped CT
                answer = generate_with_swapped_ct(
                    model, tokenizer, problem_A, problem_B,
                    swap_position=position, device=device
                )

                # Check correctness
                predicted_numeric = extract_answer(answer)
                is_correct = (predicted_numeric == problem_A['gold_numeric'])

                swapped_results.append({
                    'problem_b_idx': problem_b_idx,
                    'generated_answer': answer,
                    'predicted_numeric': predicted_numeric,
                    'correct': is_correct
                })

            # Calculate accuracy for this problem
            n_correct = sum(1 for r in swapped_results if r['correct'])
            accuracy = n_correct / len(swapped_results)

            position_results.append({
                'problem_a_idx': problem_a_idx,
                'baseline_correct': problem_A['baseline_correct'],
                'swapped_correct': [r['correct'] for r in swapped_results],
                'accuracy': accuracy
            })

            # Update progress bar with running accuracy
            running_acc = np.mean([r['accuracy'] for r in position_results])
            pbar.set_postfix({'acc': f"{running_acc:.2%}"})

        # Calculate position-level metrics
        baseline_accuracy = np.mean([r['baseline_correct'] for r in position_results])
        resampled_accuracy = np.mean([r['accuracy'] for r in position_results])
        impact = baseline_accuracy - resampled_accuracy
        std_error = np.std([r['accuracy'] for r in position_results]) / np.sqrt(n_problems)

        results[f'CT{position}'] = {
            'baseline_accuracy': float(baseline_accuracy),
            'resampled_accuracy': float(resampled_accuracy),
            'impact': float(impact),
            'std_error': float(std_error),
            'per_problem_results': position_results
        }

        print(f"\nCT{position} Results:")
        print(f"  Baseline: {baseline_accuracy:.2%}")
        print(f"  Resampled: {resampled_accuracy:.2%}")
        print(f"  Impact: {impact:.2%} (±{std_error:.2%})")

        # Log to W&B
        wandb.log({
            f'pilot/CT{position}_baseline_acc': baseline_accuracy,
            f'pilot/CT{position}_resampled_acc': resampled_accuracy,
            f'pilot/CT{position}_impact': impact,
            f'pilot/CT{position}_std_error': std_error
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Run resampling experiment")
    parser.add_argument('--phase', type=str, choices=['pilot', 'full'], default='pilot',
                        help='Experiment phase')
    parser.add_argument('--cache_file', type=str, default=None,
                        help='Path to cache file (auto-detected if not provided)')
    parser.add_argument('--n_samples', type=int, default=None,
                        help='Samples per problem (5 for pilot, 10 for full)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Auto-detect cache file and samples
    if args.cache_file is None:
        args.cache_file = f"../data/ct_hidden_states_cache_{args.phase}.pkl"

    if args.n_samples is None:
        args.n_samples = 5 if args.phase == 'pilot' else 10

    # Set all seeds
    set_seed(args.seed)

    # Initialize W&B
    wandb.init(
        project="codi-resampling",
        name=f"{args.phase}_resampling",
        tags=[args.phase, "resampling", "llama-1b"],
        config={
            "phase": args.phase,
            "n_samples_per_problem": args.n_samples,
            "random_seed": args.seed
        }
    )

    print(f"\n{'='*60}")
    print(f"Story 1.4: Resampling Experiment ({args.phase.upper()})")
    print(f"{'='*60}")
    print(f"Phase: {args.phase}")
    print(f"Cache file: {args.cache_file}")
    print(f"Samples per problem: {args.n_samples}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # Load cache
    cache_path = Path(__file__).parent / args.cache_file
    print(f"Loading cache from {cache_path}...")

    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)

    print(f"✓ Loaded cache with {len(cache)} problems")

    # Load model
    model, tokenizer = load_model(device=args.device)

    # Run resampling experiment
    results = run_resampling_experiment(
        model, tokenizer, cache,
        n_samples_per_problem=args.n_samples,
        device=args.device
    )

    # Add metadata
    output = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'phase': args.phase,
            'n_problems': len(cache),
            'n_samples_per_problem': args.n_samples,
            'random_seed': args.seed,
            'model': 'llama-1b-gsm8k',
            'total_generations': 6 * len(cache) * args.n_samples
        },
        'results': results
    }

    # Save results
    results_dir = Path(__file__).parent / '../results'
    results_dir.mkdir(parents=True, exist_ok=True)

    results_filename = f"resampling_{args.phase}_results.json"
    results_path = results_dir / results_filename

    print(f"\nSaving results to {results_path}...")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESAMPLING COMPLETE")
    print(f"{'='*60}")
    print(f"Total generations: {output['metadata']['total_generations']}")
    print(f"Results file: {results_path}")
    print(f"W&B run: {wandb.run.url}")

    print(f"\nPer-Position Impacts:")
    for pos in range(6):
        impact = results[f'CT{pos}']['impact']
        std_err = results[f'CT{pos}']['std_error']
        print(f"  CT{pos}: {impact:+.2%} (±{std_err:.2%})")

    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
