"""
CORRECTED Experiment Runner - Only patches valid intervention cases.

Key Change: Only patch when clean baseline is CORRECT.
This ensures we're injecting good reasoning, not bad reasoning.

Usage:
    python run_experiment_corrected.py --model_path ~/codi_ckpt/gpt2_gsm8k/ \
                                        --problem_pairs problem_pairs.json \
                                        --output_dir results_corrected/
"""

import json
import os
import re
import argparse
from typing import Dict, List
from tqdm import tqdm
import wandb
import torch

from cache_activations import ActivationCacher, LAYER_CONFIG
from patch_and_eval import ActivationPatcher


def extract_answer_number(text: str) -> int:
    """Extract the numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*(-?\d+)',
        r'\$?\s*(-?\d+)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])

    return None


class CorrectedExperimentRunner:
    """Runs corrected activation patching experiment."""

    def __init__(self, model_path: str, output_dir: str, wandb_project: str = None):
        """Initialize experiment runner.

        Args:
            model_path: Path to CODI checkpoint
            output_dir: Directory to save results
            wandb_project: WandB project name (optional)
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("Loading CODI model...")
        self.cacher = ActivationCacher(model_path)
        self.patcher = ActivationPatcher(self.cacher)
        print("✓ Model loaded")

        # Initialize WandB
        self.use_wandb = wandb_project is not None
        if self.use_wandb:
            print("Initializing WandB...")
            wandb.init(
                project=wandb_project,
                name="corrected-patching-valid-cases-only",
                config={
                    "experiment": "activation_patching_corrected",
                    "model": model_path,
                    "layers": list(LAYER_CONFIG.keys()),
                    "note": "Only patches cases where clean baseline is CORRECT"
                }
            )
            print(f"✓ WandB initialized: {wandb.run.url}")

    def run_pair(self, pair: Dict) -> Dict:
        """Run experiment on single problem pair.

        Returns dict with results and validity flag.
        """
        pair_id = pair['pair_id']
        clean_q = pair['clean']['question']
        corrupted_q = pair['corrupted']['question']
        clean_ans = pair['clean']['answer']
        corrupted_ans = pair['corrupted']['answer']

        # 1. Run clean baseline
        clean_generated = self.patcher.run_without_patch(clean_q, max_new_tokens=200)
        clean_predicted = extract_answer_number(clean_generated)
        clean_correct = (clean_predicted == clean_ans)

        # 2. Check validity: Only continue if clean is CORRECT
        if not clean_correct:
            return {
                'pair_id': pair_id,
                'valid': False,
                'reason': 'clean_baseline_wrong',
                'clean': {
                    'correct': False,
                    'expected': clean_ans,
                    'extracted': clean_predicted,
                    'generated': clean_generated
                }
            }

        # 3. Run corrupted baseline
        corrupted_generated = self.patcher.run_without_patch(corrupted_q, max_new_tokens=200)
        corrupted_predicted = extract_answer_number(corrupted_generated)
        corrupted_correct = (corrupted_predicted == corrupted_ans)

        # 4. Cache clean activations
        clean_activations = self.cacher.cache_problem_activations(clean_q, pair_id)

        # 5. Run patched versions for each layer
        patched_results = {}
        for layer_name in LAYER_CONFIG.keys():
            clean_act = clean_activations[layer_name]
            patched_generated = self.patcher.run_with_patch(
                corrupted_q,
                clean_act,
                layer_name,
                max_new_tokens=200
            )
            patched_predicted = extract_answer_number(patched_generated)
            patched_correct = (patched_predicted == clean_ans)  # Should match CLEAN answer

            patched_results[layer_name] = {
                'correct': patched_correct,
                'extracted': patched_predicted,
                'generated': patched_generated
            }

        return {
            'pair_id': pair_id,
            'valid': True,
            'clean': {
                'correct': clean_correct,
                'expected': clean_ans,
                'extracted': clean_predicted,
                'generated': clean_generated
            },
            'corrupted': {
                'correct': corrupted_correct,
                'expected': corrupted_ans,
                'extracted': corrupted_predicted,
                'generated': corrupted_generated
            },
            'patched': patched_results
        }

    def run_experiment(self, problem_pairs: List[Dict]) -> Dict:
        """Run full experiment on all problem pairs."""

        print(f"\n{'='*60}")
        print("CORRECTED EXPERIMENT")
        print(f"{'='*60}")
        print(f"Total pairs: {len(problem_pairs)}")
        print("Strategy: Only patch when clean baseline is CORRECT")
        print(f"{'='*60}\n")

        all_results = []
        valid_results = []
        invalid_results = []

        for pair in tqdm(problem_pairs, desc="Processing problem pairs"):
            try:
                result = self.run_pair(pair)
                all_results.append(result)

                if result['valid']:
                    valid_results.append(result)
                else:
                    invalid_results.append(result)

                # Log to WandB
                if self.use_wandb and result['valid']:
                    wandb.log({
                        'pair_id': result['pair_id'],
                        'clean_correct': 1,  # Always true for valid cases
                        'corrupted_correct': int(result['corrupted']['correct']),
                        'early_correct': int(result['patched']['early']['correct']),
                        'middle_correct': int(result['patched']['middle']['correct']),
                        'late_correct': int(result['patched']['late']['correct']),
                    })

            except Exception as e:
                print(f"\nERROR on pair {pair['pair_id']}: {e}")
                continue

        # Calculate metrics on VALID cases only
        if len(valid_results) == 0:
            print("\n⚠️  No valid results! All clean baselines were wrong.")
            return {'error': 'no_valid_results'}

        metrics = self._calculate_metrics(valid_results)

        # Save results
        results_data = {
            'summary': metrics,
            'valid_results': valid_results,
            'invalid_results': invalid_results,
            'config': {
                'total_pairs': len(problem_pairs),
                'valid_pairs': len(valid_results),
                'invalid_pairs': len(invalid_results),
                'layers': list(LAYER_CONFIG.keys()),
                'note': 'Only valid interventions (clean correct) included in metrics'
            }
        }

        results_path = os.path.join(self.output_dir, 'experiment_results_corrected.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        # Log summary to WandB
        if self.use_wandb:
            wandb.log({
                'summary/valid_pairs': len(valid_results),
                'summary/target_pairs': metrics['total_targets'],
                'summary/clean_accuracy': metrics['clean_accuracy'],
                'summary/corrupted_accuracy': metrics['corrupted_accuracy'],
                'summary/early_accuracy': metrics['early']['accuracy'],
                'summary/early_recovery': metrics['early']['recovery_rate'],
                'summary/middle_accuracy': metrics['middle']['accuracy'],
                'summary/middle_recovery': metrics['middle']['recovery_rate'],
                'summary/late_accuracy': metrics['late']['accuracy'],
                'summary/late_recovery': metrics['late']['recovery_rate'],
            })

        return results_data

    def _calculate_metrics(self, valid_results: List[Dict]) -> Dict:
        """Calculate metrics on TARGET cases only (Clean ✓, Corrupted ✗).

        Valid cases = Clean correct (all intervention candidates)
        Target cases = Clean correct AND Corrupted wrong (where patching should help)
        """
        total_valid = len(valid_results)

        # Filter to TARGET cases only: Clean ✓, Corrupted ✗
        target_cases = [r for r in valid_results if not r['corrupted']['correct']]
        total_targets = len(target_cases)

        # Overall baselines (on all valid cases for context)
        clean_correct_all = total_valid  # All valid cases have clean correct by definition
        corrupted_correct_all = sum(1 for r in valid_results if r['corrupted']['correct'])

        # If no target cases, return early (all corrupted were already correct)
        if total_targets == 0:
            return {
                'clean_accuracy': 1.0,
                'corrupted_accuracy': 1.0,
                'clean_correct': clean_correct_all,
                'corrupted_correct': corrupted_correct_all,
                'total_valid': total_valid,
                'total_targets': 0,
                'note': 'No target cases - all corrupted problems were already correct',
                'early': {'accuracy': 1.0, 'recovery_rate': 0.0, 'correct_count': 0, 'total_count': 0},
                'middle': {'accuracy': 1.0, 'recovery_rate': 0.0, 'correct_count': 0, 'total_count': 0},
                'late': {'accuracy': 1.0, 'recovery_rate': 0.0, 'correct_count': 0, 'total_count': 0}
            }

        # Per-layer metrics on TARGET cases only
        layer_metrics = {}
        for layer_name in LAYER_CONFIG.keys():
            # Count how many target cases were recovered (patched → correct)
            patched_correct_targets = sum(
                1 for r in target_cases if r['patched'][layer_name]['correct']
            )

            # Recovery rate = what % of target cases did patching fix?
            recovery_rate = patched_correct_targets / total_targets

            # Also calculate overall accuracy on all valid cases (for comparison)
            patched_correct_all = sum(
                1 for r in valid_results if r['patched'][layer_name]['correct']
            )
            patched_acc_all = patched_correct_all / total_valid

            layer_metrics[layer_name] = {
                'accuracy': patched_acc_all,  # On all valid cases
                'recovery_rate': recovery_rate,  # On target cases only
                'correct_count': patched_correct_targets,  # On target cases
                'total_count': total_targets  # Target cases only
            }

        return {
            'clean_accuracy': 1.0,  # 100% by definition (filtered)
            'corrupted_accuracy': corrupted_correct_all / total_valid,  # On all valid
            'clean_correct': clean_correct_all,
            'corrupted_correct': corrupted_correct_all,
            'total_valid': total_valid,
            'total_targets': total_targets,  # NEW: How many cases need intervention
            **layer_metrics
        }

    def print_summary(self, results: Dict):
        """Print experiment summary."""
        if 'error' in results:
            print(f"\n⚠️  Experiment failed: {results['error']}")
            return

        metrics = results['summary']
        config = results['config']

        print(f"\n\n{'='*60}")
        print("CORRECTED EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Total pairs tested: {config['total_pairs']}")
        print(f"Valid pairs (clean ✓): {config['valid_pairs']}")
        print(f"  └─ Already correct (corrupted ✓): {metrics['corrupted_correct']}")
        print(f"  └─ TARGET cases (corrupted ✗): {metrics['total_targets']}")
        print(f"Invalid pairs (clean ✗): {config['invalid_pairs']}")
        print(f"{'='*60}\n")

        print("BASELINE PERFORMANCE:")
        print(f"  Clean problems:      {100*metrics['clean_accuracy']:.1f}% ({metrics['clean_correct']}/{metrics['total_valid']})")
        print(f"  Corrupted problems:  {100*metrics['corrupted_accuracy']:.1f}% ({metrics['corrupted_correct']}/{metrics['total_valid']})")
        print()

        print(f"PATCHING RESULTS (on {metrics['total_targets']} target cases):")
        for layer_name in LAYER_CONFIG.keys():
            layer_idx = LAYER_CONFIG[layer_name]
            m = metrics[layer_name]
            print(f"  {layer_name.capitalize():7s} (L{layer_idx:2d}) - "
                  f"Recovery: {100*m['recovery_rate']:5.1f}% "
                  f"({m['correct_count']}/{m['total_count']} fixed)")

        print(f"\n{'='*60}")
        best_layer = max(LAYER_CONFIG.keys(), key=lambda l: metrics[l]['recovery_rate'])
        best_recovery = metrics[best_layer]['recovery_rate']

        if best_recovery > 0:
            print("✓ POSITIVE RECOVERY DETECTED")
            print(f"{'='*60}")
            print(f"Patching shows causal effect on reasoning!")
            print(f"Best layer: {best_layer.capitalize()} (L{LAYER_CONFIG[best_layer]}) - {100*best_recovery:.1f}% recovery")
        else:
            print("⚠️  NO POSITIVE RECOVERY")
            print(f"{'='*60}")
            print(f"Patching does not improve performance on target cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--problem_pairs', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_corrected/')
    parser.add_argument('--wandb_project', type=str, default='codi-activation-patching')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    # Load problem pairs
    print(f"Loading problem pairs from {args.problem_pairs}...")
    with open(args.problem_pairs, 'r') as f:
        pairs = json.load(f)
    print(f"✓ Loaded {len(pairs)} problem pairs")

    # Initialize runner
    wandb_project = None if args.no_wandb else args.wandb_project
    runner = CorrectedExperimentRunner(
        args.model_path,
        args.output_dir,
        wandb_project
    )

    # Run experiment
    results = runner.run_experiment(pairs)

    # Print summary
    runner.print_summary(results)

    if runner.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
