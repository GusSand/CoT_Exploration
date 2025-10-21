"""
Main Experiment Runner
Runs activation patching experiment with WandB tracking.

Usage:
    python run_experiment.py --model_path ~/codi_ckpt/gpt2_gsm8k_6latent/ \
                             --problem_pairs problem_pairs.json \
                             --output_dir results/
"""

import json
import os
import re
import argparse
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
import wandb
import torch

from cache_activations import ActivationCacher, LAYER_CONFIG
from patch_and_eval import ActivationPatcher

# Get script directory for wandb logs
SCRIPT_DIR = Path(__file__).parent.absolute()


def extract_answer_number(text: str) -> int:
    """Extract the numerical answer from generated text.

    Args:
        text: Generated answer text

    Returns:
        Extracted number or None
    """
    # Look for patterns like "####" 21 or "The answer is 21"
    patterns = [
        r'####\s*(-?\d+)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*(-?\d+)',
        r'\$?\s*(-?\d+)\s*$',  # Number at end
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))

    # Fall back to last number in text
    numbers = re.findall(r'-?\d+', text)
    if numbers:
        return int(numbers[-1])

    return None


class ExperimentRunner:
    """Runs the full activation patching experiment."""

    def __init__(
        self,
        model_path: str,
        problem_pairs_file: str,
        output_dir: str,
        wandb_project: str = "codi-activation-patching"
    ):
        """Initialize experiment runner.

        Args:
            model_path: Path to CODI checkpoint
            problem_pairs_file: Path to problem pairs JSON
            output_dir: Directory for outputs
            wandb_project: WandB project name
        """
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load problem pairs
        print(f"Loading problem pairs from {problem_pairs_file}...")
        with open(problem_pairs_file, 'r') as f:
            self.pairs = json.load(f)
        print(f"✓ Loaded {len(self.pairs)} problem pairs")

        # Initialize cacher and patcher
        print(f"Loading CODI model...")
        self.cacher = ActivationCacher(model_path)
        self.patcher = ActivationPatcher(self.cacher)
        print(f"✓ Model loaded")

        # Initialize WandB
        print(f"Initializing WandB...")
        wandb.init(
            project=wandb_project,
            name=f"direct-patching-3layers-{len(self.pairs)}pairs",
            dir=str(SCRIPT_DIR / "wandb_runs"),
            config={
                "experiment": "direct_activation_patching",
                "num_pairs": len(self.pairs),
                "layers": list(LAYER_CONFIG.keys()),
                "layer_indices": LAYER_CONFIG,
                "model": "gpt2-codi",
                "dataset": "gsm8k",
                "model_path": model_path
            },
            tags=["causal-analysis", "mechanistic-interpretability"]
        )
        print(f"✓ WandB initialized: {wandb.run.url}")

        self.results = []

    def run_pair(self, pair: Dict) -> Dict:
        """Run all conditions for a single problem pair.

        Args:
            pair: Problem pair dict

        Returns:
            Results dict with all conditions
        """
        pair_id = pair['pair_id']
        clean_question = pair['clean']['question']
        clean_answer = pair['clean']['answer']
        corrupted_question = pair['corrupted']['question']
        corrupted_answer = pair['corrupted']['answer']

        # 1. Run clean problem and cache activations
        clean_activations = self.cacher.cache_problem_activations(
            clean_question,
            pair_id
        )

        clean_generated = self.patcher.run_without_patch(clean_question)
        clean_extracted = extract_answer_number(clean_generated)
        clean_correct = (clean_extracted == clean_answer)

        # 2. Run corrupted problem (baseline)
        corrupted_generated = self.patcher.run_without_patch(corrupted_question)
        corrupted_extracted = extract_answer_number(corrupted_generated)
        corrupted_correct = (corrupted_extracted == corrupted_answer)

        # 3. Run patched at each layer
        patched_results = {}
        for layer_name in LAYER_CONFIG.keys():
            patched_generated = self.patcher.run_with_patch(
                corrupted_question,
                clean_activations[layer_name],
                layer_name
            )
            patched_extracted = extract_answer_number(patched_generated)

            # Patched is "correct" if it recovers clean answer
            patched_correct = (patched_extracted == clean_answer)

            patched_results[layer_name] = {
                'generated': patched_generated,
                'extracted': patched_extracted,
                'correct': patched_correct
            }

        return {
            'pair_id': pair_id,
            'clean': {
                'generated': clean_generated,
                'extracted': clean_extracted,
                'expected': clean_answer,
                'correct': clean_correct
            },
            'corrupted': {
                'generated': corrupted_generated,
                'extracted': corrupted_extracted,
                'expected': corrupted_answer,
                'correct': corrupted_correct
            },
            'patched': patched_results
        }

    def run_experiment(self):
        """Run the full experiment on all pairs."""
        print(f"\n{'='*60}")
        print(f"STARTING EXPERIMENT")
        print(f"{'='*60}")
        print(f"Problem pairs: {len(self.pairs)}")
        print(f"Conditions per pair: 5 (clean, corrupted, 3 patched layers)")
        print(f"Total forward passes: {len(self.pairs) * 5}")
        print(f"{'='*60}\n")

        for pair in tqdm(self.pairs, desc="Processing problem pairs"):
            try:
                # Run all conditions for this pair
                result = self.run_pair(pair)
                self.results.append(result)

                # Log to WandB (per-problem)
                wandb.log({
                    'pair_id': result['pair_id'],
                    'clean_correct': int(result['clean']['correct']),
                    'corrupted_correct': int(result['corrupted']['correct']),
                    'early_correct': int(result['patched']['early']['correct']),
                    'middle_correct': int(result['patched']['middle']['correct']),
                    'late_correct': int(result['patched']['late']['correct']),
                })

                # Checkpoint every 10 problems
                if (result['pair_id'] + 1) % 10 == 0:
                    self.save_checkpoint(result['pair_id'] + 1)
                    tqdm.write(f"✓ Checkpoint saved at problem {result['pair_id'] + 1}")

            except Exception as e:
                print(f"\nERROR on pair {pair['pair_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Calculate final metrics
        self.calculate_and_log_metrics()

        # Save final results
        self.save_final_results()

        print(f"\n{'='*60}")
        print(f"EXPERIMENT COMPLETE")
        print(f"✓ Results saved to {self.output_dir}")
        print(f"✓ WandB run: {wandb.run.url}")
        print(f"{'='*60}\n")

        wandb.finish()

    def calculate_and_log_metrics(self):
        """Calculate summary metrics and log to WandB."""
        if not self.results:
            print("No results to calculate metrics from!")
            return

        # Calculate accuracies
        clean_acc = sum(r['clean']['correct'] for r in self.results) / len(self.results)
        corrupted_acc = sum(r['corrupted']['correct'] for r in self.results) / len(self.results)

        layer_metrics = {}
        for layer_name in LAYER_CONFIG.keys():
            patched_acc = sum(r['patched'][layer_name]['correct'] for r in self.results) / len(self.results)

            # Recovery rate: how much of the accuracy drop is recovered?
            if (clean_acc - corrupted_acc) > 0:
                recovery_rate = (patched_acc - corrupted_acc) / (clean_acc - corrupted_acc)
            else:
                recovery_rate = 0.0

            layer_metrics[layer_name] = {
                'accuracy': patched_acc,
                'recovery_rate': recovery_rate
            }

        # Log summary to WandB
        wandb.log({
            'summary/clean_accuracy': clean_acc,
            'summary/corrupted_accuracy': corrupted_acc,
            'summary/early_accuracy': layer_metrics['early']['accuracy'],
            'summary/middle_accuracy': layer_metrics['middle']['accuracy'],
            'summary/late_accuracy': layer_metrics['late']['accuracy'],
            'summary/early_recovery': layer_metrics['early']['recovery_rate'],
            'summary/middle_recovery': layer_metrics['middle']['recovery_rate'],
            'summary/late_recovery': layer_metrics['late']['recovery_rate'],
        })

        # Print results
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Clean Accuracy:      {clean_acc:.2%}")
        print(f"Corrupted Accuracy:  {corrupted_acc:.2%}")
        print(f"\nPer-Layer Patching Results:")
        for layer_name, metrics in layer_metrics.items():
            layer_idx = LAYER_CONFIG[layer_name]
            print(f"  {layer_name:8s} (L{layer_idx:2d}) - Acc: {metrics['accuracy']:.2%}, Recovery: {metrics['recovery_rate']:.1%}")
        print(f"{'='*60}\n")

        self.metrics = {
            'clean_accuracy': clean_acc,
            'corrupted_accuracy': corrupted_acc,
            'layer_results': layer_metrics
        }

    def save_checkpoint(self, checkpoint_id: int):
        """Save checkpoint of current results.

        Args:
            checkpoint_id: Checkpoint identifier
        """
        checkpoint = {'results': self.results}
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_{checkpoint_id}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def save_final_results(self):
        """Save final results to JSON and as WandB artifact."""
        output = {
            'summary': self.metrics,
            'per_problem': self.results,
            'config': {
                'num_pairs': len(self.pairs),
                'layers': LAYER_CONFIG,
                'model_path': self.model_path
            }
        }

        # Save to JSON
        results_path = os.path.join(self.output_dir, 'experiment_results.json')
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)

        # Save as WandB artifact
        artifact = wandb.Artifact('experiment_results', type='results')
        artifact.add_file(results_path)
        wandb.log_artifact(artifact)

        print(f"✓ Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Run activation patching experiment")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to CODI checkpoint')
    parser.add_argument('--problem_pairs', type=str, default='problem_pairs.json',
                        help='Path to problem pairs JSON')
    parser.add_argument('--output_dir', type=str, default='results/',
                        help='Output directory for results')
    parser.add_argument('--wandb_project', type=str, default='codi-activation-patching',
                        help='WandB project name')

    args = parser.parse_args()

    # Run experiment
    runner = ExperimentRunner(
        model_path=args.model_path,
        problem_pairs_file=args.problem_pairs,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project
    )

    runner.run_experiment()


if __name__ == "__main__":
    main()
