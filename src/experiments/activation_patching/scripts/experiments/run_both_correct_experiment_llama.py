"""
Both-Correct Activation Patching Experiment (LLaMA Version)

Hypothesis: If the model learns to reason, then patching CLEAN activations into
CORRUPTED question processing should cause the model to output the CLEAN answer.

Filtering: Only use pairs where BOTH clean and corrupted are answered correctly.
Direction: Patch CLEAN activations → into CORRUPTED question processing.
Classify: Output as clean_answer / corrupted_answer / other / gibberish

Usage:
    python run_both_correct_experiment_llama.py --model_path ~/codi_ckpt/llama_commonsense/gsm8k_llama1b_latent_baseline/.../seed_11/ \
                                                --problem_pairs data/problem_pairs.json \
                                                --output_dir results_both_correct_llama/
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

from cache_activations_llama import ActivationCacherLLaMA as ActivationCacher, LAYER_CONFIG
from patch_and_eval_llama import ActivationPatcher

# Get script directory for wandb logs
SCRIPT_DIR = Path(__file__).parent.absolute()


def extract_answer_number(text: str) -> int:
    """Extract the numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
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


def answers_match(predicted, expected) -> bool:
    """Check if predicted answer matches expected (handles floats)."""
    if predicted is None or expected is None:
        return False

    # Convert both to float for comparison
    try:
        pred_float = float(predicted)
        exp_float = float(expected)
        # Allow small floating point differences
        return abs(pred_float - exp_float) < 0.01
    except (ValueError, TypeError):
        return False


def classify_output(generated_text: str, clean_answer, corrupted_answer) -> str:
    """Classify the output as clean_answer, corrupted_answer, other_coherent, or gibberish."""
    extracted = extract_answer_number(generated_text)

    if extracted is None:
        # Check if text seems coherent or gibberish
        if len(generated_text) < 10 or not any(c.isalpha() for c in generated_text):
            return "gibberish"
        return "other_coherent"

    if answers_match(extracted, clean_answer):
        return "clean_answer"
    elif answers_match(extracted, corrupted_answer):
        return "corrupted_answer"
    else:
        return "other_coherent"


class BothCorrectExperimentRunner:
    """Runs both-correct activation patching experiment."""

    def __init__(self, model_path: str, output_dir: str, wandb_project: str = None):
        """Initialize experiment runner."""
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
                name="llama-both-correct-patching",
                dir=str(SCRIPT_DIR / "wandb_runs"),
                config={
                    "experiment": "llama_both_correct_activation_patching",
                    "model": "Llama-3.2-1B-Instruct + CODI",
                    "layers": list(LAYER_CONFIG.keys()),
                    "hypothesis": "Patching clean→corrupted should produce clean answer if reasoning is learned",
                    "filtering": "Both clean and corrupted correct"
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
        clean_correct = answers_match(clean_predicted, clean_ans)

        # 2. Run corrupted baseline
        corrupted_generated = self.patcher.run_without_patch(corrupted_q, max_new_tokens=200)
        corrupted_predicted = extract_answer_number(corrupted_generated)
        corrupted_correct = answers_match(corrupted_predicted, corrupted_ans)

        # 3. Check validity: Only continue if BOTH are CORRECT
        if not (clean_correct and corrupted_correct):
            return {
                'pair_id': pair_id,
                'valid': False,
                'reason': f'clean_correct={clean_correct}, corrupted_correct={corrupted_correct}',
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
                }
            }

        # 4. Cache clean activations
        clean_activations = self.cacher.cache_problem_activations(clean_q, pair_id)

        # 5. Run patched versions for each layer: CLEAN → CORRUPTED
        patched_results = {}
        for layer_name in LAYER_CONFIG.keys():
            clean_act = clean_activations[layer_name]
            patched_generated = self.patcher.run_with_patch(
                corrupted_q,  # Run on corrupted question
                clean_act,    # Patch in clean activations
                layer_name,
                max_new_tokens=200
            )
            patched_predicted = extract_answer_number(patched_generated)

            # Classify the output
            classification = classify_output(patched_generated, clean_ans, corrupted_ans)

            patched_results[layer_name] = {
                'classification': classification,
                'extracted': patched_predicted,
                'generated': patched_generated,
                'matches_clean': answers_match(patched_predicted, clean_ans),
                'matches_corrupted': answers_match(patched_predicted, corrupted_ans)
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
        print("BOTH-CORRECT ACTIVATION PATCHING EXPERIMENT")
        print(f"{'='*60}")
        print(f"Total pairs: {len(problem_pairs)}")
        print("Filtering: Both clean AND corrupted must be correct")
        print("Direction: CLEAN activations → CORRUPTED question")
        print("Hypothesis: Should produce CLEAN answer if reasoning is learned")
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
                    log_data = {
                        'pair_id': result['pair_id'],
                        'clean_correct': 1,
                        'corrupted_correct': 1,
                    }

                    for layer_name in LAYER_CONFIG.keys():
                        classification = result['patched'][layer_name]['classification']
                        log_data[f'{layer_name}_classification'] = classification
                        log_data[f'{layer_name}_matches_clean'] = int(result['patched'][layer_name]['matches_clean'])
                        log_data[f'{layer_name}_matches_corrupted'] = int(result['patched'][layer_name]['matches_corrupted'])

                    wandb.log(log_data)

            except Exception as e:
                print(f"\nERROR on pair {pair['pair_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Calculate metrics on VALID cases only
        if len(valid_results) == 0:
            print("\n⚠️  No valid results! No pairs where both clean AND corrupted were correct.")
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
                'hypothesis': 'Patching clean→corrupted should produce clean answer',
                'filtering': 'Both clean and corrupted correct'
            }
        }

        results_path = os.path.join(self.output_dir, 'experiment_results_both_correct.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        # Log summary to WandB
        if self.use_wandb:
            summary_log = {
                'summary/valid_pairs': len(valid_results),
            }

            for layer_name in LAYER_CONFIG.keys():
                for classification in ['clean_answer', 'corrupted_answer', 'other_coherent', 'gibberish']:
                    count = metrics[layer_name]['classifications'][classification]
                    pct = metrics[layer_name]['classification_pcts'][classification]
                    summary_log[f'summary/{layer_name}_{classification}_count'] = count
                    summary_log[f'summary/{layer_name}_{classification}_pct'] = pct

            wandb.log(summary_log)

        return results_data

    def _calculate_metrics(self, valid_results: List[Dict]) -> Dict:
        """Calculate metrics on valid both-correct cases."""
        total_valid = len(valid_results)

        layer_metrics = {}
        for layer_name in LAYER_CONFIG.keys():
            # Count classifications
            classifications = {
                'clean_answer': 0,
                'corrupted_answer': 0,
                'other_coherent': 0,
                'gibberish': 0
            }

            for r in valid_results:
                classification = r['patched'][layer_name]['classification']
                classifications[classification] += 1

            # Calculate percentages
            classification_pcts = {
                k: v / total_valid for k, v in classifications.items()
            }

            layer_metrics[layer_name] = {
                'classifications': classifications,
                'classification_pcts': classification_pcts
            }

        return {
            'total_valid': total_valid,
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
        print("BOTH-CORRECT EXPERIMENT RESULTS")
        print(f"{'='*60}")
        print(f"Total pairs tested: {config['total_pairs']}")
        print(f"Valid pairs (both ✓): {config['valid_pairs']}")
        print(f"Invalid pairs: {config['invalid_pairs']}")
        print(f"{'='*60}\n")

        print("HYPOTHESIS: Patching CLEAN→CORRUPTED should produce CLEAN answer\n")

        print("CLASSIFICATION BREAKDOWN BY LAYER:")
        print(f"{'Layer':<10} {'Clean %':<10} {'Corrupt %':<12} {'Other %':<10} {'Gibberish %':<12}")
        print(f"{'-'*60}")

        for layer_name in LAYER_CONFIG.keys():
            layer_idx = LAYER_CONFIG[layer_name]
            m = metrics[layer_name]
            clean_pct = 100 * m['classification_pcts']['clean_answer']
            corrupt_pct = 100 * m['classification_pcts']['corrupted_answer']
            other_pct = 100 * m['classification_pcts']['other_coherent']
            gibberish_pct = 100 * m['classification_pcts']['gibberish']

            print(f"{layer_name.capitalize()} (L{layer_idx:2d})  "
                  f"{clean_pct:5.1f}%     "
                  f"{corrupt_pct:5.1f}%       "
                  f"{other_pct:5.1f}%     "
                  f"{gibberish_pct:5.1f}%")

        print(f"\n{'='*60}")

        # Find best layer for producing clean answer
        best_layer = max(LAYER_CONFIG.keys(),
                        key=lambda l: metrics[l]['classification_pcts']['clean_answer'])
        best_clean_pct = metrics[best_layer]['classification_pcts']['clean_answer']

        if best_clean_pct > 0.5:
            print("✓ HYPOTHESIS SUPPORTED")
            print(f"{'='*60}")
            print(f"Patching CLEAN activations causes model to output CLEAN answer!")
            print(f"Best layer: {best_layer.capitalize()} (L{LAYER_CONFIG[best_layer]}) - "
                  f"{100*best_clean_pct:.1f}% clean answers")
        elif metrics[best_layer]['classification_pcts']['corrupted_answer'] > 0.5:
            print("⚠️  UNEXPECTED RESULT")
            print(f"{'='*60}")
            print(f"Model still produces CORRUPTED answer despite clean activations")
        else:
            print("⚠️  INCONCLUSIVE")
            print(f"{'='*60}")
            print(f"Model produces mixed results - neither clearly clean nor corrupted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--problem_pairs', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results_both_correct/')
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
    runner = BothCorrectExperimentRunner(
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
