#!/usr/bin/env python3
"""
Gradient Attribution Sweep - Main Runner

Processes 532 problems from LLaMA Steering Dataset (Full) and computes
integrated gradients for all (layer, token) positions.

Features:
- Checkpoint system for resuming interrupted runs
- Error handling with detailed logging
- Progress tracking with tqdm
- Separate results for correct/incorrect predictions

Usage:
    # Full run on 532 problems (4-6 hours on A100)
    python run_gradient_attribution_sweep.py

    # Test run on 10 problems
    python run_gradient_attribution_sweep.py --test_mode --num_problems 10

    # Resume from checkpoint
    python run_gradient_attribution_sweep.py --resume

Author: Generated for CoT Exploration Project
Date: 2025-10-24
"""

import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time
from datetime import datetime

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
activation_patching_core = project_root / 'src' / 'experiments' / 'activation_patching' / 'core'
sys.path.insert(0, str(activation_patching_core))

# Import infrastructure
from cache_activations_llama import ActivationCacherLLaMA
from gradient_attribution import extract_answer_number, answers_match
from ablation_attribution import AblationAttributor


# ========================================
# STORY 4: Data Pipeline & Parallelization
# ========================================

@dataclass
class ExperimentConfig:
    """Configuration for ablation attribution experiment."""
    model_path: str
    dataset_path: str
    output_dir: str
    target_layers: List[int]
    target_tokens: List[int]
    checkpoint_interval: int
    test_mode: bool
    num_test_problems: int


class GradientAttributionRunner:
    """
    Main runner for gradient attribution sweep across dataset.

    Handles:
    - Dataset loading
    - Checkpointing
    - Error handling
    - Progress tracking
    - Results aggregation
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize the runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "attribution_results.json"
        self.errors_file = self.output_dir / "errors.json"

        # Statistics
        self.stats = {
            'total_problems': 0,
            'processed': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }

        # Load model and attributor
        print(f"\nLoading LLaMA CODI model from {config.model_path}...")
        self.cacher = ActivationCacherLLaMA(config.model_path)

        print(f"\nInitializing ablation attributor...")
        self.attributor = AblationAttributor(
            self.cacher,
            target_layers=config.target_layers,
            target_tokens=config.target_tokens
        )

        print(f"\nOutput directory: {self.output_dir}")

    def load_dataset(self) -> List[Dict]:
        """
        Load problem pairs and construct dataset.

        Returns:
            List of problem dictionaries with question and expected answer
        """
        print(f"\nLoading dataset from {self.config.dataset_path}...")

        # Load steering dataset
        with open(self.config.dataset_path, 'r') as f:
            steering_data = json.load(f)

        # Load problem pairs to get questions
        pairs_path = '/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching/problem_pairs_gpt4_answers.json'
        with open(pairs_path, 'r') as f:
            problem_pairs = json.load(f)

        # Create lookup dict for pairs
        pairs_dict = {p['pair_id']: p for p in problem_pairs}

        # Construct dataset from train_correct and train_wrong
        dataset = []

        # Process correct answers
        for item in steering_data.get('train_correct', []):
            pair_id = item['pair_id']
            expected = item['expected']

            if pair_id in pairs_dict:
                pair = pairs_dict[pair_id]
                # Use clean variant for correct answers
                dataset.append({
                    'problem_id': f"pair_{pair_id}_clean",
                    'pair_id': pair_id,
                    'question': pair['clean']['question'],
                    'expected_answer': expected,
                    'variant': 'clean',
                    'label': 'correct'
                })

        # Process incorrect answers
        for item in steering_data.get('train_wrong', []):
            pair_id = item['pair_id']
            expected = item['expected']

            if pair_id in pairs_dict:
                pair = pairs_dict[pair_id]
                # Use corrupted variant for wrong answers
                dataset.append({
                    'problem_id': f"pair_{pair_id}_corrupted",
                    'pair_id': pair_id,
                    'question': pair['corrupted']['question'],
                    'expected_answer': expected,
                    'variant': 'corrupted',
                    'label': 'wrong'
                })

        # Also add test split
        for item in steering_data.get('test_correct', []):
            pair_id = item['pair_id']
            expected = item['expected']

            if pair_id in pairs_dict:
                pair = pairs_dict[pair_id]
                dataset.append({
                    'problem_id': f"pair_{pair_id}_clean_test",
                    'pair_id': pair_id,
                    'question': pair['clean']['question'],
                    'expected_answer': expected,
                    'variant': 'clean',
                    'label': 'correct',
                    'split': 'test'
                })

        for item in steering_data.get('test_wrong', []):
            pair_id = item['pair_id']
            expected = item['expected']

            if pair_id in pairs_dict:
                pair = pairs_dict[pair_id]
                dataset.append({
                    'problem_id': f"pair_{pair_id}_corrupted_test",
                    'pair_id': pair_id,
                    'question': pair['corrupted']['question'],
                    'expected_answer': expected,
                    'variant': 'corrupted',
                    'label': 'wrong',
                    'split': 'test'
                })

        print(f"Loaded {len(dataset)} problems:")
        train_count = sum(1 for p in dataset if p.get('split') != 'test')
        test_count = sum(1 for p in dataset if p.get('split') == 'test')
        print(f"  Train: {train_count} problems")
        print(f"  Test: {test_count} problems")

        # Limit to test size if in test mode
        if self.config.test_mode:
            dataset = dataset[:self.config.num_test_problems]
            print(f"\n⚠️  TEST MODE: Limited to {len(dataset)} problems")

        return dataset

    def load_checkpoint(self) -> Dict:
        """
        Load checkpoint if it exists.

        Returns:
            Checkpoint data with processed results and stats
        """
        if self.checkpoint_file.exists():
            print(f"\n📁 Found checkpoint: {self.checkpoint_file}")
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            print(f"   Processed: {len(checkpoint['results'])} problems")
            return checkpoint
        return {'results': [], 'errors': [], 'stats': {}}

    def save_checkpoint(self, results: List[Dict], errors: List[Dict]):
        """
        Save checkpoint with current results.

        Args:
            results: List of attribution results
            errors: List of error records
        """
        checkpoint = {
            'results': results,
            'errors': errors,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }

        # Atomic write
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        temp_file.replace(self.checkpoint_file)

    def process_problem(self, problem: Dict) -> Optional[Dict]:
        """
        Process a single problem and compute attributions.

        Args:
            problem: Problem dictionary with question and expected answer

        Returns:
            Attribution result dictionary, or None if error
        """
        problem_id = problem['problem_id']
        question = problem['question']
        expected_answer = problem['expected_answer']

        try:
            # Compute ablation attributions (also returns baseline logit diff)
            ablation_attributions, baseline_logit_diff = self.attributor.compute_ablation_attributions(
                question,
                expected_answer,
                show_progress=False
            )

            # Compute baselines
            random_attributions = self.attributor.compute_random_baseline()
            attention_attributions = self.attributor.compute_attention_attribution(question)

            # Check if model got it right by looking at baseline logit diff
            # Positive logit diff means model prefers correct answer
            correct = baseline_logit_diff > 0

            # For predicted answer, we'd need to actually generate - skip for now
            predicted_answer = None  # Can add if needed

            # Sum of ablation attributions (should be related to total importance)
            total_importance = sum(sum(scores) for scores in ablation_attributions.values())

            # Construct result
            result = {
                'problem_id': problem_id,
                'pair_id': problem.get('pair_id'),
                'variant': problem.get('variant'),
                'label': problem.get('label'),
                'split': problem.get('split', 'train'),
                'question': question,
                'expected_answer': expected_answer,
                'predicted_answer': predicted_answer,
                'correct': correct,
                'baseline_logit_diff': baseline_logit_diff,
                'total_importance': total_importance,
                'ablation_attributions': {f"layer_{k}": v for k, v in ablation_attributions.items()},
                'attention_baseline': {f"layer_{k}": v for k, v in attention_attributions.items()},
                'random_baseline': {f"layer_{k}": v for k, v in random_attributions.items()}
            }

            return result

        except Exception as e:
            print(f"\n❌ Error processing {problem_id}: {e}")
            import traceback
            error_record = {
                'problem_id': problem_id,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().isoformat()
            }
            return None, error_record

    def run(self):
        """
        Main execution loop.

        Processes all problems with checkpointing and error handling.
        """
        print("="*80)
        print("GRADIENT ATTRIBUTION SWEEP")
        print("="*80)

        # Load dataset
        dataset = self.load_dataset()
        self.stats['total_problems'] = len(dataset)

        # Load checkpoint
        checkpoint = self.load_checkpoint()
        results = checkpoint.get('results', [])
        errors = checkpoint.get('errors', [])

        # Get already processed problem IDs
        processed_ids = {r['problem_id'] for r in results}

        # Filter to unprocessed problems
        remaining = [p for p in dataset if p['problem_id'] not in processed_ids]

        print(f"\n📊 Progress:")
        print(f"   Total problems: {len(dataset)}")
        print(f"   Already processed: {len(processed_ids)}")
        print(f"   Remaining: {len(remaining)}")

        if len(remaining) == 0:
            print("\n✅ All problems already processed!")
            self._finalize_results(results, errors)
            return

        # Start timer
        self.stats['start_time'] = datetime.now().isoformat()
        start_time = time.time()

        # Process remaining problems
        print(f"\n🚀 Starting processing...")
        print(f"   Checkpoint interval: {self.config.checkpoint_interval} problems")

        pbar = tqdm(remaining, desc="Processing problems")

        for i, problem in enumerate(pbar):
            problem_id = problem['problem_id']

            # Process problem
            result_or_error = self.process_problem(problem)

            if result_or_error is None:
                # Error occurred but was already logged
                self.stats['errors'] += 1
                continue

            if isinstance(result_or_error, tuple):
                # Error case
                result, error = result_or_error
                errors.append(error)
                self.stats['errors'] += 1
            else:
                # Success case
                result = result_or_error
                results.append(result)
                self.stats['processed'] += 1

                if result['correct']:
                    self.stats['correct_predictions'] += 1
                else:
                    self.stats['incorrect_predictions'] += 1

            # Update progress bar
            elapsed = time.time() - start_time
            rate = (len(results) + len(errors)) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'correct': self.stats['correct_predictions'],
                'incorrect': self.stats['incorrect_predictions'],
                'errors': self.stats['errors'],
                'rate': f"{rate:.2f} prob/s"
            })

            # Checkpoint periodically
            if (i + 1) % self.config.checkpoint_interval == 0:
                print(f"\n💾 Saving checkpoint at {len(results)} problems...")
                self.save_checkpoint(results, errors)

        pbar.close()

        # Final save
        self.stats['end_time'] = datetime.now().isoformat()
        elapsed_total = time.time() - start_time
        print(f"\n💾 Saving final results...")
        self.save_checkpoint(results, errors)

        # Finalize
        self._finalize_results(results, errors)

        # Print summary
        print("\n" + "="*80)
        print("ATTRIBUTION SWEEP COMPLETE")
        print("="*80)
        print(f"\n📊 Statistics:")
        print(f"   Total problems: {self.stats['total_problems']}")
        print(f"   Successfully processed: {self.stats['processed']}")
        print(f"   Correct predictions: {self.stats['correct_predictions']}")
        print(f"   Incorrect predictions: {self.stats['incorrect_predictions']}")
        print(f"   Errors: {self.stats['errors']}")
        print(f"   Success rate: {100 * self.stats['processed'] / max(self.stats['total_problems'], 1):.1f}%")
        print(f"\n⏱️  Time:")
        print(f"   Total: {elapsed_total / 3600:.2f} hours")
        print(f"   Rate: {self.stats['processed'] / elapsed_total * 60:.2f} problems/min")
        print(f"\n📁 Output files:")
        print(f"   Results: {self.results_file}")
        print(f"   Errors: {self.errors_file}")
        print(f"   Checkpoint: {self.checkpoint_file}")
        print("="*80)

    def _finalize_results(self, results: List[Dict], errors: List[Dict]):
        """
        Finalize and save results with aggregation.

        Args:
            results: List of attribution results
            errors: List of error records
        """
        # Save individual results
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Save errors
        if errors:
            with open(self.errors_file, 'w') as f:
                json.dump(errors, f, indent=2)

        # Compute and save aggregate statistics
        if results:
            self._compute_aggregate_stats(results)

    def _compute_aggregate_stats(self, results: List[Dict]):
        """
        Compute aggregate statistics across all results.

        Args:
            results: List of attribution results
        """
        print(f"\n📊 Computing aggregate statistics...")

        # Separate by correct/incorrect
        correct_results = [r for r in results if r['correct']]
        incorrect_results = [r for r in results if not r['correct']]

        # Aggregate by layer and token
        layers = self.config.target_layers
        tokens = self.config.target_tokens

        # Initialize accumulators
        ablation_by_layer = {layer: np.zeros(len(tokens)) for layer in layers}
        ablation_by_token = np.zeros(len(tokens))

        for result in correct_results:
            for layer in layers:
                layer_key = f"layer_{layer}"
                if layer_key in result['ablation_attributions']:
                    scores = result['ablation_attributions'][layer_key]
                    ablation_by_layer[layer] += np.array(scores)
                    ablation_by_token += np.array(scores)

        # Compute means
        n_correct = len(correct_results)
        if n_correct > 0:
            ablation_by_layer = {k: (v / n_correct).tolist() for k, v in ablation_by_layer.items()}
            ablation_by_token = (ablation_by_token / (n_correct * len(layers))).tolist()

        # Create aggregate statistics
        aggregate_stats = {
            'n_problems': len(results),
            'n_correct': len(correct_results),
            'n_incorrect': len(incorrect_results),
            'accuracy': len(correct_results) / len(results) if results else 0,
            'mean_ablation_by_layer': ablation_by_layer,
            'mean_ablation_by_token': ablation_by_token,
            'mean_baseline_logit_diff': np.mean([r['baseline_logit_diff'] for r in correct_results]) if correct_results else 0,
            'mean_total_importance': np.mean([r['total_importance'] for r in correct_results]) if correct_results else 0,
            'config': {
                'layers': layers,
                'tokens': tokens,
                'method': 'ablation'
            }
        }

        # Save aggregate stats
        aggregate_file = self.output_dir / "aggregate_stats.json"
        with open(aggregate_file, 'w') as f:
            json.dump(aggregate_stats, f, indent=2)

        print(f"   Saved to: {aggregate_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run gradient attribution sweep on LLaMA steering dataset"
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
        help='Path to CODI LLaMA checkpoint'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching/results/steering_dataset_llama_full.json',
        help='Path to steering dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/home/paperspace/dev/CoT_Exploration/src/experiments/codi_attention_interp/results/gradient_attribution',
        help='Output directory for results'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default='4,5,6,7,8,9,10,11',
        help='Comma-separated list of layer indices (default: 4-11)'
    )
    parser.add_argument(
        '--tokens',
        type=str,
        default='0,1,2,3,4,5',
        help='Comma-separated list of token indices (default: 0-5)'
    )
    parser.add_argument(
        '--checkpoint_interval',
        type=int,
        default=10,
        help='Save checkpoint every N problems (default: 10)'
    )
    parser.add_argument(
        '--test_mode',
        action='store_true',
        help='Run on small subset for testing'
    )
    parser.add_argument(
        '--num_problems',
        type=int,
        default=10,
        help='Number of problems for test mode (default: 10)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint (default: auto-detect)'
    )

    args = parser.parse_args()

    # Parse layer and token lists
    target_layers = [int(x) for x in args.layers.split(',')]
    target_tokens = [int(x) for x in args.tokens.split(',')]

    # Create config
    config = ExperimentConfig(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_layers=target_layers,
        target_tokens=target_tokens,
        checkpoint_interval=args.checkpoint_interval,
        test_mode=args.test_mode,
        num_test_problems=args.num_problems
    )

    # Create runner and execute
    runner = GradientAttributionRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
