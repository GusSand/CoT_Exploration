#!/usr/bin/env python3
"""
GPT-2 Activation Steering Experiment

This script:
1. Loads test set (43 correct + 43 wrong = 86 problems)
2. Runs baseline inference (no steering)
3. Runs steered inference with amplification (alpha > 0)
4. Runs steered inference with suppression (alpha < 0)
5. Analyzes results and generates report

Expected outcomes:
- Baseline: ~50% accuracy (mix of correct/wrong)
- Amplified: 62%+ accuracy (+12 points)
- Suppressed: 38% or lower (-12 points)
"""

import json
import sys
import re
import torch
import numpy as np
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


class SteeringInference:
    """Run inference with activation steering."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize with CODI model."""
        print("Initializing Steering Inference Engine...")
        self.cacher = ActivationCacher(model_path, device=device)
        self.device = device
        self.layer_idx = LAYER_CONFIG['middle']

    def run_with_steering(
        self,
        problem_text: str,
        steering_direction: np.ndarray,
        alpha: float = 0.0,
        max_new_tokens: int = 200
    ) -> str:
        """Run inference with steering applied to continuous thoughts.

        Args:
            problem_text: Problem question
            steering_direction: Direction to steer [6, 768]
            alpha: Amplification factor (0 = no steering, >0 = amplify, <0 = suppress)
            max_new_tokens: Max tokens to generate

        Returns:
            Generated output text
        """
        with torch.no_grad():
            # Convert steering direction to tensor
            steering_tensor = torch.tensor(
                steering_direction, dtype=torch.float32, device=self.device
            )

            # Tokenize input
            inputs = self.cacher.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.cacher.model.get_embd(
                self.cacher.model.codi,
                self.cacher.model.model_name
            )(input_ids).to(self.device)

            # Forward through model to get initial context
            outputs = self.cacher.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # Get BOT (Beginning of Thought) embedding
            bot_emb = self.cacher.model.get_embd(
                self.cacher.model.codi,
                self.cacher.model.model_name
            )(
                torch.tensor([self.cacher.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Process latent thoughts with steering
            latent_embd = bot_emb

            for latent_step in range(self.cacher.num_latent):
                outputs = self.cacher.model.codi(
                    inputs_embeds=latent_embd,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )

                past_key_values = outputs.past_key_values

                # Get next latent embedding
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # APPLY STEERING HERE
                if alpha != 0.0:
                    # Add steering to this token's representation
                    steer_vector = steering_tensor[latent_step].unsqueeze(0).unsqueeze(0)  # [1, 1, 768]
                    latent_embd = latent_embd + alpha * steer_vector

                # Apply projection if model uses it
                if self.cacher.model.use_prj:
                    latent_embd = self.cacher.model.prj(latent_embd)

            # Get EOT (End of Thought) embedding
            eot_emb = self.cacher.model.get_embd(
                self.cacher.model.codi,
                self.cacher.model.model_name
            )(
                torch.tensor([self.cacher.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Forward through EOT
            outputs = self.cacher.model.codi(
                inputs_embeds=eot_emb,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values

            # Generate answer
            generated_ids = self.cacher.model.codi.generate(
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,
                eos_token_id=self.cacher.tokenizer.eos_token_id,
                pad_token_id=self.cacher.tokenizer.pad_token_id,
                do_sample=False
            )

            # Decode
            output_text = self.cacher.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )

            return output_text


def load_test_set():
    """Load test problems."""
    print("="*80)
    print("LOADING TEST SET")
    print("="*80)

    dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_gpt2.json'
    problem_pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'

    with open(dataset_file) as f:
        dataset = json.load(f)

    with open(problem_pairs_file) as f:
        all_pairs = json.load(f)

    # Create lookup
    pairs_lookup = {p['pair_id']: p for p in all_pairs}

    # Get test problems
    test_correct = dataset['test_correct']
    test_wrong = dataset['test_wrong']

    # Add problem text
    for prob in test_correct + test_wrong:
        pair = pairs_lookup[prob['pair_id']]
        prob['question'] = pair['clean']['question']

    print(f"\nTest CORRECT: {len(test_correct)} problems")
    print(f"Test WRONG: {len(test_wrong)} problems")
    print(f"Total test set: {len(test_correct) + len(test_wrong)} problems")

    return test_correct, test_wrong


def load_steering_direction():
    """Load computed reasoning direction."""
    print("\n" + "="*80)
    print("LOADING STEERING DIRECTION")
    print("="*80)

    direction_file = Path(__file__).parent / 'results' / 'steering_activations' / 'reasoning_direction.npz'

    data = np.load(direction_file)
    direction = data['direction']  # [6, 768]

    print(f"\nDirection shape: {direction.shape}")
    print(f"Direction magnitude: {np.linalg.norm(direction):.4f}")

    return direction


def run_experiments(engine, test_correct, test_wrong, direction):
    """Run all steering experiments."""
    print("\n" + "="*80)
    print("RUNNING STEERING EXPERIMENTS")
    print("="*80)

    # Combine test sets
    all_test = test_correct + test_wrong

    # Alpha values to test
    alphas = [0.0] + [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] + [-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]

    results = {alpha: [] for alpha in alphas}

    print(f"\nTesting {len(alphas)} alpha values on {len(all_test)} problems")
    print(f"Alpha values: {alphas}\n")

    for alpha in alphas:
        print(f"\n{'='*80}")
        print(f"Alpha = {alpha:+.1f} {'(BASELINE)' if alpha == 0 else '(AMPLIFIED)' if alpha > 0 else '(SUPPRESSED)'}")
        print(f"{'='*80}")

        correct_count = 0

        for prob in tqdm(all_test, desc=f"Alpha {alpha:+.1f}"):
            try:
                # Run inference with steering
                output = engine.run_with_steering(
                    problem_text=prob['question'],
                    steering_direction=direction,
                    alpha=alpha,
                    max_new_tokens=200
                )

                # Extract and check answer
                predicted = extract_answer_number(output)
                expected = prob['expected']
                correct = answers_match(predicted, expected)

                if correct:
                    correct_count += 1

                # Store result
                results[alpha].append({
                    'pair_id': prob['pair_id'],
                    'expected': expected,
                    'predicted': predicted,
                    'correct': correct,
                    'output': output
                })

            except Exception as e:
                print(f"\nError on pair {prob['pair_id']}: {e}")
                results[alpha].append({
                    'pair_id': prob['pair_id'],
                    'error': str(e)
                })

        accuracy = 100 * correct_count / len(all_test)
        print(f"\nAlpha {alpha:+.1f}: {correct_count}/{len(all_test)} correct ({accuracy:.1f}%)")

    return results, alphas


def analyze_results(results, alphas):
    """Analyze and summarize results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    summary = {}

    for alpha in alphas:
        result_list = results[alpha]
        total = len(result_list)
        correct = sum(1 for r in result_list if r.get('correct', False))
        accuracy = 100 * correct / total if total > 0 else 0

        summary[alpha] = {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        }

        status = ""
        if alpha == 0:
            status = " (BASELINE)"
        elif alpha > 0:
            status = " (AMPLIFIED)"
        else:
            status = " (SUPPRESSED)"

        print(f"Alpha {alpha:+5.1f}{status:15s}: {correct:3d}/{total:3d} = {accuracy:5.1f}%")

    # Calculate improvements
    baseline_acc = summary[0.0]['accuracy']
    best_amplified_alpha = max([a for a in alphas if a > 0], key=lambda a: summary[a]['accuracy'])
    best_amplified_acc = summary[best_amplified_alpha]['accuracy']
    worst_suppressed_alpha = min([a for a in alphas if a < 0], key=lambda a: summary[a]['accuracy'])
    worst_suppressed_acc = summary[worst_suppressed_alpha]['accuracy']

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"Baseline (α=0.0):           {baseline_acc:.1f}%")
    print(f"Best Amplified (α={best_amplified_alpha:+.1f}):  {best_amplified_acc:.1f}% ({best_amplified_acc - baseline_acc:+.1f} points)")
    print(f"Worst Suppressed (α={worst_suppressed_alpha:+.1f}): {worst_suppressed_acc:.1f}% ({worst_suppressed_acc - baseline_acc:+.1f} points)")

    # Success criteria check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)

    amplified_improvement = best_amplified_acc - baseline_acc
    suppressed_degradation = baseline_acc - worst_suppressed_acc

    if amplified_improvement >= 12:
        print(f"✅ AMPLIFIED improvement: {amplified_improvement:+.1f} points (target: +12)")
    else:
        print(f"❌ AMPLIFIED improvement: {amplified_improvement:+.1f} points (target: +12)")

    if suppressed_degradation >= 12:
        print(f"✅ SUPPRESSED degradation: {suppressed_degradation:.1f} points (target: -12)")
    else:
        print(f"❌ SUPPRESSED degradation: {suppressed_degradation:.1f} points (target: -12)")

    return summary


def save_results(results, summary, alphas):
    """Save experiment results."""
    output_dir = Path(__file__).parent / 'results' / 'steering_experiments'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = output_dir / 'steering_results_detailed.json'
    with open(results_file, 'w') as f:
        json.dump({
            'alphas': alphas,
            'results': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    print(f"\n✓ Saved detailed results: {results_file}")

    # Save summary
    summary_file = output_dir / 'steering_results_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': {str(k): v for k, v in summary.items()}
        }, f, indent=2)
    print(f"✓ Saved summary: {summary_file}")


def main():
    """Main experiment pipeline."""
    print("="*80)
    print("GPT-2 ACTIVATION STEERING EXPERIMENT")
    print("="*80)

    # Load data
    test_correct, test_wrong = load_test_set()
    direction = load_steering_direction()

    # Initialize engine
    model_path = str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    engine = SteeringInference(model_path)

    # Run experiments
    results, alphas = run_experiments(engine, test_correct, test_wrong, direction)

    # Analyze
    summary = analyze_results(results, alphas)

    # Save
    save_results(results, summary, alphas)

    print("\n" + "="*80)
    print("✅ STEERING EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
