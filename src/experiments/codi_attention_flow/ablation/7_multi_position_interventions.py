#!/usr/bin/env python3
"""
Multi-Position Attention Interventions

Research Question: What happens when multiple CT positions are blocked simultaneously?

Tests:
- CT0 + CT1: Both early positions
- CT0 + CT2: CT0 + middle position
- CT0 + CT4: CT0 + late position (control comparison)
- CT1 + CT2: Non-CT0 combinations
- CT2 + CT3: Middle positions
- CT0 + CT1 + CT2: Three positions

Expected outcomes:
- Additive: Combined drop ≈ sum of individual drops
- Super-additive: Combined drop > sum (synergistic)
- Sub-additive: Combined drop < sum (compensation)

Time: 2-3 hours
"""

import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Optional, Callable

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_model, extract_answer, validate_model_architecture


class MultiPositionAblator:
    """Block attention to multiple CT positions simultaneously."""

    def __init__(self, model, model_name: str, tokenizer):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.positions_to_block: List[int] = []

    def set_positions(self, positions: List[int]):
        """
        Set which CT positions to block.

        Args:
            positions: List of position indices (0-5) to block
        """
        self.positions_to_block = positions

    def _create_ct_attention_mask(self, full_seq_len: int, ct_start: int, ct_end: int, current_pos: int) -> torch.Tensor:
        """
        Create attention mask blocking multiple positions.

        Returns mask where blocked positions have -10000.0
        """
        # Start with no masking
        mask = torch.zeros(1, full_seq_len, device=self.device)

        # Block attention TO specified CT positions
        for pos_idx in self.positions_to_block:
            absolute_pos = ct_start + pos_idx
            if absolute_pos < full_seq_len:
                mask[0, absolute_pos] = -10000.0

        return mask

    def generate_with_ablation(self, question: str, max_new_tokens: int = 100) -> str:
        """Generate answer with multi-position blocking active."""

        # Tokenize question
        inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']

        # Question forward pass
        question_outputs = self.model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = question_outputs.past_key_values
        question_len = input_ids.shape[1]

        # Generate BOT token
        bot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        latent_embd = bot_emb
        ct_start_pos = question_len + 1  # After question + BOT

        # Generate 6 continuous thought tokens WITH MULTI-POSITION BLOCKING
        for step in range(6):
            current_pos = ct_start_pos + step
            ct_end_pos = ct_start_pos + 6  # Will have 6 CTs eventually
            full_seq_len = current_pos + 1

            # Create attention mask blocking specified positions
            attention_mask = self._create_ct_attention_mask(
                full_seq_len, ct_start_pos, ct_end_pos, current_pos
            )

            outputs = self.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=attention_mask if len(self.positions_to_block) > 0 else None
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.model.use_prj:
                latent_embd = self.model.prj(latent_embd)

        # Generate EOT token
        eot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        output_emb = eot_emb
        pred_tokens = []

        for _ in range(max_new_tokens):
            out = self.model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :self.model.codi.config.vocab_size-1]
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = self.model.get_embd(self.model.codi, self.model_name)(
                next_token_id
            ).unsqueeze(1)

        # Decode answer
        generated_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
        return generated_text.strip()


def load_single_position_results():
    """Load existing single-position ablation results."""
    results_dir = Path(__file__).parent.parent / 'results'

    single_pos_results = {}
    for pos in range(6):
        result_file = results_dir / f'llama_attention_pattern_position_{pos}.json'
        if result_file.exists():
            with open(result_file) as f:
                single_pos_results[f'CT{pos}'] = json.load(f)

    # Also load baseline
    baseline_file = results_dir / 'llama_baseline.json'
    if baseline_file.exists():
        with open(baseline_file) as f:
            single_pos_results['baseline'] = json.load(f)

    return single_pos_results


def run_multi_position_experiment(
    model_name: str = 'llama',
    n_problems: int = 100,
    position_combinations: List[List[int]] = None
) -> dict:
    """
    Run multi-position intervention experiments.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems
        position_combinations: List of position lists to test

    Returns:
        Results dict with all combinations tested
    """

    if position_combinations is None:
        # Default combinations to test
        position_combinations = [
            [0, 1],      # CT0 + CT1 (both early)
            [0, 2],      # CT0 + CT2
            [0, 4],      # CT0 + CT4 (CT0 + late)
            [1, 2],      # CT1 + CT2 (non-CT0)
            [2, 3],      # CT2 + CT3 (middle positions)
            [0, 1, 2],   # Three positions
        ]

    print(f"\n{'='*60}")
    print(f"Multi-Position Attention Interventions")
    print(f"Model: {model_name}")
    print(f"Problems: {n_problems}")
    print(f"Combinations: {len(position_combinations)}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Validate architecture
    print("\nValidating model architecture...")
    validate_model_architecture(model, model_name, tokenizer)
    print("✓ Architecture validated")

    # Load dataset
    print("\nLoading GSM8K test set...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = list(dataset.select(range(min(n_problems, len(dataset)))))
    print(f"Loaded {len(test_problems)} problems")

    # Load single-position results for comparison
    print("\nLoading single-position results for comparison...")
    single_results = load_single_position_results()
    baseline_acc = single_results.get('baseline', {}).get('accuracy', 0.0)
    print(f"Baseline accuracy: {baseline_acc:.2%}")

    # Single position drops
    single_drops = {}
    for pos in range(6):
        key = f'CT{pos}'
        if key in single_results:
            single_acc = single_results[key].get('accuracy', 0.0)
            single_drops[pos] = baseline_acc - single_acc
            print(f"  CT{pos} single drop: {single_drops[pos]:.2%}")

    # Initialize ablator
    ablator = MultiPositionAblator(model, model_name, tokenizer)

    # Results storage
    all_results = {
        'model': model_name,
        'n_problems': len(test_problems),
        'baseline_accuracy': baseline_acc,
        'single_position_drops': single_drops,
        'multi_position_results': []
    }

    # Test each combination
    for combo in position_combinations:
        combo_str = '+'.join([f'CT{p}' for p in combo])

        print(f"\n{'='*60}")
        print(f"Testing: {combo_str}")
        print(f"Positions: {combo}")
        print(f"{'='*60}\n")

        # Set positions to block
        ablator.set_positions(combo)

        # Test on problems
        n_correct = 0
        n_failed = 0
        results_detail = []

        for problem in tqdm(test_problems, desc=combo_str):
            question = problem['question']
            gold_answer = extract_answer(problem['answer'])

            try:
                # Generate with multi-position blocking
                generated_text = ablator.generate_with_ablation(question)
                pred_answer = extract_answer(generated_text)

                correct = (pred_answer == gold_answer)
                if correct:
                    n_correct += 1

                results_detail.append({
                    'question': question[:100] + '...',
                    'gold_answer': gold_answer,
                    'pred_answer': pred_answer,
                    'correct': correct
                })

            except Exception as e:
                print(f"\nFailed on problem: {e}")
                n_failed += 1
                results_detail.append({
                    'question': question[:100] + '...',
                    'gold_answer': gold_answer,
                    'pred_answer': None,
                    'correct': False
                })

        accuracy = n_correct / len(test_problems)
        accuracy_drop = baseline_acc - accuracy

        # Compute expected drop (sum of individual drops)
        expected_drop = sum(single_drops.get(p, 0.0) for p in combo)

        # Interaction effect
        interaction = accuracy_drop - expected_drop

        combo_result = {
            'combination': combo,
            'combination_str': combo_str,
            'n_positions': len(combo),
            'n_correct': n_correct,
            'n_failed': n_failed,
            'accuracy': accuracy,
            'accuracy_drop': accuracy_drop,
            'expected_drop_additive': expected_drop,
            'interaction_effect': interaction,
            'interaction_type': 'super-additive' if interaction > 0.02 else ('sub-additive' if interaction < -0.02 else 'additive'),
            'results_detail': results_detail[:10]  # Save first 10
        }

        all_results['multi_position_results'].append(combo_result)

        print(f"\nResults for {combo_str}:")
        print(f"  Correct: {n_correct}/{len(test_problems)}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Drop from baseline: {accuracy_drop:.2%}")
        print(f"  Expected drop (additive): {expected_drop:.2%}")
        print(f"  Interaction effect: {interaction:+.2%}")
        print(f"  Type: {combo_result['interaction_type']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Multi-Position Interaction Effects")
    print(f"{'='*60}\n")

    print(f"Baseline: {baseline_acc:.2%}\n")
    print(f"{'Combination':<15} {'Accuracy':<10} {'Drop':<10} {'Expected':<10} {'Interaction':<12} {'Type':<15}")
    print('-' * 80)

    for result in all_results['multi_position_results']:
        print(f"{result['combination_str']:<15} {result['accuracy']:<10.2%} {result['accuracy_drop']:<10.2%} "
              f"{result['expected_drop_additive']:<10.2%} {result['interaction_effect']:+<12.2%} {result['interaction_type']:<15}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Multi-position attention interventions')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    parser.add_argument('--n_problems', type=int, default=100)
    args = parser.parse_args()

    # Run experiment
    results = run_multi_position_experiment(
        model_name=args.model,
        n_problems=args.n_problems
    )

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f'{args.model}_multi_position_interventions.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
