#!/usr/bin/env python3
"""
Individual Head Ablation - Story 3

Test whether heads 2-10 are individually critical (ablating each one separately).
This answers: "Is L4H5 the only bottleneck, or are other heads also critical?"

Usage:
    python 3_ablate_individual_heads.py [--model MODEL] [--n_problems N]

Output:
    ../results/{model}_ablation_individual_heads.json
"""
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from utils import load_model, extract_answer, validate_model_architecture, get_model_architecture_info, get_attention_output_layer
from typing import List, Tuple, Set


class AttentionAblator:
    """
    Hook-based attention head ablation.

    Zeros out specific attention heads by intervening on attention output projections.
    """

    def __init__(self, model, model_name: str, tokenizer):
        """
        Initialize ablator.

        Args:
            model: CODI model
            model_name: 'llama' or 'gpt2'
            tokenizer: Model tokenizer
        """
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Get architecture info
        self.arch_info = get_model_architecture_info(model_name)
        self.n_layers = self.arch_info['n_layers']
        self.n_heads = self.arch_info['n_heads']
        self.hidden_dim = self.arch_info['hidden_dim']
        self.head_dim = self.hidden_dim // self.n_heads

        # Ablation state
        self.heads_to_ablate: Set[Tuple[int, int]] = set()
        self.hook_handles = []

    def _get_attention_output_layer(self, layer_idx: int):
        """Get attention output projection layer."""
        return get_attention_output_layer(self.model.codi, layer_idx, self.model_name)

    def _create_ablation_hook(self, layer_idx: int):
        """
        Create hook that zeros out specific heads at this layer.

        Args:
            layer_idx: Layer index

        Returns:
            Hook function
        """
        def hook(module, input, output):
            # Output shape: [batch, seq_len, hidden_dim]
            batch_size, seq_len, _ = output.shape

            # Reshape to heads: [batch, seq_len, n_heads, head_dim]
            output_heads = output.view(batch_size, seq_len, self.n_heads, self.head_dim)

            # Zero out specified heads
            for head_idx in range(self.n_heads):
                if (layer_idx, head_idx) in self.heads_to_ablate:
                    output_heads[:, :, head_idx, :] = 0.0

            # Reshape back: [batch, seq_len, hidden_dim]
            output_ablated = output_heads.view(batch_size, seq_len, self.hidden_dim)

            return output_ablated

        return hook

    def register_hooks(self, heads_to_ablate: List[Tuple[int, int]]):
        """
        Register ablation hooks for specified heads.

        Args:
            heads_to_ablate: List of (layer_idx, head_idx) tuples
        """
        self.heads_to_ablate = set(heads_to_ablate)

        # Get unique layers
        layers_to_hook = set(layer_idx for layer_idx, _ in heads_to_ablate)

        # Register hooks
        for layer_idx in layers_to_hook:
            layer = self._get_attention_output_layer(layer_idx)
            hook = self._create_ablation_hook(layer_idx)
            handle = layer.register_forward_hook(hook)
            self.hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.heads_to_ablate = set()

    def generate_with_ablation(self, question: str, max_new_tokens: int = 100) -> str:
        """
        Generate answer with ablation active.

        Args:
            question: Input question
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer text
        """
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

        # Generate BOT token
        bot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Generate 6 continuous thought tokens (WITH ABLATION)
        for step in range(6):
            outputs = self.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
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


def load_top_heads(model_name: str) -> List[dict]:
    """Load top 10 critical heads from Phase 2 results."""
    # Load from top 10 ablation results (which contains the ranked list)
    results_dir = Path(__file__).parent.parent / 'results'
    results_file = results_dir / f'{model_name}_ablation_top10.json'

    with open(results_file) as f:
        data = json.load(f)

    return data['ablated_heads']


def run_individual_ablation_experiment(
    model_name: str = 'llama',
    n_problems: int = 100
) -> dict:
    """
    Run individual head ablation for heads 2-10.

    Tests each head separately to see which ones are individually critical.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Individual Head Ablation - Story 3")
    print(f"Model: {model_name}")
    print(f"Problems: {n_problems}")
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

    # Load top heads
    print("\nLoading top critical heads...")
    top_heads = load_top_heads(model_name)
    print(f"Loaded {len(top_heads)} heads")

    # Skip head 1 (L4H5) since we already know it causes 100% failure
    heads_to_test = top_heads[1:10]  # Heads 2-10

    print(f"\nTesting {len(heads_to_test)} heads individually:")
    for i, head in enumerate(heads_to_test, start=2):
        print(f"  {i}. L{head['layer']}H{head['head']} (score: {head['composite_score']:.4f})")

    # Initialize ablator
    ablator = AttentionAblator(model, model_name, tokenizer)

    # Results storage
    all_results = {
        'model': model_name,
        'n_problems': len(test_problems),
        'heads_tested': [],
        'individual_results': []
    }

    # Test each head individually
    for head_idx, head_info in enumerate(heads_to_test, start=2):
        layer_idx = head_info['layer']
        head_num = head_info['head']
        composite_score = head_info['composite_score']

        print(f"\n{'='*60}")
        print(f"Testing Head {head_idx}/10: L{layer_idx}H{head_num}")
        print(f"Composite Score: {composite_score:.4f}")
        print(f"{'='*60}\n")

        # Register hook for this head only
        ablator.register_hooks([(layer_idx, head_num)])

        # Test on all problems
        n_correct = 0
        n_failed = 0
        results_detail = []

        for problem in tqdm(test_problems, desc=f"L{layer_idx}H{head_num}"):
            question = problem['question']
            gold_answer = extract_answer(problem['answer'])

            try:
                # Generate with ablation
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

        head_result = {
            'rank': head_idx,
            'layer': layer_idx,
            'head': head_num,
            'composite_score': composite_score,
            'n_correct': n_correct,
            'n_failed': n_failed,
            'accuracy': accuracy,
            'accuracy_drop_from_baseline': 0.59 - accuracy,  # Baseline was 59%
            'results_detail': results_detail[:10]  # Save first 10 for inspection
        }

        all_results['individual_results'].append(head_result)
        all_results['heads_tested'].append({
            'rank': head_idx,
            'layer': layer_idx,
            'head': head_num,
            'composite_score': composite_score
        })

        print(f"\nResults for L{layer_idx}H{head_num}:")
        print(f"  Correct: {n_correct}/{len(test_problems)}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Drop from baseline: {head_result['accuracy_drop_from_baseline']:.2%}")

        # Remove hooks for next test
        ablator.remove_hooks()

    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY: Individual Head Ablation Results")
    print(f"{'='*60}\n")

    print(f"Baseline accuracy: 59.00%")
    print(f"\nIndividual head ablation results:")
    print(f"{'Rank':<6} {'Head':<10} {'Score':<8} {'Accuracy':<10} {'Drop':<10}")
    print('-' * 50)

    for result in all_results['individual_results']:
        rank = result['rank']
        head_str = f"L{result['layer']}H{result['head']}"
        score = result['composite_score']
        acc = result['accuracy']
        drop = result['accuracy_drop_from_baseline']
        print(f"{rank:<6} {head_str:<10} {score:<8.4f} {acc:<10.2%} {drop:<10.2%}")

    # Add summary stats
    accuracies = [r['accuracy'] for r in all_results['individual_results']]
    all_results['summary'] = {
        'baseline_accuracy': 0.59,
        'n_heads_tested': len(heads_to_test),
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'min_accuracy': float(np.min(accuracies)),
        'max_accuracy': float(np.max(accuracies)),
        'mean_drop': float(np.mean([r['accuracy_drop_from_baseline'] for r in all_results['individual_results']])),
        'critical_heads_count': sum(1 for acc in accuracies if acc < 0.30)  # <30% = critical
    }

    print(f"\nSummary Statistics:")
    print(f"  Mean accuracy: {all_results['summary']['mean_accuracy']:.2%} ± {all_results['summary']['std_accuracy']:.2%}")
    print(f"  Range: {all_results['summary']['min_accuracy']:.2%} - {all_results['summary']['max_accuracy']:.2%}")
    print(f"  Mean drop: {all_results['summary']['mean_drop']:.2%}")
    print(f"  Heads causing >50% drop (critical): {all_results['summary']['critical_heads_count']}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Individual head ablation experiment')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    parser.add_argument('--n_problems', type=int, default=100)
    args = parser.parse_args()

    # Run experiment
    results = run_individual_ablation_experiment(
        model_name=args.model,
        n_problems=args.n_problems
    )

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f'{args.model}_ablation_individual_heads.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
