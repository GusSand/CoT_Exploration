#!/usr/bin/env python3
"""
Score-Stratified Head Ablation - Story 4

Test whether criticality is unique to top 10 heads or if any head causes failure.

Sample heads across different composite score ranges:
- Ranks 11-20 (just below top 10)
- Ranks 50-60 (middle tier)
- Ranks 100-110 (lower tier)
- Ranks 500-510 (bottom tier)

Usage:
    python 4_ablate_score_stratified_heads.py [--model MODEL] [--n_problems N]

Output:
    ../results/{model}_ablation_stratified_heads.json
"""
import json
import argparse
import torch
import numpy as np
import pandas as pd
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

        # Generate answer (manual token-by-token)
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


def load_stratified_heads(model_name: str) -> dict:
    """
    Load heads stratified by score range.

    Returns dict with keys: 'ranks_11_20', 'ranks_50_60', 'ranks_100_110', 'ranks_500_510'
    """
    # Load ranked heads CSV
    results_dir = Path(__file__).parent.parent / 'results' / model_name
    ranked_file = results_dir / 'ranked_heads.csv'

    df = pd.read_csv(ranked_file)

    # Sample from different score ranges
    stratified = {}

    # Ranks 11-20 (just below top 10)
    stratified['ranks_11_20'] = df.iloc[10:20][['layer', 'head', 'composite_score']].to_dict('records')

    # Ranks 50-60 (middle tier)
    stratified['ranks_50_60'] = df.iloc[49:59][['layer', 'head', 'composite_score']].to_dict('records')

    # Ranks 100-110 (lower tier)
    stratified['ranks_100_110'] = df.iloc[99:109][['layer', 'head', 'composite_score']].to_dict('records')

    # Ranks 500-510 (bottom tier) - near zero scores
    stratified['ranks_500_510'] = df.iloc[499:509][['layer', 'head', 'composite_score']].to_dict('records')

    return stratified


def run_stratified_ablation_experiment(
    model_name: str = 'llama',
    n_problems: int = 100
) -> dict:
    """
    Run stratified head ablation across score ranges.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Score-Stratified Head Ablation - Story 4")
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

    # Load stratified heads
    print("\nLoading stratified heads...")
    stratified_heads = load_stratified_heads(model_name)

    print(f"\nTesting {sum(len(v) for v in stratified_heads.values())} heads across 4 strata:")
    for stratum, heads in stratified_heads.items():
        scores = [h['composite_score'] for h in heads]
        print(f"  {stratum}: {len(heads)} heads, scores {min(scores):.4f} - {max(scores):.4f}")

    # Initialize ablator
    ablator = AttentionAblator(model, model_name, tokenizer)

    # Results storage
    all_results = {
        'model': model_name,
        'n_problems': len(test_problems),
        'baseline_accuracy': 0.59,  # From Story 0
        'strata_results': {}
    }

    # Test each stratum
    for stratum_name, stratum_heads in stratified_heads.items():
        print(f"\n{'='*60}")
        print(f"Testing Stratum: {stratum_name}")
        print(f"{'='*60}\n")

        stratum_results = []

        # Test each head in this stratum
        for i, head_info in enumerate(stratum_heads, start=1):
            layer_idx = int(head_info['layer'])
            head_num = int(head_info['head'])
            composite_score = head_info['composite_score']

            print(f"\nTesting {i}/{len(stratum_heads)}: L{layer_idx}H{head_num} (score: {composite_score:.6f})")

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
                    n_failed += 1
                    results_detail.append({
                        'question': question[:100] + '...',
                        'gold_answer': gold_answer,
                        'pred_answer': None,
                        'correct': False
                    })

            accuracy = n_correct / len(test_problems)

            head_result = {
                'layer': layer_idx,
                'head': head_num,
                'composite_score': composite_score,
                'n_correct': n_correct,
                'n_failed': n_failed,
                'accuracy': accuracy,
                'accuracy_drop_from_baseline': 0.59 - accuracy,
                'results_detail': results_detail[:5]  # Save first 5 for inspection
            }

            stratum_results.append(head_result)

            print(f"  Accuracy: {accuracy:.2%} (drop: {head_result['accuracy_drop_from_baseline']:.2%})")

            # Remove hooks for next test
            ablator.remove_hooks()

        all_results['strata_results'][stratum_name] = stratum_results

    # Summary statistics by stratum
    print(f"\n{'='*60}")
    print("SUMMARY: Score-Stratified Ablation Results")
    print(f"{'='*60}\n")

    print(f"Baseline accuracy: 59.00%\n")

    for stratum_name, stratum_results in all_results['strata_results'].items():
        accuracies = [r['accuracy'] for r in stratum_results]
        drops = [r['accuracy_drop_from_baseline'] for r in stratum_results]

        print(f"{stratum_name}:")
        print(f"  Mean accuracy: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
        print(f"  Range: {np.min(accuracies):.2%} - {np.max(accuracies):.2%}")
        print(f"  Mean drop: {np.mean(drops):.2%}")
        print(f"  Critical heads (>50% drop): {sum(1 for acc in accuracies if acc < 0.30)}/{len(accuracies)}\n")

    # Overall summary
    all_accuracies = []
    all_scores = []
    for stratum_results in all_results['strata_results'].values():
        for r in stratum_results:
            all_accuracies.append(r['accuracy'])
            all_scores.append(r['composite_score'])

    all_results['summary'] = {
        'total_heads_tested': len(all_accuracies),
        'mean_accuracy': float(np.mean(all_accuracies)),
        'std_accuracy': float(np.std(all_accuracies)),
        'min_accuracy': float(np.min(all_accuracies)),
        'max_accuracy': float(np.max(all_accuracies)),
        'critical_heads_count': sum(1 for acc in all_accuracies if acc < 0.30),
        'score_range': [float(min(all_scores)), float(max(all_scores))]
    }

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Score-stratified head ablation experiment')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    parser.add_argument('--n_problems', type=int, default=100)
    args = parser.parse_args()

    # Run experiment
    results = run_stratified_ablation_experiment(
        model_name=args.model,
        n_problems=args.n_problems
    )

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f'{args.model}_ablation_stratified_heads.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
