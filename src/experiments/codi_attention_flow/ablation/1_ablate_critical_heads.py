#!/usr/bin/env python3
"""
Critical Head Ablation - Story 1

Test causal necessity of critical attention heads by zeroing their outputs.

Usage:
    python 1_ablate_critical_heads.py [--model MODEL] [--n_problems N] [--top_n TOP_N]

Output:
    ../results/{model}_ablation_top{n}.json
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
            target_layer = self._get_attention_output_layer(layer_idx)
            hook = self._create_ablation_hook(layer_idx)
            handle = target_layer.register_forward_hook(hook)
            self.hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.heads_to_ablate = set()

    def generate_with_ablation(self, question: str, max_new_tokens: int = 256) -> str:
        """
        Generate answer with ablation active.

        Args:
            question: Problem question
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer text
        """
        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward through question
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            # BOT token
            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Generate 6 continuous thoughts (WITH ABLATION)
            for step in range(6):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Update embedding for next step
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            # EOT token
            eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            output_emb = eot_emb

            # Generate answer tokens
            pred_tokens = []
            for _ in range(max_new_tokens):
                out = self.model.codi(
                    inputs_embeds=output_emb,
                    use_cache=True,
                    past_key_values=past_key_values
                )

                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :self.model.codi.config.vocab_size-1]

                # Greedy decoding
                next_token_id = torch.argmax(logits, dim=-1)

                if next_token_id.item() == self.tokenizer.eos_token_id:
                    break

                pred_tokens.append(next_token_id.item())
                output_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                    next_token_id
                ).unsqueeze(1)

            # Decode answer
            answer = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            return answer


def load_critical_heads(model_name: str, top_n: int) -> List[Tuple[int, int, float]]:
    """
    Load top N critical heads from Phase 2 results.

    Args:
        model_name: 'llama' or 'gpt2'
        top_n: Number of top heads to load

    Returns:
        List of (layer_idx, head_idx, composite_score) tuples
    """
    import pandas as pd

    results_dir = Path(__file__).parent.parent / 'results'
    metrics_file = results_dir / model_name / 'ranked_heads.csv'

    # Read CSV (already sorted by composite score)
    df = pd.read_csv(metrics_file)

    # Return top N
    top_heads = []
    for _, row in df.head(top_n).iterrows():
        top_heads.append((
            int(row['layer']),
            int(row['head']),
            float(row['composite_score'])
        ))

    return top_heads


def extract_numeric_answer(generated_text: str) -> int:
    """Extract numeric answer from generated text."""
    import re

    # Try to find #### marker
    if '####' in generated_text:
        try:
            answer_str = generated_text.split('####')[1].strip()
            answer_str = answer_str.split()[0]
            answer_str = answer_str.replace(',', '')
            return int(answer_str)
        except (IndexError, ValueError):
            pass

    # Extract last number
    numbers = re.findall(r'-?\\d+(?:,\\d{3})*', generated_text)
    if numbers:
        last_number = numbers[-1].replace(',', '')
        try:
            return int(last_number)
        except ValueError:
            pass

    return None


def run_ablation_experiment(model_name='llama', n_problems=100, top_n=10):
    """
    Run critical head ablation experiment.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems
        top_n: Number of top critical heads to ablate

    Returns:
        dict: Ablation results
    """
    print("=" * 80)
    print(f"CRITICAL HEAD ABLATION - Story 1")
    print("=" * 80)
    print(f"\\nModel: {model_name.upper()}")
    print(f"Test problems: {n_problems}")
    print(f"Top N heads to ablate: {top_n}")

    # Load model
    print(f"\\nLoading CODI model...")
    model, tokenizer = load_model(model_name)
    print(f"✓ Model loaded")

    # Validate architecture
    print(f"\\nValidating architecture...")
    validate_model_architecture(model, model_name, tokenizer)

    # Load critical heads
    print(f"\\nLoading top {top_n} critical heads...")
    critical_heads = load_critical_heads(model_name, top_n)
    print(f"✓ Loaded {len(critical_heads)} critical heads")
    print(f"\\nTop {min(5, top_n)} heads:")
    for i, (layer, head, score) in enumerate(critical_heads[:5]):
        print(f"  {i+1}. L{layer}H{head}: {score:.4f}")

    # Create ablator
    ablator = AttentionAblator(model, model_name, tokenizer)

    # Load test set
    print(f"\\nLoading GSM8K test set...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = dataset.select(range(n_problems))
    print(f"✓ Loaded {len(test_problems)} problems")

    # Run ablation
    print(f"\\nRunning ablation experiment...")
    print(f"Ablating top {top_n} heads simultaneously...")

    # Register hooks for all top N heads
    heads_to_ablate = [(layer, head) for layer, head, _ in critical_heads]
    ablator.register_hooks(heads_to_ablate)

    n_correct = 0
    n_failed = 0
    results_detail = []

    model.eval()
    for i, problem in enumerate(tqdm(test_problems, desc="Evaluating")):
        question = problem['question']
        gold_answer = extract_answer(problem['answer'])

        if gold_answer is None:
            n_failed += 1
            continue

        try:
            # Generate with ablation
            generated_text = ablator.generate_with_ablation(question)
            pred_answer = extract_numeric_answer(generated_text)

            is_correct = (pred_answer == gold_answer)
            if is_correct:
                n_correct += 1

            results_detail.append({
                'problem_idx': i,
                'question': question[:100] + '...',
                'gold_answer': gold_answer,
                'pred_answer': pred_answer,
                'correct': is_correct
            })

        except Exception as e:
            print(f"\\nError on problem {i}: {e}")
            n_failed += 1
            results_detail.append({
                'problem_idx': i,
                'error': str(e)
            })

    # Remove hooks
    ablator.remove_hooks()

    # Calculate accuracy
    accuracy = n_correct / (n_problems - n_failed) if (n_problems - n_failed) > 0 else 0.0

    # Load baseline for comparison
    results_dir = Path(__file__).parent.parent / 'results'
    baseline_file = results_dir / f'{model_name}_baseline.json'
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    baseline_accuracy = baseline['accuracy']

    # Save results
    results = {
        'model': model_name,
        'n_problems': n_problems,
        'top_n': top_n,
        'ablated_heads': [
            {'layer': layer, 'head': head, 'composite_score': score}
            for layer, head, score in critical_heads
        ],
        'n_correct': n_correct,
        'n_failed': n_failed,
        'accuracy': accuracy,
        'baseline_accuracy': baseline_accuracy,
        'accuracy_drop': baseline_accuracy - accuracy,
        'results_detail': results_detail[:10]
    }

    output_dir = Path(__file__).parent.parent / 'results'
    output_path = output_dir / f'{model_name}_ablation_top{top_n}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\\n" + "=" * 80)
    print("ABLATION RESULTS")
    print("=" * 80)
    print(f"\\nBaseline Accuracy: {baseline_accuracy:.2%}")
    print(f"Ablation Accuracy: {accuracy:.2%} ({n_correct}/{n_problems - n_failed})")
    print(f"Accuracy Drop: {baseline_accuracy - accuracy:.2%} ({baseline_accuracy - accuracy:.4f})")
    print(f"Failed problems: {n_failed}")
    print(f"\\n✓ Saved: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run critical head ablation experiment')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to test (llama or gpt2)')
    parser.add_argument('--n_problems', type=int, default=100,
                        help='Number of test problems (default: 100)')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top critical heads to ablate (default: 10)')
    args = parser.parse_args()

    run_ablation_experiment(
        model_name=args.model,
        n_problems=args.n_problems,
        top_n=args.top_n
    )


if __name__ == '__main__':
    main()
