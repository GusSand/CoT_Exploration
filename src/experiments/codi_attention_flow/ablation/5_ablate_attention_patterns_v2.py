#!/usr/bin/env python3
"""
Attention Pattern Ablation - Story 5 (v2: Clean implementation)

Test which attention patterns between CT tokens are critical by masking specific
connections in the 6×6 CT attention matrix during generation.

This version uses attention_mask to control attention patterns instead of hooks.

Experiments:
1. Zero hub attention (CT_all → CT0): Block all tokens from attending to CT0
2. Zero skip connections: Block CT_i → CT_j where j < i-1 (only allow sequential)
3. Zero backward attention: Block CT_i → CT_j where j > i (causal only)
4. Random position ablation: Block attention to random CT positions

Usage:
    python 5_ablate_attention_patterns_v2.py [--model MODEL] [--n_problems N] [--pattern PATTERN]

Patterns:
    - hub_to_ct0: Zero all attention TO CT0
    - skip_connections: Zero all skip connections (i → j where j < i-1)
    - backward: Zero backward attention (future tokens)
    - position_N: Zero attention to specific position N (0-5)

Output:
    ../results/{model}_attention_pattern_{pattern}.json
"""
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, Callable

from utils import load_model, extract_answer, validate_model_architecture


class AttentionPatternAblator:
    """
    Ablate specific attention patterns during CT generation using attention masks.

    This approach modifies the attention_mask parameter to control which tokens
    can attend to which other tokens in the CT sequence.
    """

    def __init__(self, model, model_name: str, tokenizer):
        """
        Initialize pattern ablator.

        Args:
            model: CODI model
            model_name: 'llama' or 'gpt2'
            tokenizer: Model tokenizer
        """
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Ablation pattern
        self.pattern_name: Optional[str] = None
        self.pattern_mask_fn: Optional[Callable] = None

    def set_pattern(self, pattern_name: str):
        """
        Set the attention pattern to ablate.

        Args:
            pattern_name: Pattern to ablate
        """
        self.pattern_name = pattern_name

        if pattern_name == "hub_to_ct0":
            # Zero all attention TO CT0 (column 0 in CT matrix)
            def mask_fn(n_ct_tokens):
                # mask[i,j] = 1 means i CAN attend to j
                # mask[i,j] = 0 means i CANNOT attend to j
                mask = torch.ones(n_ct_tokens, n_ct_tokens)
                mask[:, 0] = 0  # Zero column 0 (attention to CT0)
                return mask
            self.pattern_mask_fn = mask_fn

        elif pattern_name == "skip_connections":
            # Zero skip connections (only allow sequential i→i-1 and i→i)
            def mask_fn(n_ct_tokens):
                mask = torch.zeros(n_ct_tokens, n_ct_tokens)
                for i in range(n_ct_tokens):
                    mask[i, i] = 1  # Self-attention
                    if i > 0:
                        mask[i, i-1] = 1  # Attend to previous token
                return mask
            self.pattern_mask_fn = mask_fn

        elif pattern_name == "backward":
            # Zero backward attention (causal only)
            def mask_fn(n_ct_tokens):
                # Lower triangular = causal (can attend to past and self)
                mask = torch.tril(torch.ones(n_ct_tokens, n_ct_tokens))
                return mask
            self.pattern_mask_fn = mask_fn

        elif pattern_name.startswith("position_"):
            # Zero attention to specific position
            pos = int(pattern_name.split("_")[1])
            def mask_fn(n_ct_tokens):
                mask = torch.ones(n_ct_tokens, n_ct_tokens)
                if pos < n_ct_tokens:
                    mask[:, pos] = 0
                return mask
            self.pattern_mask_fn = mask_fn

        else:
            raise ValueError(f"Unknown pattern: {pattern_name}")

    def _create_ct_attention_mask(self, full_seq_len: int, ct_start: int, ct_end: int, current_pos: int) -> torch.Tensor:
        """
        Create attention mask for current token with CT pattern applied.

        When using KV cache, we only need the mask for the current token attending to all previous tokens.

        Args:
            full_seq_len: Total sequence length including current token
            ct_start: Start position of CT tokens in full sequence
            ct_end: End position of CT tokens (exclusive) in full sequence
            current_pos: Current token position in full sequence

        Returns:
            Attention mask [1, full_seq_len] where 0 = masked (cannot attend)
        """
        # Current token can attend to all previous tokens by default
        # Shape: [1, full_seq_len] - attention from current token to all previous
        mask = torch.ones(1, full_seq_len, device=self.device)

        # If current token is a CT token, apply pattern restrictions
        if ct_start <= current_pos < ct_end and self.pattern_mask_fn is not None:
            # Which CT is this? (0-5)
            current_ct_idx = current_pos - ct_start
            n_ct_tokens = ct_end - ct_start

            # Get full pattern mask
            ct_pattern_mask = self.pattern_mask_fn(n_ct_tokens).to(self.device)

            # Extract the row for current CT token (what it can attend to)
            # This row tells us which other CT tokens it CAN attend to
            current_ct_attention_row = ct_pattern_mask[current_ct_idx]  # [n_ct_tokens]

            # Apply this pattern to the CT positions in the full sequence
            # Zero out attention to CT positions that pattern forbids
            for ct_idx in range(current_ct_idx + 1):  # Only consider past/current CTs
                ct_position = ct_start + ct_idx
                if ct_pattern_mask[current_ct_idx, ct_idx] == 0:
                    mask[0, ct_position] = 0

        # Convert to attention mask format (0 = can attend, large negative = cannot attend)
        attention_mask = torch.zeros_like(mask)
        attention_mask[mask == 0] = -10000.0

        return attention_mask

    def generate_with_pattern_ablation(self, question: str, max_new_tokens: int = 100) -> str:
        """
        Generate answer with attention pattern ablation.

        Args:
            question: Input question
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer text
        """
        # Tokenize question
        inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']
        question_length = input_ids.shape[1]

        # Question forward pass (no ablation)
        question_outputs = self.model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = question_outputs.past_key_values

        # BOT token (no ablation yet)
        bot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Track CT token positions
        # Sequence: [question tokens] [BOT] [CT0] [CT1] ... [CT5]
        bot_position = question_length
        ct_start = bot_position + 1

        # Generate 6 continuous thought tokens (WITH PATTERN ABLATION)
        for step in range(6):
            current_seq_len = question_length + 1 + step + 1  # question + BOT + CTs so far + current CT
            current_pos = current_seq_len - 1  # Current token position (0-indexed)
            ct_end = ct_start + step + 1

            # Create attention mask with pattern applied to CT tokens
            if self.pattern_mask_fn is not None:
                attention_mask = self._create_ct_attention_mask(current_seq_len, ct_start, ct_end, current_pos)
                # Reshape for batch: [batch, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(0)
            else:
                attention_mask = None

            # Forward pass with pattern-masked attention
            outputs = self.model.codi(
                inputs_embeds=latent_embd,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.model.use_prj:
                latent_embd = self.model.prj(latent_embd)

        # EOT token (no ablation)
        eot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        output_emb = eot_emb

        # Generate answer (no ablation)
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


def run_attention_pattern_experiment(
    model_name: str = 'llama',
    pattern: str = 'hub_to_ct0',
    n_problems: int = 100
) -> dict:
    """
    Run attention pattern ablation experiment.

    Args:
        model_name: 'llama' or 'gpt2'
        pattern: Attention pattern to ablate
        n_problems: Number of test problems

    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Attention Pattern Ablation - Story 5")
    print(f"Model: {model_name}")
    print(f"Pattern: {pattern}")
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

    # Initialize ablator
    print(f"\nSetting up attention pattern ablation: {pattern}")
    ablator = AttentionPatternAblator(model, model_name, tokenizer)
    ablator.set_pattern(pattern)

    # Test on all problems
    n_correct = 0
    n_failed = 0
    results_detail = []

    print(f"\nTesting with pattern ablation...")
    for problem in tqdm(test_problems, desc=f"Pattern: {pattern}"):
        question = problem['question']
        gold_answer = extract_answer(problem['answer'])

        try:
            # Generate with pattern ablation
            generated_text = ablator.generate_with_pattern_ablation(question)
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
                'correct': False,
                'error': str(e)
            })

    accuracy = n_correct / len(test_problems)

    results = {
        'model': model_name,
        'pattern': pattern,
        'n_problems': len(test_problems),
        'n_correct': n_correct,
        'n_failed': n_failed,
        'accuracy': accuracy,
        'baseline_accuracy': 0.59,
        'accuracy_drop': 0.59 - accuracy,
        'results_detail': results_detail  # Save ALL predictions for qualitative analysis
    }

    print(f"\n{'='*60}")
    print(f"Results for pattern: {pattern}")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Drop from baseline: {results['accuracy_drop']:.2%}")
    print(f"Correct: {n_correct}/{len(test_problems)}")
    print(f"Failed: {n_failed}/{len(test_problems)}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Attention pattern ablation experiment')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    parser.add_argument('--pattern', type=str, default='hub_to_ct0',
                       choices=['hub_to_ct0', 'skip_connections', 'backward',
                               'position_0', 'position_1', 'position_2',
                               'position_3', 'position_4', 'position_5'])
    parser.add_argument('--n_problems', type=int, default=100)
    args = parser.parse_args()

    # Run experiment
    results = run_attention_pattern_experiment(
        model_name=args.model,
        pattern=args.pattern,
        n_problems=args.n_problems
    )

    # Save results
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / f'{args.model}_attention_pattern_{args.pattern}.json'

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == '__main__':
    main()
