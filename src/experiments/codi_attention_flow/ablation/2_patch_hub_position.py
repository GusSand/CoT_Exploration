#!/usr/bin/env python3
"""
Hub Position Activation Patching - Story 2

Test whether patching hub position activations from correct examples can restore
reasoning capability. Complementary to ablation experiments.

Hypothesis: If hub position (CT0 for LLaMA) is the critical bottleneck, patching
it with activations from a correct answer should improve accuracy.

Usage:
    python 2_patch_hub_position.py [--model MODEL] [--n_problems N]

Output:
    ../results/{model}_patching_results.json
"""
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Tuple

from utils import load_model, extract_answer, validate_model_architecture, get_model_architecture_info


class HubPositionPatcher:
    """
    Activation patching at hub position (CT0 for LLaMA, CT1 for GPT-2).

    Replaces activations at the hub position with activations from a correct example,
    testing whether this restores reasoning capability.
    """

    def __init__(self, model, model_name: str, tokenizer):
        """
        Initialize patcher.

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
        self.hidden_dim = self.arch_info['hidden_dim']

        # Hub position (from Phase 2 findings)
        self.hub_position = 0 if model_name == 'llama' else 1

        # Patching state
        self.patch_activation = None
        self.patch_layer_idx = None
        self.current_step = 0
        self.hook_handle = None

    def cache_hub_activation(self, question: str, layer_idx: int) -> torch.Tensor:
        """
        Generate and cache hub position activation from a question.

        Args:
            question: Problem question
            layer_idx: Which layer to cache from

        Returns:
            Cached activation tensor [1, hidden_dim]
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

            # Generate continuous thoughts and cache hub position
            for step in range(6):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Cache if this is the hub position
                if step == self.hub_position:
                    # Get activation from specified layer
                    hub_activation = outputs.hidden_states[layer_idx][:, -1, :]  # [1, hidden_dim]
                    return hub_activation.detach().clone()

                # Update embedding for next step
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

        raise RuntimeError(f"Failed to cache hub activation at position {self.hub_position}")

    def _create_patch_hook(self, layer_idx: int):
        """
        Create hook that patches hub position activation.

        Args:
            layer_idx: Layer index to patch

        Returns:
            Hook function
        """
        def hook(module, input, output):
            # Only patch at hub position
            if self.current_step == self.hub_position and self.patch_activation is not None:
                # output is tuple (hidden_states, ) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                # Replace activation at last position
                hidden_states[:, -1, :] = self.patch_activation

                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                else:
                    return hidden_states

            return output

        return hook

    def generate_with_patching(self, question: str, patch_layer_idx: int, max_new_tokens: int = 256) -> str:
        """
        Generate answer with hub position patching.

        Args:
            question: Problem question
            patch_layer_idx: Which layer to patch
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer text
        """
        # Register hook on target layer
        if self.model_name == 'llama':
            target_layer = self.model.codi.model.model.layers[patch_layer_idx]
        else:
            target_layer = self.model.codi.transformer.h[patch_layer_idx]

        hook = self._create_patch_hook(patch_layer_idx)
        self.hook_handle = target_layer.register_forward_hook(hook)
        self.patch_layer_idx = patch_layer_idx
        self.current_step = 0

        try:
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

                # Generate 6 continuous thoughts (WITH PATCHING at hub position)
                for step in range(6):
                    self.current_step = step

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

        finally:
            # Always remove hook
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None
            self.current_step = 0


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
    numbers = re.findall(r'-?\d+(?:,\d{3})*', generated_text)
    if numbers:
        last_number = numbers[-1].replace(',', '')
        try:
            return int(last_number)
        except ValueError:
            pass

    return None


def run_patching_experiment(model_name='llama', n_problems=100, patch_layer=4):
    """
    Run hub position patching experiment.

    Strategy:
    1. Find pairs of similar problems (one correct, one incorrect from baseline)
    2. Cache hub activation from correct example
    3. Patch incorrect example with correct activation
    4. Measure if accuracy improves

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems
        patch_layer: Which layer to patch (default: 4 for LLaMA L4H5)

    Returns:
        dict: Patching results
    """
    print("=" * 80)
    print(f"HUB POSITION PATCHING - Story 2")
    print("=" * 80)
    print(f"\nModel: {model_name.upper()}")
    print(f"Test problems: {n_problems}")
    print(f"Patch layer: {patch_layer}")

    # Load model
    print(f"\nLoading CODI model...")
    model, tokenizer = load_model(model_name)
    print(f"✓ Model loaded")

    # Validate architecture
    print(f"\nValidating architecture...")
    validate_model_architecture(model, model_name, tokenizer)

    # Create patcher
    patcher = HubPositionPatcher(model, model_name, tokenizer)
    hub_pos = patcher.hub_position

    print(f"\nHub position: CT{hub_pos}")
    print(f"Patching at: Layer {patch_layer}")

    # Load baseline results to identify correct/incorrect pairs
    results_dir = Path(__file__).parent.parent / 'results'
    baseline_file = results_dir / f'{model_name}_baseline.json'

    print(f"\nLoading baseline results...")
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    # Load test set
    print(f"\nLoading GSM8K test set...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = dataset.select(range(n_problems))
    print(f"✓ Loaded {len(test_problems)} problems")

    # Strategy: Use first correct problem as "donor" for all incorrect problems
    print(f"\nFinding donor problem (first correct from baseline)...")

    donor_idx = None
    for i in range(min(n_problems, 10)):  # Check first 10
        question = test_problems[i]['question']
        gold_answer = extract_answer(test_problems[i]['answer'])

        # Quick inference to verify it's correct
        try:
            generated = generate_baseline(model, tokenizer, question, model_name)
            pred = extract_numeric_answer(generated)
            if pred == gold_answer:
                donor_idx = i
                print(f"✓ Found donor: Problem {donor_idx} (answer: {gold_answer})")
                break
        except:
            continue

    if donor_idx is None:
        print("✗ No correct donor found in first 10 problems")
        return None

    # Cache donor activation
    print(f"\nCaching donor activation from problem {donor_idx}...")
    donor_question = test_problems[donor_idx]['question']
    donor_activation = patcher.cache_hub_activation(donor_question, patch_layer)
    print(f"✓ Cached activation shape: {donor_activation.shape}")

    # Set patch activation
    patcher.patch_activation = donor_activation

    # Run patching on all problems
    print(f"\nRunning patching experiment...")

    n_correct_baseline = 0
    n_correct_patched = 0
    n_failed = 0
    results_detail = []

    model.eval()
    for i, problem in enumerate(tqdm(test_problems, desc="Evaluating")):
        if i == donor_idx:
            continue  # Skip donor itself

        question = problem['question']
        gold_answer = extract_answer(problem['answer'])

        if gold_answer is None:
            n_failed += 1
            continue

        try:
            # Baseline (no patching)
            baseline_answer = generate_baseline(model, tokenizer, question, model_name)
            baseline_pred = extract_numeric_answer(baseline_answer)
            baseline_correct = (baseline_pred == gold_answer)

            if baseline_correct:
                n_correct_baseline += 1

            # Patched
            patched_answer = patcher.generate_with_patching(question, patch_layer)
            patched_pred = extract_numeric_answer(patched_answer)
            patched_correct = (patched_pred == gold_answer)

            if patched_correct:
                n_correct_patched += 1

            results_detail.append({
                'problem_idx': i,
                'question': question[:100] + '...',
                'gold_answer': gold_answer,
                'baseline_pred': baseline_pred,
                'baseline_correct': baseline_correct,
                'patched_pred': patched_pred,
                'patched_correct': patched_correct,
                'effect': 'fix' if (not baseline_correct and patched_correct) else
                         'break' if (baseline_correct and not patched_correct) else
                         'neutral'
            })

        except Exception as e:
            print(f"\nError on problem {i}: {e}")
            n_failed += 1

    # Calculate accuracies
    n_tested = n_problems - n_failed - 1  # -1 for donor
    baseline_accuracy = n_correct_baseline / n_tested if n_tested > 0 else 0.0
    patched_accuracy = n_correct_patched / n_tested if n_tested > 0 else 0.0

    # Count effects
    n_fixed = sum(1 for r in results_detail if r['effect'] == 'fix')
    n_broken = sum(1 for r in results_detail if r['effect'] == 'break')
    n_neutral = sum(1 for r in results_detail if r['effect'] == 'neutral')

    # Save results
    results = {
        'model': model_name,
        'n_problems': n_problems,
        'patch_layer': patch_layer,
        'hub_position': hub_pos,
        'donor_idx': donor_idx,
        'n_tested': n_tested,
        'baseline_accuracy': baseline_accuracy,
        'patched_accuracy': patched_accuracy,
        'accuracy_change': patched_accuracy - baseline_accuracy,
        'n_fixed': n_fixed,
        'n_broken': n_broken,
        'n_neutral': n_neutral,
        'results_detail': results_detail[:20]  # Save first 20
    }

    output_dir = Path(__file__).parent.parent / 'results'
    output_path = output_dir / f'{model_name}_patching_L{patch_layer}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("PATCHING RESULTS")
    print("=" * 80)
    print(f"\nDonor: Problem {donor_idx}")
    print(f"Tested: {n_tested} problems")
    print(f"\nBaseline Accuracy: {baseline_accuracy:.2%} ({n_correct_baseline}/{n_tested})")
    print(f"Patched Accuracy:  {patched_accuracy:.2%} ({n_correct_patched}/{n_tested})")
    print(f"Change: {patched_accuracy - baseline_accuracy:+.2%} ({patched_accuracy - baseline_accuracy:+.4f})")
    print(f"\nEffects:")
    print(f"  Fixed (incorrect→correct): {n_fixed}")
    print(f"  Broken (correct→incorrect): {n_broken}")
    print(f"  Neutral (no change): {n_neutral}")
    print(f"\n✓ Saved: {output_path}")

    return results


def generate_baseline(model, tokenizer, question: str, model_name: str) -> str:
    """Generate baseline answer without patching."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Get embeddings
        input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

        # Forward through question
        outputs = model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = outputs.past_key_values

        # BOT token
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Generate 6 continuous thoughts
        for step in range(6):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)

        # EOT token
        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        output_emb = eot_emb

        # Generate answer tokens
        pred_tokens = []
        for _ in range(256):
            out = model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = model.get_embd(model.codi, model.model_name)(
                next_token_id
            ).unsqueeze(1)

        return tokenizer.decode(pred_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description='Run hub position patching experiment')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to test (llama or gpt2)')
    parser.add_argument('--n_problems', type=int, default=100,
                        help='Number of test problems (default: 100)')
    parser.add_argument('--patch_layer', type=int, default=4,
                        help='Which layer to patch (default: 4 for L4H5)')
    args = parser.parse_args()

    run_patching_experiment(
        model_name=args.model,
        n_problems=args.n_problems,
        patch_layer=args.patch_layer
    )


if __name__ == '__main__':
    main()
