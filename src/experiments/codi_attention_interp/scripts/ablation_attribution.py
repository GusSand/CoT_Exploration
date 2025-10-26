#!/usr/bin/env python3
"""
Ablation-Based Attribution for Continuous Thought Tokens

Instead of using gradients, this measures importance by zeroing out each
(layer, token) position and measuring the impact on logit difference.

Importance = baseline_logit_diff - ablated_logit_diff

If importance > 0: Removing this position hurts performance (it's important)
If importance < 0: Removing this position helps performance (it's detrimental)
If importance ≈ 0: This position has no effect

This approach:
- Requires no gradients (more robust)
- Is interpretable (direct causal effect)
- Follows proven patterns from existing experiments

Author: Generated for CoT Exploration Project
Date: 2025-10-24
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
activation_patching_core = project_root / 'src' / 'experiments' / 'activation_patching' / 'core'
sys.path.insert(0, str(activation_patching_core))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG


class AblationAttributor:
    """
    Computes ablation-based attributions for continuous thought tokens.

    For each (layer, token) position:
    1. Compute baseline logit difference (all tokens active)
    2. Zero out the target position
    3. Recompute logit difference
    4. Importance = baseline - ablated
    """

    def __init__(
        self,
        cacher: ActivationCacherLLaMA,
        target_layers: List[int] = None,
        target_tokens: List[int] = None
    ):
        """
        Initialize the ablation attributor.

        Args:
            cacher: Model cacher with loaded CODI model
            target_layers: List of layer indices (default: 4-11)
            target_tokens: List of token positions (default: 0-5)
        """
        self.cacher = cacher
        self.model = cacher.model
        self.tokenizer = cacher.tokenizer
        self.device = cacher.device
        self.num_latent = cacher.num_latent

        self.target_layers = target_layers if target_layers is not None else list(range(4, 12))
        self.target_tokens = target_tokens if target_tokens is not None else list(range(6))

        print(f"Ablation Attributor initialized:")
        print(f"  Target layers: {self.target_layers}")
        print(f"  Target tokens: {self.target_tokens}")

    def _get_answer_token_ids(self, answer: int) -> List[int]:
        """Get token IDs for an answer number."""
        answer_str = str(int(answer))
        token_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
        return token_ids

    def _forward_to_answer_position(
        self,
        problem_text: str
    ) -> Tuple:
        """
        Generate forward until we reach the answer position.

        Returns:
            Tuple of (past_key_values, tokens_generated)
        """
        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Process latent thoughts
            for _ in range(self.num_latent):
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

            # EOT token
            eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            output_emb = eot_emb

            # Generate tokens until we see ":" (which comes before the answer)
            pred_tokens = []
            colon_token_id = self.tokenizer.encode(":", add_special_tokens=False)[0]

            for _ in range(20):
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
                output_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                    next_token_id
                ).unsqueeze(1)

                if next_token_id.item() == colon_token_id:
                    out = self.model.codi(
                        inputs_embeds=output_emb,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = out.past_key_values
                    break

        return past_key_values, len(pred_tokens) + 1

    def compute_logit_difference(
        self,
        past_key_values,
        correct_answer: int
    ) -> float:
        """
        Compute logit difference at answer position.

        Args:
            past_key_values: KV cache at answer position
            correct_answer: The correct numerical answer

        Returns:
            Logit difference (correct - max_incorrect)
        """
        self.model.eval()

        with torch.no_grad():
            # Use space token as input (typically before number)
            space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
            space_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([space_token_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            outputs = self.model.codi(
                inputs_embeds=space_emb,
                use_cache=True,
                past_key_values=past_key_values
            )

            logits = outputs.logits[:, -1, :self.model.codi.config.vocab_size-1]

            # Get correct token
            correct_token_ids = self._get_answer_token_ids(correct_answer)
            correct_token_id = correct_token_ids[0]

            correct_logit = logits[0, correct_token_id].item()

            # Get max incorrect
            logits_copy = logits.clone()
            logits_copy[0, correct_token_id] = float('-inf')
            max_incorrect_logit = torch.max(logits_copy).item()

            logit_diff = correct_logit - max_incorrect_logit

        return logit_diff

    def _forward_with_ablation(
        self,
        problem_text: str,
        ablate_layer_idx: int,
        ablate_token_idx: int
    ) -> Tuple:
        """
        Forward pass with ONE position ablated (zeroed out).

        Args:
            problem_text: The problem question
            ablate_layer_idx: Layer index to ablate
            ablate_token_idx: Token position to ablate

        Returns:
            past_key_values at answer position
        """
        self.model.eval()

        # Create hook to zero out the target position
        hook_handle = None
        hook_called = [0]  # Track if hook was called

        def ablation_hook(module, input, output):
            # Check if this is the latent token we want to ablate
            if hook_called[0] == ablate_token_idx:
                if isinstance(output, tuple):
                    hidden_states = output[0].clone()
                    # Zero out the last token (current latent token)
                    hidden_states[:, -1, :] = 0.0
                    return (hidden_states,) + output[1:]
                else:
                    hidden_states = output.clone()
                    hidden_states[:, -1, :] = 0.0
                    return hidden_states

            hook_called[0] += 1
            return output

        # Get the target layer module
        try:
            target_layer = self.model.codi.base_model.model.model.layers[ablate_layer_idx]
        except AttributeError:
            target_layer = self.model.codi.model.layers[ablate_layer_idx]

        # Register hook
        hook_handle = target_layer.register_forward_hook(ablation_hook)

        try:
            with torch.no_grad():
                inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]

                input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

                outputs = self.model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values

                bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                    torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
                ).unsqueeze(0)

                latent_embd = bot_emb

                # Process latent thoughts (hook will ablate target position)
                for _ in range(self.num_latent):
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

                # Reset hook counter for generation phase
                hook_called[0] = 999  # Don't ablate during generation

                # EOT token
                eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                    torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
                ).unsqueeze(0)

                output_emb = eot_emb

                # Generate to answer position
                pred_tokens = []
                colon_token_id = self.tokenizer.encode(":", add_special_tokens=False)[0]

                for _ in range(20):
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
                    output_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                        next_token_id
                    ).unsqueeze(1)

                    if next_token_id.item() == colon_token_id:
                        out = self.model.codi(
                            inputs_embeds=output_emb,
                            use_cache=True,
                            past_key_values=past_key_values
                        )
                        past_key_values = out.past_key_values
                        break

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        return past_key_values

    def compute_ablation_attributions(
        self,
        problem_text: str,
        correct_answer: int,
        show_progress: bool = False
    ) -> Dict[int, List[float]]:
        """
        Compute ablation-based attributions for all (layer, token) positions.

        Args:
            problem_text: The problem question
            correct_answer: The correct numerical answer
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping layer_idx -> List of 6 attribution scores
        """
        # Step 1: Compute baseline logit difference
        baseline_kv, _ = self._forward_to_answer_position(problem_text)
        baseline_logit_diff = self.compute_logit_difference(baseline_kv, correct_answer)

        # Step 2: Compute ablated logit difference for each position
        attributions = {layer_idx: [] for layer_idx in self.target_layers}

        positions = [(layer_idx, token_idx)
                     for layer_idx in self.target_layers
                     for token_idx in self.target_tokens]

        iterator = tqdm(positions, desc="Ablating positions") if show_progress else positions

        for layer_idx, token_idx in iterator:
            # Forward with this position ablated
            ablated_kv = self._forward_with_ablation(problem_text, layer_idx, token_idx)
            ablated_logit_diff = self.compute_logit_difference(ablated_kv, correct_answer)

            # Importance = how much performance dropped when we removed this position
            importance = baseline_logit_diff - ablated_logit_diff

            attributions[layer_idx].append(importance)

        return attributions, baseline_logit_diff

    def compute_random_baseline(self) -> Dict[int, List[float]]:
        """Random baseline (sanity check)."""
        random_attributions = {}
        for layer_idx in self.target_layers:
            random_scores = np.random.randn(len(self.target_tokens)).tolist()
            random_attributions[layer_idx] = random_scores
        return random_attributions

    def compute_attention_attribution(self, problem_text: str) -> Dict[int, List[float]]:
        """Attention-based attribution."""
        self.model.eval()

        with torch.no_grad():
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True
            )

            past_key_values = outputs.past_key_values

            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            attention_scores = {layer_idx: [] for layer_idx in self.target_layers}
            latent_embd = bot_emb

            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                if latent_step in self.target_tokens and outputs.attentions is not None:
                    for layer_idx in self.target_layers:
                        attn = outputs.attentions[layer_idx]
                        attn_avg = attn.mean(dim=1)[0, -1, :]
                        total_attention = attn_avg.sum().item()
                        attention_scores[layer_idx].append(total_attention)

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            # Fill missing values with zeros
            for layer_idx in self.target_layers:
                if len(attention_scores[layer_idx]) == 0:
                    attention_scores[layer_idx] = [0.0] * len(self.target_tokens)

        return attention_scores


def main():
    """Test ablation attribution."""
    print("="*80)
    print("ABLATION-BASED ATTRIBUTION TEST")
    print("="*80)

    # Load model
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    print(f"\nLoading model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)

    # Initialize attributor
    print("\nInitializing ablation attributor...")
    attributor = AblationAttributor(
        cacher,
        target_layers=[4, 8],  # Test with 2 layers only
        target_tokens=list(range(6))
    )

    # Test problem
    test_problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"
    test_answer = 21

    print(f"\n{'='*80}")
    print(f"Test problem: {test_problem}")
    print(f"Expected answer: {test_answer}")
    print(f"{'='*80}")

    # Compute ablation attributions
    print("\n[Test] Computing ablation attributions...")
    print("  (This will zero out each position and measure impact)")

    ablation_attr, baseline_logit_diff = attributor.compute_ablation_attributions(
        test_problem,
        test_answer,
        show_progress=True
    )

    print(f"\nBaseline logit difference: {baseline_logit_diff:.4f}")
    print(f"\nAblation attributions:")
    for layer_idx, scores in ablation_attr.items():
        print(f"  Layer {layer_idx}: {[f'{s:.3f}' for s in scores]}")

    # Compute baselines
    print("\n[Test] Computing random baseline...")
    random_attr = attributor.compute_random_baseline()
    for layer_idx, scores in random_attr.items():
        print(f"  Layer {layer_idx}: {[f'{s:.3f}' for s in scores]}")

    print("\n[Test] Computing attention baseline...")
    attn_attr = attributor.compute_attention_attribution(test_problem)
    for layer_idx, scores in attn_attr.items():
        print(f"  Layer {layer_idx}: {[f'{s:.3f}' for s in scores]}")

    # Summary
    total_importance = sum(sum(scores) for scores in ablation_attr.values())
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Baseline logit diff: {baseline_logit_diff:.4f}")
    print(f"Total importance: {total_importance:.4f}")
    print(f"Mean importance per position: {total_importance / (len(ablation_attr) * 6):.4f}")

    # Check if attributions are non-zero
    non_zero_count = sum(1 for scores in ablation_attr.values() for s in scores if abs(s) > 0.01)
    print(f"Non-zero positions: {non_zero_count}/{len(ablation_attr) * 6}")

    if non_zero_count > 0:
        print("\n✅ SUCCESS: Ablation attributions are non-zero!")
    else:
        print("\n⚠️  WARNING: All attributions are near zero")

    print("="*80)


if __name__ == "__main__":
    main()
