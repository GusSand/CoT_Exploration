"""
Activation Patching Script
Injects cached activations at specific layers during forward pass.

Usage:
    Used by run_experiment.py to patch activations and measure causal effects.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Dict, Optional

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI
from cache_activations import LAYER_CONFIG, ActivationCacher

class ActivationPatcher:
    """Patches activations at specific layers during CODI forward pass."""

    def __init__(self, cacher: ActivationCacher):
        """Initialize with an ActivationCacher instance.

        Args:
            cacher: ActivationCacher with loaded model
        """
        self.cacher = cacher
        self.model = cacher.model
        self.tokenizer = cacher.tokenizer
        self.device = cacher.device

        # State for patching
        self.patch_activation = None
        self.patch_layer_idx = None
        self.patch_step = None
        self.current_step = 0
        self.hook_handle = None

    def _get_layer_module(self, layer_idx: int):
        """Get the transformer layer module at the specified index.

        Args:
            layer_idx: Layer index (0-11 for GPT-2)

        Returns:
            The transformer layer module
        """
        # For GPT-2 with LoRA
        try:
            # Try LoRA path first
            return self.model.codi.base_model.model.transformer.h[layer_idx]
        except AttributeError:
            # Fall back to non-LoRA path
            return self.model.codi.transformer.h[layer_idx]

    def _create_patch_hook(self):
        """Create a forward hook that patches activations at the right step.

        Returns:
            Hook function
        """
        def patch_hook(module, input, output):
            # output is a tuple: (hidden_states, ...)
            # We want to patch hidden_states at the current position
            if self.current_step == self.patch_step and self.patch_activation is not None:
                # Clone the output to avoid in-place modification
                if isinstance(output, tuple):
                    hidden_states = output[0].clone()
                    # Patch the last token's activation
                    hidden_states[:, -1, :] = self.patch_activation.to(self.device)
                    # Return modified tuple
                    return (hidden_states,) + output[1:]
                else:
                    # Just hidden states
                    hidden_states = output.clone()
                    hidden_states[:, -1, :] = self.patch_activation.to(self.device)
                    return hidden_states

            return output

        return patch_hook

    def run_with_patch(
        self,
        problem_text: str,
        patch_activation: torch.Tensor,
        patch_layer_name: str,
        max_new_tokens: int = 200
    ) -> str:
        """Run problem with activation patched at specified layer.

        Args:
            problem_text: The problem question
            patch_activation: Activation tensor to inject
            patch_layer_name: Which layer to patch ('early', 'middle', 'late')
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated answer text
        """
        patch_layer_idx = LAYER_CONFIG[patch_layer_name]

        # Set up patching state
        self.patch_activation = patch_activation
        self.patch_layer_idx = patch_layer_idx
        self.patch_step = 0  # Patch at first [THINK] token
        self.current_step = 0

        # Register hook on target layer
        target_layer = self._get_layer_module(patch_layer_idx)
        hook = self._create_patch_hook()
        self.hook_handle = target_layer.register_forward_hook(hook)

        try:
            # Run generation with patching
            answer = self._generate_with_patching(problem_text, max_new_tokens)
        finally:
            # Always remove hook
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None

        # Reset state
        self.patch_activation = None
        self.current_step = 0

        return answer

    def _generate_with_patching(self, problem_text: str, max_new_tokens: int) -> str:
        """Generate answer with patching active.

        Args:
            problem_text: Problem question
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids)

            # Forward through model
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            # BOT embedding
            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Process latent thoughts (with patching on first step)
            for latent_step in range(self.model.num_latent):
                self.current_step = latent_step

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

            # EOT embedding
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

    def run_without_patch(self, problem_text: str, max_new_tokens: int = 200) -> str:
        """Run problem normally without any patching (baseline).

        Args:
            problem_text: Problem question
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer
        """
        return self._generate_with_patching(problem_text, max_new_tokens)


def main():
    """Test activation patching."""
    import argparse
    from cache_activations import ActivationCacher

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()

    print("Initializing cacher...")
    cacher = ActivationCacher(args.model_path)

    print("Initializing patcher...")
    patcher = ActivationPatcher(cacher)

    # Test problems
    clean_problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"
    corrupted_problem = "John has 4 bags with 7 apples each. How many apples does he have in total?"

    print(f"\n{'='*60}")
    print("TEST: Activation Patching")
    print(f"{'='*60}")

    # Run clean and cache activation
    print("\n1. Running clean problem and caching activation...")
    print(f"   Problem: {clean_problem}")
    clean_activations = cacher.cache_problem_activations(clean_problem, problem_id=0)
    clean_answer = patcher.run_without_patch(clean_problem)
    print(f"   Answer: {clean_answer}")

    # Run corrupted
    print("\n2. Running corrupted problem (baseline)...")
    print(f"   Problem: {corrupted_problem}")
    corrupted_answer = patcher.run_without_patch(corrupted_problem)
    print(f"   Answer: {corrupted_answer}")

    # Run corrupted with clean activation patched
    print("\n3. Running corrupted problem with clean activation patched...")
    for layer_name in ['early', 'middle', 'late']:
        print(f"\n   Patching at {layer_name} layer (L{LAYER_CONFIG[layer_name]})...")
        patched_answer = patcher.run_with_patch(
            corrupted_problem,
            clean_activations[layer_name],
            layer_name
        )
        print(f"   Answer: {patched_answer}")

    print(f"\n{'='*60}")
    print("Expected: Patched answers should shift toward clean answer (21)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
