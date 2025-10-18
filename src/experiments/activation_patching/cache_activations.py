"""
Activation Caching Script
Extracts and caches hidden state activations from [THINK] tokens at multiple layers.

Usage:
    python cache_activations.py --problem_file problem_pairs.json --output_dir results/activations/
"""

import torch
import torch.nn.functional as F
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType

# Layer configuration - which layers to cache
LAYER_CONFIG = {
    'early': 3,    # Early layer (1/4 through 12-layer GPT-2)
    'middle': 6,   # Middle layer (1/2 through model)
    'late': 11     # Late layer (near final)
}

class ActivationCacher:
    """Caches activations from CODI model at specified layers."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize the cacher with CODI model.

        Args:
            model_path: Path to CODI checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"Loading CODI model from {model_path}...")

        # Parse arguments for CODI model
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=[
                '--model_name_or_path', 'gpt2',
                '--output_dir', './tmp',
                '--num_latent', '6',
                '--use_lora', 'True',
                '--ckpt_dir', model_path,
                '--use_prj', 'True',
                '--prj_dim', '768',  # GPT-2 uses 768, not default 2048
                '--lora_r', '128',
                '--lora_alpha', '32',
                '--lora_init', 'True',
            ]
        )

        # Modify for inference
        model_args.train = False
        training_args.greedy = True

        # Create LoRA config for GPT-2
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", 'c_fc'],  # GPT-2 specific
            init_lora_weights=True,
        )

        # Load model with lora_config
        self.model = CODI(model_args, training_args, lora_config)

        # Load checkpoint weights
        import os
        from safetensors.torch import load_file
        try:
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        except Exception:
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')

        self.model.load_state_dict(state_dict, strict=False)
        self.model.codi.tie_weights()

        # Convert to float32 to avoid dtype mismatches (checkpoint has mixed bfloat16/float32)
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = training_args.num_latent

        print("Model loaded successfully!")

    def cache_problem_activations(
        self,
        problem_text: str,
        problem_id: int,
        layer_indices: Dict[str, int] = None
    ) -> Dict[str, torch.Tensor]:
        """Run problem through model and cache activations at specified layers.

        Args:
            problem_text: The problem question text
            problem_id: Unique identifier for this problem
            layer_indices: Dict mapping layer names to indices (e.g., {'early': 3})

        Returns:
            Dict mapping layer names to activation tensors
        """
        if layer_indices is None:
            layer_indices = LAYER_CONFIG

        activations = {}

        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward through model to get initial context
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # Get BOT (Beginning of Thought) embedding
            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Process latent thoughts
            latent_embd = bot_emb

            # We'll cache the activation from the FIRST latent iteration
            # This is the key [THINK] token we'll patch later
            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Cache activations from specified layers (only first step)
                if latent_step == 0:  # Cache first [THINK] token
                    for layer_name, layer_idx in layer_indices.items():
                        # hidden_states is a tuple: (layer_0, layer_1, ..., layer_N)
                        # Each layer: [batch_size, seq_len, hidden_dim]
                        # We want the last token ([:, -1, :])
                        activation = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                        activations[layer_name] = activation

                # Update latent embedding for next iteration
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection if used
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

        return activations

    def save_activations(
        self,
        activations: Dict[str, torch.Tensor],
        problem_id: int,
        condition: str,  # 'clean' or 'corrupted'
        output_dir: str
    ):
        """Save activations to disk.

        Args:
            activations: Dict of layer_name -> activation tensor
            problem_id: Problem identifier
            condition: 'clean' or 'corrupted'
            output_dir: Directory to save to
        """
        os.makedirs(output_dir, exist_ok=True)

        for layer_name, activation in activations.items():
            save_path = os.path.join(
                output_dir,
                f"pair_{problem_id}_{condition}_{layer_name}.pt"
            )
            torch.save(activation, save_path)

    def load_activation(
        self,
        problem_id: int,
        condition: str,
        layer_name: str,
        activation_dir: str
    ) -> torch.Tensor:
        """Load a cached activation from disk.

        Args:
            problem_id: Problem identifier
            condition: 'clean' or 'corrupted'
            layer_name: Layer to load ('early', 'middle', 'late')
            activation_dir: Directory containing activations

        Returns:
            Activation tensor
        """
        load_path = os.path.join(
            activation_dir,
            f"pair_{problem_id}_{condition}_{layer_name}.pt"
        )
        return torch.load(load_path)


def main():
    """Test the activation caching on a sample problem."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to CODI checkpoint')
    parser.add_argument('--test', action='store_true', help='Run test on single problem')
    args = parser.parse_args()

    # Initialize cacher
    cacher = ActivationCacher(args.model_path)

    if args.test:
        # Test on a simple problem
        problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"

        print(f"\nTesting activation caching on problem: {problem}")
        print(f"Caching activations from layers: {LAYER_CONFIG}")

        activations = cacher.cache_problem_activations(
            problem_text=problem,
            problem_id=0
        )

        print("\nCached activations:")
        for layer_name, activation in activations.items():
            print(f"  {layer_name:8s}: shape={activation.shape}, device={activation.device}")

        # Test save/load
        test_dir = "./results/activations_test"
        print(f"\nTesting save to {test_dir}...")
        cacher.save_activations(activations, problem_id=0, condition='clean', output_dir=test_dir)

        print("Testing load...")
        loaded = cacher.load_activation(0, 'clean', 'middle', test_dir)
        print(f"Loaded activation shape: {loaded.shape}")

        # Verify they match
        assert torch.allclose(activations['middle'], loaded), "Saved and loaded activations don't match!"
        print("✓ Save/load test passed!")


if __name__ == "__main__":
    main()
