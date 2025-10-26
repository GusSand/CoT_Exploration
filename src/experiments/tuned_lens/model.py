"""
Model wrappers for integrating tuned-lens library with CODI.

This module provides wrappers around the tuned-lens library's TunedLens and LogitLens
classes, adapted to work with CODI's pre-extracted continuous thought activations.

Key Differences from Standard tuned-lens:
- CODI's continuous thoughts are generated iteratively, not via standard forward pass
- We use pre-extracted activations instead of running model forward passes
- Custom training loop needed to work with cached activations
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional
from tuned_lens.nn import TunedLens, LogitLens, TunedLensConfig, Unembed
from transformers import AutoTokenizer, AutoModelForCausalLM


class CODITunedLensWrapper:
    """Wrapper for TunedLens to work with CODI's continuous thought activations.

    This class provides a bridge between CODI's architecture and the tuned-lens library:
    1. Loads CODI model to extract unembedding matrix
    2. Creates TunedLens with correct configuration for CODI
    3. Provides methods for training and inference on pre-extracted activations
    """

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 2048,
        num_layers: int = 16,
        vocab_size: int = 32000,
        device: str = "cuda",
        bias: bool = True,
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ):
        """Initialize CODI Tuned Lens wrapper.

        Args:
            model_path: Path to CODI checkpoint
            hidden_size: Hidden dimension size (LLaMA: 2048, GPT-2: 768)
            num_layers: Number of layers (LLaMA: 16, GPT-2: 12)
            vocab_size: Vocabulary size (LLaMA: ~32000, GPT-2: 50257)
            device: Device to use ('cuda' or 'cpu')
            bias: Whether to use bias in affine transformations
            base_model_name: Base model name for tokenizer (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        """
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.bias = bias
        self.base_model_name = base_model_name

        # Load tokenizer from base model (CODI checkpoint doesn't include tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Create unembedding matrix from CODI model
        self.unembed = self._create_unembed()

        # Create TunedLens and LogitLens
        self.tuned_lens = self._create_tuned_lens()
        self.logit_lens = self._create_logit_lens()

    def _create_unembed(self) -> Unembed:
        """Create Unembed object from base model's output embedding.

        We extract the unembedding matrix (lm_head) from the base LLaMA model.
        CODI uses the same unembedding as its base model.
        """
        # Load base model temporarily to extract lm_head
        # Note: This loads the full model but we only need lm_head weights
        # The tuned-lens Unembed class will handle extracting the weights
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=self.device
        )

        # Create Unembed object using the base model
        unembed = Unembed(base_model)

        # Clean up base model to free memory (keep only unembed)
        del base_model
        torch.cuda.empty_cache()

        return unembed

    def _create_tuned_lens(self) -> TunedLens:
        """Create TunedLens with CODI configuration."""
        config = TunedLensConfig(
            base_model_name_or_path=self.model_path,
            d_model=self.hidden_size,
            num_hidden_layers=self.num_layers,
            bias=self.bias,
            base_model_revision=None,
        )

        tuned_lens = TunedLens(self.unembed, config)
        tuned_lens = tuned_lens.to(self.device)

        return tuned_lens

    def _create_logit_lens(self) -> LogitLens:
        """Create LogitLens for baseline comparison."""
        logit_lens = LogitLens(self.unembed)
        logit_lens = logit_lens.to(self.device)

        return logit_lens

    def decode_with_tuned_lens(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Decode hidden states using Tuned Lens.

        Args:
            hidden_states: Hidden states to decode (batch_size, hidden_size)
            layer_idx: Layer index these hidden states come from
            top_k: Number of top tokens to return

        Returns:
            Dictionary with:
            - logits: Full logit distribution
            - token_ids: Top-k token IDs
            - tokens: Top-k decoded tokens
            - probs: Top-k probabilities
        """
        with torch.no_grad():
            # Apply tuned lens transformation and unembedding
            logits = self.tuned_lens(hidden_states, layer_idx)

            # Get top-k predictions
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

            # Decode tokens
            tokens = [
                self.tokenizer.decode(token_id) for token_id in top_ids[0].cpu().tolist()
            ]

            return {
                'logits': logits,
                'token_ids': top_ids,
                'tokens': tokens,
                'probs': top_probs
            }

    def decode_with_logit_lens(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Decode hidden states using Logit Lens (baseline).

        Args:
            hidden_states: Hidden states to decode (batch_size, hidden_size)
            layer_idx: Layer index (not used for logit lens, but kept for API consistency)
            top_k: Number of top tokens to return

        Returns:
            Dictionary with:
            - logits: Full logit distribution
            - token_ids: Top-k token IDs
            - tokens: Top-k decoded tokens
            - probs: Top-k probabilities
        """
        with torch.no_grad():
            # Apply logit lens (just unembedding, no transformation)
            logits = self.logit_lens(hidden_states, layer_idx)

            # Get top-k predictions
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

            # Decode tokens
            tokens = [
                self.tokenizer.decode(token_id) for token_id in top_ids[0].cpu().tolist()
            ]

            return {
                'logits': logits,
                'token_ids': top_ids,
                'tokens': tokens,
                'probs': top_probs
            }

    def save_tuned_lens(self, output_path: str):
        """Save trained TunedLens to disk.

        Args:
            output_path: Directory to save lens to
        """
        self.tuned_lens.save(output_path)
        print(f"Tuned Lens saved to: {output_path}")

    def load_tuned_lens(self, lens_path: str):
        """Load pre-trained TunedLens from disk.

        Args:
            lens_path: Path to saved lens directory
        """
        # Load state dict
        state_dict = torch.load(Path(lens_path) / "params.pt", map_location=self.device)
        self.tuned_lens.layer_translators.load_state_dict(state_dict)
        self.tuned_lens = self.tuned_lens.to(self.device)
        print(f"Tuned Lens loaded from: {lens_path}")

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer.

        Returns:
            List of trainable parameters
        """
        return [p for p in self.tuned_lens.parameters() if p.requires_grad]

    def num_parameters(self) -> int:
        """Count number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.get_trainable_parameters())


def create_codi_tuned_lens(config: Dict[str, Any], device: str = "cuda") -> CODITunedLensWrapper:
    """Factory function to create CODI Tuned Lens wrapper from config.

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        Initialized CODITunedLensWrapper
    """
    # Determine base model name based on model type
    base_model_names = {
        'llama': 'meta-llama/Llama-3.2-1B-Instruct',
        'gpt2': 'openai-community/gpt2'
    }
    base_model_name = base_model_names.get(config['model']['name'], 'meta-llama/Llama-3.2-1B-Instruct')

    return CODITunedLensWrapper(
        model_path=config['model']['checkpoint_path'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        vocab_size=config['model']['vocab_size'],
        device=device,
        bias=config['tuned_lens']['use_layer_norm'],  # Use layer norm setting as bias flag
        base_model_name=base_model_name
    )
