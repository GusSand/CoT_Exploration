"""
Attention extraction module for analyzing CoT position attention patterns
"""

import torch
import numpy as np
from typing import Dict, List, Tuple

class AttentionExtractor:
    """
    Extracts and analyzes attention patterns for CoT positions

    Key functionality:
    1. Hook into model layers to capture attention weights
    2. Filter attention to only CoT positions
    3. Aggregate across heads and layers as needed
    """

    def __init__(self, model, num_layers=16, num_heads=32, num_latent=5):
        """
        Args:
            model: CODI model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            num_latent: Number of CoT positions
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_latent = num_latent

        # Storage for attention weights
        self.attention_weights = {}  # layer_idx -> attention tensor
        self.hooks = []

    def _get_layer(self, layer_idx):
        """Get the transformer layer at the given index"""
        if hasattr(self.model, 'codi'):
            if hasattr(self.model.codi, 'base_model'):
                return self.model.codi.base_model.model.model.layers[layer_idx]
            else:
                return self.model.codi.model.layers[layer_idx]
        else:
            return self.model.model.layers[layer_idx]

    def _create_attention_hook(self, layer_idx):
        """
        Create a hook to capture attention weights from a layer

        The hook captures the attention weights from the self-attention module
        """
        def hook(module, input, output):
            # LLaMA attention output format: (hidden_states, attention_weights, ...)
            # attention_weights shape: [batch, num_heads, seq_len, seq_len]
            if len(output) > 1 and output[1] is not None:
                attn_weights = output[1].detach()  # [batch, num_heads, seq_len, seq_len]
                self.attention_weights[layer_idx] = attn_weights
        return hook

    def register_hooks(self):
        """Register forward hooks on all layers to capture attention"""
        for layer_idx in range(self.num_layers):
            layer = self._get_layer(layer_idx)
            # Hook into the self_attn module
            hook = layer.self_attn.register_forward_hook(self._create_attention_hook(layer_idx))
            self.hooks.append(hook)
        print(f"Registered {len(self.hooks)} attention hooks")

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}

    def extract_cot_attention(self, input_ids, attention_mask, cot_positions):
        """
        Run forward pass and extract attention patterns for CoT positions

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            cot_positions: List of CoT token positions (e.g., [10, 11, 12, 13, 14])

        Returns:
            cot_attention: Dict mapping layer_idx to attention matrix
                          Each matrix is [num_heads, num_latent, num_latent]
                          where num_latent = len(cot_positions)
        """
        # Clear previous attention weights
        self.attention_weights = {}

        # Run forward pass (this triggers hooks)
        with torch.no_grad():
            outputs = self.model.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True  # Request attention weights
            )

        # Extract CoT-to-CoT attention for each layer
        cot_attention = {}
        for layer_idx in range(self.num_layers):
            if layer_idx in self.attention_weights:
                # Full attention: [batch, num_heads, seq_len, seq_len]
                full_attn = self.attention_weights[layer_idx]

                # Extract CoT positions only
                # Shape: [batch, num_heads, num_latent, num_latent]
                cot_attn = full_attn[0, :, cot_positions, :][:, :, cot_positions]

                # Convert to numpy and store
                cot_attention[layer_idx] = cot_attn.float().cpu().numpy()

        return cot_attention

    def aggregate_across_heads(self, attention_dict):
        """
        Average attention across all heads

        Args:
            attention_dict: Dict mapping layer_idx to [num_heads, num_latent, num_latent]

        Returns:
            aggregated: Dict mapping layer_idx to [num_latent, num_latent]
        """
        aggregated = {}
        for layer_idx, attn in attention_dict.items():
            # Average across heads (axis 0)
            aggregated[layer_idx] = np.mean(attn, axis=0)
        return aggregated

    def aggregate_across_layers(self, attention_dict):
        """
        Average attention across all layers

        Args:
            attention_dict: Dict mapping layer_idx to [num_heads, num_latent, num_latent]
                           or [num_latent, num_latent] if already head-aggregated

        Returns:
            aggregated: [num_heads, num_latent, num_latent] or [num_latent, num_latent]
        """
        # Stack all layers
        all_layers = np.stack([attention_dict[i] for i in sorted(attention_dict.keys())])
        # Average across layers (axis 0)
        return np.mean(all_layers, axis=0)
