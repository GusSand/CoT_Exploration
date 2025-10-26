"""
Position-Specific Tuned Lens Model (Lite Version).

This module implements position-specific affine transformations for critical layers only.
Non-critical layers use shared position-agnostic transformations.

Critical Layers: 6, 9, 14, 15 (showed best performance in initial experiments)
Positions: 0-5 (all 6 continuous thought positions)

Architecture:
- Critical layers: 4 layers x 6 positions x 2048^2 parameters ~= 100M params
- Other layers: 12 layers x 2048^2 parameters ~= 50M params
- Total: ~150M parameters (vs 403M for full position-specific)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM


class AffineTransform(nn.Module):
    """Affine transformation: y = Wx + b"""

    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.register_parameter('bias', None)

        # Initialize near identity as per tuned-lens best practices
        nn.init.eye_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out = out + self.bias
        return out


class PositionSpecificTunedLens(nn.Module):
    """Position-specific Tuned Lens with critical layer specialization.

    Critical layers (6, 9, 14, 15) have separate transformations for each position.
    Non-critical layers use position-agnostic transformations.
    """

    def __init__(
        self,
        num_layers: int = 16,
        num_positions: int = 6,
        hidden_size: int = 2048,
        vocab_size: int = 32000,
        critical_layers: Optional[List[int]] = None,
        bias: bool = True,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_positions = num_positions
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.critical_layers = critical_layers if critical_layers is not None else [6, 9, 14, 15]
        self.use_layer_norm = use_layer_norm

        # Create layer normalization (applied before unembedding)
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)

        # Position-specific transformations for critical layers
        self.position_specific_transforms = nn.ModuleDict({
            str(layer): nn.ModuleList([
                AffineTransform(hidden_size, bias)
                for _ in range(num_positions)
            ])
            for layer in self.critical_layers
        })

        # Position-agnostic transformations for non-critical layers
        self.position_agnostic_transforms = nn.ModuleList([
            AffineTransform(hidden_size, bias)
            for layer in range(num_layers)
            if layer not in self.critical_layers
        ])

        # Map non-critical layer indices to transform indices
        self.layer_to_transform_idx = {}
        transform_idx = 0
        for layer in range(num_layers):
            if layer not in self.critical_layers:
                self.layer_to_transform_idx[layer] = transform_idx
                transform_idx += 1

    def forward(self, hidden_states, layer_idx, position_idx=None):
        """Apply layer and position-specific transformation.

        Args:
            hidden_states: (batch_size, hidden_size)
            layer_idx: Layer index (0-15)
            position_idx: Position index (0-5), required for critical layers

        Returns:
            Transformed hidden states (batch_size, hidden_size)
        """
        # Check if this is a critical layer
        if layer_idx in self.critical_layers:
            if position_idx is None:
                raise ValueError(f"position_idx required for critical layer {layer_idx}")

            # Apply position-specific transformation
            transform = self.position_specific_transforms[str(layer_idx)][position_idx]
            transformed = transform(hidden_states)
        else:
            # Apply position-agnostic transformation
            transform_idx = self.layer_to_transform_idx[layer_idx]
            transform = self.position_agnostic_transforms[transform_idx]
            transformed = transform(hidden_states)

        # Apply layer normalization
        if self.use_layer_norm:
            transformed = self.layer_norm(transformed)

        return transformed


class PositionSpecificTunedLensWrapper:
    """Wrapper for Position-Specific Tuned Lens with CODI."""

    def __init__(
        self,
        model_path: str,
        hidden_size: int = 2048,
        num_layers: int = 16,
        num_positions: int = 6,
        vocab_size: int = 32000,
        device: str = "cuda",
        bias: bool = True,
        use_layer_norm: bool = True,
        critical_layers: Optional[List[int]] = None,
        base_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    ):
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_positions = num_positions
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.bias = bias
        self.use_layer_norm = use_layer_norm
        self.critical_layers = critical_layers if critical_layers is not None else [6, 9, 14, 15]
        self.base_model_name = base_model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load unembedding matrix
        self.unembedding = self._load_unembedding()

        # Create position-specific tuned lens
        self.tuned_lens = PositionSpecificTunedLens(
            num_layers=num_layers,
            num_positions=num_positions,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            critical_layers=critical_layers,
            bias=bias,
            use_layer_norm=use_layer_norm
        ).to(self.device)

    def _load_unembedding(self):
        """Load unembedding matrix from base model."""
        print(f"Loading unembedding from {self.base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            device_map=self.device
        )

        # Extract unembedding weight (lm_head)
        unembedding = base_model.lm_head.weight.clone().detach()

        # Clean up
        del base_model
        torch.cuda.empty_cache()

        print(f"Unembedding shape: {unembedding.shape}")
        return unembedding.to(self.device)

    def forward(self, hidden_states, layer_idx, position_idx=None):
        """Forward pass through tuned lens and unembedding.

        Args:
            hidden_states: (batch_size, hidden_size)
            layer_idx: Layer index
            position_idx: Position index (required for critical layers)

        Returns:
            logits: (batch_size, vocab_size)
        """
        # Apply tuned lens transformation
        transformed = self.tuned_lens(hidden_states, layer_idx, position_idx)

        # Apply unembedding: logits = transformed @ unembedding.T
        logits = torch.matmul(transformed, self.unembedding.t())

        return logits

    def decode(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        position_idx: int,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Decode hidden states to tokens.

        Args:
            hidden_states: (batch_size, hidden_size)
            layer_idx: Layer index
            position_idx: Position index
            top_k: Number of top tokens to return

        Returns:
            Dictionary with logits, token_ids, tokens, probs
        """
        with torch.no_grad():
            logits = self.forward(hidden_states, layer_idx, position_idx)

            # Get top-k predictions
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

            # Decode tokens
            tokens = [
                self.tokenizer.decode([token_id])
                for token_id in top_ids[0].cpu().tolist()
            ]

            return {
                'logits': logits,
                'token_ids': top_ids,
                'tokens': tokens,
                'probs': top_probs
            }

    def save_tuned_lens(self, output_path: str):
        """Save trained position-specific tuned lens."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model state dict
        torch.save(
            self.tuned_lens.state_dict(),
            output_dir / "params.pt"
        )

        # Save config
        config = {
            'num_layers': self.num_layers,
            'num_positions': self.num_positions,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'critical_layers': self.critical_layers,
            'bias': self.bias,
            'use_layer_norm': self.use_layer_norm,
        }
        torch.save(config, output_dir / "config.pt")

        print(f"Position-specific tuned lens saved to: {output_path}")

    def load_tuned_lens(self, lens_path: str):
        """Load pre-trained position-specific tuned lens."""
        lens_dir = Path(lens_path)

        # Load state dict
        state_dict = torch.load(lens_dir / "params.pt", map_location=self.device, weights_only=False)
        self.tuned_lens.load_state_dict(state_dict)
        self.tuned_lens = self.tuned_lens.to(self.device)

        print(f"Position-specific tuned lens loaded from: {lens_path}")

    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer."""
        return [p for p in self.tuned_lens.parameters() if p.requires_grad]

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())


def create_position_specific_tuned_lens(
    config: Dict[str, Any],
    device: str = "cuda"
) -> PositionSpecificTunedLensWrapper:
    """Factory function to create position-specific tuned lens.

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        PositionSpecificTunedLensWrapper instance
    """
    # Determine base model name
    base_model_names = {
        'llama': 'meta-llama/Llama-3.2-1B-Instruct',
        'gpt2': 'openai-community/gpt2'
    }
    base_model_name = base_model_names.get(
        config['model']['name'],
        'meta-llama/Llama-3.2-1B-Instruct'
    )

    # Get critical layers from config or use default
    critical_layers = config.get('tuned_lens', {}).get('critical_layers', [6, 9, 14, 15])

    return PositionSpecificTunedLensWrapper(
        model_path=config['model']['checkpoint_path'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_positions=config['model'].get('num_ct_tokens', 6),
        vocab_size=config['model']['vocab_size'],
        device=device,
        bias=config['tuned_lens'].get('use_layer_norm', True),
        use_layer_norm=config['tuned_lens'].get('use_layer_norm', True),
        critical_layers=critical_layers,
        base_model_name=base_model_name
    )
