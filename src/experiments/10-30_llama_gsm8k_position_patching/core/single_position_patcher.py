"""
Single-position activation patching infrastructure for fine-grained interventions
"""

import torch
from pathlib import Path
import sys

# Add layer_patching to path to reuse get_layer_module
layer_patching_path = Path(__file__).parent.parent.parent / "10-30_llama_gsm8k_layer_patching" / "core"
sys.path.insert(0, str(layer_patching_path))
from model_loader import get_layer_module


class SinglePositionPatcher:
    """
    Context manager for patching activation at a single (layer, position) combination
    """

    def __init__(self, model, layer_idx, position, replacement_activation):
        """
        Args:
            model: CODI model
            layer_idx: Which layer to patch (0-15)
            position: Single token position to patch (integer)
            replacement_activation: Tensor of activation to use [batch, hidden_dim]
                                   Note: Single position, so shape is [batch, hidden_dim]
        """
        self.model = model
        self.layer_idx = layer_idx
        self.position = position
        self.replacement_activation = replacement_activation.to(model.codi.device)
        self.hook_handle = None

    def patch_hook(self, module, input, output):
        """
        Hook function that replaces activation at specified position
        """
        # output is tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Replace activation at single position
        # replacement_activation shape: [batch, hidden_dim]
        hidden_states[:, self.position, :] = self.replacement_activation

        # Return modified output in same format
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        else:
            return hidden_states

    def __enter__(self):
        """Register the patching hook"""
        layer = get_layer_module(self.model, self.layer_idx)
        self.hook_handle = layer.register_forward_hook(self.patch_hook)
        return self

    def __exit__(self, *args):
        """Remove the patching hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()


def run_with_single_position_patching(model, input_ids, attention_mask, layer_idx,
                                      position, replacement_activation):
    """
    Run model forward pass with patched activation at single (layer, position)

    Args:
        model: CODI model
        input_ids: Input token IDs
        attention_mask: Attention mask
        layer_idx: Layer to patch (0-15)
        position: Token position to patch (integer)
        replacement_activation: Activation to use for patching [batch, hidden_dim]

    Returns:
        output: Model output with patched activation
    """
    with SinglePositionPatcher(model, layer_idx, position, replacement_activation):
        with torch.no_grad():
            output = model.codi(input_ids=input_ids, attention_mask=attention_mask)

    return output
