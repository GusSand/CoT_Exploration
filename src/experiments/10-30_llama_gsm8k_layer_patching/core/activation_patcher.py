"""
Activation patching infrastructure for layer-wise interventions
"""

import torch
from .model_loader import get_layer_module


class ActivationPatcher:
    """
    Context manager for patching activations at a specific layer
    """

    def __init__(self, model, layer_idx, positions, replacement_activations):
        """
        Args:
            model: CODI model
            layer_idx: Which layer to patch
            positions: List of token positions to patch
            replacement_activations: Tensor of activations to use [batch, len(positions), hidden_dim]
        """
        self.model = model
        self.layer_idx = layer_idx
        self.positions = positions
        self.replacement_activations = replacement_activations.to(model.codi.device)
        self.hook_handle = None

    def patch_hook(self, module, input, output):
        """
        Hook function that replaces activations at specified positions
        """
        # output is tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Replace activations at specified positions
        for i, pos in enumerate(self.positions):
            hidden_states[:, pos, :] = self.replacement_activations[:, i, :]

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


def run_with_patching(model, input_ids, attention_mask, layer_idx, positions, replacement_activations):
    """
    Run model forward pass with patched activations at specific layer

    Args:
        model: CODI model
        input_ids: Input token IDs
        attention_mask: Attention mask
        layer_idx: Layer to patch
        positions: Token positions to patch
        replacement_activations: Activations to use for patching

    Returns:
        output: Model output with patched activations
    """
    with ActivationPatcher(model, layer_idx, positions, replacement_activations):
        with torch.no_grad():
            output = model.codi(input_ids=input_ids, attention_mask=attention_mask)

    return output


def extract_answer_logits(output, answer_start_pos, answer_length):
    """
    Extract logits for answer tokens only

    Args:
        output: Model output
        answer_start_pos: Starting position of answer tokens
        answer_length: Number of answer tokens

    Returns:
        answer_logits: Logits for answer tokens [batch, answer_length, vocab_size]
    """
    logits = output.logits
    answer_logits = logits[:, answer_start_pos:answer_start_pos + answer_length, :]
    return answer_logits
