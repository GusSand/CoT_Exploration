"""
Iterative activation patcher with activation chaining

This implements the core innovation of Experiment 2:
- Sequential patching: Patch position N, then use the generated activation
  at position N+1 from that run to inform the next patch
- This tests whether CoT positions depend on each other sequentially vs in parallel
"""

import torch
import numpy as np
from typing import List, Dict, Optional


class IterativePatcher:
    """
    Patches CoT activations iteratively, chaining generated activations

    Strategy:
    1. Start with corrupted input, baseline run
    2. Patch position 0 at target layer with clean activation
    3. Run forward pass, extract generated activation at position 1
    4. Patch position 1 with clean activation
    5. Run forward pass, extract generated activation at position 2
    6. ... continue for all 5 positions

    This tests sequential dependency: does position N+1 depend on position N's value?
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        clean_activations: Dict[int, np.ndarray],  # {pos_idx: activation}
        num_positions: int = 5
    ):
        """
        Initialize iterative patcher

        Args:
            model: CODI model
            layer_idx: Layer index to patch at
            clean_activations: Dict mapping position index to clean activation
                              {0: [batch, hidden_dim], 1: [...], ...}
            num_positions: Number of CoT positions (default 5)
        """
        self.model = model
        self.layer_idx = layer_idx
        self.clean_activations = clean_activations
        self.num_positions = num_positions
        # Navigate to the correct layer in CODI wrapped model
        if hasattr(model.codi, 'base_model'):
            self.layer = model.codi.base_model.model.model.layers[layer_idx]
        else:
            self.layer = model.codi.model.layers[layer_idx]

        # Convert numpy arrays to tensors
        # Get device from the underlying model
        device = next(model.parameters()).device
        self.clean_activations_tensor = {}
        for pos_idx, act in clean_activations.items():
            if isinstance(act, np.ndarray):
                self.clean_activations_tensor[pos_idx] = torch.from_numpy(act).to(device)
            else:
                self.clean_activations_tensor[pos_idx] = act.to(device)

    def run_iterative_patching(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cot_positions: List[int]
    ) -> Dict:
        """
        Run iterative patching: patch positions sequentially, chaining activations

        Process:
        - For each position i from 0 to num_positions-1:
            1. Patch position i with clean activation
            2. Run forward pass
            3. Extract activation at position i+1 (if exists)
            4. Store results

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            cot_positions: List of CoT token positions in sequence

        Returns:
            results: Dict containing:
                - 'logits': Final logits after all patches [batch, seq_len, vocab_size]
                - 'trajectory': List of intermediate states
                - 'generated_activations': Dict of generated activations per position
        """
        trajectory = []
        generated_activations = {}

        # Current patched positions (cumulative)
        patched_positions_so_far = []

        for step in range(self.num_positions):
            # Patch position `step` with clean activation
            patched_positions_so_far.append(step)

            # Run forward pass with current patches
            logits, intermediate_activations = self._run_with_patches(
                input_ids,
                attention_mask,
                cot_positions,
                patched_positions_so_far
            )

            # Extract generated activation at next position (if exists)
            if step < self.num_positions - 1:
                next_pos = step + 1
                # Generated activation at position next_pos from this run
                generated_act = intermediate_activations[:, next_pos, :].detach()
                # Convert BFloat16 to Float32 for numpy compatibility
                generated_activations[next_pos] = generated_act.float().cpu().numpy()

            # Store trajectory
            trajectory.append({
                'step': step,
                'patched_positions': patched_positions_so_far.copy(),
                'logits': logits.detach().cpu(),
                'intermediate_activations': intermediate_activations.detach().cpu()
            })

        # Final logits after all patches
        final_logits = logits

        return {
            'logits': final_logits,
            'trajectory': trajectory,
            'generated_activations': generated_activations
        }

    def _run_with_patches(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cot_positions: List[int],
        positions_to_patch: List[int]
    ):
        """
        Run forward pass with specified positions patched

        Args:
            input_ids: Input IDs
            attention_mask: Attention mask
            cot_positions: All CoT positions
            positions_to_patch: Which positions to patch (indices into cot_positions)

        Returns:
            logits: Output logits
            intermediate_activations: Activations at the target layer
        """
        intermediate_activations = None

        def patch_hook(module, input, output):
            nonlocal intermediate_activations
            # output[0] is hidden states [batch, seq_len, hidden_dim]
            hidden_states = output[0].clone()

            # Patch specified positions
            for pos_idx in positions_to_patch:
                absolute_position = cot_positions[pos_idx]
                clean_act = self.clean_activations_tensor[pos_idx]
                hidden_states[:, absolute_position, :] = clean_act

            # Store intermediate for analysis
            intermediate_activations = hidden_states[:, cot_positions, :].detach()

            # Return modified output
            return (hidden_states,) + output[1:]

        # Register hook
        handle = self.layer.register_forward_hook(patch_hook)

        # Forward pass
        with torch.no_grad():
            outputs = self.model.codi(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Remove hook
        handle.remove()

        return logits, intermediate_activations


class ParallelPatcher:
    """
    Patches all CoT positions at once (for comparison with iterative)

    This is essentially the same as layer-wise patching from the previous experiment,
    included here for direct comparison.
    """

    def __init__(
        self,
        model,
        layer_idx: int,
        clean_activations: Dict[int, np.ndarray],
        num_positions: int = 5
    ):
        """
        Initialize parallel patcher

        Args:
            model: CODI model
            layer_idx: Layer to patch
            clean_activations: Clean activations for all positions
            num_positions: Number of positions
        """
        self.model = model
        self.layer_idx = layer_idx
        self.clean_activations = clean_activations
        self.num_positions = num_positions
        # Navigate to the correct layer in CODI wrapped model
        if hasattr(model.codi, 'base_model'):
            self.layer = model.codi.base_model.model.model.layers[layer_idx]
        else:
            self.layer = model.codi.model.layers[layer_idx]

        # Convert to tensors
        device = next(model.parameters()).device
        self.clean_activations_tensor = {}
        for pos_idx, act in clean_activations.items():
            if isinstance(act, np.ndarray):
                self.clean_activations_tensor[pos_idx] = torch.from_numpy(act).to(device)
            else:
                self.clean_activations_tensor[pos_idx] = act.to(device)

    def run_parallel_patching(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cot_positions: List[int]
    ):
        """
        Run parallel patching: patch all positions at once

        Args:
            input_ids: Input IDs
            attention_mask: Attention mask
            cot_positions: CoT positions

        Returns:
            logits: Output logits after patching all positions
        """
        patched_logits = None

        def patch_hook(module, input, output):
            nonlocal patched_logits
            # output[0] is hidden states
            hidden_states = output[0].clone()

            # Patch all positions at once
            for pos_idx in range(self.num_positions):
                absolute_position = cot_positions[pos_idx]
                clean_act = self.clean_activations_tensor[pos_idx]
                hidden_states[:, absolute_position, :] = clean_act

            return (hidden_states,) + output[1:]

        # Register hook
        handle = self.layer.register_forward_hook(patch_hook)

        # Forward pass
        with torch.no_grad():
            outputs = self.model.codi(input_ids=input_ids, attention_mask=attention_mask)
            patched_logits = outputs.logits

        # Remove hook
        handle.remove()

        return patched_logits
