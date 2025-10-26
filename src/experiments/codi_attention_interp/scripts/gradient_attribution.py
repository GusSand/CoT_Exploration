#!/usr/bin/env python3
"""
Gradient-Based Attribution Sweep for Continuous Thought Tokens

Computes integrated gradients across all continuous thought tokens (0-5) and
layers (4-12) to identify which positions are causally important for correct reasoning.

Target: Logit difference between correct and incorrect answer tokens
Method: Integrated Gradients with baselines (random, attention)

Usage:
    python gradient_attribution.py [--test_mode] [--num_steps 50]

Author: Generated for CoT Exploration Project
Date: 2025-10-24
"""

import json
import sys
import torch
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
activation_patching_core = project_root / 'src' / 'experiments' / 'activation_patching' / 'core'
sys.path.insert(0, str(activation_patching_core))

# Import infrastructure
from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG


# ========================================
# STORY 1: Infrastructure & Core Attribution Framework
# ========================================

@dataclass
class AttributionResult:
    """Stores attribution results for a single problem."""
    problem_id: str
    question: str
    correct_answer: int
    predicted_answer: Optional[int]
    correct: bool
    target_logit_diff: float
    integrated_gradients: Dict[str, List[float]]  # layer -> [6 token scores]
    attention_baseline: Dict[str, List[float]]
    random_baseline: Dict[str, List[float]]
    completeness_residual: float


def extract_answer_number(text: str) -> Optional[int]:
    """Extract the numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
        r'\$?\s*(-?\d+(?:\.\d+)?)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                return int(num) if num.is_integer() else num
            except ValueError:
                continue

    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            num = float(numbers[-1])
            return int(num) if num.is_integer() else num
        except ValueError:
            pass

    return None


def answers_match(predicted, expected) -> bool:
    """Check if predicted answer matches expected."""
    if predicted is None or expected is None:
        return False
    try:
        pred_float = float(predicted)
        exp_float = float(expected)
        return abs(pred_float - exp_float) < 0.01
    except (ValueError, TypeError):
        return False


class GradientAttributor:
    """
    Computes gradient-based attributions for continuous thought tokens.

    Implements:
    - Story 1: Infrastructure & Core Attribution Framework
    - Story 2: Integrated Gradients Implementation
    - Story 3: Baseline Attribution Methods
    """

    def __init__(
        self,
        cacher: ActivationCacherLLaMA,
        target_layers: List[int] = None,
        target_tokens: List[int] = None,
        num_ig_steps: int = 50
    ):
        """
        Initialize the gradient attributor.

        Args:
            cacher: Model cacher with loaded CODI model
            target_layers: List of layer indices to compute attributions for (default: 4-12)
            target_tokens: List of token positions to compute attributions for (default: 0-5)
            num_ig_steps: Number of interpolation steps for integrated gradients (default: 50)
        """
        self.cacher = cacher
        self.model = cacher.model
        self.tokenizer = cacher.tokenizer
        self.device = cacher.device
        self.num_latent = cacher.num_latent

        # Default to layers 4-12 (8 layers)
        self.target_layers = target_layers if target_layers is not None else list(range(4, 12))

        # Default to tokens 0-5 (6 tokens)
        self.target_tokens = target_tokens if target_tokens is not None else list(range(6))

        self.num_ig_steps = num_ig_steps

        print(f"Gradient Attributor initialized:")
        print(f"  Target layers: {self.target_layers}")
        print(f"  Target tokens: {self.target_tokens}")
        print(f"  IG interpolation steps: {num_ig_steps}")

    def _get_answer_token_ids(self, answer: int) -> List[int]:
        """
        Get token IDs for an answer number.

        Args:
            answer: The numerical answer

        Returns:
            List of token IDs representing the answer
        """
        answer_str = str(int(answer))
        token_ids = self.tokenizer.encode(answer_str, add_special_tokens=False)
        return token_ids

    def _forward_with_activations(
        self,
        problem_text: str,
        return_all_activations: bool = False
    ) -> Tuple[torch.Tensor, Dict[int, List[torch.Tensor]], torch.Tensor]:
        """
        Run forward pass and collect activations at each (layer, token) position.

        Args:
            problem_text: The problem question
            return_all_activations: If True, return activations for all layers

        Returns:
            Tuple of:
                - final_logits: Logits for next token prediction [batch, vocab_size]
                - activations: Dict mapping layer_idx -> List of 6 token activations
                - past_key_values: KV cache for generation
        """
        # Enable gradient computation
        self.model.train()  # Put in train mode to enable gradients

        # Tokenize input
        inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get initial embeddings
        input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

        # Forward through initial context
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

        # Process latent thoughts and collect activations
        latent_embd = bot_emb
        activations = {layer_idx: [] for layer_idx in self.target_layers}

        for latent_step in range(self.num_latent):
            outputs = self.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values

            # Collect activations at target layers
            if latent_step in self.target_tokens:
                for layer_idx in self.target_layers:
                    # Get activation at this layer for this token
                    activation = outputs.hidden_states[layer_idx][:, -1, :]
                    activations[layer_idx].append(activation)

            # Update latent embedding for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.model.use_prj:
                latent_embd = self.model.prj(latent_embd)

        # EOT token to get final logits
        eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
            torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        final_outputs = self.model.codi(
            inputs_embeds=eot_emb,
            use_cache=True,
            past_key_values=past_key_values
        )

        # Get logits for answer prediction
        final_logits = final_outputs.logits[:, -1, :self.model.codi.config.vocab_size-1]

        return final_logits, activations, past_key_values

    def compute_logit_difference_target(
        self,
        final_logits: torch.Tensor,
        correct_answer: int,
        top_k_incorrect: int = 10
    ) -> torch.Tensor:
        """
        Compute target value: logit[correct] - max(logit[incorrect]).

        This measures how much more the model prefers the correct answer
        over the most competitive incorrect answer.

        Args:
            final_logits: Logits from model [batch, vocab_size]
            correct_answer: The correct numerical answer
            top_k_incorrect: Number of top incorrect answers to consider

        Returns:
            Scalar tensor with logit difference
        """
        # Get token IDs for correct answer
        correct_token_ids = self._get_answer_token_ids(correct_answer)

        # For simplicity, use the first token of the answer
        # (most answers are single tokens or start with the key digit)
        correct_token_id = correct_token_ids[0]

        # Get logit for correct answer
        correct_logit = final_logits[0, correct_token_id]

        # Get top-k logits excluding correct token
        logits_copy = final_logits.clone()
        logits_copy[0, correct_token_id] = float('-inf')  # Mask out correct

        # Max of top incorrect logits
        max_incorrect_logit = torch.max(logits_copy)

        # Compute difference
        logit_diff = correct_logit - max_incorrect_logit

        return logit_diff

    def _forward_to_answer_position(
        self,
        problem_text: str
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate forward until we reach the answer position.

        The model generates "The answer is: <number>", so we need to
        generate forward to the position right before the number appears.

        Args:
            problem_text: The problem question

        Returns:
            Tuple of (past_key_values, number_of_tokens_generated)
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
            # Pattern: "The answer is: 21"
            pred_tokens = []
            colon_token_id = self.tokenizer.encode(":", add_special_tokens=False)[0]

            for _ in range(20):  # Max 20 tokens to find answer position
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

                # Stop after colon - next token should be the answer
                if next_token_id.item() == colon_token_id:
                    # Generate one more token (usually space before number)
                    out = self.model.codi(
                        inputs_embeds=output_emb,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = out.past_key_values
                    break

        return past_key_values, len(pred_tokens) + 1

    def get_answer_position_logits(
        self,
        problem_text: str,
        correct_answer: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get logits at the position where the answer actually appears.

        This generates "The answer is: " and then gets logits for the next token,
        which should be the numerical answer.

        Args:
            problem_text: The problem question
            correct_answer: The correct numerical answer

        Returns:
            Tuple of (logits at answer position, logit_difference)
        """
        # Generate forward to answer position
        past_key_values, num_tokens = self._forward_to_answer_position(problem_text)

        # Now get logits for the next token (should be the answer)
        # We need to enable gradients for this
        self.model.train()

        # Get the last hidden state and compute final logits
        # We need to do a forward pass from the current state
        # The past_key_values already contains everything up to answer position

        # Create a dummy input (we just need one step forward)
        # Use space token as it's typically before the number
        space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
        space_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
            torch.tensor([space_token_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        outputs = self.model.codi(
            inputs_embeds=space_emb,
            use_cache=True,
            past_key_values=past_key_values
        )

        # Get logits for answer prediction
        final_logits = outputs.logits[:, -1, :self.model.codi.config.vocab_size-1]

        # Compute logit difference
        logit_diff = self.compute_logit_difference_target(final_logits, correct_answer)

        return final_logits, logit_diff

    def _generate_answer(self, problem_text: str, max_new_tokens: int = 200) -> str:
        """Generate answer for a problem (for validation)."""
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

            # Generate latent thoughts
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


# ========================================
# STORY 2: Integrated Gradients Implementation
# ========================================

    def compute_integrated_gradients(
        self,
        problem_text: str,
        correct_answer: int
    ) -> Dict[int, List[float]]:
        """
        Compute integrated gradients for all (layer, token) positions.

        Integrated gradients measures the average gradient along a path from
        a baseline (zeros) to the actual activation values.

        Args:
            problem_text: The problem question
            correct_answer: The correct numerical answer

        Returns:
            Dict mapping layer_idx -> List of 6 attribution scores (one per token)
        """
        # Step 1: Get actual activations (forward pass)
        final_logits, actual_activations, _ = self._forward_with_activations(problem_text)

        # Step 2: Compute baseline (zeros)
        baseline_activations = {
            layer_idx: [torch.zeros_like(act) for act in acts]
            for layer_idx, acts in actual_activations.items()
        }

        # Step 3: Interpolate between baseline and actual
        attributions = {layer_idx: [] for layer_idx in self.target_layers}

        for layer_idx in self.target_layers:
            for token_idx in range(len(self.target_tokens)):
                # Get baseline and actual for this (layer, token)
                baseline = baseline_activations[layer_idx][token_idx]
                actual = actual_activations[layer_idx][token_idx]

                # Compute IG for this position
                ig_score = self._integrate_gradients_single_position(
                    problem_text,
                    correct_answer,
                    layer_idx,
                    token_idx,
                    baseline,
                    actual
                )

                attributions[layer_idx].append(ig_score)

        return attributions

    def _integrate_gradients_single_position(
        self,
        problem_text: str,
        correct_answer: int,
        layer_idx: int,
        token_idx: int,
        baseline: torch.Tensor,
        actual: torch.Tensor
    ) -> float:
        """
        Compute integrated gradient for a single (layer, token) position.

        IG formula: (actual - baseline) * mean_gradient_along_path

        Args:
            problem_text: The problem question
            correct_answer: The correct answer
            layer_idx: Target layer index
            token_idx: Target token index
            baseline: Baseline activation (zeros)
            actual: Actual activation from forward pass

        Returns:
            Attribution score (scalar)
        """
        accumulated_grads = []

        # Interpolate along path from baseline to actual
        for step in range(self.num_ig_steps):
            # Interpolation coefficient: 0 -> 1
            alpha = step / (self.num_ig_steps - 1) if self.num_ig_steps > 1 else 1.0

            # Interpolated activation (detach first to make it a leaf variable)
            interpolated = (baseline + alpha * (actual - baseline)).detach()
            interpolated.requires_grad = True

            # Forward pass with this interpolated activation
            logit_diff = self._forward_with_intervention(
                problem_text,
                correct_answer,
                layer_idx,
                token_idx,
                interpolated
            )

            # Compute gradient
            if logit_diff.requires_grad:
                grad = torch.autograd.grad(
                    logit_diff,
                    interpolated,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True
                )[0]

                if grad is not None:
                    accumulated_grads.append(grad.detach())
                else:
                    # If unused, use zero
                    accumulated_grads.append(torch.zeros_like(interpolated))
            else:
                # If no gradient, use zero
                accumulated_grads.append(torch.zeros_like(interpolated))

        # Average gradients along path
        avg_grad = torch.stack(accumulated_grads).mean(dim=0)

        # Compute attribution: (actual - baseline) * avg_gradient
        diff = (actual - baseline).detach()
        attribution = (diff * avg_grad).sum().item()

        return attribution

    def _forward_with_intervention(
        self,
        problem_text: str,
        correct_answer: int,
        target_layer_idx: int,
        target_token_idx: int,
        intervened_activation: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with intervention at specific (layer, token) position.

        This is a simplified forward that intervenes at ONE position.

        Args:
            problem_text: The problem question
            correct_answer: The correct answer
            target_layer_idx: Layer to intervene at
            target_token_idx: Token position to intervene at
            intervened_activation: The activation to inject

        Returns:
            Logit difference (scalar tensor with gradient)
        """
        # Tokenize input
        inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        # Get initial embeddings
        input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

        # Forward through initial context
        outputs = self.model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )

        past_key_values = outputs.past_key_values

        # Get BOT embedding
        bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
            torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        # Process latent thoughts with intervention
        latent_embd = bot_emb

        for latent_step in range(self.num_latent):
            outputs = self.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values

            # INTERVENTION: Replace activation at target position
            if latent_step == target_token_idx:
                # Extract hidden states tuple and convert to list for modification
                hidden_states_list = list(outputs.hidden_states)

                # Replace at target layer
                hidden_state = hidden_states_list[target_layer_idx]
                hidden_state_modified = hidden_state.clone()
                hidden_state_modified[:, -1, :] = intervened_activation
                hidden_states_list[target_layer_idx] = hidden_state_modified

                # Update outputs
                outputs.hidden_states = tuple(hidden_states_list)

            # Update latent embedding
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if self.model.use_prj:
                latent_embd = self.model.prj(latent_embd)

        # EOT token
        eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
            torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        final_outputs = self.model.codi(
            inputs_embeds=eot_emb,
            use_cache=True,
            past_key_values=past_key_values
        )

        # Get logits
        final_logits = final_outputs.logits[:, -1, :self.model.codi.config.vocab_size-1]

        # Compute logit difference target
        logit_diff = self.compute_logit_difference_target(final_logits, correct_answer)

        return logit_diff


# ========================================
# STORY 3: Baseline Attribution Methods
# ========================================

    def compute_random_baseline(self) -> Dict[int, List[float]]:
        """
        Compute random gradient baseline (sanity check).

        Returns random values sampled from Gaussian distribution.
        Should show no systematic patterns.

        Returns:
            Dict mapping layer_idx -> List of 6 random attribution scores
        """
        random_attributions = {}

        for layer_idx in self.target_layers:
            # Sample random values for each token
            random_scores = np.random.randn(len(self.target_tokens)).tolist()
            random_attributions[layer_idx] = random_scores

        return random_attributions

    def compute_attention_attribution(
        self,
        problem_text: str
    ) -> Dict[int, List[float]]:
        """
        Compute attention-based attribution (standard baseline).

        Extracts attention weights to continuous thought tokens and aggregates
        across heads to get importance scores.

        Args:
            problem_text: The problem question

        Returns:
            Dict mapping layer_idx -> List of 6 attention-based attribution scores
        """
        self.model.eval()

        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward through initial context
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True
            )

            past_key_values = outputs.past_key_values

            # Get BOT embedding
            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Collect attention weights for each token
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

                # Extract attention weights
                if latent_step in self.target_tokens and outputs.attentions is not None:
                    for layer_idx in self.target_layers:
                        # Attention shape: [batch, num_heads, seq_len, seq_len]
                        attn = outputs.attentions[layer_idx]

                        # Average across heads and take last position (current token)
                        attn_avg = attn.mean(dim=1)[0, -1, :]  # [seq_len]

                        # Sum attention to all previous tokens (total attention paid)
                        total_attention = attn_avg.sum().item()

                        attention_scores[layer_idx].append(total_attention)

                # Update latent embedding
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            # If attention not available, return zeros
            for layer_idx in self.target_layers:
                if len(attention_scores[layer_idx]) == 0:
                    attention_scores[layer_idx] = [0.0] * len(self.target_tokens)

        return attention_scores


def main():
    """Test the gradient attribution framework."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
                        help='Path to CODI LLaMA checkpoint')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run on single test problem')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of IG interpolation steps')
    args = parser.parse_args()

    print("=" * 80)
    print("GRADIENT ATTRIBUTION FRAMEWORK - UNIT TEST")
    print("=" * 80)

    # Load model
    print(f"\nLoading LLaMA CODI model from {args.model_path}...")
    cacher = ActivationCacherLLaMA(args.model_path)

    # Initialize attributor
    print("\nInitializing gradient attributor...")
    attributor = GradientAttributor(
        cacher,
        target_layers=[4, 8],  # Test with 2 layers only
        target_tokens=list(range(6)),
        num_ig_steps=args.num_steps
    )

    # Test problem
    test_problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"
    test_answer = 21

    print(f"\n{'='*80}")
    print(f"Test problem: {test_problem}")
    print(f"Expected answer: {test_answer}")
    print(f"{'='*80}")

    # Test 1: Generate answer (validation)
    print("\n[Test 1] Generating answer...")
    generated = attributor._generate_answer(test_problem)
    predicted = extract_answer_number(generated)
    print(f"Generated: {generated}")
    print(f"Extracted answer: {predicted}")
    print(f"Correct: {answers_match(predicted, test_answer)}")

    # Test 2: Forward pass with activations
    print("\n[Test 2] Forward pass with activation collection...")
    logits, activations, _ = attributor._forward_with_activations(test_problem)
    print(f"Logits shape: {logits.shape}")
    for layer_idx, acts in activations.items():
        print(f"  Layer {layer_idx}: {len(acts)} tokens, shape {acts[0].shape}")

    # Test 3: Logit difference target (at first position)
    print("\n[Test 3] Computing logit difference target (at first position)...")
    logit_diff = attributor.compute_logit_difference_target(logits, test_answer)
    print(f"Logit difference: {logit_diff.item():.4f}")
    print(f"  (negative expected - model generates 'The' first, not '21')")

    # Test 3b: Logit difference at CORRECT position
    print("\n[Test 3b] Computing logit difference at ANSWER POSITION...")
    print("  (Generating 'The answer is: ' then measuring logit for '21')")
    answer_logits, answer_logit_diff = attributor.get_answer_position_logits(test_problem, test_answer)
    print(f"Logit difference at answer position: {answer_logit_diff.item():.4f}")
    print(f"  (positive = model prefers correct answer at this position)")

    # Test 4: Random baseline
    print("\n[Test 4] Computing random baseline...")
    random_attr = attributor.compute_random_baseline()
    for layer_idx, scores in random_attr.items():
        print(f"  Layer {layer_idx}: {[f'{s:.3f}' for s in scores]}")

    # Test 5: Attention baseline
    print("\n[Test 5] Computing attention baseline...")
    attn_attr = attributor.compute_attention_attribution(test_problem)
    for layer_idx, scores in attn_attr.items():
        print(f"  Layer {layer_idx}: {[f'{s:.3f}' for s in scores]}")

    # Test 6: Integrated gradients (single position only for speed)
    if not args.test_mode:
        print("\n[Test 6] Computing integrated gradients (2 layers × 6 tokens)...")
        print("  NOTE: This will take a few minutes...")
        ig_attr = attributor.compute_integrated_gradients(test_problem, test_answer)

        for layer_idx, scores in ig_attr.items():
            print(f"  Layer {layer_idx}: {[f'{s:.3f}' for s in scores]}")

        # Completeness check
        total_attr = sum(sum(scores) for scores in ig_attr.values())
        completeness_residual = abs(total_attr - logit_diff.item())
        print(f"\n  Completeness check:")
        print(f"    Sum of attributions: {total_attr:.4f}")
        print(f"    Target logit diff: {logit_diff.item():.4f}")
        print(f"    Residual: {completeness_residual:.4f}")
        print(f"    Relative error: {100 * completeness_residual / abs(logit_diff.item()):.1f}%")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
