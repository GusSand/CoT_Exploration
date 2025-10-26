"""
CODI Interface for Mechanistic Interpretability

Provides clean interface for:
1. Loading CODI model
2. Extracting continuous thoughts
3. Manipulating continuous thoughts (position-wise zeroing)
4. Measuring step importance via KL divergence

Adapted from:
- activation_patching/core/cache_activations_llama.py (ActivationCacherLLaMA)
- scripts/experiments/run_ablation_N_tokens_llama.py (NTokenPatcher)
"""

import torch
import torch.nn.functional as F
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType
from safetensors.torch import load_file


class CODIInterface:
    """
    Interface for loading CODI and extracting continuous thoughts.

    Usage:
        interface = CODIInterface(model_path="~/codi_ckpt/llama_gsm8k/")
        thoughts = interface.extract_continuous_thoughts("problem text")
    """

    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Load CODI LLaMA model.

        Args:
            model_path: Path to CODI checkpoint directory
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        self.model_path = model_path

        print(f"Loading CODI LLaMA model from {model_path}...")

        # Parse arguments for CODI model
        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=[
                '--model_name_or_path', 'meta-llama/Llama-3.2-1B-Instruct',
                '--output_dir', './tmp',
                '--num_latent', '6',
                '--use_lora', 'True',
                '--ckpt_dir', model_path,
                '--use_prj', 'True',
                '--prj_dim', '2048',
                '--lora_r', '128',
                '--lora_alpha', '32',
                '--lora_init', 'True',
            ]
        )

        # Modify for inference
        model_args.train = False
        training_args.greedy = True

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj',
                          'gate_proj', 'up_proj', 'down_proj'],
            init_lora_weights=True,
        )

        # Load model
        self.model = CODI(model_args, training_args, lora_config)

        # Load checkpoint weights
        try:
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        except Exception:
            state_dict = torch.load(
                os.path.join(model_path, "pytorch_model.bin"),
                map_location='cpu'
            )

        self.model.load_state_dict(state_dict, strict=False)
        self.model.codi.tie_weights()

        # Convert to float32 (checkpoint has mixed precision)
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = 6  # CODI uses 6 continuous thought tokens

        print("✅ CODI model loaded successfully!")
        print(f"   Architecture: Llama-3.2-1B-Instruct")
        print(f"   Layers: 16, Hidden dim: 2048")
        print(f"   Latent tokens: {self.num_latent}")

    def extract_continuous_thoughts(
        self,
        problem_text: str,
        layer_idx: int = 8  # Middle layer by default
    ) -> List[torch.Tensor]:
        """
        Extract all 6 continuous thought activations at specified layer.

        Args:
            problem_text: Problem question text
            layer_idx: Layer to extract from (0-15, default=8 for middle)

        Returns:
            List of 6 activation tensors, each shape (1, 2048)
        """
        continuous_thoughts = []

        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(
                self.model.codi,
                self.model.model_name
            )(input_ids).to(self.device)

            # Forward through model for context
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            # Get BOT (Beginning of Thought) embedding
            bot_emb = self.model.get_embd(
                self.model.codi,
                self.model.model_name
            )(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Process 6 latent tokens
            latent_embd = bot_emb

            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Extract activation at specified layer
                activation = outputs.hidden_states[layer_idx][:, -1, :]
                continuous_thoughts.append(activation.cpu())

                # Update for next step
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

        return continuous_thoughts

    def generate_answer(
        self,
        problem_text: str,
        max_new_tokens: int = 200
    ) -> str:
        """
        Generate answer for a problem (baseline - no intervention).

        Args:
            problem_text: Problem question text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated answer text
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(
                self.model.codi,
                self.model.model_name
            )(input_ids).to(self.device)

            # Forward through model for context
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            # Get BOT (Beginning of Thought) embedding
            bot_emb = self.model.get_embd(
                self.model.codi,
                self.model.model_name
            )(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Process 6 latent thoughts
            for latent_step in range(self.num_latent):
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

            # Get EOT (End of Thought) embedding
            eot_emb = self.model.get_embd(
                self.model.codi,
                self.model.model_name
            )(
                torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            output_emb = eot_emb

            # Generate answer tokens (greedy decoding)
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
                output_emb = self.model.get_embd(
                    self.model.codi,
                    self.model.model_name
                )(next_token_id).unsqueeze(1)

            # Decode answer
            answer = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
            return answer



class StepImportanceMeasurer:
    """
    Measure step importance using position-wise zeroing and KL divergence.

    Usage:
        measurer = StepImportanceMeasurer(codi_interface)
        importance = measurer.measure_position_importance(
            problem_text="...",
            position=0  # Measure importance of position 0
        )
    """

    def __init__(self, codi_interface: CODIInterface, layer_idx: int = 8, debug: bool = False):
        """
        Initialize measurer.

        Args:
            codi_interface: CODIInterface instance
            layer_idx: Layer to intervene at (default=8 for middle layer)
            debug: Enable debug output
        """
        self.codi = codi_interface
        self.model = codi_interface.model
        self.tokenizer = codi_interface.tokenizer
        self.device = codi_interface.device
        self.layer_idx = layer_idx
        self.debug = debug

        # Hook state
        self.zero_until_position = None  # Zero positions [0...i-1]
        self.current_step = 0
        self.hook_handle = None

    def _get_layer_module(self):
        """Get transformer layer module for intervention."""
        try:
            return self.model.codi.base_model.model.model.layers[self.layer_idx]
        except AttributeError:
            return self.model.codi.model.layers[self.layer_idx]

    def _create_zeroing_hook(self):
        """
        Create hook that zeros positions [0...zero_until_position-1].

        For example, if zero_until_position=3:
        - Positions 0, 1, 2 are zeroed
        - Positions 3, 4, 5 are kept
        """
        def zeroing_hook(module, input, output):
            if self.zero_until_position is not None:
                # Check if we're in the range to zero
                if self.current_step < self.zero_until_position:
                    # Zero this position
                    if isinstance(output, tuple):
                        hidden_states = output[0].clone()
                        original_norm = torch.norm(hidden_states[:, -1, :]).item()
                        hidden_states[:, -1, :] = torch.zeros_like(hidden_states[:, -1, :])
                        if self.debug:
                            print(f"    [DEBUG] Zeroed position {self.current_step} (norm: {original_norm:.2f} → 0.00)")
                        return (hidden_states,) + output[1:]
                    else:
                        hidden_states = output.clone()
                        original_norm = torch.norm(hidden_states[:, -1, :]).item()
                        hidden_states[:, -1, :] = torch.zeros_like(hidden_states[:, -1, :])
                        if self.debug:
                            print(f"    [DEBUG] Zeroed position {self.current_step} (norm: {original_norm:.2f} → 0.00)")
                        return hidden_states

            return output

        return zeroing_hook

    def _generate_with_zeroing(
        self,
        problem_text: str,
        zero_until: int,
        max_new_tokens: int = 200
    ) -> str:
        """
        Generate answer with positions [0...zero_until-1] zeroed.

        Args:
            problem_text: Problem text
            zero_until: Zero positions [0...zero_until-1]
            max_new_tokens: Max tokens to generate

        Returns:
            Generated answer text
        """
        self.zero_until_position = zero_until
        self.current_step = 0

        # Register hook
        target_layer = self._get_layer_module()
        hook = self._create_zeroing_hook()
        self.hook_handle = target_layer.register_forward_hook(hook)

        try:
            with torch.no_grad():
                inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]

                # Get initial embeddings
                input_embd = self.model.get_embd(
                    self.model.codi,
                    self.model.model_name
                )(input_ids).to(self.device)

                # Forward for context
                outputs = self.model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values

                # Get BOT embedding
                bot_emb = self.model.get_embd(
                    self.model.codi,
                    self.model.model_name
                )(
                    torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
                ).unsqueeze(0)

                latent_embd = bot_emb

                # Generate 6 latent thoughts WITH ZEROING
                for latent_step in range(6):
                    self.current_step = latent_step  # Hook checks this

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

                # Disable hook before generation
                self.current_step = 999

                # Get EOT (End of Thought) embedding
                eot_emb = self.model.get_embd(
                    self.model.codi,
                    self.model.model_name
                )(
                    torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
                ).unsqueeze(0)

                outputs = self.model.codi(
                    inputs_embeds=eot_emb,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                output_emb = eot_emb

                # Generate answer tokens (greedy decoding)
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
                    output_emb = self.model.get_embd(
                        self.model.codi,
                        self.model.model_name
                    )(next_token_id).unsqueeze(1)

                # Decode answer
                answer = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)

        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None

        self.zero_until_position = None
        self.current_step = 0

        return answer

    def measure_position_importance(
        self,
        problem_text: str,
        position: int,
        use_kl_divergence: bool = True
    ) -> Dict:
        """
        Measure importance of a single position.

        Methodology:
        1. Baseline: Full continuous thoughts [0...5] → answer distribution
        2. Ablated: Zero positions [0...position-1], keep [position...5]
        3. Importance: KL(baseline || ablated)

        For example, position=2:
        - Baseline: Use all positions [0, 1, 2, 3, 4, 5]
        - Ablated: Zero [0, 1], keep [2, 3, 4, 5]
        - High KL → positions 0,1 were important

        Args:
            problem_text: Problem text
            position: Position to measure (0-5)
            use_kl_divergence: Use KL divergence (True) or answer match (False)

        Returns:
            Dictionary with importance score and metadata
        """
        assert 0 <= position <= 5, f"Position must be 0-5, got {position}"

        # Generate baseline answer
        baseline_answer = self.codi.generate_answer(problem_text)

        # Generate ablated answer (zero positions [0...position-1])
        if position == 0:
            # Can't zero anything before position 0
            ablated_answer = baseline_answer
            kl_divergence = 0.0
        else:
            ablated_answer = self._generate_with_zeroing(
                problem_text,
                zero_until=position
            )

            # For now, use simple answer matching
            # TODO: Implement proper KL divergence with logits
            kl_divergence = 0.0 if baseline_answer == ablated_answer else 1.0

        return {
            'position': position,
            'importance_score': kl_divergence,
            'baseline_answer': baseline_answer,
            'ablated_answer': ablated_answer,
            'answers_match': baseline_answer == ablated_answer,
            'method': 'position_wise_zeroing'
        }

    def measure_all_positions(
        self,
        problem_text: str
    ) -> List[Dict]:
        """
        Measure importance of all 6 positions.

        Args:
            problem_text: Problem text

        Returns:
            List of importance dictionaries for positions 0-5
        """
        results = []

        for position in range(6):
            result = self.measure_position_importance(problem_text, position)
            results.append(result)

        return results


def test_interface():
    """Test the CODI interface on a sample problem."""

    model_path = str(Path.home() / 'codi_ckpt/llama_gsm8k')

    print("="*60)
    print("Testing CODI Interface")
    print("="*60)

    # Load interface
    print("\n1. Loading CODI model...")
    interface = CODIInterface(model_path)

    # Test problem
    problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"

    # Test continuous thought extraction
    print(f"\n2. Extracting continuous thoughts for: {problem[:50]}...")
    thoughts = interface.extract_continuous_thoughts(problem)

    print(f"\n   ✅ Extracted {len(thoughts)} continuous thoughts")
    for i, thought in enumerate(thoughts):
        print(f"      Position {i}: shape={thought.shape}")

    # Test baseline generation
    print(f"\n3. Generating baseline answer...")
    baseline = interface.generate_answer(problem)
    print(f"   Answer: {baseline[:100]}...")

    # Test step importance measurement
    print(f"\n4. Testing step importance measurement...")
    measurer = StepImportanceMeasurer(interface)

    # Measure position 0
    result = measurer.measure_position_importance(problem, position=0)
    print(f"\n   Position 0 importance: {result['importance_score']:.3f}")
    print(f"   Baseline: {result['baseline_answer'][:50]}...")
    print(f"   Ablated:  {result['ablated_answer'][:50]}...")

    print("\n" + "="*60)
    print("✅ CODI Interface Test Complete!")
    print("="*60)


if __name__ == "__main__":
    test_interface()
