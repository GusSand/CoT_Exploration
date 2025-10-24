"""
Activation Intervention for Operation-Specific Circuits

This script performs causal interventions by patching Token 1 at Layer 8
with operation-specific activation vectors to test if this controls operation type.

Based on extract_continuous_thoughts.py with added intervention capability.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add CODI to path
codi_path = Path(__file__).parent.parent.parent.parent / "codi"
sys.path.insert(0, str(codi_path))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType


# Layer configuration
LAYER_CONFIG = {
    'early': 4,
    'middle': 8,
    'late': 14
}


class OperationIntervener:
    """Performs activation interventions on CODI model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize with CODI LLaMA model."""
        self.device = device
        print(f"Loading CODI LLaMA model from {model_path}...")

        # Parse arguments
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

        model_args.train = False
        training_args.greedy = True

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            init_lora_weights=True,
        )

        # Load model
        self.model = CODI(model_args, training_args, lora_config)

        # Load checkpoint
        import os
        from safetensors.torch import load_file
        try:
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        except Exception:
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')

        self.model.load_state_dict(state_dict, strict=False)
        self.model.codi.tie_weights()
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = training_args.num_latent

        # Intervention state
        self.intervention_vector = None
        self.intervention_token = None
        self.intervention_layer = None

        print("Model loaded successfully!")

    def run_with_intervention(
        self,
        problem_text: str,
        intervention_vector: Optional[torch.Tensor] = None,
        intervention_token: int = 1,
        intervention_layer: str = 'middle',
        max_tokens: int = 200
    ) -> str:
        """
        Run inference with optional activation intervention.

        Args:
            problem_text: Problem question
            intervention_vector: Vector to inject (None for baseline)
            intervention_token: Which latent token to intervene on (0-5)
            intervention_layer: Which layer ('early', 'middle', 'late')
            max_tokens: Max tokens to generate

        Returns:
            Generated answer text
        """
        self.intervention_vector = intervention_vector
        self.intervention_token = intervention_token
        self.intervention_layer = intervention_layer

        with torch.no_grad():
            # Tokenize
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward for context
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

                # INTERVENTION: Replace activation if this is the target token/layer
                if (self.intervention_vector is not None and
                    latent_step == self.intervention_token):

                    layer_idx = LAYER_CONFIG[self.intervention_layer]

                    # Get hidden states as a list (to allow modification)
                    hidden_states_list = list(outputs.hidden_states)

                    # Clone the target layer's hidden states
                    target_hidden = hidden_states_list[layer_idx].clone()

                    # Replace the last token's activation with intervention vector
                    target_hidden[:, -1, :] = self.intervention_vector.to(self.device)

                    # Update the hidden states
                    hidden_states_list[layer_idx] = target_hidden

                    # For next iteration, use the modified hidden state
                    latent_embd = target_hidden[:, -1, :].unsqueeze(1)
                else:
                    # Normal processing
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            # Generate answer
            # Get EOT (End of Thought) embedding
            eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Forward with EOT
            outputs = self.model.codi(
                inputs_embeds=eot_emb,
                use_cache=True,
                past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values

            # Generate answer tokens
            generated_ids = []
            for _ in range(max_tokens):
                outputs = self.model.codi(
                    input_ids=torch.tensor([[generated_ids[-1] if generated_ids else self.tokenizer.pad_token_id]],
                                          device=self.device),
                    use_cache=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()

                # Stop at EOS
                if next_token == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token)

            # Decode
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return answer

    def extract_answer_number(self, answer_text: str) -> Optional[str]:
        """Extract the final numerical answer from generated text."""
        # Look for #### pattern (GSM8k format)
        if '####' in answer_text:
            parts = answer_text.split('####')
            if len(parts) > 1:
                return parts[-1].strip()

        # Otherwise try to find last number
        import re
        numbers = re.findall(r'-?\d+\.?\d*', answer_text)
        if numbers:
            return numbers[-1]

        return None


def main():
    """Test intervention on single problem."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='~/codi_ckpt/llama_gsm8k', help='Path to CODI checkpoint')
    parser.add_argument('--test', action='store_true', help='Run on single test problem')
    args = parser.parse_args()

    # Initialize
    intervener = OperationIntervener(args.model_path)

    if args.test:
        # Test problem
        test_problem = {
            'question': 'If there are 3 cars in a parking lot and 2 more arrive, how many cars are there in total?',
            'operation_type': 'pure_addition'
        }

        print(f"\nTest Problem: {test_problem['question']}")
        print(f"Operation Type: {test_problem['operation_type']}")

        # Baseline
        print("\n=== BASELINE ===")
        answer = intervener.run_with_intervention(test_problem['question'])
        print(f"Answer: {answer}")
        print(f"Extracted: {intervener.extract_answer_number(answer)}")

        # Load intervention vectors
        vectors_path = Path(__file__).parent / 'activation_vectors.json'
        if vectors_path.exists():
            vectors = json.load(open(vectors_path))

            # Intervene with multiplication mean
            mult_mean = torch.tensor(vectors['operation_means']['pure_multiplication'])
            print("\n=== INTERVENTION: Pure Multiplication Mean ===")
            answer = intervener.run_with_intervention(
                test_problem['question'],
                intervention_vector=mult_mean,
                intervention_token=1,
                intervention_layer='middle'
            )
            print(f"Answer: {answer}")
            print(f"Extracted: {intervener.extract_answer_number(answer)}")


if __name__ == '__main__':
    main()
