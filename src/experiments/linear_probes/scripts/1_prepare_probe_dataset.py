"""
Story 1.1: Prepare Balanced Test Dataset

This script:
1. Loads the existing error_analysis_dataset.json (452 correct, 462 incorrect)
2. Samples 50 correct + 50 incorrect solutions
3. Extracts layer 15 activations (final layer) for these 100 samples
4. Combines with existing layer 8 and layer 14 activations
5. Saves balanced probe training dataset

Output: src/experiments/linear_probes/data/probe_dataset_100.json

Shape: 100 samples x 3 layers (8, 14, 15) x 6 tokens x 2048 dims
"""

import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType


class Layer15Extractor:
    """Extracts layer 15 activations from CODI model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize the extractor with CODI LLaMA model."""
        self.device = device
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
            target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            init_lora_weights=True,
        )

        # Load model
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

        # Convert to float32
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = training_args.num_latent

        print("Model loaded successfully!")
        print(f"  Architecture: Llama-3.2-1B-Instruct (16 layers)")
        print(f"  Hidden dim: 2048")
        print(f"  Extracting layer: 15 (final layer)")

    def extract_layer15_thoughts(self, problem_text: str) -> List[List[float]]:
        """
        Extract layer 15 continuous thought representations.

        Args:
            problem_text: The problem question text

        Returns:
            List of 6 thought vectors (one per latent token), each with 2048 dims
        """
        layer_idx = 15  # Final layer
        thoughts = []

        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward through model
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

            # Process latent thoughts
            latent_embd = bot_emb

            # Extract thoughts from all 6 latent iterations
            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Extract layer 15 thought (last token)
                thought = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                thoughts.append(thought.squeeze(0).tolist())

                # Update latent embedding for next iteration
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection if used
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

        return thoughts


def load_error_dataset(path: str) -> Dict:
    """Load the error analysis dataset."""
    print(f"Loading error analysis dataset from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)

    print(f"  Correct solutions: {len(data['correct_solutions'])}")
    print(f"  Incorrect solutions: {len(data['incorrect_solutions'])}")
    print(f"  Existing layers: {data['metadata']['layers']}")

    return data


def sample_balanced_dataset(data: Dict, n_correct: int = 50, n_incorrect: int = 50, seed: int = 42) -> Dict:
    """Sample balanced dataset."""
    random.seed(seed)
    np.random.seed(seed)

    print(f"\nSampling balanced dataset...")
    print(f"  Target: {n_correct} correct + {n_incorrect} incorrect")

    correct = random.sample(data['correct_solutions'], n_correct)
    incorrect = random.sample(data['incorrect_solutions'], n_incorrect)

    print(f"  Sampled: {len(correct)} correct + {len(incorrect)} incorrect")

    return {
        'correct': correct,
        'incorrect': incorrect
    }


def extract_and_combine(
    sampled_data: Dict,
    extractor: Layer15Extractor,
    output_path: str
) -> Dict:
    """
    Extract layer 15 activations and combine with existing layers.

    Args:
        sampled_data: Dict with 'correct' and 'incorrect' keys
        extractor: Layer15Extractor instance
        output_path: Path to save final dataset

    Returns:
        Combined dataset with all layers
    """
    print("\nExtracting layer 15 activations...")

    final_dataset = []
    total = len(sampled_data['correct']) + len(sampled_data['incorrect'])
    count = 0

    for label, samples in [('correct', sampled_data['correct']), ('incorrect', sampled_data['incorrect'])]:
        print(f"\nProcessing {label} samples...")

        for sample in samples:
            count += 1
            print(f"  [{count}/{total}] {sample['pair_id']}")

            # Extract layer 15
            layer15_thoughts = extractor.extract_layer15_thoughts(sample['question'])

            # Combine with existing layers
            combined_sample = {
                'pair_id': sample['pair_id'],
                'variant': sample['variant'],
                'question': sample['question'],
                'ground_truth': sample['ground_truth'],
                'predicted': sample['predicted'],
                'is_correct': sample['is_correct'],
                'thoughts': {
                    'layer_8': sample['continuous_thoughts']['middle'],   # Layer 8 (middle)
                    'layer_14': sample['continuous_thoughts']['late'],    # Layer 14 (late)
                    'layer_15': layer15_thoughts                          # Layer 15 (final)
                }
            }

            final_dataset.append(combined_sample)

    # Save dataset
    print(f"\nSaving dataset to {output_path}...")

    output_data = {
        'metadata': {
            'n_samples': len(final_dataset),
            'n_correct': len(sampled_data['correct']),
            'n_incorrect': len(sampled_data['incorrect']),
            'layers': ['layer_8', 'layer_14', 'layer_15'],
            'layer_indices': {'layer_8': 8, 'layer_14': 14, 'layer_15': 15},
            'n_tokens': 6,
            'hidden_dim': 2048,
            'extraction_date': datetime.now().isoformat(),
            'source': 'error_analysis_dataset.json + layer 15 extraction'
        },
        'samples': final_dataset
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"Total samples: {len(final_dataset)}")
    print(f"  Correct: {output_data['metadata']['n_correct']}")
    print(f"  Incorrect: {output_data['metadata']['n_incorrect']}")
    print(f"Layers: {output_data['metadata']['layers']}")
    print(f"Tokens per layer: {output_data['metadata']['n_tokens']}")
    print(f"Hidden dimension: {output_data['metadata']['hidden_dim']}")
    print("="*60)

    return output_data


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent.parent
    input_path = project_root / "src/experiments/sae_error_analysis/data/error_analysis_dataset.json"
    output_path = project_root / "src/experiments/linear_probes/data/probe_dataset_100.json"
    model_path = Path.home() / "codi_ckpt" / "llama_gsm8k"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load error dataset
    error_data = load_error_dataset(input_path)

    # Sample balanced dataset
    sampled_data = sample_balanced_dataset(error_data, n_correct=50, n_incorrect=50)

    # Initialize extractor
    extractor = Layer15Extractor(str(model_path))

    # Extract layer 15 and combine
    final_data = extract_and_combine(sampled_data, extractor, output_path)

    print(f"\n✅ Success! Probe dataset ready at:")
    print(f"   {output_path}")

    return final_data


if __name__ == "__main__":
    main()
