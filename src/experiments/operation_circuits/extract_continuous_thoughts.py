"""
Extract continuous thoughts from CODI model for operation classification.

This script extracts the continuous thought representations (hidden states)
from the CODI model's [THINK] tokens across multiple layers.

Adapted from cache_activations_llama.py for operation-specific circuits analysis.
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType


# Layer configuration for LLaMA-3.2-1B (16 layers total)
LAYER_CONFIG = {
    'early': 4,    # Early layer (25% through model)
    'middle': 8,   # Middle layer (50% through model)
    'late': 14     # Late layer (87.5% - near final)
}


class ContinuousThoughtExtractor:
    """Extracts continuous thought representations from CODI model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize the extractor with CODI LLaMA model.

        Args:
            model_path: Path to CODI LLaMA checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
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

        # Create LoRA config for LLaMA
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

        # Convert to float32 to avoid dtype mismatches
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = training_args.num_latent

        print("Model loaded successfully!")
        print(f"  Architecture: Llama-3.2-1B-Instruct")
        print(f"  Layers: 16 (extracting from L{LAYER_CONFIG['early']}, L{LAYER_CONFIG['middle']}, L{LAYER_CONFIG['late']})")
        print(f"  Hidden dim: 2048")
        print(f"  Latent tokens: {self.num_latent}")

    def extract_continuous_thoughts(
        self,
        problem_text: str,
        layer_indices: Dict[str, int] = None
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Extract continuous thought representations for all latent tokens.

        Args:
            problem_text: The problem question text
            layer_indices: Dict mapping layer names to indices

        Returns:
            Dict mapping layer names to lists of thought tensors
            Each list contains num_latent tensors of shape [1, hidden_dim]
        """
        if layer_indices is None:
            layer_indices = LAYER_CONFIG

        # Store thoughts for each layer
        thoughts = {layer_name: [] for layer_name in layer_indices.keys()}

        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward through model to get initial context
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

            # Process latent thoughts
            latent_embd = bot_emb

            # Extract thoughts from ALL latent iterations
            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Extract thoughts from specified layers
                for layer_name, layer_idx in layer_indices.items():
                    # hidden_states is a tuple: (layer_0, layer_1, ..., layer_N)
                    # We want the last token ([:, -1, :])
                    thought = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                    thoughts[layer_name].append(thought)

                # Update latent embedding for next iteration
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection if used
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

        return thoughts

    def extract_problem(self, problem: Dict) -> Dict:
        """
        Extract continuous thoughts for a single problem.

        Args:
            problem: Dict with 'id', 'question', 'answer', etc.

        Returns:
            Dict with problem info and extracted thoughts
        """
        thoughts = self.extract_continuous_thoughts(problem['question'])

        # Convert tensors to lists for JSON serialization
        thoughts_serialized = {}
        for layer_name, thought_list in thoughts.items():
            thoughts_serialized[layer_name] = [
                t.squeeze(0).tolist() for t in thought_list
            ]

        return {
            'id': problem['id'],
            'question': problem['question'],
            'answer': problem['answer'],
            'thoughts': thoughts_serialized
        }

    def extract_dataset(
        self,
        dataset: Dict[str, List[Dict]],
        output_path: str,
        save_frequency: int = 10
    ) -> List[Dict]:
        """
        Extract continuous thoughts for entire dataset.

        Args:
            dataset: Dict mapping operation types to lists of problems
            output_path: Path to save results
            save_frequency: Save checkpoint every N problems

        Returns:
            List of results (one per problem)
        """
        results = []
        total_problems = sum(len(v) for v in dataset.values())
        problem_count = 0

        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        for operation_type, problems in dataset.items():
            print(f"\nProcessing {operation_type} ({len(problems)} problems)...")

            for i, problem in enumerate(problems):
                problem_count += 1
                print(f"  [{problem_count}/{total_problems}] {problem['id']}")

                # Extract thoughts
                result = self.extract_problem(problem)
                result['operation_type'] = operation_type
                results.append(result)

                # Save checkpoint
                if problem_count % save_frequency == 0:
                    checkpoint_path = output_path.replace('.json', f'_checkpoint_{problem_count}.json')
                    with open(checkpoint_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"    Saved checkpoint: {checkpoint_path}")

        # Save final results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save metadata
        metadata = {
            'total_problems': len(results),
            'operation_types': {
                op_type: len([r for r in results if r['operation_type'] == op_type])
                for op_type in dataset.keys()
            },
            'shape_info': {
                'num_tokens': self.num_latent,
                'num_layers': len(LAYER_CONFIG),
                'hidden_dim': 2048,
                'layers': LAYER_CONFIG
            }
        }

        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nExtraction complete!")
        print(f"Results: {output_path}")
        print(f"Metadata: {metadata_path}")

        return results


def main():
    """Test the extraction on a sample problem."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
                        help='Path to CODI LLaMA checkpoint')
    parser.add_argument('--test', action='store_true', help='Run test on single problem')
    args = parser.parse_args()

    # Initialize extractor
    extractor = ContinuousThoughtExtractor(args.model_path)

    if args.test:
        # Test on a simple problem
        problem = {
            'id': 'test_0',
            'question': "John has 3 bags with 7 apples each. How many apples does he have in total?",
            'answer': "John has 3 bags. Each bag has 7 apples. So he has 3 * 7 = 21 apples. #### 21"
        }

        print(f"\nTesting extraction on problem: {problem['question']}")

        result = extractor.extract_problem(problem)

        print("\nExtracted thoughts:")
        for layer_name, thoughts in result['thoughts'].items():
            print(f"  {layer_name:8s}: {len(thoughts)} tokens x {len(thoughts[0])} dims")

        print("\nTest successful!")


if __name__ == "__main__":
    main()
