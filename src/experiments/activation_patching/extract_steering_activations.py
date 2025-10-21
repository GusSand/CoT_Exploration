#!/usr/bin/env python3
"""
Extract Continuous Thought Activations for Steering

This script:
1. Loads the training problems (CORRECT and WRONG)
2. Runs GPT-2 CODI model on each problem
3. Extracts continuous thought activations (6 tokens x hidden_dim)
4. Saves activations to disk for computing steering direction

Runtime: ~10-15 minutes for 344 problems
"""

import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))

from cache_activations import ActivationCacher, LAYER_CONFIG


class ContinuousThoughtExtractor:
    """Extracts continuous thought activations from CODI model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize with CODI model."""
        print("Initializing Continuous Thought Extractor...")
        self.cacher = ActivationCacher(model_path, device=device)
        self.device = device

    def extract_continuous_thoughts(
        self,
        problem_text: str,
        layer_name: str = 'middle',
        max_new_tokens: int = 200
    ) -> torch.Tensor:
        """Extract continuous thought activations for a problem.

        Args:
            problem_text: Problem question
            layer_name: Which layer to extract from ('early', 'middle', 'late')
            max_new_tokens: Max tokens to generate

        Returns:
            Tensor of shape [6, hidden_dim] containing activations for all 6 latent tokens
        """
        layer_idx = LAYER_CONFIG[layer_name]

        with torch.no_grad():
            # Tokenize input
            inputs = self.cacher.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.cacher.model.get_embd(
                self.cacher.model.codi,
                self.cacher.model.model_name
            )(input_ids).to(self.device)

            # Forward through model to get initial context
            outputs = self.cacher.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # Get BOT (Beginning of Thought) embedding
            bot_emb = self.cacher.model.get_embd(
                self.cacher.model.codi,
                self.cacher.model.model_name
            )(
                torch.tensor([self.cacher.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Process latent thoughts and collect activations
            latent_embd = bot_emb
            continuous_thoughts = []

            for latent_step in range(self.cacher.num_latent):
                outputs = self.cacher.model.codi(
                    inputs_embeds=latent_embd,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )

                # Extract hidden state from target layer
                hidden_states = outputs.hidden_states
                target_activation = hidden_states[layer_idx][:, -1, :]  # Last position

                # Save this latent token's activation
                continuous_thoughts.append(target_activation.squeeze(0).cpu())

                # Update for next iteration
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection if model uses it
                if self.cacher.model.use_prj:
                    latent_embd = self.cacher.model.prj(latent_embd)

            # Stack into [6, hidden_dim] tensor
            continuous_thoughts_tensor = torch.stack(continuous_thoughts, dim=0)

            return continuous_thoughts_tensor


def load_problem_text(pair_id: int, problem_pairs_file: str) -> str:
    """Load problem text from problem pairs file."""
    with open(problem_pairs_file) as f:
        all_pairs = json.load(f)

    # Find the pair
    for pair in all_pairs:
        if pair['pair_id'] == pair_id:
            return pair['clean']['question']

    raise ValueError(f"pair_id {pair_id} not found in {problem_pairs_file}")


def extract_activations_for_dataset(
    dataset_file: str,
    model_path: str,
    problem_pairs_file: str,
    output_dir: str,
    layer_name: str = 'middle'
):
    """Extract continuous thought activations for all training problems."""

    print("="*80)
    print("EXTRACTING CONTINUOUS THOUGHT ACTIVATIONS")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset from {dataset_file}...")
    with open(dataset_file) as f:
        dataset = json.load(f)

    train_correct = dataset['train_correct']
    train_wrong = dataset['train_wrong']

    print(f"Train CORRECT: {len(train_correct)} problems")
    print(f"Train WRONG: {len(train_wrong)} problems")
    print(f"Total: {len(train_correct) + len(train_wrong)} problems")

    # Initialize extractor
    extractor = ContinuousThoughtExtractor(model_path)

    # Extract activations for CORRECT problems
    print(f"\n{'='*80}")
    print(f"Extracting CORRECT problem activations...")
    print(f"{'='*80}")

    correct_activations = []
    correct_ids = []

    for problem_info in tqdm(train_correct, desc="CORRECT"):
        try:
            pair_id = problem_info['pair_id']
            problem_text = load_problem_text(pair_id, problem_pairs_file)

            activations = extractor.extract_continuous_thoughts(
                problem_text,
                layer_name=layer_name
            )

            correct_activations.append(activations.numpy())
            correct_ids.append(pair_id)

        except Exception as e:
            print(f"\nError on pair {pair_id}: {e}")
            continue

    # Extract activations for WRONG problems
    print(f"\n{'='*80}")
    print(f"Extracting WRONG problem activations...")
    print(f"{'='*80}")

    wrong_activations = []
    wrong_ids = []

    for problem_info in tqdm(train_wrong, desc="WRONG"):
        try:
            pair_id = problem_info['pair_id']
            problem_text = load_problem_text(pair_id, problem_pairs_file)

            activations = extractor.extract_continuous_thoughts(
                problem_text,
                layer_name=layer_name
            )

            wrong_activations.append(activations.numpy())
            wrong_ids.append(pair_id)

        except Exception as e:
            print(f"\nError on pair {pair_id}: {e}")
            continue

    # Save activations
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("Saving activations...")
    print(f"{'='*80}")

    # Save CORRECT
    correct_output = {
        'layer_name': layer_name,
        'num_problems': len(correct_activations),
        'pair_ids': correct_ids,
        'activations': np.array(correct_activations),  # Shape: [num_problems, 6, hidden_dim]
        'shape': f"[{len(correct_activations)}, 6, {correct_activations[0].shape[-1]}]"
    }

    correct_file = output_dir_path / f'correct_activations_{layer_name}.npz'
    np.savez_compressed(
        correct_file,
        activations=correct_output['activations'],
        pair_ids=np.array(correct_ids),
        layer_name=layer_name
    )
    print(f"✓ Saved CORRECT activations: {correct_file}")
    print(f"  Shape: {correct_output['shape']}")

    # Save WRONG
    wrong_output = {
        'layer_name': layer_name,
        'num_problems': len(wrong_activations),
        'pair_ids': wrong_ids,
        'activations': np.array(wrong_activations),
        'shape': f"[{len(wrong_activations)}, 6, {wrong_activations[0].shape[-1]}]"
    }

    wrong_file = output_dir_path / f'wrong_activations_{layer_name}.npz'
    np.savez_compressed(
        wrong_file,
        activations=wrong_output['activations'],
        pair_ids=np.array(wrong_ids),
        layer_name=layer_name
    )
    print(f"✓ Saved WRONG activations: {wrong_file}")
    print(f"  Shape: {wrong_output['shape']}")

    # Save metadata
    metadata = {
        'model_path': model_path,
        'layer_name': layer_name,
        'dataset_file': dataset_file,
        'problem_pairs_file': problem_pairs_file,
        'num_correct': len(correct_activations),
        'num_wrong': len(wrong_activations),
        'activation_shape': f"[6, {correct_activations[0].shape[-1]}]"
    }

    metadata_file = output_dir_path / 'extraction_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_file}")

    print(f"\n{'='*80}")
    print("✅ ACTIVATION EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"✅ CORRECT: {len(correct_activations)} problems")
    print(f"✅ WRONG: {len(wrong_activations)} problems")
    print(f"✅ Activations saved to: {output_dir_path}")


def main():
    """Main extraction pipeline."""

    # Paths
    BASE_DIR = Path(__file__).parent
    dataset_file = BASE_DIR / 'results' / 'steering_dataset_gpt2.json'
    model_path = Path.home() / 'codi_ckpt' / 'gpt2_gsm8k'
    problem_pairs_file = BASE_DIR / 'problem_pairs_gpt4_answers.json'
    output_dir = BASE_DIR / 'results' / 'steering_activations'

    # Extract activations from middle layer
    extract_activations_for_dataset(
        dataset_file=str(dataset_file),
        model_path=str(model_path),
        problem_pairs_file=str(problem_pairs_file),
        output_dir=str(output_dir),
        layer_name='middle'  # Layer 6 (middle of 12-layer GPT-2)
    )


if __name__ == "__main__":
    main()
