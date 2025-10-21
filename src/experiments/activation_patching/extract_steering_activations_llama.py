#!/usr/bin/env python3
"""
Extract LLaMA Steering Activations

Extracts continuous thought activations from LLaMA for steering dataset.
For each problem, extracts [6, 2048] activations from all 6 latent tokens.

Usage:
    python extract_steering_activations_llama.py
"""

import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG


def extract_continuous_thought_activations(cacher, question: str, layer_name: str):
    """Extract activations from all 6 continuous thought tokens.

    Returns:
        torch.Tensor: [6, 2048] tensor of activations
    """
    device = cacher.device

    with torch.no_grad():
        # Tokenize input
        inputs = cacher.tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Get initial embeddings
        input_embd = cacher.model.get_embd(cacher.model.codi, cacher.model.model_name)(input_ids).to(device)

        # Forward through model to get initial context
        outputs = cacher.model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )

        past_key_values = outputs.past_key_values

        # Get BOT (Beginning of Thought) embedding
        bot_emb = cacher.model.get_embd(cacher.model.codi, cacher.model.model_name)(
            torch.tensor([cacher.model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        # Collect activations from all 6 latent tokens
        layer_idx = LAYER_CONFIG[layer_name]
        all_activations = []

        latent_embd = bot_emb

        for latent_step in range(cacher.num_latent):
            outputs = cacher.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values

            # Extract activation from specified layer
            # hidden_states[layer_idx] shape: [batch_size, seq_len, hidden_dim]
            # We want the last token: [:, -1, :]
            activation = outputs.hidden_states[layer_idx][:, -1, :].cpu()  # [1, 2048]
            all_activations.append(activation)

            # Update latent embedding for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Apply projection if used
            if cacher.model.use_prj:
                latent_embd = cacher.model.prj(latent_embd)

        # Stack to [6, 2048]
        result = torch.cat(all_activations, dim=0)  # [6, 2048]

    return result


def main():
    """Extract activations for all problems in steering dataset."""
    print("="*80)
    print("LLAMA ACTIVATION EXTRACTION FOR STEERING")
    print("="*80)

    # Load steering dataset
    dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_llama.json'
    with open(dataset_file) as f:
        dataset = json.load(f)

    print(f"\nDataset loaded:")
    print(f"  Training: {len(dataset['train_correct'])} correct + {len(dataset['train_wrong'])} wrong")
    print(f"  Test:     {len(dataset['test_correct'])} correct + {len(dataset['test_wrong'])} wrong")

    # Load problem pairs
    pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'
    with open(pairs_file) as f:
        all_problems = json.load(f)

    # Create problem lookup
    problem_lookup = {p['pair_id']: p for p in all_problems}

    # Initialize LLaMA
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    print(f"\nLoading LLaMA model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path, device='cuda')

    # Extract at all three layers
    for layer_name in ['early', 'middle', 'late']:
        print(f"\n{'='*80}")
        print(f"EXTRACTING ACTIVATIONS AT LAYER: {layer_name.upper()} (L{LAYER_CONFIG[layer_name]})")
        print("="*80)

        output_dir = Path(__file__).parent / 'results' / 'steering_activations_llama' / layer_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process all subsets
        for subset_name, subset_data in [
            ('train_correct', dataset['train_correct']),
            ('train_wrong', dataset['train_wrong']),
            ('test_correct', dataset['test_correct']),
            ('test_wrong', dataset['test_wrong'])
        ]:
            print(f"\nProcessing {subset_name}: {len(subset_data)} problems")

            for item in tqdm(subset_data, desc=subset_name):
                pair_id = item['pair_id']
                problem = problem_lookup[pair_id]

                # Extract from CLEAN version
                question = problem['clean']['question']

                # Extract [6, 2048] activations
                activations = extract_continuous_thought_activations(cacher, question, layer_name)

                # Save
                save_path = output_dir / f"pair_{pair_id}_{subset_name}.pt"
                torch.save(activations, save_path)

        print(f"\n✓ Saved activations for layer {layer_name} to {output_dir}")

    print("\n" + "="*80)
    print("✅ ACTIVATION EXTRACTION COMPLETE")
    print("="*80)
    print("\nExtracted [6, 2048] activations from all 6 continuous thought tokens")
    print("Layers: early (L4), middle (L8), late (L14)")


if __name__ == "__main__":
    main()
