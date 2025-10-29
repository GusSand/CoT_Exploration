#!/usr/bin/env python3
"""
Story 2: Extract Attention Flow Between Continuous Thoughts

Extracts 6×6 attention patterns showing how continuous thought tokens
attend to each other across all layers.

Usage:
    python 1_extract_ct_attention_flow.py --n_samples 1221
"""
import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from datasets import load_dataset

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))

from cache_activations_llama import ActivationCacherLLaMA


def extract_ct_attention_flow(n_samples=1221):
    """
    Extract 6×6 attention patterns between continuous thoughts.

    For each problem, we extract attention weights showing how each CT token
    attends to other CT tokens at each layer.
    """
    print("=" * 80)
    print("CONTINUOUS THOUGHT ATTENTION FLOW EXTRACTION")
    print("=" * 80)

    # Load model
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_commonsense' / 'gsm8k_llama1b_latent_baseline' / 'Llama-3.2-1B-Instruct' / 'ep_3' / 'lr_0.0008' / 'seed_11')
    print(f"\nLoading CommonsenseQA CODI model from {model_path}...")

    cacher = ActivationCacherLLaMA(model_path)
    model = cacher.model
    tokenizer = cacher.tokenizer
    device = cacher.device

    # Load dataset
    print(f"\nLoading CommonsenseQA validation dataset...")
    dataset = load_dataset('tau/commonsense_qa', split='validation')

    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    print(f"Processing {len(dataset)} examples\n")

    # Storage for attention patterns
    # Shape: [n_problems, n_layers, 6, 6]
    all_attention_patterns = []
    metadata = []

    num_layers = 16  # LLaMA-3.2-1B has 16 layers

    for idx, example in enumerate(tqdm(dataset, desc="Extracting attention")):
        try:
            # Format question
            question = example['question']
            choices = example['choices']
            formatted = f"Question: {question}\nChoices:\n"
            for label, text in zip(choices['label'], choices['text']):
                formatted += f"{label}: {text}\n"

            with torch.no_grad():
                # Tokenize
                inputs = tokenizer(formatted.strip(), return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]
                input_len = input_ids.size(1)

                # Get embeddings
                input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

                # Forward through question
                outputs = model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True
                )

                past_key_values = outputs.past_key_values

                # BOT token
                bot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.bot_id], dtype=torch.long, device=device)
                ).unsqueeze(0)

                latent_embd = bot_emb

                # Generate 6 continuous thoughts and collect attention at each step
                ct_attention_by_layer = np.zeros((num_layers, 6, 6))  # [layers, from_ct, to_ct]

                for ct_step in range(6):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values

                    # Extract attention weights
                    # outputs.attentions: tuple of tensors [batch, num_heads, seq_len, seq_len]
                    # We want attention from current CT token (last position) to previous CT tokens

                    # CT tokens are at the END of the sequence (most recent tokens)
                    # When generating CT_i, we have: [...question..., BOT, CT_0, CT_1, ..., CT_(i-1), CT_i (current)]
                    # The current token is at position -1 in the sequence
                    # Previous CTs are at positions -(ct_step+1) to -1

                    from_ct_idx = ct_step  # Current CT index (0-5)

                    for layer_idx in range(num_layers):
                        attn = outputs.attentions[layer_idx]  # [1, num_heads, seq_len, seq_len]

                        # Average over heads
                        attn_avg = attn[0].mean(dim=0)  # [seq_len, seq_len]

                        # Current CT is at position -1
                        # Previous CTs (including current) span the last (ct_step+1) positions
                        # Positions: ..., CT_0, CT_1, ..., CT_i (current)
                        #            -(ct_step+1), -(ct_step), ..., -1

                        # Extract attention from current token (-1) to all CT tokens generated so far
                        seq_len = attn_avg.shape[0]
                        ct_start_idx = seq_len - (ct_step + 1)  # Start of CT tokens in this sequence

                        # Attention from current position to CT positions
                        ct_attn = attn_avg[-1, ct_start_idx:]  # Last (ct_step+1) positions

                        # Store in matrix
                        for to_ct_idx in range(ct_step + 1):
                            if to_ct_idx < len(ct_attn):
                                ct_attention_by_layer[layer_idx, from_ct_idx, to_ct_idx] = ct_attn[to_ct_idx].item()

                    # Update latent embedding
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if model.use_prj:
                        latent_embd = model.prj(latent_embd)

                # Store results
                all_attention_patterns.append(ct_attention_by_layer)
                metadata.append({
                    'id': example['id'],
                    'question_concept': example['question_concept'],
                    'answer_key': example['answerKey']
                })

        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            import traceback
            traceback.print_exc()
            # Store zeros for failed examples
            all_attention_patterns.append(np.zeros((num_layers, 6, 6)))
            metadata.append({
                'id': example.get('id', f'error_{idx}'),
                'error': str(e)
            })

    # Convert to numpy array
    all_attention_patterns = np.array(all_attention_patterns)  # [n_problems, n_layers, 6, 6]

    # Compute statistics
    mean_attention = all_attention_patterns.mean(axis=0)  # [n_layers, 6, 6]
    std_attention = all_attention_patterns.std(axis=0)  # [n_layers, 6, 6]

    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw patterns
    np.save(output_dir / 'commonsense_attention_patterns_raw.npy', all_attention_patterns)

    # Save aggregated patterns
    np.save(output_dir / 'commonsense_attention_patterns_avg.npy', mean_attention)
    np.save(output_dir / 'commonsense_attention_patterns_std.npy', std_attention)

    # Save metadata
    with open(output_dir / 'commonsense_attention_metadata.json', 'w') as f:
        json.dump({
            'model_name': 'llama_commonsense',
            'n_problems': len(dataset),
            'n_layers': num_layers,
            'continuous_thought_positions': list(range(6)),
            'dataset': 'tau/commonsense_qa',
            'split': 'validation'
        }, f, indent=2)

    # Save summary statistics
    summary = {
        'mean_attention_by_layer': {},
        'std_attention_by_layer': {}
    }

    for layer_idx in range(num_layers):
        summary['mean_attention_by_layer'][f'layer_{layer_idx}'] = mean_attention[layer_idx].tolist()
        summary['std_attention_by_layer'][f'layer_{layer_idx}'] = std_attention[layer_idx].tolist()

    with open(output_dir / 'commonsense_attention_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("ATTENTION EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Processed: {len(dataset)} examples")
    print(f"Layers analyzed: {num_layers}")
    print(f"Attention matrix shape: 6×6 per layer")
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - commonsense_attention_patterns_raw.npy ({all_attention_patterns.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  - commonsense_attention_patterns_avg.npy")
    print(f"  - commonsense_attention_stats.json")
    print(f"  - commonsense_attention_metadata.json")

    return str(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=1221, help='Number of samples to process')
    args = parser.parse_args()

    extract_ct_attention_flow(args.n_samples)
