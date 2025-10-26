#!/usr/bin/env python3
"""
Decode final layer activations to tokens using unembedding matrix.

For each continuous thought position, compute:
  decoded_token = argmax(W_U @ activation)

Then classify whether the token is a number or not.
"""

import json
import sys
import torch
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
activation_patching_core = project_root / 'src' / 'experiments' / 'activation_patching' / 'core'
sys.path.insert(0, str(activation_patching_core))

from cache_activations import ActivationCacher
from cache_activations_llama import ActivationCacherLLaMA


def is_numeric_token(text: str) -> bool:
    """Check if token contains numeric information.

    Args:
        text: Decoded token text

    Returns:
        True if token contains digits
    """
    # Remove whitespace
    text = text.strip()

    # Match digits (including multi-digit numbers)
    # Examples: "42", "7", "120", "3.14"
    has_digit = bool(re.search(r'\d+', text))

    return has_digit


def decode_activations_to_tokens(
    activations: torch.Tensor,  # Shape: (n_positions, hidden_dim)
    W_U: torch.Tensor,          # Shape: (vocab_size, hidden_dim)
    tokenizer
) -> List[Dict]:
    """Decode activations to tokens using unembedding matrix.

    Args:
        activations: Activation vectors for each position
        W_U: Unembedding matrix
        tokenizer: HuggingFace tokenizer

    Returns:
        List of dicts with {position, token_id, token_text, is_number}
    """
    # Compute logits: activations @ W_U.T
    logits = activations @ W_U.T  # (n_positions, vocab_size)

    # Get top token per position
    token_ids = torch.argmax(logits, dim=-1)  # (n_positions,)

    results = []
    for pos, token_id in enumerate(token_ids):
        token_text = tokenizer.decode([token_id.item()])
        is_number = is_numeric_token(token_text)

        results.append({
            'position': pos,
            'token_id': token_id.item(),
            'token_text': token_text,
            'is_number': is_number
        })

    return results


def extract_gpt2_unembedding():
    """Extract GPT-2 unembedding matrix W_U."""
    print("\n" + "="*60)
    print("EXTRACTING GPT-2 UNEMBEDDING MATRIX")
    print("="*60)

    model_path = str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')

    # Load model
    print(f"Loading GPT-2 model from {model_path}...")
    cacher = ActivationCacher(model_path)

    # Extract W_U (lm_head or embedding transposed)
    model = cacher.model
    W_U = model.lm_head.weight  # Shape: (vocab_size, hidden_dim)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    print(f"✓ Extracted W_U shape: {W_U.shape}")
    print(f"  Vocab size: {W_U.shape[0]}")
    print(f"  Hidden dim: {W_U.shape[1]}")

    return {
        'W_U': W_U.cpu(),
        'vocab_size': W_U.shape[0],
        'hidden_dim': W_U.shape[1],
        'tokenizer_name': 'gpt2'
    }, tokenizer


def extract_llama_unembedding():
    """Extract LLaMA unembedding matrix W_U."""
    print("\n" + "="*60)
    print("EXTRACTING LLAMA UNEMBEDDING MATRIX")
    print("="*60)

    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    # Load model
    print(f"Loading LLaMA model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)

    # Extract W_U
    model = cacher.model
    W_U = model.lm_head.weight  # Shape: (vocab_size, hidden_dim)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

    print(f"✓ Extracted W_U shape: {W_U.shape}")
    print(f"  Vocab size: {W_U.shape[0]}")
    print(f"  Hidden dim: {W_U.shape[1]}")

    return {
        'W_U': W_U.cpu(),
        'vocab_size': W_U.shape[0],
        'hidden_dim': W_U.shape[1],
        'tokenizer_name': 'meta-llama/Llama-3.2-1B-Instruct'
    }, tokenizer


def decode_gpt2_final_layer(W_U: torch.Tensor, tokenizer):
    """Decode all GPT-2 final layer activations."""
    print("\n" + "="*60)
    print("DECODING GPT-2 FINAL LAYER (L11)")
    print("="*60)

    # Load data
    input_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_shared_data/gpt2_predictions_1000.json")
    output_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/results/gpt2_final_layer_decoding.json")

    print(f"Loading GPT-2 predictions from {input_file.name}...")
    with open(input_file, 'r') as f:
        gpt2_data = json.load(f)

    results = []
    number_position_counts = [0] * 6  # Count numbers per position

    for sample in tqdm(gpt2_data['samples'], desc="Decoding GPT-2"):
        # Extract layer 11 activations (final layer)
        layer_11_acts = torch.tensor(sample['thoughts']['layer_11'])  # (6, 768)

        # Decode
        decoded = decode_activations_to_tokens(layer_11_acts, W_U, tokenizer)

        # Count numbers per position
        for d in decoded:
            if d['is_number']:
                number_position_counts[d['position']] += 1

        results.append({
            'id': sample['id'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'is_correct': sample['is_correct'],
            'decoded_positions': decoded
        })

    # Calculate statistics
    n_samples = len(results)
    position_stats = [
        {
            'position': i,
            'number_count': number_position_counts[i],
            'pct_number': round(100 * number_position_counts[i] / n_samples, 1)
        }
        for i in range(6)
    ]

    # Save
    output = {
        'model': 'GPT-2',
        'layer': 11,
        'n_samples': len(results),
        'n_positions': 6,
        'position_statistics': position_stats,
        'samples': results
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Decoded {len(results)} samples")
    print(f"  Position number-decoding rates:")
    for stat in position_stats:
        print(f"    Position {stat['position']}: {stat['pct_number']}% ({stat['number_count']}/{n_samples})")
    print(f"  Saved to: {output_file}")

    return output


def decode_llama_final_layer(W_U: torch.Tensor, tokenizer):
    """Decode all LLaMA final layer activations."""
    print("\n" + "="*60)
    print("DECODING LLAMA FINAL LAYER (L15)")
    print("="*60)

    # Load data
    input_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/data/llama_activations_cot_dependent.json")
    output_file = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_token_ablation/results/llama_final_layer_decoding.json")

    print(f"Loading LLaMA activations from {input_file.name}...")
    with open(input_file, 'r') as f:
        llama_data = json.load(f)

    # Combine correct and incorrect samples
    all_samples = llama_data['correct_solutions'] + llama_data['incorrect_solutions']

    results = []
    number_position_counts = [0] * 6

    for sample in tqdm(all_samples, desc="Decoding LLaMA"):
        # Extract L16 (final layer, 0-indexed layer 15)
        layer_15_acts = torch.tensor(sample['continuous_thoughts']['L16'])  # (6, 2048)

        # Decode
        decoded = decode_activations_to_tokens(layer_15_acts, W_U, tokenizer)

        # Count numbers per position
        for d in decoded:
            if d['is_number']:
                number_position_counts[d['position']] += 1

        results.append({
            'pair_id': sample['pair_id'],
            'variant': sample['variant'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'is_correct': sample['is_correct'],
            'decoded_positions': decoded
        })

    # Calculate statistics
    n_samples = len(results)
    position_stats = [
        {
            'position': i,
            'number_count': number_position_counts[i],
            'pct_number': round(100 * number_position_counts[i] / n_samples, 1)
        }
        for i in range(6)
    ]

    # Save
    output = {
        'model': 'LLaMA-3.2-1B',
        'layer': 15,
        'n_samples': len(results),
        'n_positions': 6,
        'position_statistics': position_stats,
        'samples': results
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Decoded {len(results)} samples")
    print(f"  Position number-decoding rates:")
    for stat in position_stats:
        print(f"    Position {stat['position']}: {stat['pct_number']}% ({stat['number_count']}/{n_samples})")
    print(f"  Saved to: {output_file}")

    return output


def main():
    """Run token decoding pipeline."""
    print("=" * 60)
    print("FINAL LAYER TOKEN DECODING")
    print("=" * 60)

    # Extract unembedding matrices
    gpt2_unembed, gpt2_tokenizer = extract_gpt2_unembedding()
    llama_unembed, llama_tokenizer = extract_llama_unembedding()

    # Decode GPT-2
    gpt2_results = decode_gpt2_final_layer(gpt2_unembed['W_U'], gpt2_tokenizer)

    # Decode LLaMA
    llama_results = decode_llama_final_layer(llama_unembed['W_U'], llama_tokenizer)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"GPT-2: Decoded {gpt2_results['n_samples']} samples at layer {gpt2_results['layer']}")
    print(f"LLaMA: Decoded {llama_results['n_samples']} samples at layer {llama_results['layer']}")
    print("\n✓ Token decoding complete!")


if __name__ == "__main__":
    main()
