"""
Extract GPT-2 continuous thought activations using PROPER held-out questions.

This corrects Sprint 1 results by using question-level held-out splits.
Previous version had 100% overlap between training and probe questions.

New methodology (following arxiv.org/abs/2502.03407):
- Train probes on 144 questions (288 samples)
- Test probes on different 144 questions (288 samples)
- Zero overlap with CODI training questions

Author: Claude Code
Date: 2025-10-28
Fixes: Sprint 1 data leakage issue
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_probe_datasets():
    """Load probe train and test datasets."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "processed"

    print("Loading probe datasets...")

    with open(data_dir / "probe_train_proper.json", 'r') as f:
        train_data = json.load(f)

    with open(data_dir / "probe_test_proper.json", 'r') as f:
        test_data = json.load(f)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Verify balance
    train_honest = sum(1 for s in train_data if s['is_honest'])
    test_honest = sum(1 for s in test_data if s['is_honest'])

    print(f"  Train balance: {train_honest}/{len(train_data)-train_honest}")
    print(f"  Test balance: {test_honest}/{len(test_data)-test_honest}")

    return train_data, test_data


def load_gpt2_codi_model():
    """Load trained GPT-2 CODI model."""
    print("\nLoading GPT-2 CODI model...")

    checkpoint_path = Path.home() / "codi_ckpt" / "gpt2_liars_bench" / "liars_bench_gpt2_codi" / "gpt2" / "ep_20" / "lr_0.003" / "seed_42"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please train GPT-2 CODI first.")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"  Model loaded on {device}")
    print(f"  Layers: {model.config.n_layer}")
    print(f"  Hidden dim: {model.config.n_embd}")

    return model, tokenizer


def extract_continuous_thoughts(model, tokenizer, question, answer):
    """
    Extract continuous thought activations for a question-answer pair.

    Returns:
        dict: {
            'layer_4': [tok0, tok1, ..., tok5],  # 6 tokens, 768-dim each
            'layer_8': [tok0, tok1, ..., tok5],
            'layer_11': [tok0, tok1, ..., tok5]
        }
    """
    device = next(model.parameters()).device

    # Format as CODI would see it during training
    prompt = f"Q: {question}\nA:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states from layers 4, 8, 11
    # GPT-2 has 12 layers (0-11), we want layers 4, 8, 11
    hidden_states = outputs.hidden_states  # Tuple of (layer+1) tensors

    # Get last 6 token positions (continuous thoughts during generation)
    activations = {}
    for layer_idx in [4, 8, 11]:
        layer_hidden = hidden_states[layer_idx + 1]  # +1 because includes embedding layer

        # Take last 6 tokens
        last_6_tokens = layer_hidden[0, -6:, :].cpu().numpy()  # (6, 768)

        # Split into list of 6 vectors
        activations[f'layer_{layer_idx}'] = [last_6_tokens[i].tolist() for i in range(6)]

    return activations


def extract_all_activations(model, tokenizer, samples, split_name):
    """Extract activations for all samples in a split."""
    print(f"\nExtracting activations for {split_name} split...")

    extracted_samples = []

    for sample in tqdm(samples, desc=f"Extracting {split_name}"):
        question = sample['question']
        answer = sample['answer']
        is_honest = sample['is_honest']

        # Extract continuous thought activations
        thoughts = extract_continuous_thoughts(model, tokenizer, question, answer)

        extracted_sample = {
            'question': question,
            'answer': answer,
            'is_honest': is_honest,
            'question_hash': sample['question_hash'],
            'thoughts': thoughts
        }

        extracted_samples.append(extracted_sample)

    return extracted_samples


def save_extracted_data(train_samples, test_samples):
    """Save extracted activations."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "processed"

    print("\nSaving extracted activations...")

    # Create dataset
    dataset = {
        'model': 'gpt2',
        'methodology': 'proper_held_out_splits',
        'layers': ['layer_4', 'layer_8', 'layer_11'],
        'tokens_per_layer': 6,
        'hidden_dim': 768,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'train_size': len(train_samples),
        'test_size': len(test_samples),
        'train_honest': sum(1 for s in train_samples if s['is_honest']),
        'train_deceptive': sum(1 for s in train_samples if not s['is_honest']),
        'test_honest': sum(1 for s in test_samples if s['is_honest']),
        'test_deceptive': sum(1 for s in test_samples if not s['is_honest'])
    }

    output_file = output_dir / "probe_activations_gpt2_proper.json"

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"  ✅ Saved: {output_file}")

    # Print statistics
    print(f"\nDataset statistics:")
    print(f"  Train: {dataset['train_size']} samples")
    print(f"    Honest: {dataset['train_honest']}")
    print(f"    Deceptive: {dataset['train_deceptive']}")
    print(f"  Test: {dataset['test_size']} samples")
    print(f"    Honest: {dataset['test_honest']}")
    print(f"    Deceptive: {dataset['test_deceptive']}")

    return output_file


def main():
    print("=" * 80)
    print("EXTRACTING GPT-2 ACTIVATIONS - PROPER HELD-OUT METHODOLOGY")
    print("=" * 80)
    print("\nThis corrects Sprint 1 by using question-level held-out splits.")
    print("Previous version had 100% overlap (data leakage).")
    print()

    # Load probe datasets
    train_data, test_data = load_probe_datasets()

    # Load GPT-2 CODI model
    model, tokenizer = load_gpt2_codi_model()
    if model is None:
        return

    # Extract activations
    train_samples = extract_all_activations(model, tokenizer, train_data, "train")
    test_samples = extract_all_activations(model, tokenizer, test_data, "test")

    # Save
    output_file = save_extracted_data(train_samples, test_samples)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_file}")
    print("\nNext steps:")
    print("  1. Train probes: python scripts/train_probes_proper_v2.py")
    print("  2. Compare to old results (with data leakage)")
    print("  3. Document corrected Sprint 1 findings")
    print("=" * 80)


if __name__ == "__main__":
    main()
