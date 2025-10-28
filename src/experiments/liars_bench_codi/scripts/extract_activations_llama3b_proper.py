"""
Extract LLaMA-3.2-3B continuous thought activations using PROPER held-out questions.

Uses 1-epoch checkpoint from test run (training diverged at epoch 5).
Tests if scale (3B vs 124M) enables continuous thoughts to detect deception.

Methodology (following arxiv.org/abs/2502.03407):
- Train probes on 144 questions (288 samples)
- Test probes on different 144 questions (288 samples)
- Zero overlap with CODI training questions

Author: Claude Code
Date: 2025-10-28
Sprint: 4
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


def load_llama3b_codi_model():
    """Load trained LLaMA-3.2-3B CODI model (1-epoch checkpoint from test run)."""
    print("\nLoading LLaMA-3.2-3B CODI model...")

    checkpoint_path = Path.home() / "codi_ckpt" / "llama3b_liars_bench_proper_TEST" / "liars_bench_llama3b_TEST" / "Llama-3.2-3B-Instruct" / "ep_1" / "lr_0.003" / "seed_42" / "checkpoint-50"

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please run the test training first.")
        return None, None

    print(f"  Loading from: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint weights manually
    print("  Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")

    # Load base model first
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-3B-Instruct",
        torch_dtype=torch.bfloat16
    )

    # Load the checkpoint weights (CODI trained weights)
    print("  Merging checkpoint weights...")
    model.load_state_dict(checkpoint_weights, strict=False)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"  Model loaded on {device}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")

    return model, tokenizer


def extract_continuous_thoughts(model, tokenizer, question, answer):
    """
    Extract continuous thought activations for a question-answer pair.

    LLaMA-3.2-3B has 28 layers (num_hidden_layers=28). We extract from:
    - Layer 9 (early, ~1/3)
    - Layer 18 (middle, ~2/3)
    - Layer 27 (late, last layer)

    Returns:
        dict: {
            'layer_9': [tok0, tok1, ..., tok5],   # 6 tokens, 3072-dim each
            'layer_18': [tok0, tok1, ..., tok5],
            'layer_27': [tok0, tok1, ..., tok5]
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

    # Extract hidden states from layers 9, 18, 27
    # LLaMA has 28 layers (0-27), we want layers 9, 18, 27
    hidden_states = outputs.hidden_states  # Tuple of (layer+1) tensors

    # Get last 6 token positions (continuous thoughts during generation)
    activations = {}
    for layer_idx in [9, 18, 27]:
        layer_hidden = hidden_states[layer_idx + 1]  # +1 because includes embedding layer

        # Take last 6 tokens (convert bfloat16 to float32 for numpy compatibility)
        last_6_tokens = layer_hidden[0, -6:, :].cpu().to(torch.float32).numpy()  # (6, hidden_dim)

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
    output_file = output_dir / "probe_activations_llama3b_proper.json"

    print(f"\nSaving extracted data to {output_file}...")

    output_data = {
        'model': 'llama-3.2-3b',
        'checkpoint': '1_epoch_test_run',
        'methodology': 'proper_held_out_questions',
        'layers': [9, 18, 27],
        'hidden_dim': 3072,
        'num_latent_tokens': 6,
        'train_samples': train_samples,
        'test_samples': test_samples
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved {len(train_samples)} train + {len(test_samples)} test samples")


def main():
    print("=" * 80)
    print("LLAMA-3.2-3B ACTIVATION EXTRACTION - PROPER HELD-OUT METHODOLOGY")
    print("=" * 80)

    # Load probe datasets
    train_data, test_data = load_probe_datasets()

    # Load model
    model, tokenizer = load_llama3b_codi_model()
    if model is None:
        return

    # Extract activations
    train_samples = extract_all_activations(model, tokenizer, train_data, "train")
    test_samples = extract_all_activations(model, tokenizer, test_data, "test")

    # Save
    save_extracted_data(train_samples, test_samples)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run: python scripts/train_probes_llama3b_proper.py")
    print("  2. Compare to GPT-2 results (50% accuracy)")
    print("  3. Document Sprint 4 findings")
    print("=" * 80)


if __name__ == "__main__":
    main()
