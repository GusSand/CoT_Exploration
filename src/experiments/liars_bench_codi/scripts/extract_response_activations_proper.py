"""
Extract GPT-2 response token activations using PROPER held-out questions.

Tests if response tokens maintain their 70.5% advantage with proper methodology.

Author: Claude Code
Date: 2025-10-28
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

    with open(data_dir / "probe_train_proper.json", 'r') as f:
        train_data = json.load(f)

    with open(data_dir / "probe_test_proper.json", 'r') as f:
        test_data = json.load(f)

    return train_data, test_data


def load_gpt2_model():
    """Load GPT-2 base model (not CODI)."""
    print("\nLoading GPT-2 model...")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"  Model loaded on {device}")
    return model, tokenizer


def extract_response_activation(model, tokenizer, question, answer):
    """
    Extract response token activation (final layer, mean-pooled).

    This follows the Sprint 1 methodology for response tokens.
    """
    device = next(model.parameters()).device

    # Format prompt + answer
    full_text = f"Q: {question}\nA: {answer}"

    # Tokenize
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get response token positions (everything after "A:")
    prompt_only = f"Q: {question}\nA:"
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt")
    prompt_length = prompt_tokens['input_ids'].shape[1]

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get final layer hidden states
    final_hidden = outputs.hidden_states[-1][0]  # (seq_len, 768)

    # Extract response tokens only (after prompt)
    response_hidden = final_hidden[prompt_length:, :]

    # Mean pool across response tokens
    if response_hidden.shape[0] > 0:
        response_activation = response_hidden.mean(dim=0).cpu().numpy()
    else:
        # Fallback if no response tokens
        response_activation = final_hidden[-1, :].cpu().numpy()

    return response_activation.tolist()


def extract_all_activations(model, tokenizer, samples, split_name):
    """Extract response activations for all samples."""
    print(f"\nExtracting response activations for {split_name}...")

    extracted = []

    for sample in tqdm(samples, desc=split_name):
        activation = extract_response_activation(
            model, tokenizer, sample['question'], sample['answer']
        )

        extracted.append({
            'question': sample['question'],
            'answer': sample['answer'],
            'is_honest': sample['is_honest'],
            'question_hash': sample['question_hash'],
            'response_activation': activation
        })

    return extracted


def main():
    print("=" * 80)
    print("EXTRACTING RESPONSE TOKEN ACTIVATIONS - PROPER HELD-OUT METHODOLOGY")
    print("=" * 80)

    # Load datasets
    train_data, test_data = load_probe_datasets()
    print(f"\nTrain samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Load model
    model, tokenizer = load_gpt2_model()

    # Extract activations
    train_samples = extract_all_activations(model, tokenizer, train_data, "train")
    test_samples = extract_all_activations(model, tokenizer, test_data, "test")

    # Save
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "processed"
    output_file = output_dir / "response_activations_gpt2_proper.json"

    data = {
        'model': 'gpt2',
        'extraction_type': 'response_tokens_mean_pooled',
        'methodology': 'proper_held_out_questions',
        'hidden_dim': 768,
        'train_samples': train_samples,
        'test_samples': test_samples,
        'train_size': len(train_samples),
        'test_size': len(test_samples)
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✅ Saved: {output_file}")
    print("\nNext: python scripts/train_response_probe_proper.py")


if __name__ == "__main__":
    main()
