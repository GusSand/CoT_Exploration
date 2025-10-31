#!/usr/bin/env python3
"""
Extract activations from base model for deception detection (simplified version).

This is a simplified version that extracts only regular hidden states from the base model
to validate the probe training methodology. The full CODI version will be implemented next.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import pickle

def load_contrastive_test_data(data_path):
    """Load contrastive test dataset."""
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples")
    return data

def extract_hidden_states(model, tokenizer, data, layers, device, model_type="base"):
    """Extract hidden state activations from model."""
    model.eval()
    activations = {layer: [] for layer in layers}
    labels = []
    question_hashes = []

    print(f"Extracting {model_type} hidden state activations...")

    with torch.no_grad():
        for sample in tqdm(data):
            # Get the conversation format
            messages = sample["messages"]

            # Format as chat template
            text = tokenizer.apply_chat_template(
                messages[:-1],  # Exclude assistant response
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Get model outputs with hidden states
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states  # List of (batch, seq_len, hidden_dim)

            for layer_idx in layers:
                if layer_idx < len(hidden_states):
                    layer_hidden = hidden_states[layer_idx]  # (1, seq_len, hidden_dim)

                    # Take mean across all token positions (following Apollo methodology)
                    mean_activation = layer_hidden.mean(dim=1).squeeze(0)  # (hidden_dim,)
                    # Convert to float32 for numpy compatibility
                    activations[layer_idx].append(mean_activation.float().cpu().numpy())

            # Store metadata
            labels.append(sample["is_honest"])
            question_hashes.append(sample["question_hash"])

    return activations, labels, question_hashes

def save_activations(activations, labels, question_hashes, output_path, activation_type):
    """Save extracted activations."""
    output_data = {
        'activations': {str(layer): np.stack(acts) for layer, acts in activations.items()},
        'labels': labels,
        'question_hashes': question_hashes,
        'activation_type': activation_type
    }

    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Saved {activation_type} activations: {output_path}")
    print(f"  Layers: {list(activations.keys())}")
    print(f"  Samples: {len(labels)}")
    print(f"  Honest: {sum(labels)}, Deceptive: {len(labels) - sum(labels)}")

def main():
    parser = argparse.ArgumentParser(description="Extract activations for deception detection")
    parser.add_argument("--base_model_path", default="meta-llama/Llama-3.2-1B-Instruct", help="Base model for hidden states")
    parser.add_argument("--test_data_path", required=True, help="Path to contrastive test data")
    parser.add_argument("--output_dir", required=True, help="Output directory for activations")
    parser.add_argument("--layers", nargs="+", type=int, default=[4, 5, 9, 12, 15], help="Layers to extract from")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_data = load_contrastive_test_data(args.test_data_path)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {args.base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Extract hidden state activations
    print("\\n" + "="*60)
    print("EXTRACTING HIDDEN STATE ACTIVATIONS")
    print("="*60)

    activations, labels, hashes = extract_hidden_states(
        model, tokenizer, test_data, args.layers, device, "base model"
    )

    save_activations(
        activations, labels, hashes,
        output_dir / "regular_hidden_activations.pkl",
        "regular_hidden_states"
    )

    # For now, create a copy as "CT token" activations to test the pipeline
    # In reality, these would be different activations from the CODI model
    print("\\n" + "="*60)
    print("CREATING PLACEHOLDER CT TOKEN ACTIVATIONS")
    print("="*60)
    print("NOTE: This is placeholder data. Real CT tokens require CODI model loading.")

    save_activations(
        activations, labels, hashes,
        output_dir / "ct_token_activations.pkl",
        "ct_tokens_placeholder"
    )

    print("\\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("\\nNOTE: This version uses base model activations for both 'CT tokens' and regular hidden states.")
    print("This allows us to test the probe training pipeline while we fix the CODI model loading.")
    print("Expected result: Both probes should perform similarly since they use the same features.")

if __name__ == "__main__":
    main()