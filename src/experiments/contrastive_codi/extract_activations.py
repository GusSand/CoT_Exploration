#!/usr/bin/env python3
"""
Extract activations from contrastive CODI model for deception detection.

This script extracts both:
1. CT token activations (from the trained CODI model)
2. Regular hidden state activations (baseline comparison)

Following Apollo Research methodology:
- Extract at layers [4, 5, 9, 12, 15] (equivalent to their layer 22/80 = 27.5%)
- Use test set for unbiased evaluation
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import pickle

# Add CODI path for imports
import sys
sys.path.append('/home/paperspace/dev/CoT_Exploration/codi')
from src.model import CODI, ModelArguments, TrainingArguments

def load_contrastive_test_data(data_path):
    """Load contrastive test dataset."""
    with open(data_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} test samples")
    return data

def extract_ct_token_activations(model, tokenizer, data, layers, device):
    """Extract CT token activations from CODI model using true CT steps (BOT + 6 CT iterations).

    For each sample and each requested layer, we average the last-token hidden state across the 6 CT iterations.
    """
    model.eval()
    activations = {layer: [] for layer in layers}
    labels = []
    question_hashes = []

    print("Extracting CT token activations...")

    with torch.no_grad():
        for sample in tqdm(data):
            # Build a neutral, label-agnostic prompt: user question only (no system label leakage)
            question = sample.get("question")
            if not question and "messages" in sample:
                # fallback: extract from messages
                for m in sample["messages"]:
                    if m.get("role") == "user":
                        question = m.get("content", "")
                        break
            user_only_messages = [{"role": "user", "content": question or ""}]

            text = tokenizer.apply_chat_template(
                user_only_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Tokenize
            tokenized = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            input_ids = tokenized["input_ids"]

            # Encode question
            question_outputs = model.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = question_outputs.past_key_values

            # BOT token embedding
            bot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.bot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Accumulators for this sample: per layer, list of 6 ct vectors
            per_layer_ct_vectors = {layer: [] for layer in layers}

            # Generate 6 CT iterations and capture last-token hidden state per requested layer
            for _ in range(6):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                hidden_states = outputs.hidden_states  # tuple of layers
                for layer_idx in layers:
                    if layer_idx < len(hidden_states):
                        ct_vec = hidden_states[layer_idx][:, -1, :].squeeze(0).to(torch.float32)
                        per_layer_ct_vectors[layer_idx].append(ct_vec)

                # Next latent embedding is last-layer last-token
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if getattr(model, "use_prj", False):
                    latent_embd = model.prj(latent_embd)

            # Aggregate over 6 CT steps (mean) for each requested layer
            for layer_idx in layers:
                if len(per_layer_ct_vectors[layer_idx]) == 0:
                    continue
                mean_vec = torch.stack(per_layer_ct_vectors[layer_idx], dim=0).mean(dim=0)
                activations[layer_idx].append(mean_vec.cpu().numpy())

            labels.append(sample["is_honest"])
            question_hashes.append(sample["question_hash"])            

    return activations, labels, question_hashes

def find_checkpoint_bin(ckpt_root: Path) -> Path:
    """Find a pytorch_model.bin under the given checkpoint directory."""
    candidates = list(ckpt_root.rglob('pytorch_model.bin'))
    if not candidates:
        raise FileNotFoundError(f"No pytorch_model.bin found under {ckpt_root}")
    # Prefer the file closest to the root seed directory (last training checkpoint)
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[-1]

def load_codi_from_checkpoint(ckpt_dir: str, base_model: str, device: torch.device) -> CODI:
    """Instantiate CODI and load weights from a Trainer checkpoint (pytorch_model.bin)."""
    ckpt_root = Path(ckpt_dir)
    weights_path = find_checkpoint_bin(ckpt_root)

    # Minimal arguments for inference
    model_args = ModelArguments(model_name_or_path=base_model, full_precision=False, train=False)
    training_args = TrainingArguments(output_dir=str(ckpt_root), bf16=True, num_latent=6, use_lora=False)

    model = CODI(model_args=model_args, training_args=training_args, lora_config=None).to(device)

    # Load state dict; allow missing/extra keys for robustness across minor code diffs
    state_dict = torch.load(weights_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: when loading weights: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    return model

def extract_regular_hidden_states(model_path, tokenizer, data, layers, device):
    """Extract regular hidden state activations from base model (no CODI)."""
    print(f"Loading base model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    activations = {layer: [] for layer in layers}
    labels = []
    question_hashes = []

    print("Extracting regular hidden state activations...")

    with torch.no_grad():
        for sample in tqdm(data):
            # Get the conversation format
            # Build a neutral, label-agnostic prompt: user question only
            question = sample.get("question")
            if not question and "messages" in sample:
                for m in sample["messages"]:
                    if m.get("role") == "user":
                        question = m.get("content", "")
                        break
            user_only_messages = [{"role": "user", "content": question or ""}]

            text = tokenizer.apply_chat_template(
                user_only_messages,
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
                    mean_activation = layer_hidden.mean(dim=1).squeeze(0).to(torch.float32)  # (hidden_dim,)
                    activations[layer_idx].append(mean_activation.cpu().numpy())

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
    parser = argparse.ArgumentParser(description="Extract activations from contrastive CODI model")
    parser.add_argument("--codi_model_path", required=True, help="Path to trained CODI model")
    parser.add_argument("--base_model_path", default="meta-llama/Llama-3.2-1B-Instruct", help="Base model for regular hidden states")
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

    # Load tokenizer (shared for both models)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Extract CT token activations from CODI model
    print("\\n" + "="*60)
    print("EXTRACTING CT TOKEN ACTIVATIONS")
    print("="*60)

    # Load CODI model from HF Trainer checkpoint directory
    codi_model = load_codi_from_checkpoint(
        args.codi_model_path,
        args.base_model_path,
        device
    )

    ct_activations, ct_labels, ct_hashes = extract_ct_token_activations(
        codi_model, tokenizer, test_data, args.layers, device
    )

    save_activations(
        ct_activations, ct_labels, ct_hashes,
        output_dir / "ct_token_activations.pkl",
        "ct_tokens"
    )

    # Clear GPU memory
    del codi_model
    torch.cuda.empty_cache()

    # 2. Extract regular hidden state activations from base model
    print("\\n" + "="*60)
    print("EXTRACTING REGULAR HIDDEN STATE ACTIVATIONS")
    print("="*60)

    regular_activations, regular_labels, regular_hashes = extract_regular_hidden_states(
        args.base_model_path, tokenizer, test_data, args.layers, device
    )

    save_activations(
        regular_activations, regular_labels, regular_hashes,
        output_dir / "regular_hidden_activations.pkl",
        "regular_hidden_states"
    )

    print("\\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print("Next step: Train deception detection probes")

if __name__ == "__main__":
    main()