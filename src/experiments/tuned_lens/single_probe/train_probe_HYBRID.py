#!/usr/bin/env python3
"""
Enhanced hybrid sampling strategy for probe training.
Scan 20K examples → 120K raw samples → Downsample aggressively to 70K with smart balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
import argparse
from datetime import datetime
import os
import sys
from collections import Counter
import numpy as np

# Add CODI src to path
sys.path.insert(0, '/workspace/CoT_Exploration/codi/src')
from model import CODI, ModelArguments, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


def load_codi_model_correctly(args):
    """Load CODI model using the official CODI class."""
    print("="*100)
    print("LOADING CODI MODEL")
    print("="*100)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir=args.ckpt_dir,
        full_precision=True,
        token=args.token
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=args.inf_latent_iterations,
        use_prj=True,
        prj_dim=768,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    model = CODI(model_args, training_args, lora_config)
    checkpoint_path = os.path.join(args.ckpt_dir, "pytorch_model.bin")
    print(f"Loading checkpoint from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.codi.tie_weights()
    model = model.float().to(device)
    model.eval()

    print(f"\nModel loaded successfully!")
    print(f"  n_layer: {model.codi.config.n_layer}")
    print(f"  has projection: {model.use_prj}")
    print("="*100)

    return model


def collect_activations_streaming(model, tokenizer, dataset, args):
    """
    Collect activations from 20K examples (120K raw samples).
    """
    print("\n" + "="*100)
    print(f"COLLECTING ACTIVATIONS (STREAMING)")
    print("="*100)
    print(f"Scanning: {args.num_scan_examples} examples")
    print(f"Expected raw samples: {args.num_scan_examples} × {args.inf_latent_iterations} = {args.num_scan_examples * args.inf_latent_iterations}")
    print("="*100)

    model.eval()
    lm_head = model.codi.lm_head

    source_activations = []
    target_activations = []
    target_logits = []
    target_token_ids = []

    num_examples = min(args.num_scan_examples, len(dataset['train']))

    with torch.no_grad():
        for ex_idx in range(num_examples):
            question = dataset['train'][ex_idx]['question'].strip()

            # Tokenize
            inputs = tokenizer([question], return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Initial forward pass
            outputs = model.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd_projected = model.prj(latent_embd_pre_proj)
            else:
                latent_embd_projected = latent_embd_pre_proj

            # Run continuous thought iterations
            for iteration in range(args.inf_latent_iterations):
                iter_outputs = model.codi(
                    inputs_embeds=latent_embd_projected,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values
                )

                past_key_values = iter_outputs.past_key_values
                hidden_states = iter_outputs.hidden_states

                # Extract from source and target layers
                source_hidden = hidden_states[args.source_layer + 1][0, 0, :]
                target_hidden = hidden_states[args.target_layer + 1][0, 0, :]

                # Compute target logits and token ID
                target_logit = lm_head(target_hidden.unsqueeze(0))[0]
                target_token_id = target_logit.argmax().item()

                # Store
                source_activations.append(source_hidden.cpu())
                target_activations.append(target_hidden.cpu())
                target_logits.append(target_logit.cpu())
                target_token_ids.append(target_token_id)

                # Prepare next iteration
                latent_embd_pre_proj = iter_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd_projected = model.prj(latent_embd_pre_proj)
                else:
                    latent_embd_projected = latent_embd_pre_proj

            if (ex_idx + 1) % 1000 == 0:
                print(f"Processed {ex_idx + 1}/{num_examples} examples... ({len(source_activations)} samples)")

    # Stack tensors
    source_activations = torch.stack(source_activations)
    target_activations = torch.stack(target_activations)
    target_logits = torch.stack(target_logits)
    target_token_ids = torch.tensor(target_token_ids)

    print(f"\n{'='*100}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*100}")
    print(f"Total raw samples collected: {len(source_activations)}")

    return source_activations, target_activations, target_logits, target_token_ids


def create_hybrid_balanced_subset(source_acts, target_acts, target_logits, target_token_ids,
                                   target_samples, tokenizer):
    """
    Create hybrid balanced subset with aggressive downsampling.

    Strategy:
    - Ultra-frequent tokens (>10%): Cap at 2%
    - Very frequent tokens (5-10%): Cap at 4%
    - Frequent tokens (1-5%): Keep as-is
    - Mid-frequency (0.1-1%): Keep as-is
    - Rare tokens (<0.1%): Boost to 0.3%
    """
    print("\n" + "="*100)
    print("CREATING HYBRID BALANCED SUBSET")
    print("="*100)
    print(f"Raw samples: {len(source_acts)}")
    print(f"Target samples: {target_samples}")

    # Count token frequencies
    token_counts = Counter(target_token_ids.tolist())
    total_samples = len(target_token_ids)

    print(f"\nOriginal token distribution (top 20):")
    for token_id, count in token_counts.most_common(20):
        token = tokenizer.decode([token_id])
        pct = 100.0 * count / total_samples
        print(f"  '{token}': {count} ({pct:.2f}%)")

    # Classify tokens into frequency buckets
    ultra_frequent = []  # >10%
    very_frequent = []   # 5-10%
    frequent = []        # 1-5%
    mid_frequency = []   # 0.1-1%
    rare = []            # <0.1%

    for token_id, count in token_counts.items():
        pct = 100.0 * count / total_samples
        if pct > 10.0:
            ultra_frequent.append((token_id, count, pct))
        elif pct > 5.0:
            very_frequent.append((token_id, count, pct))
        elif pct > 1.0:
            frequent.append((token_id, count, pct))
        elif pct > 0.1:
            mid_frequency.append((token_id, count, pct))
        else:
            rare.append((token_id, count, pct))

    print(f"\nFrequency bucket sizes:")
    print(f"  Ultra-frequent (>10%): {len(ultra_frequent)} tokens")
    print(f"  Very frequent (5-10%): {len(very_frequent)} tokens")
    print(f"  Frequent (1-5%): {len(frequent)} tokens")
    print(f"  Mid-frequency (0.1-1%): {len(mid_frequency)} tokens")
    print(f"  Rare (<0.1%): {len(rare)} tokens")

    # Calculate target counts for each token
    target_counts = {}

    # Ultra-frequent: cap at 2%
    for token_id, count, pct in ultra_frequent:
        target_counts[token_id] = int(target_samples * 0.02)

    # Very frequent: cap at 4%
    for token_id, count, pct in very_frequent:
        target_counts[token_id] = int(target_samples * 0.04)

    # Frequent: keep proportional
    for token_id, count, pct in frequent:
        target_counts[token_id] = int(target_samples * (pct / 100.0))

    # Mid-frequency: keep proportional
    for token_id, count, pct in mid_frequency:
        target_counts[token_id] = int(target_samples * (pct / 100.0))

    # Rare: boost to 0.3%
    for token_id, count, pct in rare:
        target_counts[token_id] = max(int(target_samples * 0.003), 1)

    # Show sampling plan for top tokens
    print(f"\nHybrid sampling plan (top 30 tokens):")
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:30]
    for token_id, orig_count in sorted_tokens:
        token = tokenizer.decode([token_id])
        orig_pct = 100.0 * orig_count / total_samples
        tgt_count = target_counts[token_id]
        tgt_pct = 100.0 * tgt_count / target_samples
        ratio = tgt_count / orig_count if orig_count > 0 else 0
        action = "oversample" if ratio > 1 else "undersample" if ratio < 0.9 else "keep"
        print(f"  '{token}': {orig_count} ({orig_pct:.2f}%) → {tgt_count} ({tgt_pct:.2f}%) [{action}]")

    # Sample indices for each token
    selected_indices = []

    for token_id, target_count in target_counts.items():
        # Find all indices with this token
        token_mask = (target_token_ids == token_id)
        token_indices = torch.where(token_mask)[0]

        if len(token_indices) == 0:
            continue

        if len(token_indices) <= target_count:
            # Oversample with replacement
            sampled = torch.randint(0, len(token_indices), (target_count,))
            selected_indices.extend(token_indices[sampled].tolist())
        else:
            # Undersample
            sampled = torch.randperm(len(token_indices))[:target_count]
            selected_indices.extend(token_indices[sampled].tolist())

    # Convert to tensor and shuffle
    selected_indices = torch.tensor(selected_indices)
    shuffle_perm = torch.randperm(len(selected_indices))
    selected_indices = selected_indices[shuffle_perm]

    # Limit to target samples
    if len(selected_indices) > target_samples:
        selected_indices = selected_indices[:target_samples]

    print(f"\nFinal hybrid balanced dataset: {len(selected_indices)} samples")

    # Create balanced subset
    balanced_source = source_acts[selected_indices]
    balanced_target = target_acts[selected_indices]
    balanced_logits = target_logits[selected_indices]

    # Show new distribution
    balanced_token_ids = target_token_ids[selected_indices]
    balanced_counts = Counter(balanced_token_ids.tolist())
    print(f"\nBalanced token distribution (top 20):")
    for token_id, count in balanced_counts.most_common(20):
        token = tokenizer.decode([token_id])
        pct = 100.0 * count / len(balanced_token_ids)
        print(f"  '{token}': {count} ({pct:.2f}%)")

    print("="*100)

    return balanced_source, balanced_target, balanced_logits


def train_probe(source_acts, target_logits, args, lm_head):
    """Train probe to minimize KL divergence."""
    print("\n" + "="*100)
    print("TRAINING PROBE")
    print("="*100)

    hidden_dim = source_acts.shape[1]
    probe = LinearProbe(hidden_dim, hidden_dim).to(device)

    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.probe_lr)

    dataset_size = len(source_acts)

    for epoch in range(args.num_epochs):
        indices = torch.randperm(dataset_size)

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, dataset_size, args.probe_batch_size):
            batch_indices = indices[i:i + args.probe_batch_size]

            batch_source = source_acts[batch_indices].to(device)
            batch_target_logits = target_logits[batch_indices].to(device)

            # Apply probe
            probed_hidden = probe(batch_source)

            # Compute logits
            probed_logits = lm_head(probed_hidden)

            # Compute target probs
            target_probs = F.softmax(batch_target_logits / args.temperature, dim=-1)

            # KL divergence loss
            log_probs = F.log_softmax(probed_logits / args.temperature, dim=-1)
            kl_loss = F.kl_div(log_probs, target_probs, reduction='batchmean')

            # Backward
            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()

            # Metrics
            total_loss += kl_loss.item()

            # Accuracy
            probed_preds = probed_logits.argmax(dim=-1)
            target_preds = batch_target_logits.argmax(dim=-1)
            correct += (probed_preds == target_preds).sum().item()
            total += len(batch_indices)

        avg_loss = total_loss / (dataset_size / args.probe_batch_size)
        accuracy = 100.0 * correct / total

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

    print(f"\n{'='*100}")
    print(f"TRAINING COMPLETE - Final Accuracy: {accuracy:.2f}%")
    print(f"{'='*100}")

    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="zen-E/GSM8k-Aug")
    parser.add_argument("--inf_latent_iterations", type=int, default=6)
    parser.add_argument("--num_scan_examples", type=int, default=20000,
                        help="Number of examples to scan for collecting samples")
    parser.add_argument("--target_samples", type=int, default=70000,
                        help="Target number of training samples after hybrid balancing")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--probe_lr", type=float, default=0.001)
    parser.add_argument("--probe_batch_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--source_layer", type=int, default=10)
    parser.add_argument("--target_layer", type=int, default=11)
    parser.add_argument("--output_dir", type=str, default="outputs/probe_hybrid")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("="*100)
    print("HYBRID BALANCED PROBE TRAINING - ENHANCED STRATEGY")
    print("="*100)
    print(f"Model: {args.model_name_or_path}")
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Dataset: {args.data_name}")
    print(f"Source layer: {args.source_layer}")
    print(f"Target layer: {args.target_layer}")
    print(f"Scan examples: {args.num_scan_examples}")
    print(f"Target samples: {args.target_samples}")
    print(f"Continuous thought iterations: {args.inf_latent_iterations}")
    print(f"Epochs: {args.num_epochs}")
    print("="*100)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load CODI model
    model = load_codi_model_correctly(args)

    # Load dataset
    dataset = load_dataset(args.data_name)

    # Collect activations
    source_acts, target_acts, target_logits, target_token_ids = collect_activations_streaming(
        model, tokenizer, dataset, args
    )

    # Create hybrid balanced subset
    balanced_source, balanced_target, balanced_logits = create_hybrid_balanced_subset(
        source_acts, target_acts, target_logits, target_token_ids,
        args.target_samples, tokenizer
    )

    # Free memory
    del source_acts, target_acts, target_logits, target_token_ids
    torch.cuda.empty_cache()

    # Train probe
    probe = train_probe(balanced_source, balanced_logits, args, model.codi.lm_head)

    # Save probe
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_path = os.path.join(
        args.output_dir,
        f"probe_L{args.source_layer}_to_L{args.target_layer}_HYBRID_{timestamp}.pt"
    )
    torch.save(probe.state_dict(), probe_path)
    print(f"\nProbe saved to: {probe_path}")


if __name__ == "__main__":
    main()
