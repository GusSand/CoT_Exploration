#!/usr/bin/env python3
"""Debug attention tensor shapes during CODI generation."""
import sys
import torch
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))

from cache_activations_llama import ActivationCacherLLaMA

# Load model
model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
print(f"Loading model from {model_path}...")
cacher = ActivationCacherLLaMA(model_path)
model = cacher.model
tokenizer = cacher.tokenizer
device = cacher.device

# Test question
question = "What is 2 + 2?"

with torch.no_grad():
    # Tokenize
    inputs = tokenizer(question, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.size(1)

    print(f"\nQuestion length: {input_len} tokens")

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
    print(f"Past key values: {len(past_key_values)} layers")
    print(f"  Layer 0 key shape: {past_key_values[0][0].shape}")  # [batch, heads, seq, head_dim]

    # BOT token
    bot_emb = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.bot_id], dtype=torch.long, device=device)
    ).unsqueeze(0)

    latent_embd = bot_emb

    # Generate first continuous thought
    print("\n=== Generating first continuous thought (step 0) ===")
    outputs = model.codi(
        inputs_embeds=latent_embd,
        use_cache=True,
        output_hidden_states=True,
        output_attentions=True,
        past_key_values=past_key_values
    )

    print(f"Number of attention tensors: {len(outputs.attentions)}")
    print(f"Attention[0] shape: {outputs.attentions[0].shape}")
    print(f"  Interpretation: [batch={outputs.attentions[0].shape[0]}, heads={outputs.attentions[0].shape[1]}, seq_new={outputs.attentions[0].shape[2]}, seq_full={outputs.attentions[0].shape[3]}]")

    # Check current sequence length
    current_seq_len = outputs.attentions[0].shape[3]
    print(f"Current full sequence length: {current_seq_len}")

    # Check where continuous thought starts
    continuous_start = input_len + 1
    print(f"Continuous thought start position: {continuous_start}")
    print(f"Continuous thought end position (step 0): {continuous_start + 0 + 1}")

    # Extract attention
    attn = outputs.attentions[0]  # Layer 0
    print(f"\nAttn shape: {attn.shape}")
    attn_from_current = attn[0, :, -1, :]  # [heads, full_seq]
    print(f"Attention from current token shape: {attn_from_current.shape}")

    # Try to extract attention to continuous thoughts
    continuous_end = continuous_start + 0 + 1  # step + 1
    print(f"Extracting attention[:, {continuous_start}:{continuous_end}]")
    attn_to_continuous = attn_from_current[:, continuous_start:continuous_end]
    print(f"Attention to continuous thoughts shape: {attn_to_continuous.shape}")
    print(f"Attention values: {attn_to_continuous[:5, :]}")

    # Check if continuous_start is within bounds
    if continuous_start >= current_seq_len:
        print(f"\n⚠️  ERROR: continuous_start ({continuous_start}) >= seq_len ({current_seq_len})")
        print("The continuous thought position hasn't been added to the sequence yet!")
