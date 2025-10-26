#!/usr/bin/env python3
"""
Contextless Layer Propagation Decoding

This script implements a different approach to decoding intermediate layers:
Instead of directly decoding layer k activations via unembedding, we:
1. Take activations from layer k
2. Pass them through subsequent layers k+1, k+2, ..., final layer
3. Process each token in ISOLATION (no attention context from other positions)
4. Decode final layer output via vocabulary embeddings

Goal: Understand what output layer k activations lead to absent influence from other tokens.

This differs from standard logit lensing which directly decodes each layer.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import re
import sys

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset

sys.path.insert(0, '/workspace/CoT_Exploration/discretization_experiment/src')
from model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_number_from_text(text):
    """Extract final numerical answer from generated text."""
    patterns = [
        r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:answer|result)(?:\s+is)?:?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*$'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num_str = match.group(1).replace(',', '')
            try:
                return float(num_str)
            except:
                pass
    return None


def propagate_through_layers_contextless(hidden_state, model, start_layer, model_type="gpt2"):
    """
    Propagate a single hidden state through subsequent layers WITHOUT context.

    Args:
        hidden_state: [1, hidden_dim] tensor from layer k
        model: The transformer model
        start_layer: Starting layer index (k)
        model_type: "gpt2" or "llama"

    Returns:
        Final hidden state after propagating through all layers from k to end
    """
    # Get the transformer blocks
    if model_type == "gpt2":
        blocks = model.transformer.h
    elif "llama" in model_type.lower():
        blocks = model.model.layers
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Start with the hidden state from layer k
    h = hidden_state.unsqueeze(0)  # [1, 1, hidden_dim]

    # Propagate through subsequent layers WITHOUT attention context
    # Key: No attention mask, no position info from other tokens
    with torch.no_grad():
        for layer_idx in range(start_layer, len(blocks)):
            block = blocks[layer_idx]

            if model_type == "gpt2":
                # GPT-2 block forward pass with isolated token
                # No attention_mask means no context from other positions
                h = block(h, attention_mask=None, use_cache=False)[0]
            elif "llama" in model_type.lower():
                # LLaMA block forward pass with isolated token
                h = block(h, attention_mask=None, position_ids=None)[0]

    return h[:, -1, :]  # Return [1, hidden_dim]


def decode_via_vocab_similarity(hidden_state, embedding_layer, topk=10, tokenizer=None):
    """
    Decode hidden state by finding nearest vocabulary embeddings.

    Args:
        hidden_state: [1, hidden_dim] tensor
        embedding_layer: Model's embedding layer
        topk: Number of top tokens to return
        tokenizer: Tokenizer for decoding

    Returns:
        Dict with topk_indices, topk_similarities, topk_tokens
    """
    with torch.no_grad():
        # Get all vocabulary embeddings [vocab_size, hidden_dim]
        vocab_size = embedding_layer.weight.shape[0]
        vocab_embeddings = embedding_layer.weight  # [vocab_size, hidden_dim]

        # Compute cosine similarities
        hidden_norm = F.normalize(hidden_state, p=2, dim=-1)  # [1, hidden_dim]
        vocab_norm = F.normalize(vocab_embeddings, p=2, dim=-1)  # [vocab_size, hidden_dim]

        similarities = torch.matmul(hidden_norm, vocab_norm.T)  # [1, vocab_size]
        similarities = similarities.squeeze(0)  # [vocab_size]

        # Get topk
        topk_vals, topk_idx = torch.topk(similarities, k=topk, dim=-1)

        topk_idx_cpu = topk_idx.cpu().tolist()
        topk_vals_cpu = topk_vals.cpu().tolist()

        # Decode tokens
        if tokenizer is not None:
            topk_tokens = [tokenizer.decode([int(tid)]) for tid in topk_idx_cpu]
        else:
            topk_tokens = [str(int(x)) for x in topk_idx_cpu]

        return {
            "topk_indices": topk_idx_cpu,
            "topk_similarities": topk_vals_cpu,
            "topk_tokens": topk_tokens,
        }


def decode_layer_contextless(hidden_states_all_layers, model, embedding_layer, tokenizer, topk=5, model_type="gpt2"):
    """
    Decode each layer by propagating through subsequent layers contextlessly.

    Args:
        hidden_states_all_layers: List of [1, hidden_dim] tensors, one per layer
        model: Transformer model
        embedding_layer: Embedding layer for vocab similarity
        tokenizer: Tokenizer
        topk: Number of top tokens
        model_type: Model architecture type

    Returns:
        List of dicts with layer_index, propagated_tokens (topk decoded tokens)
    """
    num_layers = len(hidden_states_all_layers)
    results = []

    for layer_idx in range(num_layers):
        # Get hidden state from layer k
        h_k = hidden_states_all_layers[layer_idx]  # [1, hidden_dim]

        if layer_idx == num_layers - 1:
            # Final layer: no propagation needed
            final_h = h_k
        else:
            # Propagate through layers k+1 to final WITHOUT context
            final_h = propagate_through_layers_contextless(
                h_k, model, layer_idx + 1, model_type
            )

        # Decode via vocabulary similarity
        decoded = decode_via_vocab_similarity(final_h, embedding_layer, topk, tokenizer)

        results.append({
            "layer_index": layer_idx,
            "topk_indices": decoded["topk_indices"],
            "topk_similarities": decoded["topk_similarities"],
            "topk_tokens": decoded["topk_tokens"],
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Contextless layer propagation decoding")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Must be 1")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--num_thought_iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)

    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 for this experiment")

    torch.manual_seed(args.seed)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_path / f"contextless_propagation_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")

    # Determine model configuration
    if "gpt2" in args.model_name_or_path.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]
        prj_dim = 768
        model_type = "gpt2"
    elif "llama" in args.model_name_or_path.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        prj_dim = 2048
        model_type = "llama"
    else:
        target_modules = ["c_attn", "c_proj", "c_fc"]
        prj_dim = 768
        model_type = "gpt2"

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir=args.ckpt_dir,
        full_precision=True,
        token=None
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=args.num_thought_iterations,
        use_prj=True,
        prj_dim=prj_dim,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    codi_model = CODI(model_args, training_args, lora_config)

    # Load checkpoint
    try:
        state_dict = load_file(os.path.join(args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(args.ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    codi_model.load_state_dict(state_dict, strict=False)

    if hasattr(codi_model, "tie_weights"):
        codi_model.tie_weights()
    elif hasattr(codi_model.codi, "tie_weights"):
        codi_model.codi.tie_weights()

    # Move to CUDA and set to bfloat16
    codi_model = codi_model.to('cuda')
    codi_model.to(torch.bfloat16)
    codi_model.eval()

    # Get embedding layer for vocab similarity
    embedding_layer = codi_model.get_embd(codi_model.codi, codi_model.model_name)

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Load dataset
    print("Loading dataset...")
    if "zen-E/GSM8k-Aug" in args.data_name:
        dataset = load_dataset(args.data_name)
        test_set = dataset['test']
    elif "gsm8k" in args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
    else:
        raise NotImplementedError(f"Dataset {args.data_name} not supported")

    examples = []
    for ex in test_set:
        q = ex.get("question", ex.get("question_text", None))
        ans = ex.get("answer", ex.get("answers", ""))
        if q is None:
            continue
        examples.append({"question": q, "answer": ans})
        if len(examples) >= args.max_examples:
            break

    print(f"Processing {len(examples)} examples with contextless propagation...")

    all_results = []

    for ex_idx, example in enumerate(examples):
        print(f"\nProcessing example {ex_idx+1}/{len(examples)}")

        question_text = example["question"].strip()
        input_ids = tokenizer(question_text, return_tensors="pt")["input_ids"].to('cuda')

        # Add BOT token
        if training_args.remove_eos:
            bot_tensor = torch.tensor([codi_model.bot_id], dtype=torch.long).expand(input_ids.size(0), 1).to('cuda')
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, codi_model.bot_id], dtype=torch.long).expand(input_ids.size(0), 2).to('cuda')

        input_ids = torch.cat((input_ids, bot_tensor), dim=1)
        bot_position = input_ids.shape[1] - 1

        # Generate with CODI capturing hidden states
        with torch.no_grad():
            # Initial forward pass
            outputs = codi_model.codi(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=None,
                attention_mask=torch.ones_like(input_ids)
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            continuous_thoughts = []

            # Decode initial thought with contextless propagation
            initial_layers = [hs[:, -1, :] for hs in outputs.hidden_states]
            initial_decoded = decode_layer_contextless(
                initial_layers, codi_model.codi, embedding_layer, tokenizer,
                topk=args.topk, model_type=model_type
            )

            continuous_thoughts.append({
                "iteration": 0,
                "type": "initial_bot",
                "position": bot_position,
                "layers": initial_decoded
            })

            # Apply projection
            if training_args.use_prj:
                latent_embd = codi_model.prj(latent_embd)

            # Iterate through continuous thoughts
            for i in range(training_args.inf_latent_iterations):
                outputs = codi_model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Decode with contextless propagation
                thought_layers = [hs[:, -1, :] for hs in outputs.hidden_states]
                layers_decoded = decode_layer_contextless(
                    thought_layers, codi_model.codi, embedding_layer, tokenizer,
                    topk=args.topk, model_type=model_type
                )

                continuous_thoughts.append({
                    "iteration": i + 1,
                    "type": "continuous_thought",
                    "position": bot_position + 1 + i,
                    "layers": layers_decoded
                })

                # Apply projection
                if training_args.use_prj:
                    latent_embd = codi_model.prj(latent_embd_pre_proj)
                else:
                    latent_embd = latent_embd_pre_proj

            # Generate answer
            if training_args.remove_eos:
                eot_emb = codi_model.get_embd(codi_model.codi, codi_model.model_name)(
                    torch.tensor([codi_model.eot_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0)
            else:
                eot_emb = codi_model.get_embd(codi_model.codi, codi_model.model_name)(
                    torch.tensor([codi_model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0)

            output = eot_emb
            pred_tokens = []
            max_new_tokens = 256

            for gen_i in range(max_new_tokens):
                out = codi_model.codi(
                    inputs_embeds=output,
                    use_cache=True,
                    output_hidden_states=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]

                if training_args.greedy:
                    next_token_id = torch.argmax(logits, dim=-1)
                else:
                    probs = F.softmax(logits / 0.1, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                pred_tokens.append(next_token_id.item())

                if next_token_id.item() == tokenizer.eos_token_id:
                    break

                output = codi_model.get_embd(codi_model.codi, codi_model.model_name)(
                    next_token_id.unsqueeze(0)
                )

            generated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            predicted_answer = extract_number_from_text(generated_text)

        result = {
            "example_index": ex_idx,
            "question": question_text,
            "ground_truth_answer": example["answer"],
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "bot_position": bot_position,
            "num_continuous_thoughts": len(continuous_thoughts),
            "continuous_thoughts": continuous_thoughts
        }

        all_results.append(result)
        print(f"  Generated: {generated_text[:100]}...")
        print(f"  Captured {len(continuous_thoughts)} continuous thought iterations with contextless propagation")

    # Save results
    out_file = run_dir / "contextless_propagation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved contextless propagation results to {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
