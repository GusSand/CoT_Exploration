#!/usr/bin/env python3
"""
Decode implicit CoT tokens at ALL model layers during CODI generation.

This script captures hidden states during continuous thought iterations (BOT->EOT->answer)
and decodes them at each layer to understand how information evolves.

CORRECTED VERSION: Properly follows CODI architecture with:
- BOT token addition to input
- Continuous thought generation via inputs_embeds
- Decoding hidden states BEFORE projection
- Use of past_key_values for efficiency
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import re

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset

from src.model import CODI, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_number_from_text(text):
    """Extract final numerical answer from generated text."""
    # Look for patterns like "The answer is: X" or "####X"
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


def decode_continuous_thought(hidden_state, lm_head, tokenizer, topk=10):
    """
    Decode a continuous thought hidden state to vocabulary space.

    Args:
        hidden_state: Hidden state tensor [1, hidden_dim]
        lm_head: Language model head
        tokenizer: Tokenizer for decoding
        topk: Number of top tokens to return

    Returns:
        Dict with layer_index, topk_indices, topk_probs, topk_tokens
    """
    with torch.no_grad():
        # hidden_state is [1, hidden_dim]
        logits = lm_head(hidden_state.unsqueeze(0))  # [1, 1, vocab_size]
        probs = F.softmax(logits[0, 0, :], dim=-1)  # [vocab_size]

        topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)

        topk_idx_cpu = topk_idx.cpu().tolist()
        topk_vals_cpu = topk_vals.cpu().tolist()

        # Decode tokens
        topk_tokens = [tokenizer.decode([int(tid)]) for tid in topk_idx_cpu]

        return {
            "topk_indices": topk_idx_cpu,
            "topk_probs": topk_vals_cpu,
            "topk_tokens": topk_tokens,
        }


def decode_hiddenstates_all_layers(hidden_states, lm_head, topk, max_layers=None, tokenizer=None):
    """
    Decode topk tokens/probs for all layers at a SINGLE position.

    hidden_states: list of tensors [layer, 1, hidden_dim] for a single position
    Returns: list of dicts with layer_index, topk_indices, topk_probs, topk_tokens
    """
    num_layers = len(hidden_states)
    if max_layers is None or max_layers <= 0:
        max_layers = num_layers
    decode_layers = list(range(min(num_layers, max_layers)))

    lm_device = next(lm_head.parameters()).device if any(True for _ in lm_head.parameters()) else device
    layers_decoded = []

    with torch.no_grad():
        for l in decode_layers:
            hs = hidden_states[l].to(lm_device)  # [1, hidden_dim]

            # Compute logits
            logits = lm_head(hs.unsqueeze(0))  # [1, 1, vocab_size]
            probs = F.softmax(logits[0, 0, :], dim=-1)  # [vocab_size]

            # Get topk
            topk_vals, topk_idx = torch.topk(probs, k=topk, dim=-1)

            topk_idx_cpu = topk_idx.cpu().tolist()
            topk_vals_cpu = topk_vals.cpu().tolist()

            # Decode tokens
            if tokenizer is not None:
                topk_tokens = [tokenizer.decode([int(tid)]) for tid in topk_idx_cpu]
            else:
                topk_tokens = [str(int(x)) for x in topk_idx_cpu]

            layers_decoded.append({
                "layer_index": l,
                "topk_indices": topk_idx_cpu,
                "topk_probs": topk_vals_cpu,
                "topk_tokens": topk_tokens,
            })

    return layers_decoded


def main():
    parser = argparse.ArgumentParser(description="Decode CoT hidden states at all layers during generation")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Must be 1 for generation")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max_layers", type=int, default=0, help="0 => decode all layers")
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--num_thought_iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)

    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 for generation with hidden state capture")

    torch.manual_seed(args.seed)
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_path / f"cot_layer_decoding_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")

    # Determine target modules and prj_dim
    if "gpt2" in args.model_name_or_path.lower():
        target_modules = ["c_attn", "c_proj", "c_fc"]
        prj_dim = 768
    elif "llama" in args.model_name_or_path.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        prj_dim = 2048
    else:
        target_modules = ["c_attn", "c_proj", "c_fc"]
        prj_dim = 768

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

    # Move model to CUDA and convert to bfloat16 (following section5_analysis.py)
    codi_model = codi_model.to('cuda')
    codi_model.to(torch.bfloat16)
    codi_model.eval()

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

    print(f"Processing {len(examples)} examples with CoT layer decoding...")

    all_results = []

    for ex_idx, example in enumerate(examples):
        print(f"\nProcessing example {ex_idx+1}/{len(examples)}")

        question_text = example["question"].strip()
        input_ids = tokenizer(question_text, return_tensors="pt")["input_ids"].to('cuda')

        # CRITICAL: Add BOT token following CODI architecture
        if training_args.remove_eos:
            bot_tensor = torch.tensor([codi_model.bot_id], dtype=torch.long).expand(input_ids.size(0), 1).to('cuda')
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, codi_model.bot_id], dtype=torch.long).expand(input_ids.size(0), 2).to('cuda')

        input_ids = torch.cat((input_ids, bot_tensor), dim=1)
        bot_position = input_ids.shape[1] - 1  # Position of BOT token

        # Generate with CODI capturing hidden states at each thought iteration
        with torch.no_grad():
            # Initial forward pass through question + BOT
            outputs = codi_model.codi(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=None,
                attention_mask=torch.ones_like(input_ids)
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # [1, 1, hidden_dim]

            continuous_thoughts = []

            # Decode initial thought (at BOT position, before any iterations)
            initial_layers = [hs[:, -1, :] for hs in outputs.hidden_states]  # Get last position from all layers
            initial_decoded = decode_hiddenstates_all_layers(
                hidden_states=initial_layers,
                lm_head=codi_model.codi.lm_head,
                topk=args.topk,
                max_layers=args.max_layers if args.max_layers > 0 else None,
                tokenizer=tokenizer
            )

            continuous_thoughts.append({
                "iteration": 0,
                "type": "initial_bot",
                "position": bot_position,
                "layers": initial_decoded
            })

            # Apply projection if enabled
            if training_args.use_prj:
                latent_embd = codi_model.prj(latent_embd)

            # Iterate through continuous thoughts (following section5_analysis.py)
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                outputs = codi_model.codi(
                    inputs_embeds=latent_embd,  # CRITICAL: use inputs_embeds for continuous thoughts
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # [1, 1, hidden_dim]

                # Decode BEFORE projection (this is the continuous thought)
                thought_layers = [hs[:, -1, :] for hs in outputs.hidden_states]
                layers_decoded = decode_hiddenstates_all_layers(
                    hidden_states=thought_layers,
                    lm_head=codi_model.codi.lm_head,
                    topk=args.topk,
                    max_layers=args.max_layers if args.max_layers > 0 else None,
                    tokenizer=tokenizer
                )

                continuous_thoughts.append({
                    "iteration": i + 1,
                    "type": "continuous_thought",
                    "position": bot_position + 1 + i,  # Virtual position
                    "layers": layers_decoded
                })

                # Apply projection for next iteration
                if training_args.use_prj:
                    latent_embd = codi_model.prj(latent_embd_pre_proj)
                else:
                    latent_embd = latent_embd_pre_proj

            # Generate answer after EOT
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
        print(f"  Captured {len(continuous_thoughts)} continuous thought iterations")

    # Save results
    out_file = run_dir / "cot_layer_decoding_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved CoT layer decoding results to {out_file}")
    print("Done.")


if __name__ == "__main__":
    main()
