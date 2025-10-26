#!/usr/bin/env python3
"""
CODI Discretization Analysis on Custom Dataset - UNIFIED VERSION
Works with both CODI-GPT2 and CODI-Llama models
Dataset: llama_cot_all.json (CoT activation patching dataset)
Compares: vanilla vs full discretization vs posthoc discretization
"""
import torch
import transformers
from peft import LoraConfig, TaskType
import os
import sys
import json
import time
import re
from collections import defaultdict
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import CODI, ModelArguments, DataArguments, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True, choices=["gpt2", "llama"],
                   help="Model type: gpt2 or llama")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not specified)")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
args = parser.parse_args()

# Auto-generate output dir if not specified
if args.output_dir is None:
    args.output_dir = f"/workspace/CoT_Exploration/src/experiments/custom_dataset_results_{args.model_type}"

# Load custom dataset
dataset_path = "/workspace/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_all.json"
print(f"Loading dataset from {dataset_path}...")
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} examples from custom dataset")
print(f"Model type: {args.model_type}")
print(f"Batch size: {args.batch_size}")
print(f"Device: {args.device}")
print(f"Output directory: {args.output_dir}")

def extract_answer_number(sentence: str) -> float:
    """Extract numerical answer from generated text"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])

def run_posthoc_discretization(model, tokenizer, questions, training_args, device="cuda"):
    """Post-hoc discretization with frozen continuous chain"""
    batch_size = len(questions)

    with torch.no_grad():
        # Tokenize questions
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch_size, 2).to(device)

        inputs = tokenizer(questions, return_tensors="pt", padding="longest")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

        # Phase 1: Generate vanilla continuous CoT
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values_vanilla = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        continuous_hidden_states = [latent_embd.clone()]
        batch_thought_tokens = [[] for _ in range(batch_size)]
        batch_thought_probs = [[] for _ in range(batch_size)]
        batch_discretization_actions = [[] for _ in range(batch_size)]
        batch_norm_info = [[] for _ in range(batch_size)]

        probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
        top_values, top_indices = torch.topk(probs, k=3, dim=2)

        for b in range(batch_size):
            batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
            batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Thought iterations
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values_vanilla
            )
            past_key_values_vanilla = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            continuous_hidden_states.append(latent_embd.clone())

            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top_values, top_indices = torch.topk(probs, k=3, dim=2)

            for b in range(batch_size):
                batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
                batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        # Phase 2: Apply post-hoc discretization
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values

        latent_embd_init = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        if training_args.use_prj:
            latent_embd_init = model.prj(latent_embd_init)

        for i in range(training_args.inf_latent_iterations):
            latent_embd = continuous_hidden_states[i+1].clone()
            continuous_norms = torch.norm(latent_embd[:, 0, :], dim=-1)
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top_tokens = torch.argmax(probs[:, 0, :], dim=-1)
            token_embd = model.get_embd(model.codi, model.model_name)(top_tokens.unsqueeze(1)).to(device)

            token_norms = torch.norm(token_embd[:, 0, :], dim=-1)
            scale_factors = continuous_norms / (token_norms + 1e-8)
            latent_embd_discretized = token_embd * scale_factors.unsqueeze(1).unsqueeze(2)

            for b in range(batch_size):
                chosen_token = tokenizer.decode([top_tokens[b].item()])
                batch_discretization_actions[b].append(
                    f"T{i}: discretized to '{chosen_token}' (norm: {continuous_norms[b].item():.4f}→{continuous_norms[b].item():.4f}, scale: {scale_factors[b].item():.4f})"
                )
                batch_norm_info[b].append({
                    'position': f'T{i}',
                    'continuous_norm': continuous_norms[b].item(),
                    'token_norm_before_scaling': token_norms[b].item(),
                    'scale_factor': scale_factors[b].item(),
                    'token': chosen_token
                })

            if training_args.use_prj:
                latent_embd_discretized = model.prj(latent_embd_discretized)

            outputs = model.codi(
                inputs_embeds=latent_embd_discretized,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

        # Phase 3: Generate answer
        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).expand(batch_size, 1, 1).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]], dtype=torch.long).expand(batch_size, 1, 2).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor).squeeze(1)
        output = eot_emb

        batch_pred_tokens = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(256):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
            next_token_ids = torch.argmax(logits, dim=-1)

            for b in range(batch_size):
                if not finished[b]:
                    batch_pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True

            if finished.all():
                break

            output = model.get_embd(model.codi, model.model_name)(next_token_ids.unsqueeze(1)).to(device)

        results = []
        for b in range(batch_size):
            decoded_answer = tokenizer.decode(batch_pred_tokens[b], skip_special_tokens=True)
            results.append({
                "thought_tokens": batch_thought_tokens[b],
                "thought_probs": batch_thought_probs[b],
                "discretization_actions": batch_discretization_actions[b],
                "norm_info": batch_norm_info[b],
                "generated_text": decoded_answer,
                "answer_number": extract_answer_number(decoded_answer)
            })

        return results

def run_inference_batched(model, tokenizer, questions, training_args, discretize_mode="vanilla", device="cuda"):
    """Run CODI inference with batching and norm-controlled discretization"""
    if discretize_mode == "posthoc":
        return run_posthoc_discretization(model, tokenizer, questions, training_args, device)

    batch_size = len(questions)

    with torch.no_grad():
        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch_size, 2).to(device)

        inputs = tokenizer(questions, return_tensors="pt", padding="longest")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        batch_thought_tokens = [[] for _ in range(batch_size)]
        batch_thought_probs = [[] for _ in range(batch_size)]
        batch_discretization_actions = [[] for _ in range(batch_size)]
        batch_norm_info = [[] for _ in range(batch_size)]

        probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
        top_values, top_indices = torch.topk(probs, k=3, dim=2)

        for b in range(batch_size):
            batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
            batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top_values, top_indices = torch.topk(probs, k=3, dim=2)

            for b in range(batch_size):
                batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
                batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

            should_discretize = False
            if discretize_mode == "full":
                should_discretize = True
            elif discretize_mode == "alternating":
                should_discretize = (i % 2 == 1)

            if should_discretize:
                continuous_norms = torch.norm(latent_embd[:, 0, :], dim=-1)
                top_tokens = torch.argmax(probs[:, 0, :], dim=-1)
                token_embd = model.get_embd(model.codi, model.model_name)(top_tokens.unsqueeze(1)).to(device)
                token_norms = torch.norm(token_embd[:, 0, :], dim=-1)
                scale_factors = continuous_norms / (token_norms + 1e-8)
                latent_embd = token_embd * scale_factors.unsqueeze(1).unsqueeze(2)

                for b in range(batch_size):
                    chosen_token = tokenizer.decode([top_tokens[b].item()])
                    batch_discretization_actions[b].append(
                        f"T{i}: discretized to '{chosen_token}' (norm: {continuous_norms[b].item():.4f}→{continuous_norms[b].item():.4f}, scale: {scale_factors[b].item():.4f})"
                    )
                    batch_norm_info[b].append({
                        'position': f'T{i}',
                        'continuous_norm': continuous_norms[b].item(),
                        'token_norm_before_scaling': token_norms[b].item(),
                        'scale_factor': scale_factors[b].item(),
                        'token': chosen_token
                    })
            else:
                for b in range(batch_size):
                    batch_discretization_actions[b].append(f"T{i}: continuous")

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).expand(batch_size, 1, 1).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]], dtype=torch.long).expand(batch_size, 1, 2).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor).squeeze(1)
        output = eot_emb

        batch_pred_tokens = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(256):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
            next_token_ids = torch.argmax(logits, dim=-1)

            for b in range(batch_size):
                if not finished[b]:
                    batch_pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True

            if finished.all():
                break

            output = model.get_embd(model.codi, model.model_name)(next_token_ids.unsqueeze(1)).to(device)

        results = []
        for b in range(batch_size):
            decoded_answer = tokenizer.decode(batch_pred_tokens[b], skip_special_tokens=True)
            results.append({
                "thought_tokens": batch_thought_tokens[b],
                "thought_probs": batch_thought_probs[b],
                "discretization_actions": batch_discretization_actions[b],
                "norm_info": batch_norm_info[b],
                "generated_text": decoded_answer,
                "answer_number": extract_answer_number(decoded_answer)
            })

        return results

def main():
    print("="*80)
    print(f"CODI-{args.model_type.upper()} DISCRETIZATION ANALYSIS - CUSTOM DATASET")
    print("="*80)

    os.makedirs(args.output_dir, exist_ok=True)

    # Model configuration
    if args.model_type == "gpt2":
        model_name = "openai-community/gpt2"
        ckpt_dir = "/workspace/CoT_Exploration/models/CODI-gpt2"
        target_modules = ["c_attn", "c_proj", "c_fc"]
        lora_r = 128
        lora_alpha = 32
        prj_dim = 768
    else:  # llama
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        ckpt_dir = "/workspace/CoT_Exploration/models/CODI-llama3.2-1b"
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r = 128
        lora_alpha = 32
        prj_dim = 2048

    print(f"\nInitializing CODI-{args.model_type.upper()} model...")
    print(f"Model path: {model_name}")
    print(f"Checkpoint: {ckpt_dir}")

    model_args = ModelArguments(
        model_name_or_path=model_name,
        lora_init=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        ckpt_dir=ckpt_dir,
        full_precision=True
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=prj_dim,
        remove_eos=True,
        greedy=True,
        bf16=True if args.device == "cuda" else False,
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

    model = CODI(model_args, training_args, lora_config)

    print(f"Loading checkpoint from {model_args.ckpt_dir}...")
    checkpoint_path = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    model = model.to(args.device)
    if args.device == "cuda" and training_args.bf16:
        model = model.to(torch.bfloat16)
    else:
        model = model.float()

    print("Loading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    model.eval()

    print(f"\nModel ready on {args.device}! Starting evaluation...\n")

    # Run experiments
    modes = ["vanilla", "full", "posthoc"]
    all_results = {mode: [] for mode in modes}

    start_time = time.time()

    # Process in batches
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(dataset))
        batch_indices = list(range(start_idx, end_idx))

        questions = [dataset[i]['question'] for i in batch_indices]
        # Handle cases where answer might be null
        ground_truths = []
        for i in batch_indices:
            ans = dataset[i].get('answer')
            if ans is None or ans == -1:
                ground_truths.append(float('inf'))
            else:
                ground_truths.append(float(ans))

        print(f"\n{'='*80}")
        print(f"Batch {batch_idx+1}/{num_batches} | Examples {start_idx+1}-{end_idx}/{len(dataset)}")
        print(f"{'='*80}")

        for mode in modes:
            mode_start = time.time()
            batch_results = run_inference_batched(model, tokenizer, questions, training_args,
                                                  discretize_mode=mode, device=args.device)
            mode_time = time.time() - mode_start

            for i, result in enumerate(batch_results):
                is_correct = abs(result['answer_number'] - ground_truths[i]) < 0.01 if ground_truths[i] != float('inf') else False

                result_entry = {
                    'pair_id': dataset[batch_indices[i]].get('pair_id', -1),
                    'variant': dataset[batch_indices[i]].get('variant', 'unknown'),
                    'question': questions[i],
                    'ground_truth': ground_truths[i],
                    'predicted_number': result['answer_number'],
                    'generated_text': result['generated_text'],
                    'thought_tokens': result['thought_tokens'],
                    'thought_probs': result['thought_probs'],
                    'discretization_actions': result['discretization_actions'],
                    'norm_info': result['norm_info'],
                    'correct': is_correct,
                    'time': mode_time / len(batch_results)
                }

                all_results[mode].append(result_entry)

            batch_correct = sum(1 for i, r in enumerate(batch_results)
                              if ground_truths[i] != float('inf') and abs(r['answer_number'] - ground_truths[i]) < 0.01)
            batch_total = sum(1 for gt in ground_truths if gt != float('inf'))
            print(f"  {mode:12s}: {batch_correct}/{batch_total} correct ({mode_time:.2f}s, {mode_time/len(batch_results):.3f}s/ex)")

        # Save checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            with open(os.path.join(args.output_dir, f"results_checkpoint_{end_idx}.json"), 'w') as f:
                json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time

    # Calculate statistics
    stats = {}
    for mode in modes:
        # Only count examples with valid ground truth
        valid_results = [r for r in all_results[mode] if r['ground_truth'] != float('inf')]
        correct = sum(1 for r in valid_results if r['correct'])
        total = len(valid_results)
        accuracy = 100 * correct / total if total > 0 else 0
        avg_time = sum(r['time'] for r in all_results[mode]) / len(all_results[mode]) if all_results[mode] else 0

        stats[mode] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_time_per_example': avg_time
        }

    # Print summary
    print("\n" + "="*80)
    print(f"FINAL RESULTS - CODI-{args.model_type.upper()} ON CUSTOM DATASET")
    print("="*80)
    for mode in modes:
        s = stats[mode]
        print(f"{mode:12s}: {s['accuracy']:.2f}% ({s['correct']}/{s['total']}) | Avg time: {s['avg_time_per_example']:.3f}s")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Examples per minute: {len(dataset)/(total_time/60):.2f}")

    # Save final results
    final_output = {
        'stats': stats,
        'results': all_results,
        'metadata': {
            'model': f'CODI-{args.model_type}',
            'model_path': model_name,
            'checkpoint': ckpt_dir,
            'dataset': dataset_path,
            'discretization_method': 'norm_controlled',
            'num_examples': len(dataset),
            'batch_size': args.batch_size,
            'device': args.device,
            'total_time': total_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    with open(os.path.join(args.output_dir, "final_results.json"), 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/final_results.json")

    return stats, all_results

if __name__ == "__main__":
    stats, results = main()
