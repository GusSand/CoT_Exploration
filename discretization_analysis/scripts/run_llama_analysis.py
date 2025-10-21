#!/usr/bin/env python3
"""
Comprehensive CODI Discretization Analysis - LLAMA VERSION - BATCHED GPU
Compares: vanilla vs alternating discretization vs full discretization
Dataset: GSM8K test set (1319 examples)
Uses batching for GPU speedup
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
parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to process (None = all)")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
parser.add_argument("--output_dir", type=str, default="./llama_discretization_results", help="Output directory")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
args = parser.parse_args()

# Load GSM8K dataset
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
test_set = dataset['test']

if args.num_examples:
    test_set = test_set.select(range(args.num_examples))

print(f"Processing {len(test_set)} examples from GSM8K test set")
print(f"Batch size: {args.batch_size}")
print(f"Device: {args.device}")

def extract_answer_number(sentence: str) -> float:
    """Extract numerical answer from generated text"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])

def run_inference_batched(model, tokenizer, questions, training_args, discretize_mode="vanilla", device="cuda"):
    """
    Run CODI inference with batching

    Args:
        questions: List of question strings
        discretize_mode: "vanilla" | "alternating" | "full"
    """
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

        # Encode questions
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)  # [batch, 1, hidden]

        # Track thought tokens and probabilities for entire batch
        batch_thought_tokens = [[] for _ in range(batch_size)]
        batch_thought_probs = [[] for _ in range(batch_size)]
        batch_discretization_actions = [[] for _ in range(batch_size)]

        # T-2: Before initial projection
        probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
        top_values, top_indices = torch.topk(probs, k=3, dim=2)

        for b in range(batch_size):
            batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
            batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

        # Apply initial projection
        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Thought iterations
        for i in range(training_args.inf_latent_iterations):
            # Decode latent embeddings
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Probe BEFORE projection
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top_values, top_indices = torch.topk(probs, k=3, dim=2)

            for b in range(batch_size):
                batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
                batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

            # Discretization logic
            should_discretize = False
            if discretize_mode == "full":
                should_discretize = True
            elif discretize_mode == "alternating":
                should_discretize = (i % 2 == 1)

            if should_discretize:
                # Discretize entire batch
                top_tokens = torch.argmax(probs[:, 0, :], dim=-1)  # [batch]
                for b in range(batch_size):
                    chosen_token = tokenizer.decode([top_tokens[b].item()])
                    batch_discretization_actions[b].append(f"T{i}: discretized to '{chosen_token}'")

                # Replace with discrete embeddings
                latent_embd = model.get_embd(model.codi, model.model_name)(top_tokens.unsqueeze(1)).to(device)
            else:
                for b in range(batch_size):
                    batch_discretization_actions[b].append(f"T{i}: continuous")

            # Apply projection
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        # Generate final answers
        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).expand(batch_size, 1, 1).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]], dtype=torch.long).expand(batch_size, 1, 2).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor).squeeze(1)  # [batch, 1 or 2, hidden]
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
            next_token_ids = torch.argmax(logits, dim=-1)  # [batch]

            # Handle EOS for each sequence
            for b in range(batch_size):
                if not finished[b]:
                    batch_pred_tokens[b].append(next_token_ids[b].item())
                    if next_token_ids[b] == tokenizer.eos_token_id:
                        finished[b] = True

            if finished.all():
                break

            output = model.get_embd(model.codi, model.model_name)(next_token_ids.unsqueeze(1)).to(device)

        # Decode answers
        results = []
        for b in range(batch_size):
            decoded_answer = tokenizer.decode(batch_pred_tokens[b], skip_special_tokens=True)
            results.append({
                "thought_tokens": batch_thought_tokens[b],
                "thought_probs": batch_thought_probs[b],
                "discretization_actions": batch_discretization_actions[b],
                "generated_text": decoded_answer,
                "answer_number": extract_answer_number(decoded_answer)
            })

        return results

def main():
    print("="*80)
    print("CODI-LLAMA DISCRETIZATION ANALYSIS - BATCHED GPU VERSION")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model for LLAMA
    print("\nInitializing CODI-Llama model...")
    model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/CoT_Exploration/models/CODI-llama3.2-1b",
        full_precision=True
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=2048,  # Llama hidden size
        remove_eos=True,
        greedy=True,
        bf16=True if args.device == "cuda" else False,
        inf_num_iterations=1
    )

    # Llama-specific LoRA target modules
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
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

    # Load checkpoint
    print(f"Loading checkpoint from {model_args.ckpt_dir}...")
    checkpoint_path = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    # Move to device
    model = model.to(args.device)
    if args.device == "cuda" and training_args.bf16:
        model = model.to(torch.bfloat16)

    # Load tokenizer
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
    modes = ["vanilla", "alternating", "full"]
    all_results = {mode: [] for mode in modes}

    start_time = time.time()

    # Process in batches
    num_batches = (len(test_set) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(test_set))
        batch_indices = list(range(start_idx, end_idx))

        questions = [test_set[i]['question'] for i in batch_indices]
        ground_truths = [extract_answer_number(test_set[i]['answer']) for i in batch_indices]

        print(f"\n{'='*80}")
        print(f"Batch {batch_idx+1}/{num_batches} | Examples {start_idx+1}-{end_idx}/{len(test_set)}")
        print(f"{'='*80}")

        for mode in modes:
            mode_start = time.time()
            batch_results = run_inference_batched(model, tokenizer, questions, training_args,
                                                  discretize_mode=mode, device=args.device)
            mode_time = time.time() - mode_start

            # Process results
            for i, result in enumerate(batch_results):
                is_correct = abs(result['answer_number'] - ground_truths[i]) < 0.01

                result_entry = {
                    'question': questions[i],
                    'ground_truth': ground_truths[i],
                    'predicted_number': result['answer_number'],
                    'generated_text': result['generated_text'],
                    'thought_tokens': result['thought_tokens'],
                    'thought_probs': result['thought_probs'],
                    'discretization_actions': result['discretization_actions'],
                    'correct': is_correct,
                    'time': mode_time / len(batch_results)  # Per-example time
                }

                all_results[mode].append(result_entry)

            # Calculate batch accuracy
            batch_correct = sum(1 for i, r in enumerate(batch_results) if abs(r['answer_number'] - ground_truths[i]) < 0.01)
            print(f"  {mode:12s}: {batch_correct}/{len(batch_results)} correct ({mode_time:.2f}s, {mode_time/len(batch_results):.3f}s/ex)")

        # Save checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            with open(os.path.join(args.output_dir, f"results_checkpoint_{end_idx}.json"), 'w') as f:
                json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time

    # Calculate statistics
    stats = {}
    for mode in modes:
        correct = sum(1 for r in all_results[mode] if r['correct'])
        total = len(all_results[mode])
        accuracy = 100 * correct / total if total > 0 else 0
        avg_time = sum(r['time'] for r in all_results[mode]) / total if total > 0 else 0

        stats[mode] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_time_per_example': avg_time
        }

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS - CODI-LLAMA")
    print("="*80)
    for mode in modes:
        s = stats[mode]
        print(f"{mode:12s}: {s['accuracy']:.2f}% ({s['correct']}/{s['total']}) | Avg time: {s['avg_time_per_example']:.3f}s")

    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Examples per minute: {len(test_set)/(total_time/60):.2f}")

    # Save final results
    final_output = {
        'stats': stats,
        'results': all_results,
        'metadata': {
            'model': 'CODI-Llama3.2-1B',
            'num_examples': len(test_set),
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
