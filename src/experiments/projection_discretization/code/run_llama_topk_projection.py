#!/usr/bin/env python3
"""
CODI Discretization Analysis with TOP-K PROJECTION METHOD - GPT-2 VERSION
Compares: vanilla vs different projection configurations
KEY FEATURE: Projects continuous token vector onto subspace spanned by top-k vocab embeddings
- k=1: Single token projection (equivalent to norm-controlled replacement)
- k>1: Subspace projection onto k-dimensional vocabulary subspace
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
parser.add_argument("--output_dir", type=str, default="./meta-llama/Llama-2-7b-hf_projection_results", help="Output directory")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
parser.add_argument("--k_nearest", type=int, default=1, help="Number of nearest tokens to project onto (for future use)")
parser.add_argument("--discretize_positions", type=str, default="all",
                    help="Which positions to discretize: 'all', 'bot_only', 'thought_only', or comma-separated indices like '0,2,4'")
parser.add_argument("--normalize", action="store_true", help="Normalize projected vector to match original norm")
args = parser.parse_args()

# Load GSM8K dataset
from datasets import load_dataset
dataset = load_dataset("gsm8k", "main")
test_set = dataset['test']

if args.num_examples is not None and args.num_examples > 0:
    test_set = test_set.select(range(args.num_examples))

print(f"Processing {len(test_set)} examples from GSM8K test set")
print(f"Batch size: {args.batch_size}")
print(f"Device: {args.device}")
print(f"K nearest: {args.k_nearest}")
print(f"Discretize positions: {args.discretize_positions}")
print(f"Normalize: {args.normalize}")

def extract_answer_number(sentence: str) -> float:
    """Extract numerical answer from generated text"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])

def should_discretize_position(position_idx: int, discretize_positions: str, total_iterations: int = 6) -> bool:
    """
    Determine if a position should be discretized.

    Args:
        position_idx: Position index (-1 for BoT output, 0-5 for thought tokens)
        discretize_positions: Configuration string
        total_iterations: Total number of thought iterations (default 6)

    Returns:
        True if this position should be discretized
    """
    if discretize_positions == "all":
        return True
    elif discretize_positions == "bot_only":
        return position_idx == -1
    elif discretize_positions == "thought_only":
        return position_idx >= 0
    elif discretize_positions == "none":
        return False
    else:
        # Parse comma-separated indices
        try:
            indices = [int(x.strip()) for x in discretize_positions.split(',')]
            return position_idx in indices
        except:
            print(f"Warning: Invalid discretize_positions '{discretize_positions}', defaulting to 'none'")
            return False

def project_onto_topk_vocab(continuous_vector, vocab_embeddings_topk, normalize=False):
    """
    Project continuous vector onto subspace spanned by top-k vocab embeddings.

    For k=1: Uses direct replacement formula for exact numerical equivalence
    For k>1: Uses least-squares projection onto k-dimensional subspace

    Args:
        continuous_vector: Tensor of shape [batch, hidden] - the continuous activation
        vocab_embeddings_topk: Tensor of shape [batch, k, hidden] - top k vocab token embeddings
        normalize: If True, scale result to match continuous norm
                   If False, keep raw projection magnitude

    Returns:
        Projected vector of shape [batch, hidden]
    """
    batch_size, k, hidden = vocab_embeddings_topk.shape

    if k == 1:
        # Special case: k=1 uses direct replacement for numerical stability
        vocab_embedding = vocab_embeddings_topk[:, 0, :]  # [batch, hidden]
        if normalize:
            # CRITICAL: Use direct replacement formula for EXACT numerical equivalence
            continuous_norm = torch.norm(continuous_vector, dim=-1, keepdim=True)
            vocab_norm = torch.norm(vocab_embedding, dim=-1, keepdim=True)
            result = vocab_embedding * (continuous_norm / (vocab_norm + 1e-8))
            return result
        else:
            # True projection onto vocab direction (unnormalized)
            vocab_direction = vocab_embedding / (torch.norm(vocab_embedding, dim=-1, keepdim=True) + 1e-8)
            projection_scalar = torch.sum(continuous_vector * vocab_direction, dim=-1, keepdim=True)
            projected = projection_scalar * vocab_direction
            return projected
    else:
        # General case: k>1 uses subspace projection
        # For each batch element, solve: alpha = (V V^T)^{-1} V c
        # where V is [k, hidden] matrix of vocab embeddings

        projected_batch = []
        for b in range(batch_size):
            V = vocab_embeddings_topk[b]  # [k, hidden]
            c = continuous_vector[b]  # [hidden]

            # Compute Gram matrix G = V V^T (k x k)
            G = torch.mm(V, V.t())  # [k, k]

            # Compute V c (k-dimensional vector)
            Vc = torch.mv(V, c)  # [k]

            # Solve G alpha = Vc for alpha
            try:
                alpha = torch.linalg.solve(G, Vc)  # [k]
            except:
                # If singular, use pseudo-inverse
                alpha = torch.linalg.lstsq(G, Vc).solution  # [k]

            # Compute projection: result = V^T alpha = sum_i alpha_i v_i
            projected = torch.mv(V.t(), alpha)  # [hidden]
            projected_batch.append(projected)

        projected = torch.stack(projected_batch, dim=0)  # [batch, hidden]

        if normalize:
            # Scale to match original norm
            continuous_norm = torch.norm(continuous_vector, dim=-1, keepdim=True)
            projected_norm = torch.norm(projected, dim=-1, keepdim=True)
            projected = projected * (continuous_norm / (projected_norm + 1e-8))

        return projected

def run_inference_batched(model, tokenizer, questions, training_args, discretize_positions="all",
                          normalize=False, k_nearest=1, device="cuda"):
    """
    Run CODI inference with batching and TOP-K PROJECTION-BASED discretization

    Args:
        questions: List of question strings
        discretize_positions: Which positions to discretize
        normalize: Whether to normalize projected vectors
        k_nearest: Number of top tokens to project onto (k=1: single token, k>1: subspace)
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
        batch_projection_info = [[] for _ in range(batch_size)]

        # T-2: Before initial projection - BoT output
        probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
        top_values, top_indices = torch.topk(probs, k=3, dim=2)

        for b in range(batch_size):
            batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
            batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

        # Check if we should discretize BoT output (position -1)
        if should_discretize_position(-1, discretize_positions):
            # Get continuous vector before projection
            continuous_vector = latent_embd[:, 0, :]  # [batch, hidden]
            continuous_norms = torch.norm(continuous_vector, dim=-1)  # [batch]

            # Get top-k vocab tokens
            top_k_probs, top_k_indices = torch.topk(probs[:, 0, :], k=k_nearest, dim=-1)  # [batch, k]
            vocab_embeddings_topk = model.get_embd(model.codi, model.model_name)(top_k_indices).to(device)  # [batch, k, hidden]

            # Project onto top-k vocab subspace
            projected_vector = project_onto_topk_vocab(continuous_vector, vocab_embeddings_topk, normalize=normalize)
            latent_embd = projected_vector.unsqueeze(1)  # [batch, 1, hidden]

            projected_norms = torch.norm(projected_vector, dim=-1)  # [batch]

            for b in range(batch_size):
                if k_nearest == 1:
                    chosen_tokens_str = tokenizer.decode([top_k_indices[b, 0].item()])
                else:
                    chosen_tokens_str = "[" + ", ".join([tokenizer.decode([top_k_indices[b, i].item()]) for i in range(k_nearest)]) + "]"
                norm_str = "normalized" if normalize else "unnormalized"
                batch_discretization_actions[b].append(
                    f"BoT: projected to {chosen_tokens_str} (k={k_nearest}, {norm_str}, norm: {continuous_norms[b].item():.4f}->{projected_norms[b].item():.4f})"
                )
                batch_projection_info[b].append({
                    'position': 'BoT',
                    'continuous_norm': continuous_norms[b].item(),
                    'projected_norm': projected_norms[b].item(),
                    'normalized': normalize,
                    'k': k_nearest,
                    'tokens': [tokenizer.decode([top_k_indices[b, i].item()]) for i in range(k_nearest)]
                })
        else:
            for b in range(batch_size):
                batch_discretization_actions[b].append(f"BoT: continuous")

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

            # Probe BEFORE discretization/projection
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top_values, top_indices = torch.topk(probs, k=3, dim=2)

            for b in range(batch_size):
                batch_thought_tokens[b].append([tokenizer.decode([idx.item()]) for idx in top_indices[b, 0]])
                batch_thought_probs[b].append([p.item() for p in top_values[b, 0]])

            # Check if we should discretize this thought position
            if should_discretize_position(i, discretize_positions):
                # Get continuous vector before projection
                continuous_vector = latent_embd[:, 0, :]  # [batch, hidden]
                continuous_norms = torch.norm(continuous_vector, dim=-1)  # [batch]

                # Get top-k vocab tokens
                top_k_probs, top_k_indices = torch.topk(probs[:, 0, :], k=k_nearest, dim=-1)  # [batch, k]
                vocab_embeddings_topk = model.get_embd(model.codi, model.model_name)(top_k_indices).to(device)  # [batch, k, hidden]

                # Project onto top-k vocab subspace
                projected_vector = project_onto_topk_vocab(continuous_vector, vocab_embeddings_topk, normalize=normalize)
                latent_embd = projected_vector.unsqueeze(1)  # [batch, 1, hidden]

                projected_norms = torch.norm(projected_vector, dim=-1)  # [batch]

                for b in range(batch_size):
                    if k_nearest == 1:
                        chosen_tokens_str = tokenizer.decode([top_k_indices[b, 0].item()])
                    else:
                        chosen_tokens_str = "[" + ", ".join([tokenizer.decode([top_k_indices[b, i].item()]) for i in range(k_nearest)]) + "]"
                    norm_str = "normalized" if normalize else "unnormalized"
                    batch_discretization_actions[b].append(
                        f"T{i}: projected to {chosen_tokens_str} (k={k_nearest}, {norm_str}, norm: {continuous_norms[b].item():.4f}->{projected_norms[b].item():.4f})"
                    )
                    batch_projection_info[b].append({
                        'position': f'T{i}',
                        'continuous_norm': continuous_norms[b].item(),
                        'projected_norm': projected_norms[b].item(),
                        'normalized': normalize,
                        'k': k_nearest,
                        'tokens': [tokenizer.decode([top_k_indices[b, i].item()]) for i in range(k_nearest)]
                    })
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
                "projection_info": batch_projection_info[b],
                "generated_text": decoded_answer,
                "answer_number": extract_answer_number(decoded_answer)
            })

        return results

def main():
    print("="*80)
    print("CODI-GPT2 DISCRETIZATION ANALYSIS - PROJECTION METHOD")
    print("="*80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize model
    print("\nInitializing CODI-GPT2 model...")
    model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-2-7b-hf",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="./checkpoints/CODI-meta-llama/Llama-2-7b-hf",
        full_precision=True
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=768,
        remove_eos=True,
        greedy=True,
        bf16=True if args.device == "cuda" else False,
        inf_num_iterations=1
    )

    target_modules = ["c_attn", "c_proj", 'c_fc']
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

    # Move to device and handle dtype
    if args.device == "cpu":
        model = model.to(torch.float32)  # CPU requires float32
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

    # Run experiment with current configuration
    config_name = f"{args.discretize_positions}_{'normalized' if args.normalize else 'unnormalized'}"
    all_results = []

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
        print(f"Config: {config_name}")
        print(f"{'='*80}")

        batch_start = time.time()
        batch_results = run_inference_batched(model, tokenizer, questions, training_args,
                                              discretize_positions=args.discretize_positions,
                                              normalize=args.normalize, k_nearest=args.k_nearest,
                                              device=args.device)
        batch_time = time.time() - batch_start

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
                'projection_info': result['projection_info'],
                'correct': is_correct,
                'time': batch_time / len(batch_results)  # Per-example time
            }

            all_results.append(result_entry)

        # Calculate batch accuracy
        batch_correct = sum(1 for i, r in enumerate(batch_results) if abs(r['answer_number'] - ground_truths[i]) < 0.01)
        print(f"  Accuracy: {batch_correct}/{len(batch_results)} correct ({batch_time:.2f}s, {batch_time/len(batch_results):.3f}s/ex)")

        # Save checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            with open(os.path.join(args.output_dir, f"results_checkpoint_{end_idx}.json"), 'w') as f:
                json.dump(all_results, f, indent=2)

    total_time = time.time() - start_time

    # Calculate statistics
    correct = sum(1 for r in all_results if r['correct'])
    total = len(all_results)
    accuracy = 100 * correct / total if total > 0 else 0
    avg_time = sum(r['time'] for r in all_results) / total if total > 0 else 0

    stats = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'avg_time_per_example': avg_time
    }

    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS - CODI-GPT2 PROJECTION METHOD")
    print("="*80)
    print(f"Config: {config_name}")
    print(f"Accuracy: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print(f"Avg time: {stats['avg_time_per_example']:.3f}s per example")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Examples per minute: {len(test_set)/(total_time/60):.2f}")

    # Save final results
    final_output = {
        'stats': stats,
        'results': all_results,
        'metadata': {
            'model': 'CODI-GPT2',
            'discretization_method': 'projection',
            'discretize_positions': args.discretize_positions,
            'normalize': args.normalize,
            'k_nearest': args.k_nearest,
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
