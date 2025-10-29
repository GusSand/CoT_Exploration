#!/usr/bin/env python3
"""
Evaluate probe with semantic equivalence for spacing:
- '9' = ' 9' (number with/without leading space)
- '40' = ' 40'
- '-' = ' -' (operators with/without leading space)

Report accuracy separately for:
- Numbers (digit sequences, possibly with leading space)
- Non-numbers (everything else)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
import argparse
import sys
import re

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
    checkpoint_path = f"{args.ckpt_dir}/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.codi.tie_weights()
    model = model.float().to(device)
    model.eval()

    print(f"Model loaded: n_layer={model.codi.config.n_layer}, has_prj={model.use_prj}")
    print("="*100)

    return model


def normalize_token(token_str):
    """
    Normalize token for semantic equivalence.
    Returns normalized string (stripped) and category.

    Categories:
    - 'number': Pure digit sequences (with or without leading space)
    - 'operator': Mathematical operators (with or without leading space)
    - 'other': Everything else
    """
    stripped = token_str.strip()

    # Check if it's a pure number (all digits)
    if stripped and stripped.isdigit():
        return stripped, 'number'

    # Check if it's an operator (common math/formatting symbols)
    if stripped in ['-', '+', '*', '/', '=', '>', '<', '>>', '<<', '·', '×', '÷']:
        return stripped, 'operator'

    # Everything else (words, punctuation, mixed)
    return token_str, 'other'  # Keep original for non-numbers


def tokens_semantically_equal(token1, token2):
    """Check if two tokens are semantically equivalent."""
    norm1, cat1 = normalize_token(token1)
    norm2, cat2 = normalize_token(token2)

    # If both are numbers or operators, compare normalized versions
    if cat1 in ['number', 'operator'] and cat2 in ['number', 'operator']:
        return norm1 == norm2

    # Otherwise require exact match
    return token1 == token2


def evaluate_semantic_accuracy(model, tokenizer, probe, dataset, args):
    """Evaluate probe with semantic equivalence."""
    print("\n" + "="*100)
    print(f"SEMANTIC ACCURACY EVALUATION")
    print("="*100)
    print(f"Probe: {args.probe_path}")
    print(f"Test examples: {args.num_test_examples}")
    print("="*100)
    print("\nSemantic Equivalence Rules:")
    print("  - Numbers: '9' = ' 9', '40' = ' 40', etc.")
    print("  - Operators: '-' = ' -', '+' = ' +', etc.")
    print("  - Other tokens: Require exact match")
    print("="*100)

    model.eval()
    lm_head = model.codi.lm_head

    # Aggregate metrics
    total_hard_matches = 0
    total_semantic_matches = 0

    # By category
    number_hard_matches = 0
    number_semantic_matches = 0
    number_total = 0

    operator_hard_matches = 0
    operator_semantic_matches = 0
    operator_total = 0

    other_hard_matches = 0
    other_semantic_matches = 0
    other_total = 0

    # Per-iteration tracking
    iter_hard_matches = [0] * args.inf_latent_iterations
    iter_semantic_matches = [0] * args.inf_latent_iterations

    iter_number_hard = [0] * args.inf_latent_iterations
    iter_number_semantic = [0] * args.inf_latent_iterations
    iter_number_total = [0] * args.inf_latent_iterations

    total_samples = 0

    # Examples where semantic matching helps
    semantic_rescue_examples = []

    num_examples = min(args.num_test_examples, len(dataset['test']))

    with torch.no_grad():
        for ex_idx in range(num_examples):
            question = dataset['test'][ex_idx]['question'].strip()

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

                # Extract activations
                source_hidden = hidden_states[args.source_layer + 1][0, 0, :]
                target_hidden = hidden_states[args.target_layer + 1][0, 0, :]

                # Apply probe
                probed_hidden = probe(source_hidden.unsqueeze(0)).squeeze(0)

                # Get logits
                probed_logits = lm_head(probed_hidden)
                target_logits = lm_head(target_hidden)

                # Top-1 predictions
                probed_top1_id = probed_logits.argmax().item()
                target_top1_id = target_logits.argmax().item()

                # Decode tokens
                probed_token = tokenizer.decode([probed_top1_id])
                target_token = tokenizer.decode([target_top1_id])

                # Hard match (exact)
                hard_match = (probed_top1_id == target_top1_id)

                # Semantic match
                semantic_match = tokens_semantically_equal(probed_token, target_token)

                # Categorize target token
                _, target_category = normalize_token(target_token)

                # Update counters
                total_hard_matches += int(hard_match)
                total_semantic_matches += int(semantic_match)

                iter_hard_matches[iteration] += int(hard_match)
                iter_semantic_matches[iteration] += int(semantic_match)

                if target_category == 'number':
                    number_total += 1
                    number_hard_matches += int(hard_match)
                    number_semantic_matches += int(semantic_match)

                    iter_number_total[iteration] += 1
                    iter_number_hard[iteration] += int(hard_match)
                    iter_number_semantic[iteration] += int(semantic_match)
                elif target_category == 'operator':
                    operator_total += 1
                    operator_hard_matches += int(hard_match)
                    operator_semantic_matches += int(semantic_match)
                else:
                    other_total += 1
                    other_hard_matches += int(hard_match)
                    other_semantic_matches += int(semantic_match)

                total_samples += 1

                # Track semantic rescues (semantic match but not hard match)
                if semantic_match and not hard_match:
                    semantic_rescue_examples.append({
                        'example_idx': ex_idx,
                        'iteration': iteration,
                        'question': question[:70],
                        'probed': probed_token,
                        'target': target_token,
                        'category': target_category
                    })

                # Detailed logging for first 5 examples
                if ex_idx < 5:
                    if iteration == 0:
                        print(f"\n{'-'*100}")
                        print(f"Example {ex_idx + 1}: {question[:70]}...")

                    hard_str = "✓" if hard_match else "✗"
                    sem_str = "✓" if semantic_match else "✗"
                    rescue_str = " (SEMANTIC RESCUE)" if (semantic_match and not hard_match) else ""

                    print(f"  Iter {iteration + 1}: Probe='{probed_token}' vs Target='{target_token}' | "
                          f"Hard={hard_str} Semantic={sem_str} [{target_category}]{rescue_str}")

                # Prepare next iteration
                latent_embd_pre_proj = iter_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd_projected = model.prj(latent_embd_pre_proj)
                else:
                    latent_embd_projected = latent_embd_pre_proj

            if (ex_idx + 1) % 50 == 0:
                current_hard = 100.0 * total_hard_matches / total_samples
                current_sem = 100.0 * total_semantic_matches / total_samples
                print(f"\n{'-'*100}")
                print(f"Processed {ex_idx + 1}/{num_examples} examples...")
                print(f"  Running Hard Accuracy: {current_hard:.2f}%")
                print(f"  Running Semantic Accuracy: {current_sem:.2f}%")

    # Compute final statistics
    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}")

    hard_accuracy = 100.0 * total_hard_matches / total_samples
    semantic_accuracy = 100.0 * total_semantic_matches / total_samples
    semantic_gain = semantic_accuracy - hard_accuracy

    print(f"\nOverall Metrics ({total_samples} samples):")
    print(f"  Hard Top-1 Accuracy (exact match):     {total_hard_matches}/{total_samples} = {hard_accuracy:.2f}%")
    print(f"  Semantic Top-1 Accuracy (normalized):  {total_semantic_matches}/{total_samples} = {semantic_accuracy:.2f}%")
    print(f"  Semantic Gain:                         +{semantic_gain:.2f}%")
    print(f"  Semantic Rescues:                      {len(semantic_rescue_examples)} samples")

    # By category
    print(f"\n{'='*100}")
    print("ACCURACY BY TOKEN CATEGORY")
    print(f"{'='*100}")

    if number_total > 0:
        num_hard_pct = 100.0 * number_hard_matches / number_total
        num_sem_pct = 100.0 * number_semantic_matches / number_total
        num_gain = num_sem_pct - num_hard_pct
        print(f"\nNumbers ({number_total} samples):")
        print(f"  Hard Accuracy:     {number_hard_matches}/{number_total} = {num_hard_pct:.2f}%")
        print(f"  Semantic Accuracy: {number_semantic_matches}/{number_total} = {num_sem_pct:.2f}%")
        print(f"  Semantic Gain:     +{num_gain:.2f}%")

    if operator_total > 0:
        op_hard_pct = 100.0 * operator_hard_matches / operator_total
        op_sem_pct = 100.0 * operator_semantic_matches / operator_total
        op_gain = op_sem_pct - op_hard_pct
        print(f"\nOperators ({operator_total} samples):")
        print(f"  Hard Accuracy:     {operator_hard_matches}/{operator_total} = {op_hard_pct:.2f}%")
        print(f"  Semantic Accuracy: {operator_semantic_matches}/{operator_total} = {op_sem_pct:.2f}%")
        print(f"  Semantic Gain:     +{op_gain:.2f}%")

    if other_total > 0:
        other_hard_pct = 100.0 * other_hard_matches / other_total
        other_sem_pct = 100.0 * other_semantic_matches / other_total
        other_gain = other_sem_pct - other_hard_pct
        print(f"\nOther Tokens ({other_total} samples):")
        print(f"  Hard Accuracy:     {other_hard_matches}/{other_total} = {other_hard_pct:.2f}%")
        print(f"  Semantic Accuracy: {other_semantic_matches}/{other_total} = {other_sem_pct:.2f}%")
        print(f"  Semantic Gain:     +{other_gain:.2f}%")

    # Per-iteration analysis
    print(f"\n{'='*100}")
    print("PER-ITERATION ACCURACY")
    print(f"{'='*100}")

    samples_per_iter = num_examples

    print(f"\nOverall (all token types):")
    for i in range(args.inf_latent_iterations):
        iter_hard_pct = 100.0 * iter_hard_matches[i] / samples_per_iter
        iter_sem_pct = 100.0 * iter_semantic_matches[i] / samples_per_iter
        iter_gain = iter_sem_pct - iter_hard_pct
        print(f"  Iteration {i+1}: Hard={iter_hard_pct:.2f}%  Semantic={iter_sem_pct:.2f}%  Gain=+{iter_gain:.2f}%")

    print(f"\nNumbers only:")
    for i in range(args.inf_latent_iterations):
        if iter_number_total[i] > 0:
            iter_num_hard_pct = 100.0 * iter_number_hard[i] / iter_number_total[i]
            iter_num_sem_pct = 100.0 * iter_number_semantic[i] / iter_number_total[i]
            iter_num_gain = iter_num_sem_pct - iter_num_hard_pct
            print(f"  Iteration {i+1}: Hard={iter_num_hard_pct:.2f}%  Semantic={iter_num_sem_pct:.2f}%  "
                  f"Gain=+{iter_num_gain:.2f}%  ({iter_number_total[i]} samples)")

    # Show some semantic rescue examples
    print(f"\n{'='*100}")
    print(f"SEMANTIC RESCUE EXAMPLES (first 20)")
    print(f"{'='*100}")

    for i, ex in enumerate(semantic_rescue_examples[:20]):
        print(f"\n{i+1}. Example {ex['example_idx']+1}, Iteration {ex['iteration']+1} [{ex['category']}]:")
        print(f"   Question: {ex['question']}...")
        print(f"   Probed: '{ex['probed']}' vs Target: '{ex['target']}'")

    print(f"\n{'='*100}")

    return {
        'hard_accuracy': hard_accuracy,
        'semantic_accuracy': semantic_accuracy,
        'semantic_gain': semantic_gain,
        'number_hard': num_hard_pct if number_total > 0 else 0,
        'number_semantic': num_sem_pct if number_total > 0 else 0,
        'operator_hard': op_hard_pct if operator_total > 0 else 0,
        'operator_semantic': op_sem_pct if operator_total > 0 else 0,
        'other_hard': other_hard_pct if other_total > 0 else 0,
        'other_semantic': other_sem_pct if other_total > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="zen-E/GSM8k-Aug")
    parser.add_argument("--inf_latent_iterations", type=int, default=6)
    parser.add_argument("--num_test_examples", type=int, default=100)
    parser.add_argument("--source_layer", type=int, default=10)
    parser.add_argument("--target_layer", type=int, default=11)

    args = parser.parse_args()

    print("="*100)
    print("SEMANTIC ACCURACY EVALUATION")
    print("="*100)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load CODI model
    model = load_codi_model_correctly(args)

    # Load probe
    probe = LinearProbe(768, 768).to(device)
    probe.load_state_dict(torch.load(args.probe_path, map_location=device))
    probe.eval()
    print(f"\nProbe loaded from: {args.probe_path}")

    # Load dataset
    dataset = load_dataset(args.data_name)

    # Evaluate probe
    evaluate_semantic_accuracy(model, tokenizer, probe, dataset, args)


if __name__ == "__main__":
    main()
