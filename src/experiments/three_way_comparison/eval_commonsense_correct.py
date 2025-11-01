#!/usr/bin/env python3
"""
CommonsenseQA Evaluation - MATCHING ORIGINAL CODI EVALUATION

This script matches the exact evaluation method from codi/test.py to reproduce
the documented 71.33% accuracy.

Key differences from our extract_activations.py:
1. Input format: Raw question text ONLY (no "Reasoning:" suffix)
2. Answer extraction: Takes FIRST letter after "The answer is:", defaults to "C"
3. Generation: Uses do_sample=True with temperature=0.1
4. Token flow: BOT → CT tokens × 6 → EOT → answer generation
"""
import sys
sys.path.insert(0, '/home/paperspace/dev/CoT_Exploration/codi')

import torch
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

def extract_answer_number(sentence: str) -> str:
    """
    Extract answer from CommonsenseQA response.

    MATCHES ORIGINAL CODI: codi/test.py lines 334-337
    """
    pred = sentence.split("The answer is:")[-1].strip()

    # Check if first character is valid
    if not pred or pred[0] not in "ABCDE":
        return "C"  # Default to C (like original)

    return pred[0]

def compute_accuracy(gold: list, pred: list):
    """Compute accuracy - MATCHES ORIGINAL"""
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1
    return acc / len(gold)

def main():
    print("\n" + "="*80)
    print("CommonsenseQA CODI Evaluation (Matching Original)")
    print("="*80)

    # Load model
    print("\nLoading CommonsenseQA CODI model...")

    model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
        full_precision=True,
        lora_r=128,
        lora_alpha=32,
        lora_init=True,  # For LoRA initialization
    )

    training_args = TrainingArguments(
        output_dir="/tmp/eval",
        num_latent=6,
        use_prj=True,
        prj_dim=2048,
        use_lora=True,  # This is correct for TrainingArguments
        model_max_length=512,
        remove_eos=True,
        greedy=True,  # Use greedy decoding
        bf16=True,
        inf_latent_iterations=6,
    )

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    # Load model
    model = CODI(model_args, training_args, lora_config)
    model_path = "/home/paperspace/codi_ckpt/llama_commonsense/gsm8k_llama1b_latent_baseline/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11"
    checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda')
    model.to(torch.bfloat16)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    # Load dataset
    print("Loading CommonsenseQA validation set...")
    dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
    test_set = dataset['validation']

    print(f"Total examples: {len(test_set)}")

    # Format questions - RAW QUESTIONS ONLY (like original)
    questions = [example['question'].strip().replace('  ', ' ') for example in test_set]
    answers = [example['answer'] for example in test_set]

    # Batch processing
    batch_size = 8
    eval_steps = (len(questions) + batch_size - 1) // batch_size

    print(f"\nEvaluating with batch_size={batch_size}, steps={eval_steps}")

    predictions = []

    for step in tqdm(range(eval_steps), desc="Evaluating"):
        start_idx = step * batch_size
        end_idx = min(start_idx + batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        # Tokenize
        batch = tokenizer(
            batch_questions,
            return_tensors="pt",
            padding="longest",
        )

        # Add BOT token (like original)
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

        batch = batch.to('cuda')
        current_batch_size = batch["input_ids"].size(0)

        with torch.no_grad():
            # Encode question
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=None,
                attention_mask=batch["attention_mask"]
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Generate CT tokens (6 iterations)
            for i in range(6):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Add EOT token
            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
            ).unsqueeze(0)
            eot_emb = eot_emb.expand(current_batch_size, -1, -1)

            output = eot_emb

            # Generate answer tokens
            pred_tokens = [[] for _ in range(current_batch_size)]
            finished = torch.zeros(current_batch_size, dtype=torch.bool, device="cuda")

            for i in range(256):  # max_new_tokens=256
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

                # Greedy decoding (since greedy=True)
                next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)

                # Handle EOS for each sequence
                for b in range(current_batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                # Break if all sequences finished
                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1)

        # Decode and extract answers
        for pred_token in pred_tokens:
            decoded_pred = tokenizer.decode(pred_token, skip_special_tokens=True)
            predicted_answer = extract_answer_number(decoded_pred)
            predictions.append(predicted_answer)

    # Compute accuracy
    accuracy = compute_accuracy(answers, predictions)
    n_correct = sum(p == g for p, g in zip(predictions, answers))

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total examples: {len(answers)}")
    print(f"Correct: {n_correct}/{len(answers)}")
    print(f"Accuracy: {100*accuracy:.2f}%")
    print(f"{'='*80}")

    # Save results
    results_path = Path("results/commonsense_eval_original_method.json")
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'n_correct': n_correct,
            'n_total': len(answers),
            'predictions': predictions,
            'ground_truth': answers,
            'method': 'original_codi_evaluation'
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")

    return accuracy

if __name__ == '__main__':
    main()
