#!/usr/bin/env python3
"""
Test if CODI-LLAMA can be batched with left-padding without accuracy loss.

Compares accuracy between:
1. Sequential processing (batch_size=1) - baseline
2. Batched processing (batch_size=8, 16) with left-padding
"""

import torch
import sys
import os
import json
import re
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import numpy as np

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

number_regex = re.compile(r'^\s?\d+')


def load_llama_model():
    """Load CODI-LLAMA model"""
    print("="*80)
    print("Loading CODI-LLAMA")
    print("="*80)

    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/.cache/huggingface/hub/models--zen-E--CODI-llama3.2-1b-Instruct/snapshots/b2c88ba224b06b12b52ef39b87f794b98a6eb1c8",
        full_precision=True,
        token=None
    )

    llama_training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=2048,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    llama_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=llama_model_args.lora_r,
        lora_alpha=llama_model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        init_lora_weights=True,
    )

    llama_model = CODI(llama_model_args, llama_training_args, llama_lora_config)
    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    # Set padding to left for batching
    llama_tokenizer.padding_side = 'left'
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    return llama_model, llama_tokenizer, llama_training_args


def run_single_inference(model, tokenizer, training_args, question):
    """Run inference on a single question (batch_size=1)"""
    batch_size = 1

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer([question], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    with torch.no_grad():
        # Initial encoding
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # CoT iterations
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1:, :]

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

        # Generate answer
        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]],
                                       dtype=torch.long).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor)
        output = eot_emb

        pred_tokens = []
        for step in range(128):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values

            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
            next_token_id = torch.argmax(logits, dim=-1).item()

            pred_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            current_token_str = tokenizer.decode([next_token_id])
            if number_regex.match(current_token_str.strip()):
                break

            if step >= 49:
                break

            output = model.get_embd(model.codi, model.model_name)(
                torch.tensor([[next_token_id]], device=device)
            )

    full_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    text = full_answer.replace(',', '')
    numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    predicted_number = float(numbers[-1]) if numbers else None

    return predicted_number, full_answer


def run_batched_inference(model, tokenizer, training_args, questions, batch_size=8):
    """Run inference on multiple questions in a batch with left-padding"""

    # Tokenize with left-padding
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    actual_batch_size = len(questions)

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(actual_batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(actual_batch_size, 2).to(device)

    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    with torch.no_grad():
        # Initial encoding
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # CoT iterations
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1:, :]

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    # Generate answers for each item in batch
    results = []
    for batch_idx in range(actual_batch_size):
        # Get past_key_values for this batch item
        batch_past_kv = tuple(
            tuple(kv[:, batch_idx:batch_idx+1, :, :] for kv in layer)
            for layer in past_key_values
        )

        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]],
                                       dtype=torch.long).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor)
        output = eot_emb

        pred_tokens = []
        for step in range(128):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=batch_past_kv
            )
            batch_past_kv = out.past_key_values

            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
            next_token_id = torch.argmax(logits, dim=-1).item()

            pred_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            current_token_str = tokenizer.decode([next_token_id])
            if number_regex.match(current_token_str.strip()):
                break

            if step >= 49:
                break

            output = model.get_embd(model.codi, model.model_name)(
                torch.tensor([[next_token_id]], device=device)
            )

        full_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        text = full_answer.replace(',', '')
        numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
        predicted_number = float(numbers[-1]) if numbers else None

        results.append((predicted_number, full_answer))

    return results


def main():
    # Load model
    model, tokenizer, training_args = load_llama_model()

    # Load test examples
    print("\n" + "="*80)
    print("Loading GSM8K Test Examples")
    print("="*80)

    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_examples = gsm8k_dataset['test'].select(range(100))  # First 100 examples

    questions = []
    ground_truths = []
    for example in test_examples:
        questions.append(example['question'])
        answer_num = float(example['answer'].split('####')[1].strip().replace(',', ''))
        ground_truths.append(answer_num)

    # Test 1: Sequential (baseline)
    print("\n" + "="*80)
    print("Test 1: Sequential Processing (batch_size=1)")
    print("="*80)

    sequential_results = []
    for i, question in enumerate(tqdm(questions, desc="Sequential")):
        pred, answer_text = run_single_inference(model, tokenizer, training_args, question)
        sequential_results.append(pred)

    sequential_correct = sum(1 for pred, gt in zip(sequential_results, ground_truths)
                            if pred is not None and pred == gt)
    sequential_accuracy = sequential_correct / len(questions) * 100

    print(f"Sequential Accuracy: {sequential_accuracy:.2f}% ({sequential_correct}/{len(questions)})")

    # Test 2: Batched with batch_size=8
    print("\n" + "="*80)
    print("Test 2: Batched Processing (batch_size=8)")
    print("="*80)

    batch_size_8_results = []
    for i in tqdm(range(0, len(questions), 8), desc="Batch-8"):
        batch_questions = questions[i:i+8]
        batch_results = run_batched_inference(model, tokenizer, training_args, batch_questions, batch_size=8)
        batch_size_8_results.extend([r[0] for r in batch_results])

    batch8_correct = sum(1 for pred, gt in zip(batch_size_8_results, ground_truths)
                        if pred is not None and pred == gt)
    batch8_accuracy = batch8_correct / len(questions) * 100

    print(f"Batch-8 Accuracy: {batch8_accuracy:.2f}% ({batch8_correct}/{len(questions)})")
    print(f"Accuracy Drop: {sequential_accuracy - batch8_accuracy:.2f}%")

    # Test 3: Batched with batch_size=16
    print("\n" + "="*80)
    print("Test 3: Batched Processing (batch_size=16)")
    print("="*80)

    batch_size_16_results = []
    for i in tqdm(range(0, len(questions), 16), desc="Batch-16"):
        batch_questions = questions[i:i+16]
        batch_results = run_batched_inference(model, tokenizer, training_args, batch_questions, batch_size=16)
        batch_size_16_results.extend([r[0] for r in batch_results])

    batch16_correct = sum(1 for pred, gt in zip(batch_size_16_results, ground_truths)
                         if pred is not None and pred == gt)
    batch16_accuracy = batch16_correct / len(questions) * 100

    print(f"Batch-16 Accuracy: {batch16_accuracy:.2f}% ({batch16_correct}/{len(questions)})")
    print(f"Accuracy Drop: {sequential_accuracy - batch16_accuracy:.2f}%")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Sequential (baseline):  {sequential_accuracy:.2f}%")
    print(f"Batch-8:                {batch8_accuracy:.2f}% (drop: {sequential_accuracy - batch8_accuracy:+.2f}%)")
    print(f"Batch-16:               {batch16_accuracy:.2f}% (drop: {sequential_accuracy - batch16_accuracy:+.2f}%)")

    # Save results
    results = {
        'n_examples': len(questions),
        'sequential': {
            'accuracy': sequential_accuracy,
            'correct': sequential_correct,
            'predictions': sequential_results
        },
        'batch_8': {
            'accuracy': batch8_accuracy,
            'correct': batch8_correct,
            'drop': sequential_accuracy - batch8_accuracy,
            'predictions': batch_size_8_results
        },
        'batch_16': {
            'accuracy': batch16_accuracy,
            'correct': batch16_correct,
            'drop': sequential_accuracy - batch16_accuracy,
            'predictions': batch_size_16_results
        },
        'ground_truths': ground_truths
    }

    output_path = Path('./batching_test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Verdict
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)
    if batch8_accuracy >= sequential_accuracy - 1:
        print("✅ Batching with batch_size=8 is VIABLE (< 1% accuracy drop)")
        print("   Recommend: Use batching for Phase 1")
    elif batch8_accuracy >= sequential_accuracy - 5:
        print("⚠️  Batching has small accuracy drop (1-5%)")
        print("   Decision: Depends on speed vs accuracy tradeoff")
    else:
        print("❌ Batching has significant accuracy drop (> 5%)")
        print("   Recommend: Use sequential processing")


if __name__ == '__main__':
    main()
