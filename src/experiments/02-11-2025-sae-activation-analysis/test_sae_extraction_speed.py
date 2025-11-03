#!/usr/bin/env python3
"""
Measure precise timing for SAE feature extraction (encoding + CoT only, no decoding).
"""

import torch
import sys
import os
import time
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

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


def load_llama_model():
    """Load CODI-LLAMA model"""
    print("="*80)
    print("Loading CODI-LLAMA")
    print("="*80)
    load_start = time.time()

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

    load_time = time.time() - load_start
    print(f"Model loading time: {load_time:.2f} seconds")

    return llama_model, llama_tokenizer, llama_training_args


def extract_cot_activations(model, tokenizer, training_args, question):
    """
    Extract CoT activations from multiple layers (just encoding + CoT, NO decoding).
    This is what we need for SAE feature extraction.
    """
    batch_size = 1

    # Layer indices to extract
    layer_indices = {'early': 4, 'middle': 8, 'late': 14}

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer([question], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    hidden_states_list = []

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

        # CoT iterations - extract from multiple layers
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Extract hidden states from specified layers
            position_data = {'position': i, 'layers': {}}

            for layer_name, layer_idx in layer_indices.items():
                hidden_state = outputs.hidden_states[layer_idx][:, -1, :].squeeze().cpu()
                position_data['layers'][layer_name] = {
                    'layer_idx': layer_idx,
                    'hidden_state': hidden_state
                }

            hidden_states_list.append(position_data)

            # Continue with projection for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1:, :]
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return hidden_states_list


def main():
    # Load model
    model, tokenizer, training_args = load_llama_model()

    # Load dataset
    print("\n" + "="*80)
    print("Loading GSM8K Train Set")
    print("="*80)
    dataset_start = time.time()

    gsm8k_dataset = load_dataset("gsm8k", "main")
    full_train = gsm8k_dataset['train']
    n_total = len(full_train)

    dataset_time = time.time() - dataset_start
    print(f"Dataset loading time: {dataset_time:.2f} seconds")
    print(f"Total train examples: {n_total}")

    # Test on small sample first
    print("\n" + "="*80)
    print("Timing Test: 100 examples")
    print("="*80)

    test_examples = full_train.select(range(100))
    questions = [ex['question'] for ex in test_examples]

    extraction_start = time.time()
    for i, question in enumerate(questions):
        if i % 10 == 0:
            elapsed = time.time() - extraction_start
            if i > 0:
                rate = i / elapsed
                print(f"  Processed {i}/100 examples ({elapsed:.1f}s, {rate:.2f} ex/s)")

        hidden_states = extract_cot_activations(model, tokenizer, training_args, question)

    extraction_time = time.time() - extraction_start
    rate = 100 / extraction_time

    print("\n" + "="*80)
    print("TIMING RESULTS")
    print("="*80)
    print(f"Model loading time:      {load_time:.2f} seconds")
    print(f"Dataset loading time:    {dataset_time:.2f} seconds")
    print(f"Extraction time (100 ex): {extraction_time:.2f} seconds")
    print(f"Extraction rate:         {rate:.2f} examples/second")
    print(f"Time per example:        {extraction_time/100:.3f} seconds")

    print("\n" + "="*80)
    print("FULL DATASET PROJECTION")
    print("="*80)
    print(f"Total examples:          {n_total}")
    print(f"Estimated extraction:    {n_total/rate:.1f} seconds = {n_total/rate/60:.1f} minutes")
    print(f"Total estimated time:    {(load_time + dataset_time + n_total/rate)/60:.1f} minutes")


if __name__ == '__main__':
    # Need to define load_time at module level for final report
    load_time = 0
    main()
