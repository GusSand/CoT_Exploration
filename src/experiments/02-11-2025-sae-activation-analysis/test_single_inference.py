#!/usr/bin/env python3
"""
Simple test to debug CODI-LLAMA single inference
"""

import torch
import sys
import os
import re
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
    llama_tokenizer.padding_side = 'left'
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    return llama_model, llama_tokenizer, llama_training_args


def run_single_inference(model, tokenizer, training_args, question):
    """Run inference on a single question"""
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

    print(f"\nInput shape: {inputs['input_ids'].shape}")
    print(f"Bot ID: {model.bot_id}, EoT ID: {model.eot_id}")

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

        print(f"\nAfter {training_args.inf_latent_iterations} CoT iterations")

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
                print(f"  Step {step}: EOS")
                break

            current_token_str = tokenizer.decode([next_token_id])
            print(f"  Step {step}: token={next_token_id} '{current_token_str}'")

            if number_regex.match(current_token_str.strip()):
                print(f"  -> Matched number regex, stopping")
                break

            if step >= 49:
                print(f"  -> Max steps reached")
                break

            output = model.get_embd(model.codi, model.model_name)(
                torch.tensor([[next_token_id]], device=device)
            )

    full_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
    text = full_answer.replace(',', '')
    numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    predicted_number = float(numbers[-1]) if numbers else None

    print(f"\nFull answer: '{full_answer}'")
    print(f"Extracted numbers: {numbers}")
    print(f"Predicted: {predicted_number}")

    return predicted_number, full_answer


def main():
    # Load model
    model, tokenizer, training_args = load_llama_model()

    # Load first 3 test examples
    print("\n" + "="*80)
    print("Loading GSM8K Test Examples")
    print("="*80)

    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_examples = gsm8k_dataset['test'].select(range(3))

    for idx, example in enumerate(test_examples):
        question = example['question']
        answer_text = example['answer']
        ground_truth = float(answer_text.split('####')[1].strip().replace(',', ''))

        print("\n" + "="*80)
        print(f"Example {idx + 1}")
        print("="*80)
        print(f"Q: {question}")
        print(f"Ground truth: {ground_truth}")

        pred, answer_text = run_single_inference(model, tokenizer, training_args, question)

        is_correct = pred is not None and pred == ground_truth
        print(f"Correct: {is_correct}")


if __name__ == '__main__':
    main()
