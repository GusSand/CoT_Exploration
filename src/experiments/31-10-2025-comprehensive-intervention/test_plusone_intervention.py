#!/usr/bin/env python3
"""
Test plus-one discretization intervention.
Compare: baseline, discretize (numbers), discretize_plusone (numbers)
On: Clean, GSM8K Train, GSM8K Test datasets
"""
import torch
import sys
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("[OK] Logged in to HuggingFace")

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Number detection regex
number_regex = re.compile(r'^\s?\d+')

# Results directory
RESULTS_DIR = Path('./plusone_intervention_results')
RESULTS_DIR.mkdir(exist_ok=True)

def load_llama_model():
    """Load CODI-LLaMA model"""
    print("="*80)
    print("Loading CODI-LLaMA from Local Checkpoint")
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

    return llama_model, llama_tokenizer, llama_training_args


def get_plusone_token_id(tokenizer, token_id):
    """Get token ID for number+1, preserving format"""
    token_str = tokenizer.decode([token_id])

    match = re.search(r'-?\d+', token_str)
    if not match:
        return None

    num_str = match.group()
    try:
        num = int(num_str)
    except:
        return None

    next_num = num + 1
    new_token_str = token_str.replace(num_str, str(next_num))

    encoded = tokenizer.encode(new_token_str, add_special_tokens=False)
    if len(encoded) == 1:
        return encoded[0]
    elif len(encoded) > 1:
        return encoded[0]
    else:
        return None


def run_cot_with_intervention(model, tokenizer, training_args, question, intervention_type):
    """Run CoT with specified intervention (baseline, discretize, or discretize_plusone)"""
    batch_size = 1
    questions = [question]

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer(questions, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    embedding_layer = model.codi.model.model.embed_tokens

    with torch.no_grad():
        # Initial encoding
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        # Decode BoT
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id]).strip()
        is_number = bool(number_regex.match(token_str))

        # Apply intervention at position 0 (BoT)
        if intervention_type != 'baseline' and is_number:
            A = latent_embd.squeeze(1)

            if intervention_type == 'discretize':
                # Regular discretization
                predicted_embd = embedding_layer(torch.tensor([token_id], device=device))
                A_norm = torch.norm(A, dim=-1, keepdim=True)
                E_norm = torch.norm(predicted_embd, dim=-1, keepdim=True)
                latent_embd = (predicted_embd * (A_norm / (E_norm + 1e-8))).unsqueeze(1)

            elif intervention_type == 'discretize_plusone':
                # Plus-one discretization
                plusone_token_id = get_plusone_token_id(tokenizer, token_id)
                if plusone_token_id is not None:
                    plusone_embd = embedding_layer(torch.tensor([plusone_token_id], device=device))
                    A_norm = torch.norm(A, dim=-1, keepdim=True)
                    E_norm = torch.norm(plusone_embd, dim=-1, keepdim=True)
                    latent_embd = (plusone_embd * (A_norm / (E_norm + 1e-8))).unsqueeze(1)

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

            # Decode
            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id]).strip()
            is_number = bool(number_regex.match(token_str))

            # Apply intervention
            if intervention_type != 'baseline' and is_number:
                A = latent_embd.squeeze(1)

                if intervention_type == 'discretize':
                    predicted_embd = embedding_layer(torch.tensor([token_id], device=device))
                    A_norm = torch.norm(A, dim=-1, keepdim=True)
                    E_norm = torch.norm(predicted_embd, dim=-1, keepdim=True)
                    latent_embd = (predicted_embd * (A_norm / (E_norm + 1e-8))).unsqueeze(1)

                elif intervention_type == 'discretize_plusone':
                    plusone_token_id = get_plusone_token_id(tokenizer, token_id)
                    if plusone_token_id is not None:
                        plusone_embd = embedding_layer(torch.tensor([plusone_token_id], device=device))
                        A_norm = torch.norm(A, dim=-1, keepdim=True)
                        E_norm = torch.norm(plusone_embd, dim=-1, keepdim=True)
                        latent_embd = (plusone_embd * (A_norm / (E_norm + 1e-8))).unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return past_key_values


def generate_answer(model, tokenizer, training_args, past_key_values, max_length=128):
    """Generate final answer"""
    batch_size = 1

    with torch.no_grad():
        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).expand(batch_size, 1, 1).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]],
                                       dtype=torch.long).expand(batch_size, 1, 2).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor).squeeze(1)
        output = eot_emb

        pred_tokens = []

        for step in range(max_length):
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
            next_token_id = torch.argmax(logits, dim=-1).item()

            pred_tokens.append(next_token_id)

            current_token_str = tokenizer.decode([next_token_id])

            if next_token_id == tokenizer.eos_token_id:
                break

            if number_regex.match(current_token_str.strip()):
                break

            if step >= 49:
                break

            output = model.get_embd(model.codi, model.model_name)(
                torch.tensor([[next_token_id]], device=device)
            )

        full_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)

        # Extract numerical answer
        text = full_answer.replace(',', '')
        numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
        predicted_number = float(numbers[-1]) if numbers else None

        return predicted_number


def test_intervention(model, tokenizer, training_args, dataset, dataset_name, intervention_type):
    """Test intervention on dataset"""
    correct = 0
    total = len(dataset)

    for ex_idx in tqdm(range(total), desc=f"{dataset_name} - {intervention_type}"):
        if dataset_name == 'clean':
            question = dataset[ex_idx]['question']
            ground_truth = float(dataset[ex_idx]['answer'])
        else:  # GSM8K
            question = dataset[ex_idx]['question']
            answer_text_with_number = dataset[ex_idx]['answer'].split('####')[1].strip()
            ground_truth = float(answer_text_with_number.replace(',', ''))

        past_kv = run_cot_with_intervention(model, tokenizer, training_args, question, intervention_type)
        predicted_answer = generate_answer(model, tokenizer, training_args, past_kv)

        if predicted_answer is not None and predicted_answer == ground_truth:
            correct += 1

    accuracy = (correct / total) * 100
    return accuracy, correct, total


def main():
    model, tokenizer, training_args = load_llama_model()

    # Load datasets
    print("\nLoading datasets...")
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)
    print(f"[OK] Loaded {len(clean_dataset)} examples from clean dataset")

    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train_132 = gsm8k_dataset['train'].select(range(132))
    gsm8k_test_132 = gsm8k_dataset['test'].select(range(132))
    print(f"[OK] Loaded {len(gsm8k_train_132)} examples from GSM8K train")
    print(f"[OK] Loaded {len(gsm8k_test_132)} examples from GSM8K test")

    # Define interventions to test
    interventions = ['baseline', 'discretize', 'discretize_plusone']
    datasets_to_test = [
        ('clean', clean_dataset),
        ('gsm8k_train', gsm8k_train_132),
        ('gsm8k_test', gsm8k_test_132)
    ]

    results = {}

    # Test each intervention on each dataset
    for dataset_name, dataset in datasets_to_test:
        results[dataset_name] = {}

        for intervention_type in interventions:
            print(f"\n{'='*80}")
            print(f"Testing {intervention_type} on {dataset_name}")
            print(f"{'='*80}")

            accuracy, correct, total = test_intervention(
                model, tokenizer, training_args, dataset, dataset_name, intervention_type
            )

            results[dataset_name][intervention_type] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }

            print(f"Result: {accuracy:.1f}% ({correct}/{total})")

    # Save results
    results_file = RESULTS_DIR / 'plusone_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results saved to {results_file}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Baseline vs Discretize vs Discretize+1")
    print("="*80)
    print(f"{'Dataset':15s} {'Baseline':>10s} {'Discretize':>10s} {'Discretize+1':>12s}")
    print("-"*80)
    for dataset_name in ['clean', 'gsm8k_train', 'gsm8k_test']:
        print(f"{dataset_name:15s} "
              f"{results[dataset_name]['baseline']['accuracy']:9.1f}% "
              f"{results[dataset_name]['discretize']['accuracy']:9.1f}% "
              f"{results[dataset_name]['discretize_plusone']['accuracy']:11.1f}%")


if __name__ == "__main__":
    main()
