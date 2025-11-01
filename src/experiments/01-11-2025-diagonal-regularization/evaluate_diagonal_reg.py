#!/usr/bin/env python3
"""Evaluate diagonal regularization checkpoints"""

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
    print("Logged in to HuggingFace")

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

number_regex = re.compile(r'^\s?\d+')
RESULTS_DIR = Path('./diagonal_reg_results')


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


def run_cot_with_diagonal_mapping(model, tokenizer, training_args, question,
                                    intervention_type='baseline', W=None, b=None):
    """Run CoT with diagonal mapping intervention"""
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

    decoded_tokens = []
    embedding_layer = model.codi.get_input_embeddings()

    with torch.no_grad():
        # Precompute W if using intervention
        if intervention_type == 'diagonal_mapping' and W is not None:
            W_bf16 = W.to(device).to(torch.bfloat16)
            b_bf16 = b.to(device).to(torch.bfloat16)

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

        # Decode and intervene at BoT
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        if intervention_type == 'diagonal_mapping':
            # Apply learned linear mapping: A' = W @ E + b
            token_embedding = embedding_layer(torch.tensor([token_id], device=device))
            A_modified = token_embedding @ W_bf16.T + b_bf16
            latent_embd = A_modified.unsqueeze(1)

            logits_modified = model.codi.lm_head(A_modified)
            new_token_id = torch.argmax(logits_modified, dim=-1).item()
            new_token_str = tokenizer.decode([new_token_id])
            decoded_tokens.append({'position': 0, 'token': new_token_str, 'is_number': is_number, 'intervened': True})
        else:
            decoded_tokens.append({'position': 0, 'token': token_str, 'is_number': is_number, 'intervened': False})

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

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])
            is_number = bool(number_regex.match(token_str))

            if intervention_type == 'diagonal_mapping':
                token_embedding = embedding_layer(torch.tensor([token_id], device=device))
                A_modified = token_embedding @ W_bf16.T + b_bf16
                latent_embd = A_modified.unsqueeze(1)

                logits_modified = model.codi.lm_head(A_modified)
                new_token_id = torch.argmax(logits_modified, dim=-1).item()
                new_token_str = tokenizer.decode([new_token_id])
                decoded_tokens.append({'position': i + 1, 'token': new_token_str, 'is_number': is_number, 'intervened': True})
            else:
                decoded_tokens.append({'position': i + 1, 'token': token_str, 'is_number': is_number, 'intervened': False})

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return past_key_values, decoded_tokens


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

        return full_answer, predicted_number


def main():
    # Load model
    model, tokenizer, training_args = load_llama_model()

    # Load test datasets
    print("\n" + "="*80)
    print("Loading Test Datasets")
    print("="*80)

    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)
    print(f"Loaded {len(clean_dataset)} examples from clean dataset")

    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_test_132 = gsm8k_dataset['test'].select(range(132))
    print(f"Loaded {len(gsm8k_test_132)} examples from GSM8K test")

    # Load hyperparameter search results
    with open(RESULTS_DIR / 'hyperparameter_search.json', 'r') as f:
        search_data = json.load(f)

    # Test configurations: baseline, best reconstruction, closest to identity
    configs_to_test = [
        {'idx': None, 'name': 'baseline', 'W': None, 'b': None},
        {'idx': 0, 'name': 'no_reg', 'desc': 'No regularization (best reconstruction)'},
        {'idx': 6, 'name': 'strong_reg', 'desc': 'Strong regularization (closest to identity)'},
    ]

    # Load checkpoint weights for non-baseline configs
    for config in configs_to_test:
        if config['idx'] is not None:
            checkpoint = torch.load(RESULTS_DIR / f"checkpoint_{config['idx']}.pt")
            config['W'] = checkpoint['W']
            config['b'] = checkpoint['b']
            config['metrics'] = checkpoint['metrics']

    # Evaluate on both datasets
    for dataset_name, dataset in [('clean', clean_dataset), ('gsm8k_test', gsm8k_test_132)]:
        print(f"\n{'='*80}")
        print(f"Testing on {dataset_name} dataset")
        print(f"{'='*80}")

        dataset_results = []

        for config in configs_to_test:
            if config['idx'] is None:
                print(f"\nCondition: {config['name']}")
            else:
                print(f"\nCondition: {config['name']} - {config['desc']}")
                print(f"  Frobenius: {config['metrics']['frobenius_from_identity']:.4f}")
                print(f"  Diagonal: {config['metrics']['diag_mean']:.4f}±{config['metrics']['diag_std']:.4f}")

            condition_results = []
            intervention_type = 'baseline' if config['idx'] is None else 'diagonal_mapping'

            for ex_idx in tqdm(range(len(dataset)), desc=f"Testing {config['name']}"):
                if dataset_name == 'clean':
                    example = dataset[ex_idx]
                    question = example['question']
                    ground_truth = float(example['answer'])
                else:
                    example = dataset[ex_idx]
                    question = example['question']
                    answer_text_with_number = example['answer'].split('####')[1].strip()
                    ground_truth = float(answer_text_with_number.replace(',', ''))

                past_kv, decoded_tokens = run_cot_with_diagonal_mapping(
                    model, tokenizer, training_args, question,
                    intervention_type=intervention_type,
                    W=config['W'], b=config['b']
                )

                answer_text, predicted_answer = generate_answer(
                    model, tokenizer, training_args, past_kv
                )

                correct = (predicted_answer == ground_truth) if predicted_answer is not None else False

                result = {
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted': predicted_answer,
                    'correct': correct,
                    'answer_text': answer_text
                }
                condition_results.append(result)

            accuracy = sum(r['correct'] for r in condition_results) / len(condition_results) * 100
            print(f"  Accuracy: {accuracy:.1f}%")

            dataset_results.append({
                'condition': config['name'],
                'description': config.get('desc', 'Baseline'),
                'accuracy': accuracy,
                'results': condition_results
            })

        # Save results
        output = {
            'dataset': dataset_name,
            'n_examples': len(dataset),
            'conditions': dataset_results
        }

        with open(RESULTS_DIR / f'results_{dataset_name}.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nSaved {RESULTS_DIR / f'results_{dataset_name}.json'}")


if __name__ == '__main__':
    main()
