#!/usr/bin/env python3
"""
Learned Mapping with Residual + Low-Rank + Prediction-Preserving Loss

Combines:
- Residual learning: Y = X + alpha * (U @ V^T @ X + b)
- Low-rank factorization: reduces parameters
- KL divergence loss: preserves LM head predictions
"""

import torch
import sys
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset
from datetime import datetime
from collections import defaultdict
import torch.nn.functional as F

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("✓ Logged in to HuggingFace")

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Number detection regex
number_regex = re.compile(r'^\s?\d+')

# Results directory
RESULTS_DIR = Path('./learned_mapping_residual_results')
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


def create_balanced_split(training_pairs, val_samples_per_token=5):
    """Create balanced train/validation split by token string"""
    print(f"\nCreating balanced train/validation split...")

    # Group by token string
    token_groups = defaultdict(list)
    for pair in training_pairs:
        token_groups[pair['token_string']].append(pair)

    print(f"Found {len(token_groups)} unique token strings")

    # Create balanced validation set
    val_pairs = []
    train_pairs = []

    for token_str, pairs in token_groups.items():
        np.random.shuffle(pairs)
        n_val = min(val_samples_per_token, len(pairs))
        val_pairs.extend(pairs[:n_val])
        train_pairs.extend(pairs[n_val:])

    print(f"✓ Train set: {len(train_pairs)} pairs")
    print(f"✓ Validation set: {len(val_pairs)} pairs")

    return train_pairs, val_pairs


def train_residual_lowrank_mapping(train_pairs, val_pairs, hidden_dim, lm_head,
                                    rank=128, lr=0.001, epochs=50, alpha_init=0.1,
                                    downsample_factor=1.0, device='cuda'):
    """
    Train residual low-rank mapping with prediction-preserving loss.

    Y = X + alpha * (U @ V^T @ X + b)

    Loss = KL(LM_head(Y), LM_head(Y_true))
    """
    # Downsample if requested
    if downsample_factor < 1.0:
        n_train = int(len(train_pairs) * downsample_factor)
        train_pairs = np.random.choice(train_pairs, n_train, replace=False).tolist()
        print(f"Downsampled to {n_train} training pairs")

    # Prepare data
    X_train = torch.stack([pair['embedding'] for pair in train_pairs]).to(device).float()
    Y_train = torch.stack([pair['activation'] for pair in train_pairs]).to(device).float()

    X_val = torch.stack([pair['embedding'] for pair in val_pairs]).to(device).float()
    Y_val = torch.stack([pair['activation'] for pair in val_pairs]).to(device).float()

    # Initialize low-rank factorization: W = U @ V^T
    U = torch.randn(hidden_dim, rank, device=device, dtype=torch.float32) * 0.01
    V = torch.randn(hidden_dim, rank, device=device, dtype=torch.float32) * 0.01
    b = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
    alpha = torch.tensor(alpha_init, device=device, dtype=torch.float32)

    U.requires_grad = True
    V.requires_grad = True
    b.requires_grad = True
    alpha.requires_grad = True

    optimizer = torch.optim.Adam([U, V, b, alpha], lr=lr)

    best_val_loss = float('inf')
    best_U = None
    best_V = None
    best_b = None
    best_alpha = None

    # Training loop with smaller batches for efficiency
    batch_size = 512
    n_batches = (len(X_train) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))

            X_batch = X_train[start_idx:end_idx]
            Y_batch = Y_train[start_idx:end_idx]

            optimizer.zero_grad()

            # Forward pass: Y_pred = X + alpha * (U @ V^T @ X + b)
            W = U @ V.T  # [hidden, hidden]
            residual = X_batch @ W.T + b  # [batch, hidden]
            Y_pred = X_batch + alpha * residual

            # Compute logits through LM head
            with torch.amp.autocast('cuda'):
                Y_pred_bf16 = Y_pred.to(torch.bfloat16)
                Y_batch_bf16 = Y_batch.to(torch.bfloat16)

                logits_pred = lm_head(Y_pred_bf16)  # [batch, vocab]
                logits_true = lm_head(Y_batch_bf16)  # [batch, vocab]

            # KL divergence loss (prediction-preserving)
            log_probs_pred = F.log_softmax(logits_pred.float(), dim=-1)
            probs_true = F.softmax(logits_true.float(), dim=-1)
            kl_loss = F.kl_div(log_probs_pred, probs_true, reduction='batchmean')

            # Backward
            kl_loss.backward()
            optimizer.step()

            epoch_loss += kl_loss.item()

        avg_train_loss = epoch_loss / n_batches

        # Validation
        with torch.no_grad():
            W = U @ V.T
            residual_val = X_val @ W.T + b
            Y_val_pred = X_val + alpha * residual_val

            with torch.amp.autocast('cuda'):
                Y_val_pred_bf16 = Y_val_pred.to(torch.bfloat16)
                Y_val_bf16 = Y_val.to(torch.bfloat16)

                logits_val_pred = lm_head(Y_val_pred_bf16)
                logits_val_true = lm_head(Y_val_bf16)

            log_probs_val_pred = F.log_softmax(logits_val_pred.float(), dim=-1)
            probs_val_true = F.softmax(logits_val_true.float(), dim=-1)
            val_loss = F.kl_div(log_probs_val_pred, probs_val_true, reduction='batchmean').item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_U = U.detach().clone()
                best_V = V.detach().clone()
                best_b = b.detach().clone()
                best_alpha = alpha.detach().clone()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}, alpha={alpha.item():.4f}")

    print(f"✓ Best validation loss: {best_val_loss:.6f}, alpha={best_alpha.item():.4f}")

    return best_U, best_V, best_b, best_alpha, best_val_loss


def hyperparameter_search(train_pairs, val_pairs, hidden_dim, lm_head):
    """Quick hyperparameter search"""
    print("\n" + "="*80)
    print("Hyperparameter Search (Reduced Space for Speed)")
    print("="*80)

    # Reduced search space for 1-hour target
    configs = [
        # (rank, lr, epochs, alpha_init, downsample)
        (64, 0.001, 50, 0.1, 1.0),    # Low rank, full data
        (128, 0.001, 50, 0.1, 1.0),   # Medium rank, full data
        (256, 0.001, 50, 0.1, 1.0),   # High rank, full data
        (128, 0.01, 50, 0.1, 0.5),    # Higher LR, less data
        (128, 0.0001, 50, 0.1, 0.5),  # Lower LR, less data
        (128, 0.001, 100, 0.05, 1.0), # More epochs, smaller alpha
    ]

    best_val_loss = float('inf')
    best_U = None
    best_V = None
    best_b = None
    best_alpha = None
    best_config = None

    search_results = []

    for rank, lr, epochs, alpha_init, downsample in configs:
        print(f"\nTrying: rank={rank}, lr={lr}, epochs={epochs}, alpha={alpha_init}, downsample={downsample}")

        U, V, b, alpha, val_loss = train_residual_lowrank_mapping(
            train_pairs, val_pairs, hidden_dim, lm_head,
            rank=rank, lr=lr, epochs=epochs, alpha_init=alpha_init,
            downsample_factor=downsample, device=device
        )

        search_results.append({
            'rank': rank,
            'lr': lr,
            'epochs': epochs,
            'alpha_init': alpha_init,
            'downsample': downsample,
            'val_loss': val_loss,
            'final_alpha': alpha.item()
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_U = U
            best_V = V
            best_b = b
            best_alpha = alpha
            best_config = {
                'rank': rank,
                'lr': lr,
                'epochs': epochs,
                'alpha_init': alpha_init,
                'downsample': downsample,
                'val_loss': val_loss,
                'final_alpha': alpha.item()
            }
            print(f"  ★ New best! val_loss={val_loss:.6f}, alpha={alpha.item():.4f}")

    print("\n" + "="*80)
    print("Search Results")
    print("="*80)
    print(f"Best config: {best_config}")

    # Save search results
    with open(RESULTS_DIR / 'hyperparameter_search.json', 'w') as f:
        json.dump({
            'search_results': search_results,
            'best_config': best_config
        }, f, indent=2)

    return best_U, best_V, best_b, best_alpha, best_config


def run_cot_with_residual_mapping(model, tokenizer, training_args, question,
                                  intervention_type, U=None, V=None, b=None, alpha=None):
    """Run CoT with residual low-rank mapping intervention"""
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
        if intervention_type == 'residual_mapping':
            W = (U @ V.T).to(torch.bfloat16)  # [hidden, hidden]
            b_bf16 = b.to(torch.bfloat16)
            alpha_bf16 = alpha.to(torch.bfloat16)

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

        if intervention_type == 'residual_mapping':
            token_embedding = embedding_layer(torch.tensor([token_id], device=device))
            residual = token_embedding @ W.T + b_bf16
            A_modified = token_embedding + alpha_bf16 * residual
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

            if intervention_type == 'residual_mapping':
                token_embedding = embedding_layer(torch.tensor([token_id], device=device))
                residual = token_embedding @ W.T + b_bf16
                A_modified = token_embedding + alpha_bf16 * residual
                latent_embd = A_modified.unsqueeze(1)

                logits_modified = model.codi.lm_head(A_modified)
                new_token_id = torch.argmax(logits_modified, dim=-1).item()
                new_token_str = tokenizer.decode([new_token_id])
                decoded_tokens.append({'position': i+1, 'token': new_token_str, 'is_number': is_number, 'intervened': True})
            else:
                decoded_tokens.append({'position': i+1, 'token': token_str, 'is_number': is_number, 'intervened': False})

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
    torch.manual_seed(42)
    np.random.seed(42)

    # Load model
    model, tokenizer, training_args = load_llama_model()
    hidden_dim = model.dim
    lm_head = model.codi.lm_head

    # Load existing training pairs
    print("\n" + "="*80)
    print("Loading Training Data")
    print("="*80)

    training_pairs_path = Path('../01-11-2025-learned-mapping/learned_mapping_results/training_pairs.pt')
    checkpoint = torch.load(training_pairs_path)
    training_pairs = checkpoint['training_pairs']
    print(f"✓ Loaded {len(training_pairs)} training pairs")

    # Create balanced split
    print("\n" + "="*80)
    print("Create Balanced Split")
    print("="*80)
    train_pairs, val_pairs = create_balanced_split(training_pairs, val_samples_per_token=5)

    # Hyperparameter search & training
    U, V, b, alpha, best_config = hyperparameter_search(train_pairs, val_pairs, hidden_dim, lm_head)

    # Save learned mapping
    print("\nSaving learned mapping...")
    torch.save({
        'U': U.cpu(),
        'V': V.cpu(),
        'b': b.cpu(),
        'alpha': alpha.cpu(),
        'config': best_config,
        'hidden_dim': hidden_dim
    }, RESULTS_DIR / 'learned_mapping.pt')
    print(f"✓ Saved to {RESULTS_DIR / 'learned_mapping.pt'}")

    # Convert to bfloat16 for inference
    U = U.to(torch.bfloat16)
    V = V.to(torch.bfloat16)
    b = b.to(torch.bfloat16)
    alpha = alpha.to(torch.bfloat16)

    # Evaluation
    print("\n" + "="*80)
    print("Evaluation")
    print("="*80)

    # Load test datasets
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)
    print(f"✓ Loaded {len(clean_dataset)} examples from clean dataset")

    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_test_132 = gsm8k_dataset['test'].select(range(132))
    print(f"✓ Loaded {len(gsm8k_test_132)} examples from GSM8K test")

    conditions = [
        ('baseline', None, None, None, None),
        ('residual_mapping', U, V, b, alpha)
    ]

    # Evaluate on both datasets
    for dataset_name, dataset in [('clean', clean_dataset), ('gsm8k_test', gsm8k_test_132)]:
        print(f"\n{'='*80}")
        print(f"Testing on {dataset_name} dataset")
        print(f"{'='*80}")

        dataset_results = []

        for condition_name, U_cond, V_cond, b_cond, alpha_cond in conditions:
            print(f"\nCondition: {condition_name}")

            condition_results = []

            for ex_idx in tqdm(range(len(dataset)), desc=f"Testing {condition_name}"):
                if dataset_name == 'clean':
                    example = dataset[ex_idx]
                    question = example['question']
                    ground_truth = float(example['answer'])
                else:
                    example = dataset[ex_idx]
                    question = example['question']
                    answer_text_with_number = example['answer'].split('####')[1].strip()
                    ground_truth = float(answer_text_with_number.replace(',', ''))

                past_kv, decoded_tokens = run_cot_with_residual_mapping(
                    model, tokenizer, training_args, question,
                    intervention_type=condition_name,
                    U=U_cond, V=V_cond, b=b_cond, alpha=alpha_cond
                )

                answer_text, predicted_answer = generate_answer(
                    model, tokenizer, training_args, past_kv
                )

                correct = (predicted_answer == ground_truth) if predicted_answer is not None else False

                result = {
                    'example_idx': ex_idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'answer_text': answer_text,
                    'correct': correct,
                    'decoded_tokens': decoded_tokens
                }

                condition_results.append(result)

            accuracy = sum(r['correct'] for r in condition_results) / len(condition_results) * 100
            print(f"  Accuracy: {accuracy:.1f}% ({sum(r['correct'] for r in condition_results)}/{len(condition_results)})")

            dataset_results.append({
                'condition': condition_name,
                'results': condition_results,
                'accuracy': accuracy
            })

        # Save results
        results_file = RESULTS_DIR / f'results_{dataset_name}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'dataset': dataset_name,
                'num_examples': len(dataset),
                'conditions': dataset_results
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary
        print("\n" + "="*80)
        print(f"SUMMARY: {dataset_name}")
        print("="*80)
        for result in dataset_results:
            print(f"{result['condition']:20s}: {result['accuracy']:5.1f}%")


if __name__ == "__main__":
    main()
