#!/usr/bin/env python3
"""
Learn linear mapping with diagonal regularization:
- Penalize off-diagonal elements to encourage near-identity matrix
- Optionally penalize diagonal elements deviating from 1.0
"""

import torch
import torch.nn.functional as F
import json
import sys
import re
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
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

    print(f"Train set: {len(train_pairs)} pairs")
    print(f"Validation set: {len(val_pairs)} pairs")

    return train_pairs, val_pairs


def train_diagonal_regularized_mapping(train_pairs, val_pairs, hidden_dim,
                                      lr=0.001, epochs=100, downsample_factor=1.0,
                                      lambda_offdiag=0.01, lambda_diag=0.0,
                                      device='cuda'):
    """
    Train linear mapping with diagonal regularization.

    Loss = MSE(Y_pred, Y_true)
           + lambda_offdiag * sum(W[i,j]^2 for i != j)
           + lambda_diag * sum((W[i,i] - 1)^2)
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

    # Initialize parameters close to identity
    W = torch.eye(hidden_dim, device=device, dtype=torch.float32) + torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32) * 0.01
    b = torch.zeros(hidden_dim, device=device, dtype=torch.float32)

    W.requires_grad = True
    b.requires_grad = True

    optimizer = torch.optim.Adam([W, b], lr=lr)

    batch_size = 256
    n_batches = (len(X_train) + batch_size - 1) // batch_size

    # Create mask for off-diagonal elements
    identity_mask = torch.eye(hidden_dim, device=device, dtype=torch.bool)
    offdiag_mask = ~identity_mask

    best_val_loss = float('inf')
    best_W = None
    best_b = None

    for epoch in range(epochs):
        # Training
        epoch_loss = 0
        perm = torch.randperm(len(X_train))

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(X_train))
            batch_indices = perm[start_idx:end_idx]

            X_batch = X_train[batch_indices]
            Y_batch = Y_train[batch_indices]

            # Forward pass
            Y_pred = X_batch @ W.T + b

            # MSE loss
            mse_loss = F.mse_loss(Y_pred, Y_batch)

            # Off-diagonal regularization
            offdiag_penalty = (W[offdiag_mask] ** 2).sum()

            # Diagonal regularization (encourage diagonal to be 1.0)
            diag_penalty = ((W.diag() - 1.0) ** 2).sum()

            # Total loss
            loss = mse_loss + lambda_offdiag * offdiag_penalty + lambda_diag * diag_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        with torch.no_grad():
            Y_pred_val = X_val @ W.T + b
            val_mse = F.mse_loss(Y_pred_val, Y_val).item()
            val_offdiag = (W[offdiag_mask] ** 2).sum().item()
            val_diag = ((W.diag() - 1.0) ** 2).sum().item()
            val_loss = val_mse + lambda_offdiag * val_offdiag + lambda_diag * val_diag

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = W.clone()
                best_b = b.clone()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss/n_batches:.4f}, "
                  f"Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, OffDiag: {val_offdiag:.2e}, Diag: {val_diag:.2e})")

    # Calculate final statistics
    with torch.no_grad():
        W_final = best_W
        offdiag_norm = torch.sqrt((W_final[offdiag_mask] ** 2).sum()).item()
        diag_values = W_final.diag()
        diag_mean = diag_values.mean().item()
        diag_std = diag_values.std().item()
        frobenius_norm = torch.norm(W_final - torch.eye(hidden_dim, device=device)).item()

    print(f"Best val_loss={best_val_loss:.4f}, val_mse={val_mse:.4f}")
    print(f"  Off-diagonal norm: {offdiag_norm:.4f}")
    print(f"  Diagonal mean: {diag_mean:.4f} +/- {diag_std:.4f}")
    print(f"  Frobenius distance from identity: {frobenius_norm:.4f}")

    return {
        'W': best_W.cpu(),
        'b': best_b.cpu(),
        'val_loss': best_val_loss,
        'val_mse': val_mse,
        'offdiag_norm': offdiag_norm,
        'diag_mean': diag_mean,
        'diag_std': diag_std,
        'frobenius_from_identity': frobenius_norm
    }


def hyperparameter_search(train_pairs, val_pairs, hidden_dim):
    """Search over diagonal regularization hyperparameters"""
    print("\n" + "="*80)
    print("Hyperparameter Search")
    print("="*80)

    configs = [
        # Vary off-diagonal penalty
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 0.0, 'lambda_diag': 0.0, 'downsample': 1.0},  # No regularization
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 0.001, 'lambda_diag': 0.0, 'downsample': 1.0},
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 0.01, 'lambda_diag': 0.0, 'downsample': 1.0},
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 0.1, 'lambda_diag': 0.0, 'downsample': 1.0},
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 1.0, 'lambda_diag': 0.0, 'downsample': 1.0},
        # Add diagonal penalty
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 0.01, 'lambda_diag': 0.001, 'downsample': 1.0},
        {'lr': 0.01, 'epochs': 200, 'lambda_offdiag': 0.1, 'lambda_diag': 0.01, 'downsample': 1.0},
    ]

    search_results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*80}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"lambda_offdiag={config['lambda_offdiag']}, lambda_diag={config['lambda_diag']}, "
              f"lr={config['lr']}, epochs={config['epochs']}")
        print(f"{'='*80}")

        result = train_diagonal_regularized_mapping(
            train_pairs, val_pairs, hidden_dim,
            lr=config['lr'],
            epochs=config['epochs'],
            downsample_factor=config['downsample'],
            lambda_offdiag=config['lambda_offdiag'],
            lambda_diag=config['lambda_diag'],
            device=device
        )

        search_results.append({
            **config,
            **result
        })

        # Save checkpoint
        torch.save({
            'W': result['W'],
            'b': result['b'],
            'config': config,
            'metrics': result
        }, RESULTS_DIR / f'checkpoint_{i}.pt')

    # Find best configuration
    best_idx = min(range(len(search_results)), key=lambda i: search_results[i]['val_loss'])
    best_config = search_results[best_idx]

    print(f"\n{'='*80}")
    print("BEST CONFIGURATION:")
    print(f"lambda_offdiag={best_config['lambda_offdiag']}, lambda_diag={best_config['lambda_diag']}")
    print(f"val_loss={best_config['val_loss']:.4f}, frobenius={best_config['frobenius_from_identity']:.4f}")
    print(f"{'='*80}")

    # Save search results
    search_results_serializable = []
    for res in search_results:
        res_copy = {k: v for k, v in res.items() if k not in ['W', 'b']}
        search_results_serializable.append(res_copy)

    with open(RESULTS_DIR / 'hyperparameter_search.json', 'w') as f:
        json.dump({
            'search_results': search_results_serializable,
            'best_config': {k: v for k, v in best_config.items() if k not in ['W', 'b']},
            'best_idx': best_idx
        }, f, indent=2)

    # Load best checkpoint
    best_checkpoint = torch.load(RESULTS_DIR / f'checkpoint_{best_idx}.pt')
    return best_checkpoint['W'], best_checkpoint['b'], best_config


def run_cot_with_diagonal_mapping(model, tokenizer, training_args, question,
                                    intervention_type='baseline', W=None, b=None):
    """Run CoT generation with diagonal regularization mapping intervention"""
    # Tokenize question
    inputs = tokenizer(question, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    # Get latent tokens
    with torch.no_grad():
        latent_tokens = model.latent_predictor(input_ids)

    # Run autoregressive latent reasoning
    past_kv = None
    decoded_tokens = []

    for i in range(training_args.inf_latent_iterations):
        if i == 0:
            current_latent = latent_tokens
        else:
            # Prepare input for next iteration
            current_latent = torch.tensor([[model.latent_token]], device=device)

        # Get embeddings
        with torch.no_grad():
            embeddings = model.codi.model.embed_tokens(current_latent)

            # Apply intervention
            if intervention_type == 'diagonal_mapping' and W is not None:
                embeddings_flat = embeddings.squeeze(0).float()
                W_device = W.to(device).to(torch.float32)
                b_device = b.to(device).to(torch.float32)
                mapped = embeddings_flat @ W_device.T + b_device
                embeddings = mapped.unsqueeze(0).to(torch.bfloat16)

            # Generate
            outputs = model.codi.model(
                inputs_embeds=embeddings,
                past_key_values=past_kv,
                use_cache=True
            )

            past_kv = outputs.past_key_values

            # Decode token
            logits = model.codi.lm_head(outputs.last_hidden_state[:, -1:, :])
            next_token = torch.argmax(logits, dim=-1)
            decoded_tokens.append(next_token.item())

    return past_kv, decoded_tokens


def generate_answer(model, tokenizer, training_args, past_kv):
    """Generate final answer from latent reasoning"""
    with torch.no_grad():
        # Generate from cached state
        bot_token = tokenizer.encode('<bot>', add_special_tokens=False)[0]
        bot_input = torch.tensor([[bot_token]], device=device)

        outputs = model.codi.generate(
            input_ids=bot_input,
            past_key_values=past_kv,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract number
        match = number_regex.search(answer_text)
        if match:
            try:
                predicted_answer = float(match.group().strip().replace(',', ''))
                return answer_text, predicted_answer
            except:
                return answer_text, None
        return answer_text, None


def main():
    # Load model
    model, tokenizer, training_args = load_llama_model()
    hidden_dim = model.dim

    # Load existing training pairs
    print("\n" + "="*80)
    print("Loading Training Data")
    print("="*80)

    training_pairs_path = Path('training_pairs.pt')
    checkpoint = torch.load(training_pairs_path)
    training_pairs = checkpoint['training_pairs']
    print(f"Loaded {len(training_pairs)} training pairs")

    # Create balanced split
    train_pairs, val_pairs = create_balanced_split(training_pairs, val_samples_per_token=5)

    # Hyperparameter search & training
    W, b, best_config = hyperparameter_search(train_pairs, val_pairs, hidden_dim)

    # Save learned mapping
    print("\nSaving learned mapping...")
    torch.save({
        'W': W.cpu(),
        'b': b.cpu(),
        'config': best_config,
        'hidden_dim': hidden_dim
    }, RESULTS_DIR / 'learned_mapping.pt')
    print(f"Saved to {RESULTS_DIR / 'learned_mapping.pt'}")

    # Convert to bfloat16 for inference
    W = W.to(torch.bfloat16)
    b = b.to(torch.bfloat16)

    # Evaluation
    print("\n" + "="*80)
    print("Evaluation")
    print("="*80)

    # Load test datasets
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)
    print(f"Loaded {len(clean_dataset)} examples from clean dataset")

    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_test_132 = gsm8k_dataset['test'].select(range(132))
    print(f"Loaded {len(gsm8k_test_132)} examples from GSM8K test")

    conditions = [
        ('baseline', None, None),
        ('diagonal_mapping', W, b)
    ]

    # Evaluate on both datasets
    for dataset_name, dataset in [('clean', clean_dataset), ('gsm8k_test', gsm8k_test_132)]:
        print(f"\n{'='*80}")
        print(f"Testing on {dataset_name} dataset")
        print(f"{'='*80}")

        dataset_results = []

        for condition_name, W_cond, b_cond in conditions:
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

                past_kv, decoded_tokens = run_cot_with_diagonal_mapping(
                    model, tokenizer, training_args, question,
                    intervention_type=condition_name,
                    W=W_cond, b=b_cond
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
                'condition': condition_name,
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

        print(f"Saved {RESULTS_DIR / f'results_{dataset_name}.json'}")


if __name__ == '__main__':
    main()
