#!/usr/bin/env python3
"""
Learned Linear Mapping Intervention Experiment

This experiment learns a linear mapping from token embeddings to original activations
and uses it to improve the discretization intervention.

Approach:
1. Collect (token_embedding, original_activation) pairs from GSM8K training set
2. Split data into balanced train/validation sets (balanced by token string)
3. Train linear mapping with hyperparameter search
4. Test intervention: project top-1 embedding through learned mapping
5. Compare baseline vs learned mapping intervention on clean & GSM8K test datasets
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
RESULTS_DIR = Path('./learned_mapping_results')
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


def collect_training_data(model, tokenizer, training_args, num_samples=None):
    """
    Collect (token_embedding, original_activation) pairs from GSM8K training set.

    Returns:
        List of dicts with keys: 'token_string', 'token_id', 'embedding', 'activation', 'position'
    """
    print(f"\nCollecting training data from GSM8K training set...")

    # Load GSM8K training set
    dataset = load_dataset("gsm8k", "main")
    train_set = dataset['train']
    if num_samples:
        train_set = train_set.select(range(num_samples))

    print(f"Using {len(train_set)} examples from GSM8K train")

    training_pairs = []
    embedding_layer = model.codi.get_input_embeddings()

    batch_size = 1

    for example in tqdm(train_set, desc="Collecting data"):
        question = example['question']
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

        with torch.no_grad():
            # Initial encoding (position 0: BoT)
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

            # Get top-1 token at position 0
            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])
            token_embedding = embedding_layer(torch.tensor([token_id], device=device))

            # Store pair: (embedding, activation) for position 0
            training_pairs.append({
                'token_string': token_str,
                'token_id': token_id,
                'embedding': token_embedding.squeeze(0).cpu().clone(),  # [hidden]
                'activation': latent_embd.squeeze().cpu().clone(),  # [hidden]
                'position': 0
            })

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # CoT iterations (positions 1-6)
            for i in range(training_args.inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1:, :]

                # Get top-1 token
                logits = model.codi.lm_head(latent_embd.squeeze(1))
                token_id = torch.argmax(logits, dim=-1).item()
                token_str = tokenizer.decode([token_id])
                token_embedding = embedding_layer(torch.tensor([token_id], device=device))

                # Store pair
                training_pairs.append({
                    'token_string': token_str,
                    'token_id': token_id,
                    'embedding': token_embedding.squeeze(0).cpu().clone(),
                    'activation': latent_embd.squeeze().cpu().clone(),
                    'position': i + 1
                })

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

    print(f"✓ Collected {len(training_pairs)} training pairs")
    return training_pairs


def create_balanced_split(training_pairs, val_samples_per_token=5):
    """
    Create balanced train/validation split by token string.

    Args:
        training_pairs: List of training pair dicts
        val_samples_per_token: Number of validation samples per unique token string

    Returns:
        train_pairs, val_pairs
    """
    print(f"\nCreating balanced train/validation split...")

    # Group by token string
    token_groups = defaultdict(list)
    for pair in training_pairs:
        token_groups[pair['token_string']].append(pair)

    print(f"Found {len(token_groups)} unique token strings")

    # Print distribution
    token_counts = {tok: len(pairs) for tok, pairs in token_groups.items()}
    sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 20 most frequent tokens:")
    for tok, count in sorted_tokens[:20]:
        print(f"  '{tok}': {count} samples")

    # Create balanced validation set
    val_pairs = []
    train_pairs = []

    for token_str, pairs in token_groups.items():
        # Shuffle pairs for this token
        np.random.shuffle(pairs)

        # Take up to val_samples_per_token for validation
        n_val = min(val_samples_per_token, len(pairs))
        val_pairs.extend(pairs[:n_val])
        train_pairs.extend(pairs[n_val:])

    print(f"\n✓ Train set: {len(train_pairs)} pairs")
    print(f"✓ Validation set: {len(val_pairs)} pairs")
    print(f"✓ Validation uses {val_samples_per_token} samples per token type")

    return train_pairs, val_pairs


def train_linear_mapping(train_pairs, val_pairs, hidden_dim,
                        lr=0.001, epochs=100, downsample_factor=1.0,
                        device='cuda'):
    """
    Train linear mapping: embedding -> activation

    Args:
        train_pairs: List of training pairs
        val_pairs: List of validation pairs
        hidden_dim: Hidden dimension size
        lr: Learning rate
        epochs: Number of training epochs
        downsample_factor: Fraction of training data to use (for avoiding overfitting)
        device: Device to train on

    Returns:
        Learned weight matrix W [hidden, hidden], bias b [hidden], val_loss
    """
    # Downsample training data if requested
    if downsample_factor < 1.0:
        n_train = int(len(train_pairs) * downsample_factor)
        train_pairs = np.random.choice(train_pairs, n_train, replace=False).tolist()

    # Prepare training data - convert to float32 for training
    X_train = torch.stack([pair['embedding'] for pair in train_pairs]).to(device).float()  # [N, hidden]
    Y_train = torch.stack([pair['activation'] for pair in train_pairs]).to(device).float()  # [N, hidden]

    # Prepare validation data - convert to float32 for training
    X_val = torch.stack([pair['embedding'] for pair in val_pairs]).to(device).float()
    Y_val = torch.stack([pair['activation'] for pair in val_pairs]).to(device).float()

    # Initialize linear mapping: Y = X @ W.T + b
    W = torch.randn(hidden_dim, hidden_dim, device=device, dtype=torch.float32) * 0.01
    b = torch.zeros(hidden_dim, device=device, dtype=torch.float32)
    W.requires_grad = True
    b.requires_grad = True

    optimizer = torch.optim.Adam([W, b], lr=lr)

    best_val_loss = float('inf')
    best_W = None
    best_b = None

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        Y_pred = X_train @ W.T + b  # [N, hidden]

        # MSE loss
        loss = torch.mean((Y_pred - Y_train) ** 2)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            Y_val_pred = X_val @ W.T + b
            val_loss = torch.mean((Y_val_pred - Y_val) ** 2).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_W = W.detach().clone()
                best_b = b.detach().clone()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

    print(f"✓ Best validation loss: {best_val_loss:.6f}")

    return best_W, best_b, best_val_loss


def hyperparameter_search(train_pairs, val_pairs, hidden_dim):
    """
    Search for best hyperparameters using validation set.

    Returns:
        best_W, best_b, best_config
    """
    print("\n" + "="*80)
    print("Hyperparameter Search")
    print("="*80)

    # Define search space
    lr_options = [0.001, 0.0001, 0.00001]
    epochs_options = [50, 100, 200]
    downsample_options = [1.0, 0.5, 0.2]

    best_val_loss = float('inf')
    best_W = None
    best_b = None
    best_config = None

    search_results = []

    for lr in lr_options:
        for epochs in epochs_options:
            for downsample in downsample_options:
                print(f"\nTrying: lr={lr}, epochs={epochs}, downsample={downsample}")

                W, b, val_loss = train_linear_mapping(
                    train_pairs, val_pairs, hidden_dim,
                    lr=lr, epochs=epochs, downsample_factor=downsample,
                    device=device
                )

                search_results.append({
                    'lr': lr,
                    'epochs': epochs,
                    'downsample': downsample,
                    'val_loss': val_loss
                })

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_W = W
                    best_b = b
                    best_config = {
                        'lr': lr,
                        'epochs': epochs,
                        'downsample': downsample,
                        'val_loss': val_loss
                    }
                    print(f"  ★ New best! val_loss={val_loss:.6f}")

    print("\n" + "="*80)
    print("Hyperparameter Search Results")
    print("="*80)
    print(f"Best config: {best_config}")

    # Save search results
    with open(RESULTS_DIR / 'hyperparameter_search.json', 'w') as f:
        json.dump({
            'search_results': search_results,
            'best_config': best_config
        }, f, indent=2)

    return best_W, best_b, best_config


def run_cot_with_learned_mapping(model, tokenizer, training_args, question,
                                 intervention_type, W=None, b=None):
    """
    Run CoT with learned mapping intervention.

    Args:
        intervention_type: 'baseline' or 'learned_mapping'
        W, b: Learned linear mapping parameters (required for 'learned_mapping')
    """
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
        # Initial encoding (position 0: BoT)
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

        # Decode token at position 0
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        # Apply intervention at position 0
        if intervention_type == 'learned_mapping':
            # Get top-1 embedding
            token_embedding = embedding_layer(torch.tensor([token_id], device=device))
            # Project through learned mapping
            A_mapped = token_embedding @ W.T + b  # [1, hidden]
            # Normalize to match original activation norm
            A_norm = torch.norm(latent_embd.squeeze(1), dim=-1, keepdim=True)
            mapped_norm = torch.norm(A_mapped, dim=-1, keepdim=True)
            A_modified = A_mapped * (A_norm / (mapped_norm + 1e-8))
            latent_embd = A_modified.unsqueeze(1)

            # Re-decode
            logits_modified = model.codi.lm_head(A_modified)
            new_token_id = torch.argmax(logits_modified, dim=-1).item()
            new_token_str = tokenizer.decode([new_token_id])
            decoded_tokens.append({'position': 0, 'token': new_token_str, 'is_number': is_number, 'intervened': True})
        else:
            # Baseline: no intervention
            decoded_tokens.append({'position': 0, 'token': token_str, 'is_number': is_number, 'intervened': False})

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # CoT iterations (positions 1-6)
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

            if intervention_type == 'learned_mapping':
                token_embedding = embedding_layer(torch.tensor([token_id], device=device))
                A_mapped = token_embedding @ W.T + b
                A_norm = torch.norm(latent_embd.squeeze(1), dim=-1, keepdim=True)
                mapped_norm = torch.norm(A_mapped, dim=-1, keepdim=True)
                A_modified = A_mapped * (A_norm / (mapped_norm + 1e-8))
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
    """Generate final answer using past_key_values from CoT"""
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

        return full_answer, predicted_number


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Load model
    model, tokenizer, training_args = load_llama_model()
    hidden_dim = model.dim

    # ========================================================================
    # PHASE 1: Collect Training Data
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 1: Data Collection")
    print("="*80)

    # Check if training pairs already exist
    training_pairs_path = RESULTS_DIR / 'training_pairs.pt'
    if training_pairs_path.exists():
        print(f"\nLoading existing training pairs from {training_pairs_path}...")
        checkpoint = torch.load(training_pairs_path)
        training_pairs = checkpoint['training_pairs']
        print(f"✓ Loaded {len(training_pairs)} training pairs")
    else:
        # Collect training pairs from GSM8K training set
        training_pairs = collect_training_data(
            model, tokenizer, training_args,
            num_samples=None  # Use all training data
        )

        # Save collected data
        print("\nSaving collected data...")
        torch.save({
            'training_pairs': training_pairs,
            'hidden_dim': hidden_dim
        }, RESULTS_DIR / 'training_pairs.pt')
        print(f"✓ Saved to {RESULTS_DIR / 'training_pairs.pt'}")

    # ========================================================================
    # PHASE 2: Create Balanced Train/Val Split
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 2: Create Balanced Split")
    print("="*80)

    train_pairs, val_pairs = create_balanced_split(
        training_pairs,
        val_samples_per_token=5  # Use 5 samples per token type for validation
    )

    # ========================================================================
    # PHASE 3: Hyperparameter Search & Training
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 3: Training Linear Mapping")
    print("="*80)

    W, b, best_config = hyperparameter_search(train_pairs, val_pairs, hidden_dim)

    # Save learned mapping
    print("\nSaving learned mapping...")
    torch.save({
        'W': W.cpu(),
        'b': b.cpu(),
        'config': best_config,
        'hidden_dim': hidden_dim
    }, RESULTS_DIR / 'learned_mapping.pt')
    print(f"✓ Saved to {RESULTS_DIR / 'learned_mapping.pt'}")

    # Convert to bfloat16 for inference
    W = W.to(torch.bfloat16)
    b = b.to(torch.bfloat16)

    # ========================================================================
    # PHASE 4: Evaluation on Test Sets
    # ========================================================================
    print("\n" + "="*80)
    print("PHASE 4: Evaluation")
    print("="*80)

    # Load test datasets
    print("\nLoading test datasets...")

    # Dataset 1: Clean dataset (132 examples)
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)
    print(f"✓ Loaded {len(clean_dataset)} examples from clean dataset")

    # Dataset 2: GSM8K test first 132
    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_test_132 = gsm8k_dataset['test'].select(range(132))
    print(f"✓ Loaded {len(gsm8k_test_132)} examples from GSM8K test")

    # Test conditions
    conditions = [
        ('baseline', None, None),
        ('learned_mapping', W, b)
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
                # Extract question and answer
                if dataset_name == 'clean':
                    example = dataset[ex_idx]
                    question = example['question']
                    ground_truth = float(example['answer'])
                else:  # gsm8k_test
                    example = dataset[ex_idx]
                    question = example['question']
                    answer_text_with_number = example['answer'].split('####')[1].strip()
                    ground_truth = float(answer_text_with_number.replace(',', ''))

                # Run inference
                past_kv, decoded_tokens = run_cot_with_learned_mapping(
                    model, tokenizer, training_args, question,
                    intervention_type=condition_name,
                    W=W_cond, b=b_cond
                )

                # Generate answer
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
