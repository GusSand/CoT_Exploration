#!/usr/bin/env python3
"""
Comprehensive Intervention Comparison for CODI-LLaMA
Compares multiple intervention strategies on chain-of-thought reasoning

Intervention types:
1. Baseline (no intervention)
2. Replacement (>'5') - existing implementation
3. Zero ablation
4. Average ablation (position-specific means from GSM8K train)
5. Minus ablation
6. Discretization (L2-normalized)
7. Discretization + LayerNorm (L2-normalized)
8. Projection@1 (unnormalized)
9. Projection@5 (normalized)
10. Projection@5 (unnormalized)

Each intervention (except baseline) tested with 2 scopes:
- Numbers only (intervene at number positions)
- All positions (intervene at all CoT positions)

Total: 1 + (9 × 2) = 19 conditions
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
RESULTS_DIR = Path('./intervention_comparison_results')
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


def compute_average_activations(model, tokenizer, training_args, num_samples=100):
    """
    Compute average activations at each CoT position from GSM8K training set.
    Returns dict: {position: mean_activation} for positions 0-6
    """
    print(f"\nComputing average activations from {num_samples} GSM8K train examples...")

    # Load GSM8K training set
    dataset = load_dataset("gsm8k", "main")
    train_set = dataset['train'].select(range(num_samples))

    # Accumulate activations at each position
    position_activations = {i: [] for i in range(7)}  # positions 0-6

    batch_size = 1

    for example in tqdm(train_set, desc="Collecting activations"):
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

            # Store BoT activation (position 0, BEFORE projection)
            position_activations[0].append(latent_embd.squeeze().cpu())

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

                # Store activation (BEFORE projection)
                position_activations[i+1].append(latent_embd.squeeze().cpu())

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

    # Compute means
    mean_activations = {}
    for pos in range(7):
        activations_tensor = torch.stack(position_activations[pos])
        mean_activations[pos] = activations_tensor.mean(dim=0).to(device)

    print("✓ Average activations computed")
    return mean_activations


def project_onto_topk_vocab(continuous_vector, vocab_embeddings_topk, normalize=False):
    """
    Project continuous vector onto subspace spanned by top-k vocab embeddings.

    Args:
        continuous_vector: [batch, hidden]
        vocab_embeddings_topk: [batch, k, hidden]
        normalize: If True, normalize result

    Returns:
        Projected vector [batch, hidden]
    """
    batch_size, k, hidden = vocab_embeddings_topk.shape

    if k == 1:
        vocab_embedding = vocab_embeddings_topk[:, 0, :]
        if normalize:
            continuous_norm = torch.norm(continuous_vector, dim=-1, keepdim=True)
            vocab_norm = torch.norm(vocab_embedding, dim=-1, keepdim=True)
            result = vocab_embedding * (continuous_norm / (vocab_norm + 1e-8))
            return result
        else:
            # Project onto vocab direction (unnormalized)
            vocab_direction = vocab_embedding / (torch.norm(vocab_embedding, dim=-1, keepdim=True) + 1e-8)
            projection_scalar = torch.sum(continuous_vector * vocab_direction, dim=-1, keepdim=True)
            projected = projection_scalar * vocab_direction
            return projected
    else:
        # k > 1: subspace projection using least squares
        # For each batch element, solve: alpha = (V V^T)^{-1} V c
        projected_batch = []
        for b in range(batch_size):
            V = vocab_embeddings_topk[b].float()  # [k, hidden] - convert to float for linalg
            c = continuous_vector[b].float()  # [hidden] - convert to float for linalg

            G = torch.mm(V, V.t())  # [k, k] Gram matrix
            Vc = torch.mv(V, c)  # [k]

            try:
                alpha = torch.linalg.solve(G, Vc)  # [k]
            except:
                alpha = torch.linalg.lstsq(G, Vc).solution  # [k]

            projected = torch.mv(V.t(), alpha)  # [hidden]
            projected_batch.append(projected)

        projected = torch.stack(projected_batch, dim=0).to(continuous_vector.dtype)  # Convert back to original dtype

        if normalize:
            continuous_norm = torch.norm(continuous_vector, dim=-1, keepdim=True)
            projected_norm = torch.norm(projected, dim=-1, keepdim=True)
            projected = projected * (continuous_norm / (projected_norm + 1e-8))

        return projected


def run_cot_with_intervention(model, tokenizer, training_args, question,
                              intervention_type, intervention_scope,
                              mean_activations=None, layer_norm=None):
    """
    Run CoT with specified intervention.

    Args:
        intervention_type: 'baseline', 'replacement', 'zero', 'average', 'minus',
                          'discretize', 'discretize_ln', 'proj1', 'proj5', 'proj5_unnorm'
        intervention_scope: 'numbers' or 'all'
        mean_activations: dict of mean activations per position (for 'average')
        layer_norm: LayerNorm module (for 'discretize_ln')
    """
    batch_size = 1
    questions = [question]

    # Get target embedding for replacement intervention
    if intervention_type == 'replacement':
        target_token = '5'
        target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
        embedding_layer = model.codi.get_input_embeddings()
        target_embd = embedding_layer(torch.tensor([target_token_id], device=device))
        k_replace = 3  # scaling factor for replacement

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
    intervened_positions = []
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

        # Decode and potentially intervene at BoT
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        # Determine if we should intervene at this position
        should_intervene = (intervention_type != 'baseline' and
                           (intervention_scope == 'all' or (intervention_scope == 'numbers' and is_number)))

        if should_intervene:
            A = latent_embd.squeeze(1)  # [batch, hidden]
            A_modified = apply_intervention(
                A, token_id, embedding_layer, intervention_type,
                position=0, mean_activations=mean_activations,
                layer_norm=layer_norm, target_embd=target_embd if intervention_type == 'replacement' else None,
                k_replace=k_replace if intervention_type == 'replacement' else None
            )
            latent_embd = A_modified.unsqueeze(1)

            # Re-decode after intervention
            logits_modified = model.codi.lm_head(A_modified)
            new_token_id = torch.argmax(logits_modified, dim=-1).item()
            new_token_str = tokenizer.decode([new_token_id])
            decoded_tokens.append({'position': 0, 'token': new_token_str, 'is_number': is_number, 'intervened': True})
            intervened_positions.append(0)
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

            should_intervene = (intervention_type != 'baseline' and
                               (intervention_scope == 'all' or (intervention_scope == 'numbers' and is_number)))

            if should_intervene:
                A = latent_embd.squeeze(1)
                A_modified = apply_intervention(
                    A, token_id, embedding_layer, intervention_type,
                    position=i+1, mean_activations=mean_activations,
                    layer_norm=layer_norm, target_embd=target_embd if intervention_type == 'replacement' else None,
                    k_replace=k_replace if intervention_type == 'replacement' else None
                )
                latent_embd = A_modified.unsqueeze(1)

                logits_modified = model.codi.lm_head(A_modified)
                new_token_id = torch.argmax(logits_modified, dim=-1).item()
                new_token_str = tokenizer.decode([new_token_id])
                decoded_tokens.append({'position': i+1, 'token': new_token_str, 'is_number': is_number, 'intervened': True})
                intervened_positions.append(i+1)
            else:
                decoded_tokens.append({'position': i+1, 'token': token_str, 'is_number': is_number, 'intervened': False})

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return past_key_values, decoded_tokens, intervened_positions


def apply_intervention(A, token_id, embedding_layer, intervention_type,
                      position, mean_activations=None, layer_norm=None,
                      target_embd=None, k_replace=None):
    """
    Apply the specified intervention to activation A.

    Args:
        A: activation tensor [batch, hidden] (BEFORE projection layer)
        token_id: decoded token ID
        embedding_layer: token embedding layer
        intervention_type: type of intervention
        position: CoT position (0-6)
        mean_activations: dict of mean activations (for 'average')
        layer_norm: LayerNorm module (for 'discretize_ln')
        target_embd: target embedding (for 'replacement')
        k_replace: scaling factor (for 'replacement')

    Returns:
        Modified activation tensor [batch, hidden]
    """
    if intervention_type == 'zero':
        # Zero ablation
        return torch.zeros_like(A)

    elif intervention_type == 'average':
        # Average ablation
        return mean_activations[position].unsqueeze(0).expand_as(A)

    elif intervention_type == 'minus':
        # Minus ablation: subtract projection onto top-1
        predicted_embd = embedding_layer(torch.tensor([token_id], device=A.device))
        E_pred_norm = predicted_embd / torch.norm(predicted_embd, dim=-1, keepdim=True)
        proj_predicted = torch.sum(A * E_pred_norm, dim=-1, keepdim=True) * E_pred_norm
        return A - proj_predicted

    elif intervention_type == 'discretize':
        # Discretization: replace with embedding of top-1, L2-normalized
        predicted_embd = embedding_layer(torch.tensor([token_id], device=A.device))
        A_norm = torch.norm(A, dim=-1, keepdim=True)
        E_norm = torch.norm(predicted_embd, dim=-1, keepdim=True)
        return predicted_embd * (A_norm / (E_norm + 1e-8))

    elif intervention_type == 'discretize_ln':
        # Discretization + LayerNorm, then L2-normalized
        predicted_embd = embedding_layer(torch.tensor([token_id], device=A.device))
        ln_embd = layer_norm(predicted_embd)
        A_norm = torch.norm(A, dim=-1, keepdim=True)
        LN_norm = torch.norm(ln_embd, dim=-1, keepdim=True)
        return ln_embd * (A_norm / (LN_norm + 1e-8))

    elif intervention_type == 'proj1':
        # Projection@1 (unnormalized)
        predicted_embd = embedding_layer(torch.tensor([token_id], device=A.device))
        vocab_embeddings_topk = predicted_embd.unsqueeze(1)  # [1, 1, hidden]
        return project_onto_topk_vocab(A, vocab_embeddings_topk, normalize=False)
    elif intervention_type == 'proj5':
        # Projection@5 (normalized)
        logits = embedding_layer.weight @ A.t()  # [vocab_size, batch]
        topk_values, topk_indices = torch.topk(logits, k=5, dim=0)  # [5, batch]
        topk_indices_flat = topk_indices.squeeze(-1)  # [5]
        topk_embeddings = embedding_layer.weight[topk_indices_flat]  # [5, hidden]
        vocab_embeddings_topk = topk_embeddings.unsqueeze(0)  # [1, 5, hidden]
        return project_onto_topk_vocab(A, vocab_embeddings_topk, normalize=True)

    elif intervention_type == 'proj5_unnorm':
        # Projection@5 (unnormalized)
        logits = embedding_layer.weight @ A.t()  # [vocab_size, batch]
        topk_values, topk_indices = torch.topk(logits, k=5, dim=0)  # [5, batch]
        topk_indices_flat = topk_indices.squeeze(-1)  # [5]
        topk_embeddings = embedding_layer.weight[topk_indices_flat]  # [5, hidden]
        vocab_embeddings_topk = topk_embeddings.unsqueeze(0)  # [1, 5, hidden]
        return project_onto_topk_vocab(A, vocab_embeddings_topk, normalize=False)

    elif intervention_type == 'replacement':
        # Replacement intervention (existing implementation)
        predicted_embd = embedding_layer(torch.tensor([token_id], device=A.device))
        E_pred_norm = predicted_embd / torch.norm(predicted_embd, dim=-1, keepdim=True)
        E_target_norm = target_embd / torch.norm(target_embd, dim=-1, keepdim=True)

        proj_predicted = torch.sum(A * E_pred_norm, dim=-1, keepdim=True) * E_pred_norm
        proj_target = torch.sum(A * E_target_norm, dim=-1, keepdim=True) * E_target_norm
        return A - proj_predicted + k_replace * proj_target

    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


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
    # Load model
    model, tokenizer, training_args = load_llama_model()

    # Create LayerNorm for discretize_ln intervention
    layer_norm = torch.nn.LayerNorm(model.dim).to(device).to(torch.bfloat16)

    # Compute average activations for 'average' ablation
    print("\n" + "="*80)
    print("Computing Average Activations")
    print("="*80)
    mean_activations = compute_average_activations(model, tokenizer, training_args, num_samples=100)

    # Load datasets
    print("\n" + "="*80)
    print("Loading Datasets")
    print("="*80)

    # Dataset 1: Clean dataset (132 examples that LLAMA solves correctly)
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)
    print(f"✓ Loaded {len(clean_dataset)} examples from clean dataset")

    # Dataset 2: GSM8K train first 132
    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train_132 = gsm8k_dataset['train'].select(range(132))
    print(f"✓ Loaded {len(gsm8k_train_132)} examples from GSM8K train")

    # Define intervention conditions
    intervention_conditions = [
        ('baseline', 'none'),  # Baseline
        ('replacement', 'numbers'),
        ('replacement', 'all'),
        ('zero', 'numbers'),
        ('zero', 'all'),
        ('average', 'numbers'),
        ('average', 'all'),
        ('minus', 'numbers'),
        ('minus', 'all'),
        ('discretize', 'numbers'),
        ('discretize', 'all'),
        ('discretize_ln', 'numbers'),
        ('discretize_ln', 'all'),
        ('proj1', 'numbers'),
        ('proj1', 'all'),
        ('proj5', 'numbers'),
        ('proj5', 'all'),
        ('proj5_unnorm', 'numbers'),
        ('proj5_unnorm', 'all'),
    ]

    print(f"\n✓ Testing {len(intervention_conditions)} intervention conditions")

    # Test on all 132 examples from both datasets
    num_test_clean = len(clean_dataset)  # 132
    num_test_gsm8k = len(gsm8k_train_132)  # 132

    print(f"\n" + "="*80)
    print(f"FULL EVALUATION: Testing All Examples")
    print(f"Clean Dataset: {num_test_clean} examples")
    print(f"GSM8K Train: {num_test_gsm8k} examples")
    print("="*80)

    # Run on both datasets
    for dataset_name, dataset, num_examples in [
        ('clean', clean_dataset, num_test_clean),
        ('gsm8k_train', gsm8k_train_132, num_test_gsm8k)
    ]:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset_name} ({num_examples} examples)")
        print(f"{'='*80}")

        dataset_results = []

        for condition_idx, (interv_type, interv_scope) in enumerate(intervention_conditions):
            print(f"\n[{condition_idx+1}/{len(intervention_conditions)}] Testing: {interv_type} ({interv_scope})")

            condition_results = []

            for ex_idx in range(num_examples):
                # Extract question and answer based on dataset type
                if dataset_name == 'clean':
                    example = dataset[ex_idx]
                    question = example['question']
                    ground_truth = float(example['answer'])
                    pair_id = example.get('pair_id', ex_idx)
                else:  # gsm8k_train
                    example = dataset[ex_idx]
                    question = example['question']
                    # Extract answer from GSM8K format
                    answer_text_with_number = example['answer'].split('####')[1].strip()
                    ground_truth = float(answer_text_with_number.replace(',', ''))
                    pair_id = ex_idx

                # Run intervention
                past_kv, decoded_tokens, intervened_positions = run_cot_with_intervention(
                    model, tokenizer, training_args, question,
                    intervention_type=interv_type,
                    intervention_scope=interv_scope,
                    mean_activations=mean_activations,
                    layer_norm=layer_norm
                )

                # Generate answer
                answer_text, predicted_answer = generate_answer(model, tokenizer, training_args, past_kv)

                correct = (predicted_answer == ground_truth) if predicted_answer is not None else False

                result = {
                    'example_idx': ex_idx,
                    'pair_id': pair_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted_answer,
                    'answer_text': answer_text,
                    'correct': correct,
                    'decoded_tokens': decoded_tokens,
                    'intervened_positions': intervened_positions
                }

                condition_results.append(result)

            accuracy = sum(r['correct'] for r in condition_results) / len(condition_results) * 100
            print(f"  Accuracy: {accuracy:.1f}% ({sum(r['correct'] for r in condition_results)}/{len(condition_results)})")

            dataset_results.append({
                'intervention_type': interv_type,
                'intervention_scope': interv_scope,
                'results': condition_results,
                'accuracy': accuracy
            })

        # Save results for this dataset
        results_file = RESULTS_DIR / f'full_results_{dataset_name}_{num_examples}_examples.json'
        with open(results_file, 'w') as f:
            json.dump({
                'config': {
                    'num_examples': num_examples,
                    'dataset': dataset_name,
                    'num_conditions': len(intervention_conditions)
                },
                'conditions': dataset_results
            }, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

        # Print summary for this dataset
        print("\n" + "="*80)
        print(f"RESULTS SUMMARY: {dataset_name}")
        print("="*80)
        for result in dataset_results:
            print(f"{result['intervention_type']:15s} ({result['intervention_scope']:7s}): {result['accuracy']:5.1f}%")


if __name__ == "__main__":
    main()
