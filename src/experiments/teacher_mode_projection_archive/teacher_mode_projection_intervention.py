#!/usr/bin/env python3
"""
Teacher Mode Top-K Projection Intervention Comparison

Adapts top-k projection intervention for TEACHER MODE (explicit CoT generation).
Key difference from student mode: Interventions stop after "The answer is:" is generated.

Intervention types tested:
1. Baseline (no intervention)
2. Average ablation (position-specific means - NA for teacher mode, used for comparison)
3. Projection@k (numbers only) - for k = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

Total conditions: 1 baseline + 10 k values = 11 conditions
Tested on Clean dataset (132 examples that LLAMA solves correctly)

Restriction mechanism: Token sequence detection to identify "The answer is:"
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
RESULTS_DIR = Path('./teacher_mode_projection_results')
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
        bf16=False,
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

    return llama_model, llama_tokenizer


def detect_answer_trigger(token_window, trigger_sequences):
    """
    Check if the token window ends with any of the trigger sequences.
    Returns True if "The answer is:" pattern is detected.
    """
    for trigger_seq in trigger_sequences:
        seq_len = len(trigger_seq)
        if len(token_window) >= seq_len:
            if token_window[-seq_len:] == trigger_seq:
                return True
    return False


def project_onto_topk_vocab(continuous_vector, vocab_embeddings_topk, normalize=False):
    """
    Project continuous vector onto subspace spanned by top-k vocab embeddings.

    Args:
        continuous_vector: [batch, hidden]
        vocab_embeddings_topk: [batch, k, hidden]
        normalize: If True, normalize result to preserve magnitude

    Returns:
        Projected vector [batch, hidden]
    """
    batch_size, k, hidden = vocab_embeddings_topk.shape

    if k == 1:
        # For k=1: project onto single vocabulary embedding
        vocab_embedding = vocab_embeddings_topk[:, 0, :]
        if normalize:
            # Preserve original magnitude
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
        # For each batch element, solve: alpha = (V^T V)^{-1} V^T c
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
            # Preserve original magnitude
            continuous_norm = torch.norm(continuous_vector, dim=-1, keepdim=True)
            projected_norm = torch.norm(projected, dim=-1, keepdim=True)
            projected = projected * (continuous_norm / (projected_norm + 1e-8))

        return projected


def apply_teacher_projection_intervention(hidden_states, token_id, embedding_layer, k):
    """
    Apply top-k projection intervention to hidden states in teacher mode.

    Args:
        hidden_states: [1, hidden_dim] tensor
        token_id: predicted token ID
        embedding_layer: model's token embedding layer
        k: number of top embeddings to project onto

    Returns:
        Modified hidden states [1, hidden_dim]
    """
    if k == 1:
        # Special case for k=1: use predicted token directly (equivalent to discretization)
        predicted_embd = embedding_layer(torch.tensor([token_id], device=hidden_states.device))
        vocab_embeddings_topk = predicted_embd.unsqueeze(1)  # [1, 1, hidden]
    else:
        # k > 1: find top-k vocab embeddings by computing logits
        # Compute similarity scores (logits) for all vocab embeddings
        logits = embedding_layer.weight @ hidden_states.t()  # [vocab_size, batch]
        topk_values, topk_indices = torch.topk(logits, k=k, dim=0)  # [k, batch]
        topk_indices_flat = topk_indices.squeeze(-1)  # [k]
        topk_embeddings = embedding_layer.weight[topk_indices_flat]  # [k, hidden]
        vocab_embeddings_topk = topk_embeddings.unsqueeze(0)  # [1, k, hidden]

    # Always use normalized projection (preserves magnitude)
    return project_onto_topk_vocab(hidden_states, vocab_embeddings_topk, normalize=True)


def generate_teacher_cot_with_projection(
    model, tokenizer, question,
    intervention_type='baseline',
    intervention_scope='numbers',
    k=1,
    max_length=300,
    answer_tokens_to_generate=2,
    debug=False
):
    """
    Generate explicit CoT in teacher mode with optional projection intervention.
    Interventions stop after "The answer is:" is detected.
    Generation stops after extracting a fixed number of answer tokens.

    Args:
        model: CODI model
        tokenizer: tokenizer
        question: input question string
        intervention_type: 'baseline' or 'projection'
        intervention_scope: 'numbers' or 'all'
        k: number of top embeddings to project onto (for projection intervention)
        max_length: maximum tokens to generate
        answer_tokens_to_generate: number of tokens to generate after "The answer is:"
        debug: if True, print detailed generation log

    Returns:
        dict with generation results and intervention statistics
    """

    # Pre-compute trigger sequences for "The answer is:"
    trigger_phrases = [
        "The answer is:",
        " The answer is:",
        "The answer is :",
        " The answer is :"
    ]
    trigger_sequences = [
        tokenizer.encode(phrase, add_special_tokens=False)
        for phrase in trigger_phrases
    ]
    max_trigger_length = max(len(seq) for seq in trigger_sequences)

    # Tokenize question
    inputs = tokenizer(question, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Initialize generation tracking
    generated_tokens = []
    token_window = []
    in_answer_phase = False
    answer_trigger_position = None
    answer_tokens_count = 0
    intervention_log = []

    # Get embedding layer
    embedding_layer = model.codi.get_input_embeddings()

    with torch.no_grad():
        past_key_values = None

        for step in range(max_length):
            # Forward pass
            if step == 0:
                # First step: use input_ids
                outputs = model.codi(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:
                # Subsequent steps: use embeddings
                current_embedding = embedding_layer(
                    torch.tensor([[next_token_id]], device=device)
                )
                outputs = model.codi(
                    inputs_embeds=current_embedding,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )

            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, hidden_dim]
            hidden_states = hidden_states.squeeze(1)  # [1, hidden_dim]

            # Get logits and predicted token
            logits = model.codi.lm_head(hidden_states)
            logits = logits[:, :tokenizer.vocab_size]  # Restrict to valid vocab
            next_token_id = torch.argmax(logits, dim=-1).item()

            # Decode token
            token_str = tokenizer.decode([next_token_id])
            is_number = bool(number_regex.match(token_str.strip()))

            # Update token window for trigger detection
            token_window.append(next_token_id)
            if len(token_window) > max_trigger_length:
                token_window.pop(0)

            # Check for answer trigger BEFORE appending
            if not in_answer_phase:
                if detect_answer_trigger(token_window, trigger_sequences):
                    in_answer_phase = True
                    answer_trigger_position = step
                    if debug:
                        print(f"  [Step {step}] ✓ Detected 'The answer is:' - stopping interventions")

            # Append generated token
            generated_tokens.append(next_token_id)

            # Determine if we should intervene
            should_intervene = (
                intervention_type == 'projection' and
                not in_answer_phase and
                (intervention_scope == 'all' or (intervention_scope == 'numbers' and is_number))
            )

            # Apply intervention if needed
            if should_intervene:
                hidden_states_modified = apply_teacher_projection_intervention(
                    hidden_states, next_token_id, embedding_layer, k
                )

                # Re-compute logits and token from modified hidden states
                logits_modified = model.codi.lm_head(hidden_states_modified)
                logits_modified = logits_modified[:, :tokenizer.vocab_size]
                next_token_id_modified = torch.argmax(logits_modified, dim=-1).item()

                # Log intervention
                intervention_log.append({
                    'step': step,
                    'original_token': token_str,
                    'original_token_id': next_token_id,
                    'modified_token': tokenizer.decode([next_token_id_modified]),
                    'modified_token_id': next_token_id_modified,
                    'is_number': is_number
                })

                # Use modified token for next iteration
                next_token_id = next_token_id_modified

                if debug:
                    print(f"  [Step {step}] Projection@{k}: '{token_str}' -> '{tokenizer.decode([next_token_id])}'")

            # Check stopping conditions
            if next_token_id == tokenizer.eos_token_id:
                if debug:
                    print(f"  [Step {step}] EOS token detected, stopping generation")
                break

            # Count tokens generated after answer trigger
            if in_answer_phase:
                answer_tokens_count += 1
                if answer_tokens_count > answer_tokens_to_generate:
                    if debug:
                        print(f"  [Step {step}] Generated {answer_tokens_to_generate} answer tokens, stopping")
                    break

            if step >= max_length - 1:
                if debug:
                    print(f"  [Step {step}] Max length reached, stopping generation")
                break

    # Decode full generated text
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extract numerical answer from tokens after "The answer is:"
    predicted_answer = None
    if answer_trigger_position is not None and in_answer_phase and len(generated_tokens) > answer_trigger_position:
        # Get tokens generated after trigger
        answer_start_idx = answer_trigger_position + 1
        answer_token_ids = generated_tokens[answer_start_idx:]
        answer_text = tokenizer.decode(answer_token_ids, skip_special_tokens=True).strip()

        # Extract first number from answer text
        answer_text_clean = answer_text.replace(',', '')
        numbers = re.findall(r'-?\d+\.?\d*', answer_text_clean)
        predicted_answer = float(numbers[0]) if numbers else None
    else:
        # Fallback: extract last number from full text
        text = generated_text.replace(',', '')
        numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
        predicted_answer = float(numbers[-1]) if numbers else None

    return {
        'generated_text': generated_text,
        'generated_tokens': generated_tokens,
        'predicted_answer': predicted_answer,
        'intervention_log': intervention_log,
        'answer_trigger_position': answer_trigger_position,
        'num_interventions': len(intervention_log),
        'answer_tokens_count': answer_tokens_count
    }


def main():
    print("="*80)
    print("Teacher Mode Top-K Projection Intervention Comparison")
    print("="*80)

    # Load model
    model, tokenizer = load_llama_model()

    # Load test dataset
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)

    # For testing: use first 5 examples
    # For full run: use all 132 examples
    num_test = len(clean_dataset)  # Full run on all examples
    test_examples = clean_dataset[:num_test]
    print(f"\n✓ Loaded {num_test} test examples")

    # Define k values to test
    K_VALUES = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

    # Define intervention conditions
    intervention_conditions = [
        ('baseline', 'none', None),  # Baseline (no intervention)
    ]

    # Add projection@k for each k value (numbers only)
    for k in K_VALUES:
        intervention_conditions.append(('projection', 'numbers', k))

    print(f"\n✓ Testing {len(intervention_conditions)} intervention conditions")
    print(f"   K values: {K_VALUES}")

    # Run experiments
    all_results = []

    for condition_idx, (interv_type, interv_scope, k_val) in enumerate(intervention_conditions):
        condition_name = f"{interv_type}@{k_val}" if k_val is not None else interv_type
        print(f"\n{'-'*80}")
        print(f"[{condition_idx+1}/{len(intervention_conditions)}] Testing: {condition_name} ({interv_scope})")
        print(f"{'-'*80}")

        condition_results = []

        for ex_idx, example in enumerate(test_examples):
            question = example['question']
            ground_truth = float(example['answer'])

            # Run generation with intervention
            result = generate_teacher_cot_with_projection(
                model, tokenizer, question,
                intervention_type=interv_type,
                intervention_scope=interv_scope,
                k=k_val if k_val is not None else 1,
                answer_tokens_to_generate=2,
                debug=False
            )

            correct = (result['predicted_answer'] == ground_truth) if result['predicted_answer'] is not None else False

            print(f"  [{ex_idx+1}/{num_test}] Predicted: {result['predicted_answer']}, "
                  f"GT: {ground_truth}, Correct: {correct}, "
                  f"Interventions: {result['num_interventions']}")

            condition_results.append({
                'example_idx': ex_idx,
                'question': question,
                'ground_truth': ground_truth,
                'predicted_answer': result['predicted_answer'],
                'generated_text': result['generated_text'],
                'correct': correct,
                'num_interventions': result['num_interventions'],
                'answer_trigger_position': result['answer_trigger_position'],
                'intervention_log': result['intervention_log']
            })

        # Compute accuracy for this condition
        accuracy = sum(r['correct'] for r in condition_results) / len(condition_results) * 100
        avg_interventions = np.mean([r['num_interventions'] for r in condition_results])
        print(f"\n  → Accuracy: {accuracy:.1f}% ({sum(r['correct'] for r in condition_results)}/{len(condition_results)})")
        print(f"  → Avg interventions: {avg_interventions:.1f}")

        all_results.append({
            'intervention_type': interv_type,
            'intervention_scope': interv_scope,
            'k': k_val,
            'results': condition_results,
            'accuracy': accuracy,
            'avg_interventions': avg_interventions
        })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'teacher_projection_results_{num_test}ex_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'num_examples': num_test,
                'dataset': 'clean',
                'num_conditions': len(intervention_conditions),
                'k_values': K_VALUES,
                'model': 'CODI-LLaMA-3.2-1B',
                'mode': 'teacher (explicit CoT)'
            },
            'conditions': all_results
        }, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to {results_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Teacher Mode Top-K Projection Intervention")
    print(f"{'='*80}")
    print(f"{'Intervention':<20s} {'Scope':<10s} {'Accuracy':<10s} {'Avg Interventions':<18s}")
    print(f"{'-'*80}")

    for result in all_results:
        if result['k'] is not None:
            interv_name = f"projection@{result['k']}"
        else:
            interv_name = result['intervention_type']

        print(f"{interv_name:<20s} "
              f"{result['intervention_scope']:<10s} "
              f"{result['accuracy']:>6.1f}%    "
              f"{result['avg_interventions']:>6.1f}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
