#!/usr/bin/env python3
"""
Teacher Mode Intervention Comparison with Plus-One Intervention

Adapts intervention comparison for TEACHER MODE (explicit CoT generation).
Key difference: Interventions stop after "The answer is:" is generated.

Intervention types tested:
1. Baseline (no intervention)
2. Discretize (replace with embedding of top-1 token)
3. Discretize+1 (replace with embedding of top-1 token + 1)

Scopes:
- Numbers only (intervene at number positions)
- All positions (intervene at all CoT positions)

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
RESULTS_DIR = Path('./teacher_mode_intervention_results')
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


def get_plusone_token_id(tokenizer, token_id):
    """
    Get token ID for number+1, preserving format.
    Returns None if token is not a number or encoding fails.
    """
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
    if len(encoded) >= 1:
        return encoded[0]
    else:
        return None


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


def apply_teacher_intervention(hidden_states, token_id, embedding_layer,
                               intervention_type, tokenizer):
    """
    Apply intervention to hidden states in teacher mode.

    Args:
        hidden_states: [1, hidden_dim] tensor
        token_id: predicted token ID
        embedding_layer: model's token embedding layer
        intervention_type: 'discretize' or 'discretize_plusone'
        tokenizer: tokenizer for plus-one lookup

    Returns:
        Modified hidden states [1, hidden_dim]
    """
    if intervention_type == 'discretize':
        # Replace with embedding of predicted token (L2-normalized)
        predicted_embd = embedding_layer(torch.tensor([token_id], device=hidden_states.device))
        A_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
        E_norm = torch.norm(predicted_embd, dim=-1, keepdim=True)
        return predicted_embd * (A_norm / (E_norm + 1e-8))

    elif intervention_type == 'discretize_plusone':
        # Replace with embedding of predicted token + 1 (L2-normalized)
        plusone_token_id = get_plusone_token_id(tokenizer, token_id)
        if plusone_token_id is not None:
            plusone_embd = embedding_layer(torch.tensor([plusone_token_id], device=hidden_states.device))
            A_norm = torch.norm(hidden_states, dim=-1, keepdim=True)
            E_norm = torch.norm(plusone_embd, dim=-1, keepdim=True)
            return plusone_embd * (A_norm / (E_norm + 1e-8))
        else:
            # If plus-one fails, return original
            return hidden_states

    else:
        raise ValueError(f"Unknown intervention type: {intervention_type}")


def generate_teacher_cot_with_intervention(
    model, tokenizer, question,
    intervention_type='baseline',
    intervention_scope='numbers',
    max_length=300,
    answer_tokens_to_generate=2,
    debug=False
):
    """
    Generate explicit CoT in teacher mode with optional intervention.
    Interventions stop after "The answer is:" is detected.
    Generation stops after extracting a fixed number of answer tokens.

    Args:
        model: CODI model
        tokenizer: tokenizer
        question: input question string
        intervention_type: 'baseline', 'discretize', or 'discretize_plusone'
        intervention_scope: 'numbers' or 'all'
        max_length: maximum tokens to generate
        answer_tokens_to_generate: number of tokens to generate after "The answer is:"
        debug: if True, print detailed generation log

    Returns:
        dict with:
            - 'generated_text': full generated text
            - 'generated_tokens': list of token IDs
            - 'predicted_answer': extracted numerical answer
            - 'intervention_log': list of intervention events
            - 'answer_trigger_position': position where "The answer is:" was detected
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
                intervention_type != 'baseline' and
                not in_answer_phase and
                (intervention_scope == 'all' or (intervention_scope == 'numbers' and is_number))
            )

            # Apply intervention if needed
            if should_intervene:
                hidden_states_modified = apply_teacher_intervention(
                    hidden_states, next_token_id, embedding_layer,
                    intervention_type, tokenizer
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

                # Use modified token
                next_token_id = next_token_id_modified

                if debug:
                    print(f"  [Step {step}] Intervention: '{token_str}' -> '{tokenizer.decode([next_token_id])}'")

            # Check stopping conditions
            if next_token_id == tokenizer.eos_token_id:
                if debug:
                    print(f"  [Step {step}] EOS token detected, stopping generation")
                break

            # Count tokens generated after answer trigger (for early stopping)
            # Do this AFTER appending token but BEFORE checking if we should stop
            if in_answer_phase:
                answer_tokens_count += 1
                # Stop after generating N tokens post-answer
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

    # Extract numerical answer from first few tokens after "The answer is:"
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
    print("Teacher Mode Intervention Comparison with Plus-One")
    print("="*80)

    # Load model
    model, tokenizer = load_llama_model()

    # Load test dataset (all examples from clean dataset)
    clean_data_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
    with open(clean_data_path, 'r') as f:
        clean_dataset = json.load(f)

    num_test = len(clean_dataset)
    test_examples = clean_dataset
    print(f"\n✓ Loaded {num_test} test examples")

    # Define intervention conditions
    intervention_conditions = [
        ('baseline', 'none'),
        ('discretize', 'numbers'),
        ('discretize', 'all'),
        ('discretize_plusone', 'numbers'),
        ('discretize_plusone', 'all'),
    ]

    print(f"\n✓ Testing {len(intervention_conditions)} intervention conditions")

    # Run experiments
    all_results = []

    for condition_idx, (interv_type, interv_scope) in enumerate(intervention_conditions):
        print(f"\n{'-'*80}")
        print(f"[{condition_idx+1}/{len(intervention_conditions)}] Testing: {interv_type} ({interv_scope})")
        print(f"{'-'*80}")

        condition_results = []

        for ex_idx, example in enumerate(test_examples):
            question = example['question']
            ground_truth = float(example['answer'])

            print(f"\n  Example {ex_idx+1}/{num_test}: {question[:60]}...")

            # Run generation with intervention
            result = generate_teacher_cot_with_intervention(
                model, tokenizer, question,
                intervention_type=interv_type,
                intervention_scope=interv_scope,
                answer_tokens_to_generate=2,
                debug=False  # Disable debug for full run
            )

            correct = (result['predicted_answer'] == ground_truth) if result['predicted_answer'] is not None else False

            print(f"    Predicted: {result['predicted_answer']}, Ground truth: {ground_truth}, Correct: {correct}")
            print(f"    Interventions applied: {result['num_interventions']}")
            if result['answer_trigger_position'] is not None:
                print(f"    Answer trigger detected at step: {result['answer_trigger_position']}")
                print(f"    Answer tokens generated: {result['answer_tokens_count']}")

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
        print(f"\n  → Accuracy: {accuracy:.1f}% ({sum(r['correct'] for r in condition_results)}/{len(condition_results)})")

        all_results.append({
            'intervention_type': interv_type,
            'intervention_scope': interv_scope,
            'results': condition_results,
            'accuracy': accuracy
        })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'teacher_mode_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'num_examples': num_test,
                'dataset': 'clean',
                'num_conditions': len(intervention_conditions),
                'model': 'CODI-LLaMA-3.2-1B',
                'mode': 'teacher (explicit CoT)'
            },
            'conditions': all_results
        }, f, indent=2)

    print(f"\n{('='*80)}")
    print(f"✓ Results saved to {results_file}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Teacher Mode Intervention Comparison")
    print(f"{'='*80}")
    print(f"{'Intervention':<20s} {'Scope':<10s} {'Accuracy':<10s} {'Avg Interventions':<18s}")
    print(f"{'-'*80}")

    for result in all_results:
        avg_interventions = np.mean([r['num_interventions'] for r in result['results']])
        print(f"{result['intervention_type']:<20s} "
              f"{result['intervention_scope']:<10s} "
              f"{result['accuracy']:>6.1f}%    "
              f"{avg_interventions:>6.1f}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
