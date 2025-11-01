"""
CODI Intervention Propagation Analysis
Track how interventions at different CoT positions cascade through reasoning
"""

import torch
import sys
import re
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
number_regex = re.compile(r'^\s?\d+')

def load_llama_model():
    """Load CODI-LLaMA model"""
    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/CoT_Exploration/models/CODI-llama3.2-1b",
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
    llama_model.codi.config._attn_implementation = 'eager'

    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    return llama_model, llama_tokenizer, llama_training_args


def run_cot_with_intervention(model, tokenizer, training_args, question,
                               intervention_position=-1, target_token='5', k=3):
    """
    Run CoT with optional intervention at a specific position
    intervention_position=-1 means no intervention (baseline)
    """
    batch_size = 1
    questions = [question]

    # Get target embedding
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    embedding_layer = model.codi.get_input_embeddings()
    target_embd = embedding_layer(torch.tensor([target_token_id], device=device))

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer(questions, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    result = {'intervention_position': intervention_position, 'tokens': []}

    with torch.no_grad():
        # Initial encoding (position 0)
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

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        token_data = {'position': 0, 'token': token_str, 'is_number': is_number, 'intervened': False}

        # Apply intervention if this is the target position
        if intervention_position == 0 and is_number:
            predicted_embd = embedding_layer(torch.tensor([token_id], device=device))
            A = latent_embd.squeeze(1)

            E_pred_norm = predicted_embd / torch.norm(predicted_embd, dim=-1, keepdim=True)
            E_target_norm = target_embd / torch.norm(target_embd, dim=-1, keepdim=True)

            proj_predicted = torch.sum(A * E_pred_norm, dim=-1, keepdim=True) * E_pred_norm
            proj_target = torch.norm(proj_predicted, dim=-1, keepdim=True) * E_target_norm
            A_modified = A - proj_predicted + k * proj_target

            latent_embd = A_modified.unsqueeze(1)

            logits_modified = model.codi.lm_head(A_modified)
            new_token_id = torch.argmax(logits_modified, dim=-1).item()
            new_token_str = tokenizer.decode([new_token_id])

            token_data['token'] = new_token_str
            token_data['intervened'] = True
            token_data['intervention_magnitude'] = torch.norm(A_modified - A).item()

        result['tokens'].append(token_data)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Chain-of-Thought iterations
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

            token_data = {'position': i+1, 'token': token_str, 'is_number': is_number, 'intervened': False}

            # Apply intervention if this is the target position
            if intervention_position == (i+1) and is_number:
                predicted_embd = embedding_layer(torch.tensor([token_id], device=device))
                A = latent_embd.squeeze(1)

                E_pred_norm = predicted_embd / torch.norm(predicted_embd, dim=-1, keepdim=True)
                E_target_norm = target_embd / torch.norm(target_embd, dim=-1, keepdim=True)

                proj_predicted = torch.sum(A * E_pred_norm, dim=-1, keepdim=True) * E_pred_norm
                proj_target = torch.sum(A * E_target_norm, dim=-1, keepdim=True) * E_target_norm
                A_modified = A - proj_predicted + k * proj_target

                latent_embd = A_modified.unsqueeze(1)

                logits_modified = model.codi.lm_head(A_modified)
                new_token_id = torch.argmax(logits_modified, dim=-1).item()
                new_token_str = tokenizer.decode([new_token_id])

                token_data['token'] = new_token_str
                token_data['intervened'] = True
                token_data['intervention_magnitude'] = torch.norm(A_modified - A).item()

            result['tokens'].append(token_data)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return result


def analyze_intervention_propagation(model, tokenizer, training_args, question, target_token='5', k=3):
    """
    Compare what happens when we intervene at different CoT positions
    """
    print(f"\nAnalyzing intervention propagation:")
    print(f"Target token: {target_token}, k={k}\n")

    results = []

    # Baseline: No intervention
    print("Running baseline (no intervention)...")
    baseline = run_cot_with_intervention(model, tokenizer, training_args, question, -1, target_token, k)
    results.append(baseline)
    baseline_tokens = [t['token'] for t in baseline['tokens']]
    print(f"Baseline: {' → '.join(baseline_tokens)}")

    # Intervene at each position
    for pos in range(7):  # 0-6
        print(f"Intervening at position {pos}...")
        intervened = run_cot_with_intervention(model, tokenizer, training_args, question, pos, target_token, k)
        results.append(intervened)
        intervened_tokens = [t['token'] for t in intervened['tokens']]
        print(f"  Result: {' → '.join(intervened_tokens)}")

    return results


def print_propagation_summary(results, target_token):
    """Print summary of intervention propagation"""
    print("\n" + "="*80)
    print("INTERVENTION PROPAGATION SUMMARY")
    print("="*80)

    baseline_tokens = [t['token'] for t in results[0]['tokens']]

    print("\n" + "-"*80)
    print("Int Pos | Token Sequence                              | Changes")
    print("-"*80)

    for result in results:
        int_pos = result['intervention_position']
        tokens = [t['token'] for t in result['tokens']]
        tokens_str = ' → '.join(tokens)

        # Count changes from baseline
        changes = sum(1 for i, t in enumerate(tokens) if t != baseline_tokens[i])

        pos_label = f"Baseline" if int_pos == -1 else f"Pos {int_pos}"
        print(f"{pos_label:7} | {tokens_str:43} | {changes} tokens changed")

    print("-"*80)

    # Analyze cascade effects
    print("\n" + "="*80)
    print("CASCADE ANALYSIS: How far do interventions propagate?")
    print("="*80)

    for result in results[1:]:  # Skip baseline
        int_pos = result['intervention_position']
        tokens = [t['token'] for t in result['tokens']]

        # Find how many downstream positions were affected
        affected_positions = []
        for i in range(int_pos, len(tokens)):
            if tokens[i] != baseline_tokens[i]:
                affected_positions.append(i)

        if len(affected_positions) > 0:
            cascade_length = max(affected_positions) - int_pos + 1
            print(f"Intervention at Pos {int_pos}: Affected {len(affected_positions)} positions, cascade length = {cascade_length}")
            print(f"  Affected positions: {affected_positions}")
        else:
            print(f"Intervention at Pos {int_pos}: No effect!")

    print("="*80)


def main():
    print("="*80)
    print("CODI INTERVENTION PROPAGATION ANALYSIS")
    print("="*80)

    model, tokenizer, training_args = load_llama_model()
    print("✓ Model loaded")

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    results = analyze_intervention_propagation(model, tokenizer, training_args, question, target_token='5', k=3)

    # Save results
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "intervention_propagation.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {output_dir / 'intervention_propagation.json'}")

    # Print summary
    print_propagation_summary(results, target_token='5')

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
