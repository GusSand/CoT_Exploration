"""
Enhanced CODI Intervention Cascade Analysis
- Continuous activation change measurements (not just token identity)
- Multiple examples for statistical robustness
"""

import torch
import sys
import re
import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
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


def run_cot_with_activation_tracking(model, tokenizer, training_args, question,
                                     intervention_position=-1, target_token='5', k=3):
    """
    Run CoT with optional intervention, tracking ALL hidden states
    Returns continuous activation measurements
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

    result = {
        'intervention_position': intervention_position,
        'positions': []
    }

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

        # Store clean hidden state
        clean_hidden = latent_embd.clone()

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        pos_data = {
            'position': 0,
            'token': token_str,
            'token_id': token_id,
            'is_number': is_number,
            'intervened': False,
            'hidden_state_norm': torch.norm(clean_hidden).item(),
        }

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

            pos_data['intervened'] = True
            pos_data['intervened_token'] = new_token_str
            pos_data['intervened_token_id'] = new_token_id

            # Continuous measurements
            pos_data['intervention_magnitude'] = torch.norm(A_modified - A).item()
            pos_data['hidden_state_change'] = torch.norm(latent_embd - clean_hidden).item()
            pos_data['cosine_similarity'] = torch.nn.functional.cosine_similarity(
                clean_hidden.squeeze(1), latent_embd.squeeze(1), dim=-1
            ).item()

        result['positions'].append(pos_data)

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

            clean_hidden = latent_embd.clone()

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])
            is_number = bool(number_regex.match(token_str))

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'token_id': token_id,
                'is_number': is_number,
                'intervened': False,
                'hidden_state_norm': torch.norm(clean_hidden).item(),
            }

            # Apply intervention if this is the target position
            if intervention_position == (i + 1) and is_number:
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

                pos_data['intervened'] = True
                pos_data['intervened_token'] = new_token_str
                pos_data['intervened_token_id'] = new_token_id

                # Continuous measurements
                pos_data['intervention_magnitude'] = torch.norm(A_modified - A).item()
                pos_data['hidden_state_change'] = torch.norm(latent_embd - clean_hidden).item()
                pos_data['cosine_similarity'] = torch.nn.functional.cosine_similarity(
                    clean_hidden.squeeze(1), latent_embd.squeeze(1), dim=-1
                ).item()

            result['positions'].append(pos_data)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return result


def compute_activation_cascade(baseline_run, intervened_run):
    """
    Compute continuous measures of how intervention cascades through positions

    Returns:
    - activation_perturbations: For each position, how different are the hidden states?
    - token_changes: Binary indicator if token changed
    """
    num_positions = len(baseline_run['positions'])

    cascade_metrics = {
        'intervention_position': intervened_run['intervention_position'],
        'position_metrics': []
    }

    for pos in range(num_positions):
        baseline_pos = baseline_run['positions'][pos]
        intervened_pos = intervened_run['positions'][pos]

        # Token-level comparison
        token_changed = baseline_pos['token'] != intervened_pos['token']

        # Activation-level comparison
        # We can't directly compare hidden states between runs, but we can compare norms
        baseline_norm = baseline_pos['hidden_state_norm']
        intervened_norm = intervened_pos['hidden_state_norm']
        norm_difference = abs(intervened_norm - baseline_norm)
        norm_ratio = intervened_norm / baseline_norm if baseline_norm > 0 else 1.0

        metrics = {
            'position': pos,
            'baseline_token': baseline_pos['token'],
            'intervened_token': intervened_pos['token'],
            'token_changed': token_changed,
            'baseline_hidden_norm': baseline_norm,
            'intervened_hidden_norm': intervened_norm,
            'hidden_norm_difference': norm_difference,
            'hidden_norm_ratio': norm_ratio,
        }

        # If this position was intervened, include intervention stats
        if intervened_pos.get('intervened', False):
            metrics['intervention_magnitude'] = intervened_pos.get('intervention_magnitude', 0)
            metrics['cosine_similarity'] = intervened_pos.get('cosine_similarity', 1.0)

        cascade_metrics['position_metrics'].append(metrics)

    return cascade_metrics


def analyze_multiple_examples(model, tokenizer, training_args, num_examples=20, target_token='5', k=3):
    """
    Run intervention analysis on multiple examples
    """
    print(f"\n{'='*80}")
    print(f"Enhanced Intervention Cascade Analysis")
    print(f"Running on {num_examples} examples")
    print(f"{'='*80}\n")

    # Load GSM8K test set
    print("Loading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(num_examples))
    print(f"✓ Loaded {len(dataset)} examples\n")

    all_results = []

    for example_idx in tqdm(range(len(dataset)), desc="Processing examples"):
        question = dataset[example_idx]['question']

        example_results = {
            'example_id': example_idx,
            'question': question,
            'interventions': []
        }

        # Run baseline
        baseline = run_cot_with_activation_tracking(
            model, tokenizer, training_args, question, -1, target_token, k
        )

        # Run interventions at each position
        for int_pos in range(7):
            intervened = run_cot_with_activation_tracking(
                model, tokenizer, training_args, question, int_pos, target_token, k
            )

            # Compute cascade metrics
            cascade = compute_activation_cascade(baseline, intervened)
            example_results['interventions'].append(cascade)

        all_results.append(example_results)

    return all_results


def aggregate_cascade_statistics(all_results):
    """
    Aggregate cascade metrics across all examples
    """
    num_positions = 7

    # Initialize aggregation structures
    aggregated = {
        'intervention_positions': list(range(num_positions)),
        'metrics': {}
    }

    for int_pos in range(num_positions):
        aggregated['metrics'][int_pos] = {
            'position_effects': [[] for _ in range(num_positions)]
        }

    # Aggregate across examples
    for example in all_results:
        for intervention in example['interventions']:
            int_pos = intervention['intervention_position']

            for pos_metric in intervention['position_metrics']:
                pos = pos_metric['position']

                aggregated['metrics'][int_pos]['position_effects'][pos].append({
                    'token_changed': pos_metric['token_changed'],
                    'norm_difference': pos_metric['hidden_norm_difference'],
                    'norm_ratio': pos_metric['hidden_norm_ratio'],
                })

    # Compute statistics
    statistics = {
        'intervention_positions': list(range(num_positions)),
        'position_statistics': []
    }

    for int_pos in range(num_positions):
        pos_stats = {'intervention_position': int_pos, 'downstream_effects': []}

        for target_pos in range(num_positions):
            effects = aggregated['metrics'][int_pos]['position_effects'][target_pos]

            if len(effects) > 0:
                token_change_rate = np.mean([e['token_changed'] for e in effects])
                avg_norm_diff = np.mean([e['norm_difference'] for e in effects])
                std_norm_diff = np.std([e['norm_difference'] for e in effects])
                avg_norm_ratio = np.mean([e['norm_ratio'] for e in effects])

                pos_stats['downstream_effects'].append({
                    'target_position': target_pos,
                    'token_change_rate': token_change_rate,
                    'avg_hidden_norm_difference': avg_norm_diff,
                    'std_hidden_norm_difference': std_norm_diff,
                    'avg_hidden_norm_ratio': avg_norm_ratio,
                    'num_examples': len(effects),
                })

        statistics['position_statistics'].append(pos_stats)

    return statistics


def print_statistics_summary(statistics):
    """Print summary of aggregated statistics"""
    print("\n" + "="*80)
    print("AGGREGATED INTERVENTION CASCADE STATISTICS")
    print("="*80)

    for int_stat in statistics['position_statistics']:
        int_pos = int_stat['intervention_position']

        print(f"\nIntervention at Position {int_pos}:")
        print("-" * 80)
        print(f"{'Target Pos':<12} {'Token Change %':<16} {'Avg Norm Δ':<15} {'Std Norm Δ':<15}")
        print("-" * 80)

        for effect in int_stat['downstream_effects']:
            target_pos = effect['target_position']
            token_rate = effect['token_change_rate'] * 100
            avg_diff = effect['avg_hidden_norm_difference']
            std_diff = effect['std_hidden_norm_difference']

            print(f"{target_pos:<12} {token_rate:>14.1f}% {avg_diff:>14.2f} {std_diff:>14.2f}")

    print("="*80)


def main():
    model, tokenizer, training_args = load_llama_model()
    print("✓ Model loaded\n")

    # Run analysis on 20 examples
    all_results = analyze_multiple_examples(
        model, tokenizer, training_args,
        num_examples=20, target_token='5', k=3
    )

    # Aggregate statistics
    print("\nAggregating statistics across examples...")
    statistics = aggregate_cascade_statistics(all_results)

    # Save results
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "enhanced_cascade_raw.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Raw results saved to {output_dir / 'enhanced_cascade_raw.json'}")

    with open(output_dir / "enhanced_cascade_statistics.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Statistics saved to {output_dir / 'enhanced_cascade_statistics.json'}")

    # Print summary
    print_statistics_summary(statistics)

    print("\n✓ Enhanced cascade analysis complete!")


if __name__ == "__main__":
    main()
