"""
CODI Intervention with Average Activations
- Compute average hidden states per position from GSM8K train set
- Intervene by replacing with average activation (works for ALL tokens, not just numbers)
- More realistic perturbation test
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


def compute_average_activations(model, tokenizer, training_args, num_examples=100):
    """
    Compute average hidden states at each CoT position from training examples
    """
    print(f"\n{'='*80}")
    print(f"Computing Average Activations from {num_examples} Training Examples")
    print(f"{'='*80}\n")

    # Load GSM8K training set
    dataset = load_dataset("gsm8k", "main", split="train")
    dataset = dataset.select(range(num_examples))

    num_positions = 7
    hidden_dim = 2048

    # Accumulate hidden states
    hidden_states_sum = [torch.zeros(hidden_dim, device=device, dtype=torch.float32) for _ in range(num_positions)]
    counts = [0] * num_positions

    with torch.no_grad():
        for example in tqdm(dataset, desc="Processing training examples"):
            question = example['question']

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

            # Position 0
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

            hidden_states_sum[0] += latent_embd.squeeze().float()
            counts[0] += 1

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Positions 1-6
            for i in range(training_args.inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1:, :]

                hidden_states_sum[i + 1] += latent_embd.squeeze().float()
                counts[i + 1] += 1

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

    # Compute averages
    average_activations = []
    for pos in range(num_positions):
        avg = hidden_states_sum[pos] / counts[pos]
        average_activations.append(avg)
        print(f"Position {pos}: Averaged {counts[pos]} examples, norm = {torch.norm(avg).item():.2f}")

    return average_activations


def run_cot_with_average_intervention(model, tokenizer, training_args, question,
                                      average_activations, intervention_position=-1, k=1.0):
    """
    Run CoT with intervention: replace hidden state with average activation
    Works for ALL positions (not just numbers)
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

    result = {
        'intervention_position': intervention_position,
        'positions': []
    }

    with torch.no_grad():
        # Position 0
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

        clean_hidden = latent_embd.clone()

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])

        pos_data = {
            'position': 0,
            'token': token_str,
            'token_id': token_id,
            'intervened': False,
            'hidden_state_norm': torch.norm(clean_hidden).item(),
        }

        # Apply intervention if this is the target position
        if intervention_position == 0:
            # Replace with average activation
            avg_activation = average_activations[0].unsqueeze(0).unsqueeze(0).to(latent_embd.dtype)

            # Optionally scale the replacement
            latent_embd = k * avg_activation

            logits_modified = model.codi.lm_head(latent_embd.squeeze(1))
            new_token_id = torch.argmax(logits_modified, dim=-1).item()
            new_token_str = tokenizer.decode([new_token_id])

            pos_data['intervened'] = True
            pos_data['intervened_token'] = new_token_str
            pos_data['intervened_token_id'] = new_token_id
            pos_data['intervention_magnitude'] = torch.norm(latent_embd - clean_hidden).item()
            pos_data['cosine_similarity'] = torch.nn.functional.cosine_similarity(
                clean_hidden.squeeze(1).float(), latent_embd.squeeze(1).float(), dim=-1
            ).item()

        result['positions'].append(pos_data)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Positions 1-6
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

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'token_id': token_id,
                'intervened': False,
                'hidden_state_norm': torch.norm(clean_hidden).item(),
            }

            # Apply intervention if this is the target position
            if intervention_position == (i + 1):
                avg_activation = average_activations[i + 1].unsqueeze(0).unsqueeze(0).to(latent_embd.dtype)

                latent_embd = k * avg_activation

                logits_modified = model.codi.lm_head(latent_embd.squeeze(1))
                new_token_id = torch.argmax(logits_modified, dim=-1).item()
                new_token_str = tokenizer.decode([new_token_id])

                pos_data['intervened'] = True
                pos_data['intervened_token'] = new_token_str
                pos_data['intervened_token_id'] = new_token_id
                pos_data['intervention_magnitude'] = torch.norm(latent_embd - clean_hidden).item()
                pos_data['cosine_similarity'] = torch.nn.functional.cosine_similarity(
                    clean_hidden.squeeze(1).float(), latent_embd.squeeze(1).float(), dim=-1
                ).item()

            result['positions'].append(pos_data)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return result


def analyze_multiple_examples_with_avg_intervention(model, tokenizer, training_args,
                                                     average_activations, num_test_examples=20, k=1.0):
    """
    Run intervention analysis on test examples using average activations
    """
    print(f"\n{'='*80}")
    print(f"Running Average Activation Intervention Analysis")
    print(f"Test examples: {num_test_examples}")
    print(f"{'='*80}\n")

    # Load GSM8K test set
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(num_test_examples))

    all_results = []

    for example_idx in tqdm(range(len(dataset)), desc="Processing test examples"):
        question = dataset[example_idx]['question']

        example_results = {
            'example_id': example_idx,
            'question': question,
            'interventions': []
        }

        # Run baseline
        baseline = run_cot_with_average_intervention(
            model, tokenizer, training_args, question, average_activations, -1, k
        )

        # Run interventions at each position (0-6, ALL positions now!)
        for int_pos in range(7):
            intervened = run_cot_with_average_intervention(
                model, tokenizer, training_args, question, average_activations, int_pos, k
            )

            # Compute cascade metrics
            cascade = compute_activation_cascade(baseline, intervened)
            example_results['interventions'].append(cascade)

        all_results.append(example_results)

    return all_results


def compute_activation_cascade(baseline_run, intervened_run):
    """Compute cascade metrics comparing baseline to intervened run"""
    num_positions = len(baseline_run['positions'])

    cascade_metrics = {
        'intervention_position': intervened_run['intervention_position'],
        'position_metrics': []
    }

    for pos in range(num_positions):
        baseline_pos = baseline_run['positions'][pos]
        intervened_pos = intervened_run['positions'][pos]

        token_changed = baseline_pos['token'] != intervened_pos['token']

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

        if intervened_pos.get('intervened', False):
            metrics['intervention_magnitude'] = intervened_pos.get('intervention_magnitude', 0)
            metrics['cosine_similarity'] = intervened_pos.get('cosine_similarity', 1.0)

        cascade_metrics['position_metrics'].append(metrics)

    return cascade_metrics


def aggregate_cascade_statistics(all_results):
    """Aggregate statistics across all examples"""
    num_positions = 7

    aggregated = {
        'intervention_positions': list(range(num_positions)),
        'metrics': {}
    }

    for int_pos in range(num_positions):
        aggregated['metrics'][int_pos] = {
            'position_effects': [[] for _ in range(num_positions)]
        }

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

                pos_stats['downstream_effects'].append({
                    'target_position': target_pos,
                    'token_change_rate': token_change_rate,
                    'avg_hidden_norm_difference': avg_norm_diff,
                    'std_hidden_norm_difference': std_norm_diff,
                    'num_examples': len(effects),
                })

        statistics['position_statistics'].append(pos_stats)

    return statistics


def print_statistics_summary(statistics):
    """Print summary of aggregated statistics"""
    print("\n" + "="*80)
    print("AVERAGE ACTIVATION INTERVENTION CASCADE STATISTICS")
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

    # Step 1: Compute average activations from training set
    average_activations = compute_average_activations(
        model, tokenizer, training_args, num_examples=100
    )

    # Save average activations
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    avg_acts_cpu = [act.cpu().numpy().tolist() for act in average_activations]
    with open(output_dir / "average_activations_train100.json", 'w') as f:
        json.dump({'average_activations': avg_acts_cpu}, f)
    print(f"\n✓ Average activations saved")

    # Step 2: Run intervention analysis on test set
    all_results = analyze_multiple_examples_with_avg_intervention(
        model, tokenizer, training_args, average_activations,
        num_test_examples=20, k=1.0
    )

    # Step 3: Aggregate statistics
    print("\nAggregating statistics...")
    statistics = aggregate_cascade_statistics(all_results)

    # Save results
    with open(output_dir / "avg_intervention_cascade_raw.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Raw results saved")

    with open(output_dir / "avg_intervention_cascade_statistics.json", 'w') as f:
        json.dump(statistics, f, indent=2)
    print(f"✓ Statistics saved")

    # Print summary
    print_statistics_summary(statistics)

    print("\n✓ Average activation intervention analysis complete!")


if __name__ == "__main__":
    main()
