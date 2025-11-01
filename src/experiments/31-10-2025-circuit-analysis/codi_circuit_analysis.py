"""
CODI Chain-of-Thought Circuit Analysis
Inspired by Indirect Object Identification circuit analysis techniques

This script analyzes:
1. Intervention propagation through CoT positions (PRIMARY)
2. Attention patterns across CoT (SECONDARY)
3. Direct projection vs attention contribution (TERTIARY)
"""

import torch
import sys
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# Load environment variables and login to HuggingFace
load_dotenv()
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

number_regex = re.compile(r'^\s?\d+')

def extract_answer_number(text):
    """Extract numerical answer from generated text"""
    text = text.replace(',', '')
    numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    if not numbers:
        return None
    return float(numbers[-1])

def load_llama_model():
    """Load CODI-LLaMA model"""
    print("="*80)
    print("Loading CODI-LLaMA from Local Checkpoint")
    print("="*80)

    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B", attn_implementation="eager",
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
    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    return llama_model, llama_tokenizer, llama_training_args


def analyze_intervention_propagation(model, tokenizer, training_args, question,
                                     intervention_position=0, target_token='5', k=3):
    """
    PRIMARY ANALYSIS: Track how interventions propagate through CoT

    This implements a "causal activation patching" approach:
    - Intervene at a specific CoT position
    - Track how this affects all downstream positions
    - Measure the "cascade effect" of the intervention
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

    # Storage for analysis
    analysis_data = {
        'intervention_position': intervention_position,
        'target_token': target_token,
        'k': k,
        'positions': []
    }

    with torch.no_grad():
        # Initial encoding (position 0: BoT)
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,  # Get attention patterns!
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]
        attentions = outputs.attentions  # Tuple of attention weights per layer

        # Store clean (pre-intervention) hidden state
        clean_latent = latent_embd.clone()

        # Decode and potentially intervene at BoT (position 0)
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        position_data = {
            'position': 0,
            'clean_token': token_str,
            'clean_token_id': token_id,
            'is_number': is_number,
            'intervened': False,
            'clean_hidden_state_norm': torch.norm(clean_latent).item(),
            'attention_patterns': {}
        }

        # Extract attention patterns for the last token across all layers
        for layer_idx, attn in enumerate(attentions):
            # attn shape: [batch, num_heads, seq_len, seq_len]
            # We want attention FROM the last token (BoT) TO all previous tokens
            attn_from_last = attn[0, :, -1, :].cpu().numpy()  # [num_heads, seq_len]
            position_data['attention_patterns'][f'layer_{layer_idx}'] = {
                'mean_attn': attn_from_last.mean(axis=0).tolist(),  # Average across heads
                'per_head': attn_from_last.tolist()
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

            position_data['intervened'] = True
            position_data['intervened_token'] = new_token_str
            position_data['intervened_token_id'] = new_token_id
            position_data['intervention_magnitude'] = torch.norm(A_modified - A).item()
            position_data['modified_hidden_state_norm'] = torch.norm(A_modified).item()

            # Calculate cosine similarity between clean and modified hidden states
            cos_sim = torch.nn.functional.cosine_similarity(
                clean_latent.squeeze(1), A_modified, dim=-1
            ).item()
            position_data['cosine_similarity_clean_modified'] = cos_sim

        analysis_data['positions'].append(position_data)

        if training_args.use_prj:
            # Store pre-projection state
            pre_prj_latent = latent_embd.clone()
            latent_embd = model.prj(latent_embd)

            # Analyze projection contribution
            position_data['projection_applied'] = True
            position_data['pre_projection_norm'] = torch.norm(pre_prj_latent).item()
            position_data['post_projection_norm'] = torch.norm(latent_embd).item()
            position_data['projection_change'] = torch.norm(latent_embd - pre_prj_latent).item()

        # Chain-of-Thought iterations
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1:, :]
            attentions = outputs.attentions

            clean_latent = latent_embd.clone()

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])
            is_number = bool(number_regex.match(token_str))

            position_data = {
                'position': i + 1,
                'clean_token': token_str,
                'clean_token_id': token_id,
                'is_number': is_number,
                'intervened': False,
                'clean_hidden_state_norm': torch.norm(clean_latent).item(),
                'attention_patterns': {}
            }

            # Extract attention patterns
            for layer_idx, attn in enumerate(attentions):
                attn_from_last = attn[0, :, -1, :].cpu().numpy()
                position_data['attention_patterns'][f'layer_{layer_idx}'] = {
                    'mean_attn': attn_from_last.mean(axis=0).tolist(),
                    'per_head': attn_from_last.tolist()
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

                position_data['intervened'] = True
                position_data['intervened_token'] = new_token_str
                position_data['intervened_token_id'] = new_token_id
                position_data['intervention_magnitude'] = torch.norm(A_modified - A).item()
                position_data['modified_hidden_state_norm'] = torch.norm(A_modified).item()

                cos_sim = torch.nn.functional.cosine_similarity(
                    clean_latent.squeeze(1), A_modified, dim=-1
                ).item()
                position_data['cosine_similarity_clean_modified'] = cos_sim

            analysis_data['positions'].append(position_data)

            if training_args.use_prj:
                pre_prj_latent = latent_embd.clone()
                latent_embd = model.prj(latent_embd)

                position_data['projection_applied'] = True
                position_data['pre_projection_norm'] = torch.norm(pre_prj_latent).item()
                position_data['post_projection_norm'] = torch.norm(latent_embd).item()
                position_data['projection_change'] = torch.norm(latent_embd - pre_prj_latent).item()

    return analysis_data


def compare_intervention_at_different_positions(model, tokenizer, training_args,
                                                question, target_token='5', k=3):
    """
    Compare what happens when we intervene at different CoT positions
    This is like the "activation patching" approach from IOI
    """
    results = []

    print(f"\nAnalyzing intervention propagation for question:")
    print(f"Q: {question[:100]}...")
    print(f"Target token: {target_token}, k={k}\n")

    # Baseline: No intervention
    print("Running baseline (no intervention)...")
    baseline_data = analyze_intervention_propagation(
        model, tokenizer, training_args, question,
        intervention_position=-1,  # No intervention
        target_token=target_token,
        k=k
    )
    results.append(baseline_data)

    # Intervene at each position
    for pos in range(7):  # 0-6 (BoT + 6 iterations)
        print(f"Intervening at position {pos}...")
        intervention_data = analyze_intervention_propagation(
            model, tokenizer, training_args, question,
            intervention_position=pos,
            target_token=target_token,
            k=k
        )
        results.append(intervention_data)

    return results


def visualize_intervention_cascade(results, output_path):
    """
    Visualize how interventions at different positions propagate through CoT
    Similar to the patching heatmaps in IOI analysis
    """
    # Extract token sequences for each intervention position
    num_positions = 7
    intervention_positions = list(range(-1, num_positions))  # -1 = no intervention

    # Create matrix: rows = intervention position, cols = CoT position, values = token
    token_matrix = []
    token_labels = []

    for result in results:
        int_pos = result['intervention_position']
        tokens = [pos['intervened_token'] if pos.get('intervened', False)
                  else pos['clean_token'] for pos in result['positions']]
        token_matrix.append(tokens)
        token_labels.append(tokens)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Token cascade heatmap
    ax = axes[0, 0]

    # Create numeric representation for visualization
    # Count how many positions show the target token
    target_token = results[0]['target_token']
    numeric_matrix = []
    for tokens in token_labels:
        numeric_row = [1 if token == target_token else 0 for token in tokens]
        numeric_matrix.append(numeric_row)

    im = ax.imshow(numeric_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('CoT Position', fontsize=12)
    ax.set_ylabel('Intervention Position', fontsize=12)
    ax.set_title(f'Token Cascade: Showing "{target_token}" (1) vs Other (0)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([f'Pos {i}' for i in range(num_positions)])
    ax.set_yticks(range(len(intervention_positions)))
    ax.set_yticklabels([f'Int @ {p}' if p >= 0 else 'Baseline' for p in intervention_positions])

    # Add token labels
    for i in range(len(token_labels)):
        for j in range(len(token_labels[i])):
            text = ax.text(j, i, token_labels[i][j],
                         ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax, label=f'Is "{target_token}"')

    # 2. Intervention propagation strength
    ax = axes[0, 1]

    propagation_matrix = []
    for result in results[1:]:  # Skip baseline
        row = []
        int_pos = result['intervention_position']
        for pos_data in result['positions']:
            if pos_data['position'] < int_pos:
                row.append(0)  # Before intervention
            elif pos_data['position'] == int_pos:
                row.append(1)  # Intervention point
            else:
                # After intervention: measure if token matches target
                if pos_data.get('intervened', False) or pos_data['clean_token'] == target_token:
                    row.append(0.8)
                else:
                    row.append(0.2)
        propagation_matrix.append(row)

    im = ax.imshow(propagation_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xlabel('CoT Position', fontsize=12)
    ax.set_ylabel('Intervention Position', fontsize=12)
    ax.set_title('Intervention Propagation Strength', fontsize=14, fontweight='bold')
    ax.set_xticks(range(num_positions))
    ax.set_xticklabels([f'Pos {i}' for i in range(num_positions)])
    ax.set_yticks(range(len(propagation_matrix)))
    ax.set_yticklabels([f'Int @ {i}' for i in range(num_positions)])
    plt.colorbar(im, ax=ax, label='Effect Strength')

    # 3. Hidden state norm changes
    ax = axes[1, 0]

    baseline_norms = [pos['clean_hidden_state_norm'] for pos in results[0]['positions']]

    for result in results[1:]:
        int_pos = result['intervention_position']
        norms = [pos.get('modified_hidden_state_norm', pos['clean_hidden_state_norm'])
                for pos in result['positions']]
        norm_changes = [(norms[i] - baseline_norms[i]) / baseline_norms[i] * 100
                       for i in range(len(norms))]
        ax.plot(range(num_positions), norm_changes, marker='o', label=f'Int @ {int_pos}', linewidth=2)

    ax.set_xlabel('CoT Position', fontsize=12)
    ax.set_ylabel('Hidden State Norm Change (%)', fontsize=12)
    ax.set_title('Hidden State Perturbations', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # 4. Projection layer contribution
    ax = axes[1, 1]

    baseline_prj_changes = [pos.get('projection_change', 0) for pos in results[0]['positions']]
    positions_range = range(num_positions)

    for result in results[1:]:
        int_pos = result['intervention_position']
        prj_changes = [pos.get('projection_change', 0) for pos in result['positions']]
        ax.plot(positions_range, prj_changes, marker='o', label=f'Int @ {int_pos}', linewidth=2)

    ax.plot(positions_range, baseline_prj_changes, marker='s', label='Baseline',
            linewidth=2, linestyle='--', color='black')

    ax.set_xlabel('CoT Position', fontsize=12)
    ax.set_ylabel('Projection Layer Change (L2 norm)', fontsize=12)
    ax.set_title('Direct Projection Pathway Impact', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")

    return fig


def main():
    """Main analysis function"""
    print("="*80)
    print("CODI Chain-of-Thought Circuit Analysis")
    print("="*80)

    model, tokenizer, training_args = load_llama_model()

    # Use the first example from GSM8K
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # Analyze intervention propagation
    results = compare_intervention_at_different_positions(
        model, tokenizer, training_args, question,
        target_token='5', k=3
    )

    # Save detailed results
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "intervention_propagation_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Detailed results saved to {results_file}")

    # Create visualizations
    viz_path = output_dir / "intervention_cascade_visualization.png"
    visualize_intervention_cascade(results, viz_path)

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
