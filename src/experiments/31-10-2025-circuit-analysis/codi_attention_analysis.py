"""
CODI Attention Pattern Analysis
Inspired by IOI attention head analysis

This script visualizes:
1. Attention patterns from each CoT position to previous tokens
2. Which parts of the question each CoT position attends to
3. How attention evolves through the reasoning process
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
from dotenv import load_dotenv
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
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


def extract_attention_patterns(model, tokenizer, training_args, question):
    """
    Extract attention patterns for each CoT position
    Returns attention weights showing what each CoT token attends to
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

    # Get question tokens for labeling
    question_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())

    attention_data = {
        'question': question,
        'question_tokens': question_tokens,
        'question_length': len(question_tokens),
        'cot_positions': []
    }

    with torch.no_grad():
        # Initial encoding (position 0: BoT)
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]
        attentions = outputs.attentions

        # Decode BoT token
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])

        # Store attention data for position 0
        pos_data = {
            'position': 0,
            'token': token_str,
            'sequence_length': inputs["input_ids"].shape[1],
            'layers': []
        }

        num_layers = len(attentions)
        for layer_idx, attn in enumerate(attentions):
            # attn shape: [batch, num_heads, seq_len, seq_len]
            # Extract attention FROM last token TO all previous tokens
            attn_from_last = attn[0, :, -1, :].cpu().numpy()  # [num_heads, seq_len]

            layer_data = {
                'layer_idx': layer_idx,
                'num_heads': attn_from_last.shape[0],
                'attention_weights': attn_from_last.tolist(),  # [num_heads, seq_len]
                'mean_attention': attn_from_last.mean(axis=0).tolist(),  # Average across heads
                'max_attention_per_head': attn_from_last.max(axis=1).tolist(),
            }
            pos_data['layers'].append(layer_data)

        attention_data['cot_positions'].append(pos_data)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

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

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])

            # Current sequence length (question + BoT + i previous CoT tokens)
            current_seq_len = attention_data['question_length'] + i + 1

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'sequence_length': current_seq_len,
                'layers': []
            }

            for layer_idx, attn in enumerate(attentions):
                attn_from_last = attn[0, :, -1, :].cpu().numpy()

                layer_data = {
                    'layer_idx': layer_idx,
                    'num_heads': attn_from_last.shape[0],
                    'attention_weights': attn_from_last.tolist(),
                    'mean_attention': attn_from_last.mean(axis=0).tolist(),
                    'max_attention_per_head': attn_from_last.max(axis=1).tolist(),
                }
                pos_data['layers'].append(layer_data)

            attention_data['cot_positions'].append(pos_data)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return attention_data


def visualize_attention_patterns(attention_data, output_dir):
    """
    Create comprehensive attention visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    question_tokens = attention_data['question_tokens']
    question_len = attention_data['question_length']
    num_cot_positions = len(attention_data['cot_positions'])

    # Visualization 1: Average attention from each CoT position (across all layers)
    fig, ax = plt.subplots(figsize=(20, 12))

    attention_matrix = []
    cot_token_labels = []

    for pos_data in attention_data['cot_positions']:
        cot_pos = pos_data['position']
        token = pos_data['token']
        cot_token_labels.append(f"CoT{cot_pos}:{token}")

        # Average attention across all layers and heads
        all_layers_attn = []
        for layer_data in pos_data['layers']:
            all_layers_attn.append(layer_data['mean_attention'])

        # Average across layers: [num_layers, seq_len] -> [seq_len]
        avg_attn = np.mean(all_layers_attn, axis=0)

        # Pad to include previous CoT positions
        full_seq_attn = list(avg_attn) + [0.0] * (num_cot_positions - cot_pos - 1)
        attention_matrix.append(full_seq_attn)

    attention_matrix = np.array(attention_matrix)

    # Create token labels: question tokens + CoT tokens
    all_token_labels = [f"Q{i}:{tok[:8]}" for i, tok in enumerate(question_tokens)]
    all_token_labels += [f"CoT{i}" for i in range(num_cot_positions)]

    # Trim to actual size
    all_token_labels = all_token_labels[:attention_matrix.shape[1]]

    im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto')
    ax.set_xlabel('Token Position (From)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CoT Position (Attending)', fontsize=14, fontweight='bold')
    ax.set_title('Average Attention Pattern: What Each CoT Position Attends To',
                 fontsize=16, fontweight='bold')

    # Set ticks
    ax.set_xticks(range(len(all_token_labels)))
    ax.set_xticklabels(all_token_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cot_token_labels)))
    ax.set_yticklabels(cot_token_labels, fontsize=10)

    # Add vertical line separating question from CoT
    ax.axvline(x=question_len - 0.5, color='red', linestyle='--', linewidth=2, label='Question | CoT')
    ax.legend(loc='upper right')

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(output_dir / "attention_overview.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'attention_overview.png'}")
    plt.close()

    # Visualization 2: Attention to Question vs Previous CoT
    fig, ax = plt.subplots(figsize=(12, 8))

    attention_to_question = []
    attention_to_cot = []

    for pos_data in attention_data['cot_positions']:
        cot_pos = pos_data['position']

        # Average across layers
        all_layers_attn = []
        for layer_data in pos_data['layers']:
            all_layers_attn.append(layer_data['mean_attention'])
        avg_attn = np.mean(all_layers_attn, axis=0)

        # Sum attention to question tokens
        attn_to_q = np.sum(avg_attn[:question_len])
        attention_to_question.append(attn_to_q)

        # Sum attention to previous CoT tokens (if any)
        if cot_pos > 0:
            attn_to_cot_tokens = np.sum(avg_attn[question_len:question_len + cot_pos])
            attention_to_cot.append(attn_to_cot_tokens)
        else:
            attention_to_cot.append(0.0)

    positions = range(num_cot_positions)
    width = 0.35

    ax.bar([p - width/2 for p in positions], attention_to_question, width, label='Attention to Question', alpha=0.8)
    ax.bar([p + width/2 for p in positions], attention_to_cot, width, label='Attention to Previous CoT', alpha=0.8)

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Attention Weight', fontsize=12, fontweight='bold')
    ax.set_title('Attention Distribution: Question vs Previous CoT', fontsize=14, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'CoT{i}:{attention_data["cot_positions"][i]["token"]}'
                        for i in positions], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "attention_distribution.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'attention_distribution.png'}")
    plt.close()

    # Visualization 3: Layer-wise attention for a specific CoT position
    # Let's visualize the middle CoT position
    mid_pos = num_cot_positions // 2
    pos_data = attention_data['cot_positions'][mid_pos]

    fig, ax = plt.subplots(figsize=(16, 10))

    layer_attention_matrix = []
    for layer_data in pos_data['layers']:
        layer_attention_matrix.append(layer_data['mean_attention'])

    layer_attention_matrix = np.array(layer_attention_matrix)

    im = ax.imshow(layer_attention_matrix, cmap='plasma', aspect='auto')
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax.set_title(f'Layer-wise Attention from CoT Position {mid_pos} (Token: "{pos_data["token"]}")',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(range(len(all_token_labels[:pos_data['sequence_length']])))
    ax.set_xticklabels(all_token_labels[:pos_data['sequence_length']], rotation=90, fontsize=8)
    ax.set_yticks(range(len(pos_data['layers'])))
    ax.set_yticklabels([f'Layer {i}' for i in range(len(pos_data['layers']))], fontsize=8)

    ax.axvline(x=question_len - 0.5, color='red', linestyle='--', linewidth=2, label='Question | CoT')
    ax.legend(loc='upper right')

    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    plt.savefig(output_dir / f"attention_layerwise_cot{mid_pos}.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / f'attention_layerwise_cot{mid_pos}.png'}")
    plt.close()

    # Visualization 4: Attention head diversity
    # Show how different heads attend to different things
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for idx, cot_idx in enumerate([0, 2, 4, 6]):
        if cot_idx >= num_cot_positions:
            continue

        ax = axes[idx // 2, idx % 2]
        pos_data = attention_data['cot_positions'][cot_idx]

        # Use the last layer (typically most important)
        last_layer = pos_data['layers'][-1]
        attention_per_head = np.array(last_layer['attention_weights'])  # [num_heads, seq_len]

        im = ax.imshow(attention_per_head, cmap='coolwarm', aspect='auto')
        ax.set_xlabel('Token Position', fontsize=10, fontweight='bold')
        ax.set_ylabel('Attention Head', fontsize=10, fontweight='bold')
        ax.set_title(f'CoT{cot_idx} (Token: "{pos_data["token"]}") - Final Layer Attention Heads',
                     fontsize=12, fontweight='bold')

        ax.set_xticks(range(0, len(all_token_labels[:pos_data['sequence_length']]), max(1, len(all_token_labels)//10)))
        ax.set_xticklabels([all_token_labels[i] for i in range(0, pos_data['sequence_length'],
                            max(1, len(all_token_labels)//10))], rotation=90, fontsize=7)

        ax.axvline(x=question_len - 0.5, color='yellow', linestyle='--', linewidth=1.5, alpha=0.7)

        plt.colorbar(im, ax=ax, label='Attention Weight')

    plt.tight_layout()
    plt.savefig(output_dir / "attention_head_diversity.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'attention_head_diversity.png'}")
    plt.close()


def main():
    print("="*80)
    print("CODI Attention Pattern Analysis")
    print("="*80)

    model, tokenizer, training_args = load_llama_model()

    # Example question
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print("\nExtracting attention patterns...")
    attention_data = extract_attention_patterns(model, tokenizer, training_args, question)

    # Save raw data
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "attention_patterns.json", 'w') as f:
        json.dump(attention_data, f, indent=2)
    print(f"✓ Saved attention data to {output_dir / 'attention_patterns.json'}")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_attention_patterns(attention_data, output_dir)

    print("\n" + "="*80)
    print("Attention analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
