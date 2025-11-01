"""
CODI Pathway Decomposition Analysis
Decompose contributions from direct projection vs attention

This is analogous to the "component ablation" approach in IOI analysis
We want to understand:
1. How much does the direct projection contribute to each CoT position?
2. How much does attention to previous tokens contribute?
3. What happens if we ablate one pathway?
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


def run_with_pathway_ablation(model, tokenizer, training_args, question, ablate_projection=False):
    """
    Run CoT with optional ablation of the projection pathway
    This is like activation patching - we "knock out" one component
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

    pathway_analysis = {
        'ablate_projection': ablate_projection,
        'positions': []
    }

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

        # Store pre-projection state
        pre_projection = latent_embd.clone()

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])

        pos_data = {
            'position': 0,
            'token': token_str,
            'pre_projection_norm': torch.norm(pre_projection).item(),
        }

        if training_args.use_prj and not ablate_projection:
            latent_embd = model.prj(latent_embd)
            pos_data['post_projection_norm'] = torch.norm(latent_embd).item()
            pos_data['projection_delta'] = torch.norm(latent_embd - pre_projection).item()
            pos_data['projection_applied'] = True
        elif ablate_projection:
            # Keep the same embedding without projection
            pos_data['post_projection_norm'] = pos_data['pre_projection_norm']
            pos_data['projection_delta'] = 0.0
            pos_data['projection_applied'] = False
        else:
            pos_data['projection_applied'] = False

        pathway_analysis['positions'].append(pos_data)

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

            pre_projection = latent_embd.clone()

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'pre_projection_norm': torch.norm(pre_projection).item(),
            }

            if training_args.use_prj and not ablate_projection:
                latent_embd = model.prj(latent_embd)
                pos_data['post_projection_norm'] = torch.norm(latent_embd).item()
                pos_data['projection_delta'] = torch.norm(latent_embd - pre_projection).item()
                pos_data['projection_applied'] = True
            elif ablate_projection:
                pos_data['post_projection_norm'] = pos_data['pre_projection_norm']
                pos_data['projection_delta'] = 0.0
                pos_data['projection_applied'] = False
            else:
                pos_data['projection_applied'] = False

            pathway_analysis['positions'].append(pos_data)

    return pathway_analysis


def analyze_projection_contribution(model, tokenizer, training_args, question):
    """
    Analyze the contribution of the projection pathway by comparing:
    1. Normal run (with projection)
    2. Ablated run (without projection)
    """
    print(f"\nAnalyzing pathway contributions for question:")
    print(f"Q: {question[:100]}...")

    # Run with normal projection
    print("Running with projection...")
    normal_result = run_with_pathway_ablation(model, tokenizer, training_args, question, ablate_projection=False)

    # Run with ablated projection
    print("Running with projection ablated...")
    ablated_result = run_with_pathway_ablation(model, tokenizer, training_args, question, ablate_projection=True)

    analysis = {
        'question': question,
        'normal': normal_result,
        'ablated': ablated_result,
        'comparison': []
    }

    # Compare tokens and metrics
    for i in range(len(normal_result['positions'])):
        normal_pos = normal_result['positions'][i]
        ablated_pos = ablated_result['positions'][i]

        comparison = {
            'position': i,
            'normal_token': normal_pos['token'],
            'ablated_token': ablated_pos['token'],
            'tokens_match': normal_pos['token'] == ablated_pos['token'],
            'projection_delta': normal_pos.get('projection_delta', 0.0),
            'norm_difference': abs(normal_pos['pre_projection_norm'] - ablated_pos['pre_projection_norm'])
        }
        analysis['comparison'].append(comparison)

    return analysis


def decompose_information_flow(model, tokenizer, training_args, question):
    """
    Decompose information flow by measuring:
    1. Contribution from direct projection (new information injected)
    2. Contribution from attention (information retrieved from context)

    We approximate this by:
    - Projection contribution: ||projection(h_n) - h_n||
    - Attention contribution: Changes in hidden state due to attention mechanism
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

    decomposition = {
        'question': question,
        'positions': []
    }

    with torch.no_grad():
        # Initial encoding
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values

        # Get all hidden states to analyze layer-by-layer information flow
        all_hidden_states = outputs.hidden_states  # Tuple of [batch, seq_len, hidden_dim] per layer

        latent_embd = all_hidden_states[-1][:, -1:, :]
        first_layer_hidden = all_hidden_states[0][:, -1:, :]

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])

        pos_data = {
            'position': 0,
            'token': token_str,
            'first_layer_norm': torch.norm(first_layer_hidden).item(),
            'last_layer_norm': torch.norm(latent_embd).item(),
            'layer_progression': torch.norm(latent_embd - first_layer_hidden).item(),
        }

        # Apply projection
        if training_args.use_prj:
            pre_prj = latent_embd.clone()
            latent_embd = model.prj(latent_embd)

            # Projection contribution: how much does projection change the representation?
            pos_data['projection_contribution'] = torch.norm(latent_embd - pre_prj).item()
            pos_data['projection_relative'] = pos_data['projection_contribution'] / torch.norm(pre_prj).item()
        else:
            pos_data['projection_contribution'] = 0.0
            pos_data['projection_relative'] = 0.0

        decomposition['positions'].append(pos_data)

        # CoT iterations
        for i in range(training_args.inf_latent_iterations):
            # Store input to this iteration
            input_to_iteration = latent_embd.clone()

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states

            latent_embd = all_hidden_states[-1][:, -1:, :]
            first_layer_hidden = all_hidden_states[0][:, -1:, :]

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'first_layer_norm': torch.norm(first_layer_hidden).item(),
                'last_layer_norm': torch.norm(latent_embd).item(),
                'layer_progression': torch.norm(latent_embd - first_layer_hidden).item(),
            }

            # Attention contribution: difference between input and output of transformer
            # This captures how much the attention mechanism modifies the representation
            attention_delta = torch.norm(latent_embd - input_to_iteration).item()
            pos_data['attention_contribution'] = attention_delta

            if training_args.use_prj:
                pre_prj = latent_embd.clone()
                latent_embd = model.prj(latent_embd)

                pos_data['projection_contribution'] = torch.norm(latent_embd - pre_prj).item()
                pos_data['projection_relative'] = pos_data['projection_contribution'] / torch.norm(pre_prj).item()
            else:
                pos_data['projection_contribution'] = 0.0
                pos_data['projection_relative'] = 0.0

            # Compute ratio of projection to attention contribution
            if pos_data['attention_contribution'] > 0:
                pos_data['projection_to_attention_ratio'] = (
                    pos_data['projection_contribution'] / pos_data['attention_contribution']
                )
            else:
                pos_data['projection_to_attention_ratio'] = 0.0

            decomposition['positions'].append(pos_data)

    return decomposition


def visualize_pathway_analysis(pathway_comparison, decomposition_data, output_dir):
    """
    Visualize the pathway decomposition
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. Token changes when projection is ablated
    ax = axes[0, 0]

    normal_tokens = [comp['normal_token'] for comp in pathway_comparison['comparison']]
    ablated_tokens = [comp['ablated_token'] for comp in pathway_comparison['comparison']]
    matches = [1 if comp['tokens_match'] else 0 for comp in pathway_comparison['comparison']]

    positions = range(len(normal_tokens))
    colors = ['green' if m else 'red' for m in matches]

    ax.bar(positions, [1]*len(positions), color=colors, alpha=0.6)
    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tokens Match', fontsize=12, fontweight='bold')
    ax.set_title('Effect of Ablating Projection Pathway\n(Green=Same Token, Red=Different Token)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels([f"P{i}\n{normal_tokens[i]}\n{ablated_tokens[i]}" for i in positions],
                       fontsize=9)
    ax.set_ylim([0, 1.2])

    # Add text annotations
    for i, match in enumerate(matches):
        if not match:
            ax.text(i, 1.05, 'CHANGED', ha='center', fontsize=8, fontweight='bold', color='red')

    # 2. Projection vs Attention Contribution
    ax = axes[0, 1]

    positions = [pos['position'] for pos in decomposition_data['positions']]
    projection_contrib = [pos.get('projection_contribution', 0) for pos in decomposition_data['positions']]
    attention_contrib = [pos.get('attention_contribution', 0) for pos in decomposition_data['positions']]

    width = 0.35
    ax.bar([p - width/2 for p in positions], projection_contrib, width,
           label='Direct Projection', alpha=0.8, color='steelblue')
    ax.bar([p + width/2 for p in positions], attention_contrib, width,
           label='Attention Mechanism', alpha=0.8, color='coral')

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contribution (L2 Norm)', fontsize=12, fontweight='bold')
    ax.set_title('Information Flow Decomposition:\nDirect Projection vs Attention',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Projection to Attention Ratio
    ax = axes[1, 0]

    ratios = [pos.get('projection_to_attention_ratio', 0) for pos in decomposition_data['positions'][1:]]
    positions_for_ratio = [pos['position'] for pos in decomposition_data['positions'][1:]]

    ax.plot(positions_for_ratio, ratios, marker='o', linewidth=2, markersize=8, color='purple')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label='Equal Contribution')

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Projection / Attention Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Relative Importance of Direct Projection vs Attention',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    ax.text(0.5, 0.95, 'Ratio > 1: Projection dominates\nRatio < 1: Attention dominates',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Layer progression (how much each layer changes the representation)
    ax = axes[1, 1]

    positions = [pos['position'] for pos in decomposition_data['positions']]
    layer_progression = [pos['layer_progression'] for pos in decomposition_data['positions']]
    tokens = [pos['token'] for pos in decomposition_data['positions']]

    ax.plot(positions, layer_progression, marker='s', linewidth=2, markersize=8, color='darkgreen')

    ax.set_xlabel('CoT Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Layer Progression (First → Last Layer)', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Processing Depth\n(Magnitude of Change Across Layers)',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate tokens
    for i, (pos, prog, tok) in enumerate(zip(positions, layer_progression, tokens)):
        ax.annotate(tok, (pos, prog), textcoords="offset points",
                   xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / "pathway_decomposition.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'pathway_decomposition.png'}")
    plt.close()


def main():
    print("="*80)
    print("CODI Pathway Decomposition Analysis")
    print("="*80)

    model, tokenizer, training_args = load_llama_model()

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    # Analyze projection pathway via ablation
    print("\n1. Analyzing projection pathway contribution (via ablation)...")
    pathway_comparison = analyze_projection_contribution(model, tokenizer, training_args, question)

    # Decompose information flow
    print("\n2. Decomposing information flow (projection vs attention)...")
    decomposition_data = decompose_information_flow(model, tokenizer, training_args, question)

    # Save results
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "pathway_comparison.json", 'w') as f:
        json.dump(pathway_comparison, f, indent=2)
    print(f"✓ Saved: {output_dir / 'pathway_comparison.json'}")

    with open(output_dir / "information_flow_decomposition.json", 'w') as f:
        json.dump(decomposition_data, f, indent=2)
    print(f"✓ Saved: {output_dir / 'information_flow_decomposition.json'}")

    # Visualize
    print("\n3. Creating visualizations...")
    visualize_pathway_analysis(pathway_comparison, decomposition_data, output_dir)

    print("\n" + "="*80)
    print("Pathway decomposition analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
