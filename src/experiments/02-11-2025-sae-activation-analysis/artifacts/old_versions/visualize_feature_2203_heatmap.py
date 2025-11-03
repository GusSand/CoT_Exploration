#!/usr/bin/env python3
"""
Visualize Feature 2203 activation dynamics across all layers and positions

Creates heatmaps showing:
- Feature 2203 activation at all 16 layers
- BOT + 6 continuous thought positions
- Top-3 decoded tokens with probability shading
"""

import torch
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("VISUALIZING FEATURE 2203 ACTIVATION DYNAMICS")
print("="*80)

# SAE class
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int = 2048, n_features: int = 8192,
                 l1_coefficient: float = 0.001):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.l1_coefficient = l1_coefficient
        self.encoder = torch.nn.Linear(input_dim, n_features, bias=True)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)


# Load model
print("\n[1/5] Loading CODI-LLAMA...")
model_args = ModelArguments(
    model_name_or_path="meta-llama/Llama-3.2-1B",
    lora_init=True,
    lora_r=128,
    lora_alpha=32,
    ckpt_dir="/workspace/.cache/huggingface/hub/models--zen-E--CODI-llama3.2-1b-Instruct/snapshots/b2c88ba224b06b12b52ef39b87f794b98a6eb1c8",
    full_precision=True,
    token=None
)

training_args = TrainingArguments(
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

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=model_args.lora_r,
    lora_alpha=model_args.lora_alpha,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    init_lora_weights=True,
)

model = CODI(model_args, training_args, lora_config)
checkpoint_path = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")
state_dict = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.codi.tie_weights()
model = model.to(device).to(torch.bfloat16)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Load SAE
print("\n[2/5] Loading SAE...")
sae_path = "/workspace/1_gpt2_codi_and_sae/src/experiments/sae_pilot/results/sae_weights.pt"
checkpoint = torch.load(sae_path, map_location='cpu')
config = checkpoint['config']

sae = SparseAutoencoder(
    input_dim=config['input_dim'],
    n_features=config['n_features'],
    l1_coefficient=config['l1_coefficient']
)
sae.load_state_dict(checkpoint['model_state_dict'])
sae = sae.to(device).to(torch.bfloat16)
sae.eval()

# Problem variants
problems = {
    'original': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    'variant_a': "Janet's ducks lay 16 eggs per day. She eats two for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    'variant_b': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with three. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
}


# Hook to track activations and logits
class ComprehensiveTrackingHook:
    def __init__(self, sae, layer_idx, feature_id=2203):
        self.sae = sae
        self.layer_idx = layer_idx
        self.feature_id = feature_id
        self.activations = []
        self.hidden_states = []

    def __call__(self, module, input, output):
        hidden_states = output[0]

        # Handle both 2D and 3D
        if len(hidden_states.shape) not in [2, 3]:
            return output

        with torch.no_grad():
            last_hidden = hidden_states.unsqueeze(1) if len(hidden_states.shape) == 2 else hidden_states[:, -1:, :]

            # Store hidden state for decoding
            self.hidden_states.append(last_hidden[0, 0, :].cpu().float())

            # SAE encode
            features = self.sae.encode(last_hidden)
            feat_activation = features[0, 0, self.feature_id].cpu().float().item()
            self.activations.append(feat_activation)

        return output


def extract_all_layers(model, tokenizer, question, sae, training_args):
    """Extract Feature 2203 activations from ALL 16 layers"""

    # Create hooks for ALL layers
    hooks = {layer_idx: ComprehensiveTrackingHook(sae, layer_idx, feature_id=2203)
             for layer_idx in range(16)}

    # Tokenize
    batch = tokenizer(question, return_tensors="pt", padding="longest")

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).unsqueeze(0)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).unsqueeze(0)

    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(device)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1).to(device)

    with torch.no_grad():
        # STEP 1: Encode question (captures BOT position)
        handles = []
        for layer_idx, hook in hooks.items():
            handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

        outputs = model.codi(
            input_ids=batch["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=batch["attention_mask"]
        )

        for handle in handles:
            handle.remove()

        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Mark BOT position
        bot_position = len(hooks[0].activations) - 1

        # STEP 2: All 6 continuous thought iterations
        for iteration_idx in range(6):
            handles = []
            for layer_idx, hook in hooks.items():
                handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
                handles.append(handle)

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            for handle in handles:
                handle.remove()

    # Extract activations matrix: [16 layers, 7 positions]
    activation_matrix = np.zeros((16, 7))
    hidden_states_matrix = []  # [7 positions, 2048 dims]

    for layer_idx in range(16):
        hook = hooks[layer_idx]
        # BOT + 6 CT positions
        activation_matrix[layer_idx, 0] = hook.activations[bot_position]
        for i in range(6):
            if bot_position + 1 + i < len(hook.activations):
                activation_matrix[layer_idx, 1+i] = hook.activations[bot_position + 1 + i]

    # Get hidden states from final layer (layer 15) for decoding
    final_hook = hooks[15]
    for i in range(7):
        if bot_position + i < len(final_hook.hidden_states):
            hidden_states_matrix.append(final_hook.hidden_states[bot_position + i])

    return activation_matrix, hidden_states_matrix, bot_position


def decode_top_tokens(hidden_states_list, model, tokenizer, k=3):
    """Decode top-k tokens from hidden states using model's LM head"""

    top_tokens_list = []

    with torch.no_grad():
        for hidden_state in hidden_states_list:
            # hidden_state: [2048]
            hidden_state = hidden_state.to(device).to(torch.bfloat16).unsqueeze(0)  # [1, 2048]

            # Get logits from LM head
            logits = model.codi.lm_head(hidden_state)  # [1, vocab_size]
            probs = torch.nn.functional.softmax(logits[0], dim=0)

            # Get top-k
            top_probs, top_indices = torch.topk(probs, k)

            tokens = []
            for prob, idx in zip(top_probs.cpu().float(), top_indices.cpu()):
                token_str = tokenizer.decode([idx.item()])
                tokens.append({
                    'token': token_str,
                    'prob': prob.item(),
                    'id': idx.item()
                })

            top_tokens_list.append(tokens)

    return top_tokens_list


# Run extraction
print("\n[3/5] Extracting activations from all layers...")
all_results = {}

for variant_name, question in problems.items():
    print(f"\n  Processing {variant_name}...")
    activation_matrix, hidden_states, bot_pos = extract_all_layers(
        model, tokenizer, question, sae, training_args
    )

    print(f"    Decoding top tokens...")
    top_tokens = decode_top_tokens(hidden_states, model, tokenizer, k=3)

    all_results[variant_name] = {
        'activations': activation_matrix,
        'top_tokens': top_tokens
    }


# Create visualizations
print("\n[4/5] Creating visualizations...")

position_labels = ['BOT', 'CT-1', 'CT-2', 'CT-3', 'CT-4', 'CT-5', 'CT-6']
layer_labels = [f'L{i}' for i in range(16)]

for variant_name, data in all_results.items():
    print(f"\n  Creating heatmap for {variant_name}...")

    activation_matrix = data['activations']
    top_tokens = data['top_tokens']

    # Add small epsilon to avoid log(0)
    activation_matrix_safe = np.maximum(activation_matrix, 1e-6)

    # === MATPLOTLIB VERSION (PNG) ===
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create heatmap
    im = ax.imshow(activation_matrix_safe, aspect='auto', cmap='YlOrRd',
                   norm=LogNorm(vmin=max(1e-6, activation_matrix_safe[activation_matrix_safe > 0].min()),
                               vmax=activation_matrix_safe.max()))

    # Set ticks
    ax.set_xticks(range(7))
    ax.set_xticklabels(position_labels, fontsize=12)
    ax.set_yticks(range(16))
    ax.set_yticklabels(layer_labels, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Feature 2203 Activation (log scale)', fontsize=12)

    # Add activation values as text
    for i in range(16):
        for j in range(7):
            val = activation_matrix[i, j]
            if val > 0.01:  # Only show if significant
                text = ax.text(j, i, f'{val:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

    # Add top tokens below heatmap
    token_text = "Top-3 Decoded Tokens:\n"
    for pos_idx, tokens in enumerate(top_tokens):
        token_text += f"{position_labels[pos_idx]}: "
        for tok in tokens:
            token_text += f"{repr(tok['token'])} ({tok['prob']*100:.1f}%)  "
        token_text += "\n"

    ax.text(0.5, -0.15, token_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
            family='monospace')

    ax.set_xlabel('Position', fontsize=14, labelpad=10)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title(f'Feature 2203 Activation Dynamics - {variant_name.replace("_", " ").title()}',
                fontsize=16, pad=20)

    plt.tight_layout()
    plt.savefig(f'./feature_2203_heatmap_{variant_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === PLOTLY VERSION (Interactive HTML) ===

    # Create annotation text for hover
    hover_text = []
    for i in range(16):
        row = []
        for j in range(7):
            tokens_str = "<br>".join([f"{t['token']}: {t['prob']*100:.1f}%"
                                      for t in top_tokens[j]])
            row.append(f"Layer {i}<br>Position: {position_labels[j]}<br>"
                      f"Activation: {activation_matrix[i,j]:.4f}<br><br>"
                      f"Top tokens:<br>{tokens_str}")
        hover_text.append(row)

    fig_plotly = go.Figure(data=go.Heatmap(
        z=activation_matrix_safe,
        x=position_labels,
        y=layer_labels,
        colorscale='YlOrRd',
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(title='Activation<br>(log scale)')
    ))

    fig_plotly.update_layout(
        title=f'Feature 2203 Activation Dynamics - {variant_name.replace("_", " ").title()}',
        xaxis_title='Position',
        yaxis_title='Layer',
        width=1000,
        height=800
    )

    # Update y-axis to show all layer labels
    fig_plotly.update_yaxes(
        tickmode='array',
        tickvals=list(range(16)),
        ticktext=layer_labels
    )

    fig_plotly.write_html(f'./feature_2203_heatmap_{variant_name}.html')

print("\n[5/5] Saving summary data...")

# Save raw data
output = {
    'feature_id': 2203,
    'positions': position_labels,
    'layers': layer_labels,
    'variants': {}
}

for variant_name, data in all_results.items():
    output['variants'][variant_name] = {
        'activations': data['activations'].tolist(),
        'top_tokens': data['top_tokens']
    }

with open('./feature_2203_heatmap_data.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
for variant in problems.keys():
    print(f"  - feature_2203_heatmap_{variant}.png")
    print(f"  - feature_2203_heatmap_{variant}.html")
print(f"  - feature_2203_heatmap_data.json")
