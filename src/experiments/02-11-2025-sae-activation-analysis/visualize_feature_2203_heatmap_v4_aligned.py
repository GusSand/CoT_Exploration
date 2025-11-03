#!/usr/bin/env python3
"""
Visualize Feature 2203 activation dynamics across all layers and positions

Creates heatmaps showing:
- Feature 2203 activation at all 16 layers
- BOT + 6 continuous thought positions
- Top-3 decoded tokens with probability shading as column headers ABOVE heatmap
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
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import plotly.graph_objects as go

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


# Hook to track activations only
class ActivationTrackingHook:
    def __init__(self, sae, layer_idx, feature_id=2203):
        self.sae = sae
        self.layer_idx = layer_idx
        self.feature_id = feature_id
        self.activations = []

    def __call__(self, module, input, output):
        hidden_states = output[0]

        if len(hidden_states.shape) not in [2, 3]:
            return output

        with torch.no_grad():
            last_hidden = hidden_states.unsqueeze(1) if len(hidden_states.shape) == 2 else hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)
            feat_activation = features[0, 0, self.feature_id].cpu().float().item()
            self.activations.append(feat_activation)

        return output


def decode_top_tokens(hidden_state, lm_head, tokenizer, k=3):
    """Decode top-k tokens from a single hidden state"""
    with torch.no_grad():
        logits = lm_head(hidden_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)

        tokens = []
        for prob, idx in zip(topk_probs[0, 0].cpu().float(), topk_indices[0, 0].cpu()):
            token_str = tokenizer.decode([idx.item()])
            tokens.append({
                'token': token_str,
                'prob': prob.item(),
                'id': idx.item()
            })

        return tokens


def extract_all_layers(model, tokenizer, question, sae, training_args):
    """Extract Feature 2203 activations from ALL 16 layers AND decode tokens before projection"""

    hooks = {layer_idx: ActivationTrackingHook(sae, layer_idx, feature_id=2203)
             for layer_idx in range(16)}

    decoded_tokens_per_position = []

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
        latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Decode BOT position (before projection!)
        bot_decoded = decode_top_tokens(latent_embd_pre_proj, model.codi.lm_head, tokenizer, k=3)
        decoded_tokens_per_position.append(bot_decoded)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd_pre_proj)
        else:
            latent_embd = latent_embd_pre_proj

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
            latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Decode BEFORE projection
            ct_decoded = decode_top_tokens(latent_embd_pre_proj, model.codi.lm_head, tokenizer, k=3)
            decoded_tokens_per_position.append(ct_decoded)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd_pre_proj)
            else:
                latent_embd = latent_embd_pre_proj

            for handle in handles:
                handle.remove()

    # Extract activations matrix: [16 layers, 7 positions]
    activation_matrix = np.zeros((16, 7))

    for layer_idx in range(16):
        hook = hooks[layer_idx]
        activation_matrix[layer_idx, 0] = hook.activations[bot_position]
        for i in range(6):
            if bot_position + 1 + i < len(hook.activations):
                activation_matrix[layer_idx, 1+i] = hook.activations[bot_position + 1 + i]

    return activation_matrix, decoded_tokens_per_position, bot_position


# Run extraction
print("\n[3/5] Extracting activations from all layers...")
all_results = {}

for variant_name, question in problems.items():
    print(f"\n  Processing {variant_name}...")
    activation_matrix, top_tokens, bot_pos = extract_all_layers(
        model, tokenizer, question, sae, training_args
    )

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
    # === MATPLOTLIB VERSION (PNG) ===
    fig = plt.figure(figsize=(18, 14))

    # Create grid: top rows for decoded tokens header, main area for heatmap
    # Give more space to header
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 5], hspace=0.02)

    # Main heatmap (create first to establish axis position)
    ax = fig.add_subplot(gs[1])

    # REVERSE the activation matrix so L15 is at top
    activation_matrix_safe_reversed = np.flipud(activation_matrix_safe)

    im = ax.imshow(activation_matrix_safe_reversed, aspect='auto', cmap='YlOrRd',
                   norm=LogNorm(vmin=max(1e-6, activation_matrix_safe[activation_matrix_safe > 0].min()),
                               vmax=activation_matrix_safe.max()))

    # Set ticks
    ax.set_xticks(range(7))
    ax.set_xticklabels([''] * 7)  # No labels - they're in the header above
    ax.set_yticks(range(16))
    ax.set_yticklabels([f'L{15-i}' for i in range(16)], fontsize=11)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Feature 2203 Activation (log scale)', fontsize=13)

    # Force draw to get accurate axis positions after colorbar is added
    fig.canvas.draw()

    # Top panel: Decoded tokens as table header
    # Position it to match the heatmap axis exactly (excluding colorbar space)
    ax_header = fig.add_subplot(gs[0])

    # Get the actual positions after colorbar has been added
    heatmap_bbox = ax.get_position()
    header_bbox = ax_header.get_position()

    # Adjust header panel to match heatmap's horizontal extent exactly
    ax_header.set_position([heatmap_bbox.x0, header_bbox.y0, heatmap_bbox.width, header_bbox.height])

    ax_header.set_xlim(-0.5, 6.5)  # Match heatmap column bounds
    ax_header.set_ylim(0, 1)
    ax_header.axis('off')

    # Draw column headers with decoded tokens
    for pos_idx, tokens in enumerate(top_tokens):
        x_center = pos_idx  # Align with heatmap columns (0-6)

        # Position label
        ax_header.text(x_center, 0.95, position_labels[pos_idx],
                      ha='center', va='top', fontsize=14, fontweight='bold')

        # Top-3 decoded tokens (stacked vertically, decreasing probability)
        y_positions = [0.75, 0.50, 0.25]
        for y_pos, tok in zip(y_positions, tokens):
            tok_str = tok['token']
            if len(tok_str) > 10:
                tok_str = tok_str[:9] + '…'

            # Check if token is '7' or ' 7' and make bold
            if tok_str.strip() == '7':
                text = f"$\\mathbf{{{tok_str}}}$\n({tok['prob']*100:.1f}%)"
                fontweight = 'bold'
            else:
                text = f"{tok_str}\n({tok['prob']*100:.1f}%)"
                fontweight = 'normal'

            ax_header.text(x_center, y_pos, text,
                          ha='center', va='center', fontsize=12,
                          fontweight=fontweight,
                          bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='lightblue',
                                   edgecolor='gray',
                                   alpha=0.7))

    # Draw vertical separators between columns in header (aligned with heatmap)
    for i in range(-1, 7):
        ax_header.axvline(x=i+0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # Add activation values as text
    for i in range(16):
        for j in range(7):
            val = activation_matrix[15-i, j]
            if val > 0.01:
                ax.text(j, i, f'{val:.2f}',
                       ha="center", va="center", color="black", fontsize=9,
                       fontweight='bold')

    ax.set_xlabel('Position', fontsize=15, labelpad=10)
    ax.set_ylabel('Layer', fontsize=15)
    ax.set_title(f'Feature 2203 Activation Dynamics - {variant_name.replace("_", " ").title()}',
                fontsize=18, pad=10)

    plt.savefig(f'./feature_2203_heatmap_{variant_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # === PLOTLY VERSION (Interactive HTML) ===
    activation_matrix_safe_plotly = np.flipud(activation_matrix_safe)

    # Create hover text
    hover_text = []
    for i in range(16):
        row = []
        for j in range(7):
            layer_idx = 15 - i
            tokens_str = "<br>".join([f"{t['token']}: {t['prob']*100:.1f}%"
                                      for t in top_tokens[j]])
            row.append(f"Layer {layer_idx}<br>Position: {position_labels[j]}<br>"
                      f"Activation: {activation_matrix[layer_idx,j]:.4f}<br><br>"
                      f"Top tokens:<br>{tokens_str}")
        hover_text.append(row)

    # Create column header annotations
    annotations = []
    for pos_idx, tokens in enumerate(top_tokens):
        # Position label
        annotations.append(dict(
            x=pos_idx,
            y=1.15,
            xref='x',
            yref='paper',
            text=f"<b>{position_labels[pos_idx]}</b>",
            showarrow=False,
            font=dict(size=14)
        ))

        # Top-3 tokens
        token_text = "<br>".join([f"{t['token']} ({t['prob']*100:.1f}%)" for t in tokens])
        annotations.append(dict(
            x=pos_idx,
            y=1.05,
            xref='x',
            yref='paper',
            text=token_text,
            showarrow=False,
            font=dict(size=11),
            bgcolor='lightblue',
            bordercolor='gray',
            borderwidth=1
        ))

    fig_plotly = go.Figure(data=go.Heatmap(
        z=activation_matrix_safe_plotly,
        x=position_labels,
        y=[f'L{15-i}' for i in range(16)],
        colorscale='YlOrRd',
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(title='Activation<br>(log scale)')
    ))

    fig_plotly.update_layout(
        title=f'Feature 2203 Activation Dynamics - {variant_name.replace("_", " ").title()}',
        xaxis_title='Position',
        yaxis_title='Layer',
        width=1200,
        height=900,
        annotations=annotations
    )

    fig_plotly.update_yaxes(
        tickmode='array',
        tickvals=list(range(16)),
        ticktext=[f'L{15-i}' for i in range(16)]
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
