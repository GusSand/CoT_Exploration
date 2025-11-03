#!/usr/bin/env python3
"""
Phase 9: Feature 2203 Activation Timing Analysis

Analyze WHEN Feature 2203 activates:
- Beginning of thought (BOT) position
- Each continuous thought iteration (1-6)
- Across all three layers (early, middle, late)

For all three problem variants.
"""

import torch
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np

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
print("PHASE 9: FEATURE 2203 ACTIVATION TIMING")
print("="*80)

# SAE class definition
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
        features = torch.nn.functional.relu(self.encoder(x))
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        reconstruction = self.decoder(features)
        return reconstruction, features


# Load model
print("\n[1/4] Loading CODI-LLAMA...")
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
print("\n[2/4] Loading SAE...")
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


# Define problem variants
problems = {
    'original': {
        'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        'answer': 18,
    },
    'variant_a': {
        'question': "Janet's ducks lay 16 eggs per day. She eats two for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        'answer': 20,
    },
    'variant_b': {
        'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with three. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        'answer': 20,
    }
}


# Feature tracking hook - track ALL positions
class DetailedFeatureTrackingHook:
    def __init__(self, sae, layer_name, feature_id=2203):
        self.sae = sae
        self.layer_name = layer_name
        self.feature_id = feature_id
        self.activations = []  # List of (position_label, activation_value)
        self.position_counter = 0

    def __call__(self, module, input, output):
        hidden_states = output[0]

        # Handle both 2D and 3D tensors
        if len(hidden_states.shape) not in [2, 3]:
            return output

        with torch.no_grad():
            # Handle 2D tensors from latent iterations
            last_hidden = hidden_states.unsqueeze(1) if len(hidden_states.shape) == 2 else hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)

            # Extract feature 2203 activation
            feat_activation = features[0, 0, self.feature_id].cpu().float().item()

            # Store with position label
            self.activations.append({
                'position': self.position_counter,
                'activation': feat_activation
            })
            self.position_counter += 1

        return output


def run_detailed_analysis(model, tokenizer, question, sae, training_args):
    """Run model and track Feature 2203 at ALL positions"""

    # Create hooks for all three layers
    hooks = {}
    for layer in ['early', 'middle', 'late']:
        layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
        hooks[layer_idx] = DetailedFeatureTrackingHook(sae, layer, feature_id=2203)

    # Tokenize
    batch = tokenizer(question, return_tensors="pt", padding="longest")

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).unsqueeze(0)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).unsqueeze(0)

    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(device)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1).to(device)

    with torch.no_grad():
        # STEP 1: Encode question (this processes the BOT token at the end)
        print(f"    Encoding question...")

        # Register hooks for question encoding
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

        # Remove hooks
        for handle in handles:
            handle.remove()

        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Mark BOT position
        bot_position = {}
        for layer_idx, hook in hooks.items():
            # The last activation captured is the BOT token
            bot_position[layer_idx] = len(hook.activations) - 1

        # STEP 2: All 6 latent iterations
        for iteration_idx in range(6):
            print(f"    Continuous thought iteration {iteration_idx + 1}/6...")

            # Register hooks
            handles = []
            for layer_idx, hook in hooks.items():
                handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
                handles.append(handle)

            # Forward pass
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

            # Remove hooks
            for handle in handles:
                handle.remove()

    # Package results
    results = {}
    for layer in ['early', 'middle', 'late']:
        layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
        hook = hooks[layer_idx]

        bot_pos = bot_position[layer_idx]

        # Extract activations
        bot_activation = hook.activations[bot_pos]['activation']

        # Continuous thought activations (6 iterations after BOT)
        ct_activations = []
        for i in range(6):
            if bot_pos + 1 + i < len(hook.activations):
                ct_activations.append(hook.activations[bot_pos + 1 + i]['activation'])

        results[layer] = {
            'bot_position': bot_pos,
            'bot_activation': bot_activation,
            'continuous_thought': ct_activations,
            'all_activations': hook.activations
        }

    return results


# Run experiments
print("\n[3/4] Running experiments on all variants...")
print("="*80)

all_results = {}

for variant_name, variant_data in problems.items():
    print(f"\n{'='*80}")
    print(f"PROCESSING: {variant_name.upper()}")
    print(f"{'='*80}")
    print(f"Question: {variant_data['question'][:80]}...")

    results = run_detailed_analysis(model, tokenizer, variant_data['question'], sae, training_args)

    all_results[variant_name] = {
        'question': variant_data['question'],
        'answer': variant_data['answer'],
        'feature_2203_timing': results
    }

    # Display results
    for layer in ['early', 'middle', 'late']:
        layer_data = results[layer]
        print(f"\n  {layer.upper()} LAYER:")
        print(f"    BOT position: {layer_data['bot_position']}")
        print(f"    BOT activation: {layer_data['bot_activation']:.4f}")
        print(f"    Continuous thought activations:")
        for i, act in enumerate(layer_data['continuous_thought'], 1):
            print(f"      Iteration {i}: {act:.4f}")


# Save results
print(f"\n[4/4] Saving results...")
output_path = Path('./phase9_feature_2203_timing_results.json')
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Results saved to: {output_path}")


# Summary analysis
print("\n" + "="*80)
print("SUMMARY: WHEN DOES FEATURE 2203 FIRE?")
print("="*80)

for variant_name in ['original', 'variant_a', 'variant_b']:
    print(f"\n{variant_name.upper()}:")

    for layer in ['early', 'middle', 'late']:
        data = all_results[variant_name]['feature_2203_timing'][layer]

        # Check if it fires at BOT
        bot_fires = data['bot_activation'] > 0.1

        # Check continuous thought
        ct_fires = [act > 0.1 for act in data['continuous_thought']]
        max_ct = max(data['continuous_thought']) if data['continuous_thought'] else 0

        print(f"  {layer.upper():>6s}: BOT={'YES' if bot_fires else 'NO ':>3s} ({data['bot_activation']:.3f}) | "
              f"CT max={max_ct:.3f} at iter {ct_fires.index(True)+1 if any(ct_fires) else 'none'}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
