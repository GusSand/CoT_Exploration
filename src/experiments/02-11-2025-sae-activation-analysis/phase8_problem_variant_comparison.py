#!/usr/bin/env python3
"""
Phase 8: Problem Variant Comparison

Compare SAE feature activations across three problem variants:
1. Original: 16 eggs, 3 breakfast, 4 muffins → 9 remaining × 2 = 18
2. Variant A: 16 eggs, 2 breakfast, 4 muffins → 10 remaining × 2 = 20
3. Variant B: 16 eggs, 3 breakfast, 3 muffins → 10 remaining × 2 = 20

Extract activations from first 2 continuous thought iterations.
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
print("PHASE 8: PROBLEM VARIANT COMPARISON")
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


# Load Phase 5 results for number precision data
print("\n[1/4] Loading Phase 5 precision data...")
phase5_path = Path('./phase5_simple_precision_results.json')
with open(phase5_path, 'r') as f:
    phase5_data = json.load(f)

print(f"Loaded precision data for features across {len(phase5_data['feature_reports_by_layer'])} layers")


# Load model
print("\n[2/4] Loading CODI-LLAMA...")
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
print("\n[3/4] Loading SAE...")
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
        'computation': '16 - 3 - 4 = 9, then 9 * 2 = 18',
        'modified_numbers': []
    },
    'variant_a': {
        'question': "Janet's ducks lay 16 eggs per day. She eats two for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        'answer': 20,
        'computation': '16 - 2 - 4 = 10, then 10 * 2 = 20',
        'modified_numbers': [2, 10, 20]  # Changed from original
    },
    'variant_b': {
        'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with three. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        'answer': 20,
        'computation': '16 - 3 - 3 = 10, then 10 * 2 = 20',
        'modified_numbers': [3, 10, 20]  # Changed from original
    }
}


# Feature tracking hook
class FeatureExtractionHook:
    def __init__(self, sae, layer_name):
        self.sae = sae
        self.layer_name = layer_name
        self.activations = []

    def __call__(self, module, input, output):
        hidden_states = output[0]

        # Handle both 2D and 3D tensors
        if len(hidden_states.shape) not in [2, 3]:
            return output

        with torch.no_grad():
            # Handle 2D tensors from latent iterations
            last_hidden = hidden_states.unsqueeze(1) if len(hidden_states.shape) == 2 else hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)

            # Store feature activations (shape: [1, 1, 8192])
            self.activations.append(features[0, 0, :].cpu().float().numpy())

        return output


def extract_activations(model, tokenizer, question, sae, training_args):
    """Extract SAE activations from first 2 continuous thought iterations"""

    # Create hooks for all three layers
    hooks = {}
    for layer in ['early', 'middle', 'late']:
        layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
        hooks[layer_idx] = FeatureExtractionHook(sae, layer)

    # Tokenize
    batch = tokenizer(question, return_tensors="pt", padding="longest")

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).unsqueeze(0)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).unsqueeze(0)

    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(device)
    batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1).to(device)

    with torch.no_grad():
        # Encode question
        outputs = model.codi(
            input_ids=batch["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=batch["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # FIRST 2 latent iterations only
        for iteration_idx in range(2):
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

    return hooks


# Run experiments
print("\n[4/4] Running experiments on all variants...")
print("="*80)

results = {}

for variant_name, variant_data in problems.items():
    print(f"\n{'='*80}")
    print(f"PROCESSING: {variant_name.upper()}")
    print(f"{'='*80}")
    print(f"Question: {variant_data['question']}")
    print(f"Expected answer: {variant_data['answer']}")
    print(f"Computation: {variant_data['computation']}")

    # Extract activations
    hooks = extract_activations(model, tokenizer, variant_data['question'], sae, training_args)

    # Analyze activations for each layer
    variant_results = {}

    for layer in ['early', 'middle', 'late']:
        layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
        layer_activations = hooks[layer_idx].activations

        if not layer_activations:
            continue

        # Aggregate across both iterations: max activation for each feature
        max_activations = np.maximum(layer_activations[0], layer_activations[1])

        # Get top 30 features by activation
        top_30_indices = np.argsort(max_activations)[-30:][::-1]
        top_30_activations = max_activations[top_30_indices]

        # Store feature data
        feature_data = []
        for feat_idx, feat_val in zip(top_30_indices, top_30_activations):
            feat_id_str = str(feat_idx)

            # Get number correlations from phase5
            number_correlations = []
            if feat_id_str in phase5_data['feature_reports_by_layer'][layer]:
                feat_data = phase5_data['feature_reports_by_layer'][layer][feat_id_str]
                if feat_data is not None:
                    for token_data in feat_data['top_tokens'][:20]:  # Top 20
                        token = token_data['token']
                        if token.strip().isdigit():  # Only numbers
                            number_correlations.append({
                                'number': token.strip(),
                                'precision': token_data['precision'],
                                'count': token_data['count']
                            })

            feature_data.append({
                'feature_id': int(feat_idx),
                'activation': float(feat_val),
                'iter1_activation': float(layer_activations[0][feat_idx]),
                'iter2_activation': float(layer_activations[1][feat_idx]),
                'number_correlations': number_correlations[:10]  # Top 10 numbers
            })

        variant_results[layer] = feature_data

    results[variant_name] = {
        'question': variant_data['question'],
        'answer': variant_data['answer'],
        'computation': variant_data['computation'],
        'modified_numbers': variant_data['modified_numbers'],
        'layer_features': variant_results
    }


# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_path = Path('./phase8_variant_comparison_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {output_path}")


# Print comparison analysis
print("\n" + "="*80)
print("COMPARISON ANALYSIS")
print("="*80)

for layer in ['early', 'middle', 'late']:
    print(f"\n{'='*80}")
    print(f"{layer.upper()} LAYER")
    print(f"{'='*80}")

    # Get top 10 features for each variant
    print(f"\nTop 10 most active features by variant:\n")

    for variant_name in ['original', 'variant_a', 'variant_b']:
        features = results[variant_name]['layer_features'][layer][:10]
        print(f"{variant_name.upper()}:")
        print(f"  {'Rank':<6} {'Feature':<10} {'Activation':<12} {'Top Number Correlations':<50}")
        print("  " + "-"*80)

        for rank, feat in enumerate(features, 1):
            feat_id = feat['feature_id']
            activation = feat['activation']

            # Format top 3 number correlations
            top_nums = []
            for num_data in feat['number_correlations'][:3]:
                top_nums.append(f"{num_data['number']}({num_data['precision']*100:.1f}%)")
            nums_str = ", ".join(top_nums) if top_nums else "none"

            print(f"  {rank:<6} {feat_id:<10} {activation:<12.4f} {nums_str}")
        print()

    # Compare feature overlap
    print(f"\nFeature overlap analysis:")
    original_features = set(f['feature_id'] for f in results['original']['layer_features'][layer][:30])
    variant_a_features = set(f['feature_id'] for f in results['variant_a']['layer_features'][layer][:30])
    variant_b_features = set(f['feature_id'] for f in results['variant_b']['layer_features'][layer][:30])

    overlap_orig_a = len(original_features & variant_a_features)
    overlap_orig_b = len(original_features & variant_b_features)
    overlap_a_b = len(variant_a_features & variant_b_features)

    print(f"  Original ∩ Variant A: {overlap_orig_a}/30 features")
    print(f"  Original ∩ Variant B: {overlap_orig_b}/30 features")
    print(f"  Variant A ∩ Variant B: {overlap_a_b}/30 features")

    # Features unique to each variant
    unique_to_a = variant_a_features - original_features
    unique_to_b = variant_b_features - original_features

    if unique_to_a:
        print(f"\n  Features unique to Variant A (top 5): {sorted(list(unique_to_a))[:5]}")
    if unique_to_b:
        print(f"  Features unique to Variant B (top 5): {sorted(list(unique_to_b))[:5]}")


print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
