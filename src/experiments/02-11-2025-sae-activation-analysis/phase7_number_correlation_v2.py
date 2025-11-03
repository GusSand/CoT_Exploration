#!/usr/bin/env python3
"""
Phase 7: Number Correlation for Top Active Features

Extract SAE activations during first 2 continuous thought iterations,
then show what numbers the top 30 active features correlate with using phase5 data.
"""

import torch
import json
import sys
import os
from pathlib import Path
from datasets import load_dataset
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
print("PHASE 7: NUMBER CORRELATION FOR TOP ACTIVE FEATURES")
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


# Load Phase 5 results (has number precision data)
print("\n[1/5] Loading Phase 5 precision data...")
phase5_path = Path('./phase5_simple_precision_results.json')
with open(phase5_path, 'r') as f:
    phase5_data = json.load(f)

print(f"Loaded precision data for features across {len(phase5_data['feature_reports_by_layer'])} layers")


# Load model
print("\n[2/5] Loading CODI-LLAMA...")
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
print("\n[3/5] Loading SAE...")
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


# Load test example
print("\n[4/5] Loading first test example...")
gsm8k_dataset = load_dataset("gsm8k", "main")
first_test_example = gsm8k_dataset['test'][0]
question = first_test_example['question']
answer = first_test_example['answer']

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")


# Feature tracking hook
class FeatureExtractionHook:
    def __init__(self, sae, layer_name):
        self.sae = sae
        self.layer_name = layer_name
        self.activations = []

    def __call__(self, module, input, output):
        hidden_states = output[0]

        # Only process if 3D tensor
        if len(hidden_states.shape) not in [2, 3]:
            return output

        with torch.no_grad():
            last_hidden = hidden_states.unsqueeze(1) if len(hidden_states.shape) == 2 else hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)

            # Store feature activations (shape: [1, 1, 8192])
            self.activations.append(features[0, 0, :].cpu().float().numpy())

        return output


# Run CODI and extract activations from first 2 continuous thought iterations
print("\n[5/5] Running CODI and extracting SAE activations...")

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
        print(f"\n  Continuous thought iteration {iteration_idx + 1}/2...")

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


# Analyze activations
print("\n" + "="*80)
print("ANALYZING ACTIVATIONS AND NUMBER CORRELATIONS")
print("="*80)

output = {
    'experiment': 'number_correlation_top_active_features',
    'test_example': {'question': question, 'answer': answer},
    'iterations_analyzed': 2,
    'top_k_features': 30,
    'layers': {}
}

for layer in ['early', 'middle', 'late']:
    layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
    layer_num = layer_idx

    print(f"\n{'='*80}")
    print(f"{layer.upper()} LAYER (L{layer_num})")
    print(f"{'='*80}")

    # Get activations for this layer (2 iterations × 8192 features)
    layer_activations = hooks[layer_idx].activations

    if not layer_activations:
        print("  No activations captured!")
        continue

    print(f"\nCaptured {len(layer_activations)} iterations")

    # Aggregate across both iterations: max activation for each feature
    max_activations = np.maximum(layer_activations[0], layer_activations[1])

    # Get top 30 features by activation
    top_30_indices = np.argsort(max_activations)[-30:][::-1]
    top_30_activations = max_activations[top_30_indices]

    print(f"\nTop 30 most active features:\n")
    print(f"{'Rank':<6} {'Feature':<10} {'Activation':<15} {'Iter1':<10} {'Iter2':<10}")
    print("-"*60)

    for rank, (feat_idx, feat_val) in enumerate(zip(top_30_indices, top_30_activations), 1):
        iter1_val = layer_activations[0][feat_idx]
        iter2_val = layer_activations[1][feat_idx]

        print(f"{rank:<6} {feat_idx:<10} {feat_val:<15.4f} {iter1_val:<10.4f} {iter2_val:<10.4f}")

    # Now analyze number correlations for these top 30 features
    print(f"\n{'='*80}")
    print("NUMBER CORRELATIONS (from training data)")
    print(f"{'='*80}\n")

    feature_reports = []

    for feat_idx in top_30_indices:
        feat_id_str = str(feat_idx)

        # Get precision data from phase5
        if feat_id_str not in phase5_data['feature_reports_by_layer'][layer]:
            continue

        feat_data = phase5_data['feature_reports_by_layer'][layer][feat_id_str]
        if feat_data is None:
            continue

        # Filter for number tokens only
        number_tokens = []
        for token_data in feat_data['top_tokens']:
            token = token_data['token']
            # Check if token is a number (digit or space+digit)
            if token.strip().isdigit():
                number_tokens.append(token_data)

        if number_tokens:
            feature_reports.append({
                'feature_id': int(feat_idx),
                'activation': float(max_activations[feat_idx]),
                'iter1_activation': float(layer_activations[0][feat_idx]),
                'iter2_activation': float(layer_activations[1][feat_idx]),
                'feature_frequency': feat_data['feature_frequency'],
                'number_correlations': number_tokens[:15]  # Top 15 numbers
            })

    # Display results
    for report in feature_reports:
        feat_id = report['feature_id']
        activation = report['activation']
        freq = report['feature_frequency']

        print(f"\nFeature {feat_id} (Activation: {activation:.4f}, Frequency: {100*freq:.2f}%)")
        print(f"  Top number correlations (Precision = P(number | feature fires)):")

        for num_data in report['number_correlations'][:10]:
            token = num_data['token']
            precision = num_data['precision']
            count = num_data['count']
            print(f"    '{token}': {100*precision:.1f}% precision ({count} occurrences)")

    output['layers'][layer] = feature_reports

# Save results
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output_path = Path('./phase7_number_correlation_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
