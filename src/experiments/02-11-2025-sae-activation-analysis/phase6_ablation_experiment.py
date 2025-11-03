#!/usr/bin/env python3
"""
Phase 6: Feature Ablation Experiment

Ablate multiplication-selective features and measure impact on final answer logits.
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
print("PHASE 6: MULTIPLICATION FEATURE ABLATION EXPERIMENT")
print("="*80)

# SAE class definition
class SparseAutoencoder(torch.nn.Module):
    """Sparse Autoencoder"""
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


# Load Phase 5 results to identify multiplication-selective features
print("\n[1/8] Loading Phase 5 results...")
phase5_path = Path('./phase5_simple_precision_results.json')
with open(phase5_path, 'r') as f:
    phase5_data = json.load(f)

# Identify features with >70% precision for '*' across all layers
multiplication_features = {}
for layer in ['early', 'middle', 'late']:
    layer_features = []
    for feat_id, feat_data in phase5_data['feature_reports_by_layer'][layer].items():
        if feat_data is None:
            continue

        # Find precision for '*'
        for token_data in feat_data['top_tokens']:
            if token_data['token'] == '*':
                precision = token_data['precision']
                if precision > 0.70:
                    layer_features.append({
                        'feature_id': int(feat_id),
                        'precision': precision,
                        'frequency': feat_data['feature_frequency']
                    })
                break

    multiplication_features[layer] = sorted(layer_features, key=lambda x: x['precision'], reverse=True)

print("\nMultiplication-selective features (precision > 70%):")
for layer in ['early', 'middle', 'late']:
    layer_num = {'early': 4, 'middle': 8, 'late': 14}[layer]
    print(f"\n  {layer.upper()} (L{layer_num}): {len(multiplication_features[layer])} features")
    for feat in multiplication_features[layer][:5]:  # Show top 5
        print(f"    Feature {feat['feature_id']}: {100*feat['precision']:.1f}% precision, "
              f"{100*feat['frequency']:.1f}% frequency")


# Load LLAMA model
def load_llama_model():
    """Load CODI-LLAMA model"""
    print("\n[2/8] Loading CODI-LLAMA...")

    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/.cache/huggingface/hub/models--zen-E--CODI-llama3.2-1b-Instruct/snapshots/b2c88ba224b06b12b52ef39b87f794b98a6eb1c8",
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
    llama_model = llama_model.to(device).to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_args.model_name_or_path)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token

    return llama_model, llama_tokenizer, llama_training_args

model, tokenizer, training_args = load_llama_model()

# Load SAE (single model used for all layers)
print("\n[3/8] Loading SAE...")

sae_path = "/workspace/1_gpt2_codi_and_sae/src/experiments/sae_pilot/results/sae_weights.pt"
checkpoint = torch.load(sae_path, map_location='cpu')

config = checkpoint['config']
print(f"SAE Config: {config}")

sae = SparseAutoencoder(
    input_dim=config['input_dim'],
    n_features=config['n_features'],
    l1_coefficient=config['l1_coefficient']
)

sae.load_state_dict(checkpoint['model_state_dict'])
sae = sae.to(device).to(torch.bfloat16)
sae.eval()

# Load first test example
print("\n[4/8] Loading first test example...")
gsm8k_dataset = load_dataset("gsm8k", "main")
first_test_example = gsm8k_dataset['test'][0]

question = first_test_example['question']
answer = first_test_example['answer']

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")

# Prepare prompt
prompt = f"{question}"
print(f"\nPrompt: {prompt}")

# Get token IDs for answers
answer_tokens = {
    '18': tokenizer.encode('18', add_special_tokens=False)[0],
    ' 18': tokenizer.encode(' 18', add_special_tokens=False)[0],
    '9': tokenizer.encode('9', add_special_tokens=False)[0],
    ' 9': tokenizer.encode(' 9', add_special_tokens=False)[0],
}

print("\nAnswer token IDs:")
for text, token_id in answer_tokens.items():
    print(f"  '{text}': {token_id}")

# Define ablation hook
class AblationHook:
    def __init__(self, sae, features_to_ablate):
        self.sae = sae
        self.features_to_ablate = set(features_to_ablate)
        self.enabled = True

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        # output is tuple (hidden_states,)
        hidden_states = output[0]

        # Encode to SAE features (last position only)
        with torch.no_grad():
            last_hidden = hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)

            # Ablate selected features
            for feat_id in self.features_to_ablate:
                features[:, :, feat_id] = 0.0

            # Decode back
            reconstructed = self.sae.decode(features)

            # Replace last position with ablated version
            hidden_states[:, -1:, :] = reconstructed

        return (hidden_states,) + output[1:]


# Function to run CODI with optional ablation
def run_codi_with_ablation(model, tokenizer, prompt, training_args, ablation_hooks=None):
    """Run CODI and return logits at final position after 'The answer is :'"""

    # Register hooks if provided
    handles = []
    if ablation_hooks:
        for layer_idx, hook in ablation_hooks.items():
            handle = model.model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

    try:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                          max_length=training_args.model_max_length)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Generate with CODI
        with torch.no_grad():
            outputs = model.generate_cot(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=200,
                output_scores=True,
                return_dict_in_generate=True
            )

        # Get generated sequence
        generated_ids = outputs.sequences[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\nGenerated text:\n{generated_text}\n")

        # Find position of ':' after "The answer is"
        tokens = tokenizer.convert_ids_to_tokens(generated_ids)

        # Find "answer is :" pattern
        colon_position = None
        for i in range(len(tokens) - 2):
            if 'answer' in tokens[i].lower():
                # Look for colon in next few tokens
                for j in range(i, min(i+5, len(tokens))):
                    if ':' in tokens[j]:
                        colon_position = j - len(input_ids[0]) + 1
                        print(f"Found ':' at generation position {colon_position}")
                        break
                if colon_position:
                    break

        if colon_position is None or colon_position >= len(outputs.scores):
            print(f"Warning: Could not find ':' token. Using last position.")
            colon_position = len(outputs.scores) - 1

        # Get logits at that position
        logits_at_colon = outputs.scores[colon_position][0]

        return {
            'generated_text': generated_text,
            'logits': logits_at_colon.cpu().float(),
            'colon_position': colon_position
        }

    finally:
        # Remove hooks
        for handle in handles:
            handle.remove()


# Run baseline
print("\n[5/8] Running baseline (no ablation)...")
baseline_result = run_codi_with_ablation(model, tokenizer, prompt, training_args, ablation_hooks=None)

# Extract baseline logits
baseline_logits = {}
for text, token_id in answer_tokens.items():
    baseline_logits[text] = baseline_result['logits'][token_id].item()

print("\nBaseline logits:")
for text, logit in baseline_logits.items():
    print(f"  '{text}': {logit:.4f}")

# Run individual ablations
print("\n[6/8] Running individual feature ablations...")

ablation_results = {}

for layer in ['early', 'middle', 'late']:
    layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]

    for feat_data in multiplication_features[layer][:3]:  # Top 3 features
        feat_id = feat_data['feature_id']

        print(f"\n  Ablating Feature {feat_id} in {layer} layer (L{layer_idx})...")

        # Create ablation hook
        hook = AblationHook(sae, [feat_id])

        # Run with ablation
        result = run_codi_with_ablation(
            model, tokenizer, prompt, training_args,
            ablation_hooks={layer_idx: hook}
        )

        # Extract logits
        logits = {}
        for text, token_id in answer_tokens.items():
            logits[text] = result['logits'][token_id].item()

        ablation_results[f'{layer}_feat_{feat_id}'] = {
            'layer': layer,
            'feature_id': feat_id,
            'precision': feat_data['precision'],
            'logits': logits,
            'generated_text': result['generated_text']
        }

        print(f"    Logits: 18={logits['18']:.4f}, ' 18'={logits[' 18']:.4f}, "
              f"9={logits['9']:.4f}, ' 9'={logits[' 9']:.4f}")

# Run combined ablation
print("\n[7/8] Running combined ablation (all multiplication features)...")

all_hooks = {}
for layer in ['early', 'middle', 'late']:
    layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
    features_to_ablate = [f['feature_id'] for f in multiplication_features[layer]]

    if features_to_ablate:
        hook = AblationHook(sae, features_to_ablate)
        all_hooks[layer_idx] = hook

combined_result = run_codi_with_ablation(model, tokenizer, prompt, training_args, ablation_hooks=all_hooks)

combined_logits = {}
for text, token_id in answer_tokens.items():
    combined_logits[text] = combined_result['logits'][token_id].item()

print("\nCombined ablation logits:")
for text, logit in combined_logits.items():
    print(f"  '{text}': {logit:.4f}")

# Save results
print("\n[8/8] Saving results...")

output = {
    'experiment': 'multiplication_feature_ablation',
    'test_example': {
        'question': question,
        'answer': answer
    },
    'multiplication_features': multiplication_features,
    'answer_tokens': {text: int(token_id) for text, token_id in answer_tokens.items()},
    'baseline': {
        'logits': baseline_logits,
        'generated_text': baseline_result['generated_text']
    },
    'individual_ablations': ablation_results,
    'combined_ablation': {
        'logits': combined_logits,
        'generated_text': combined_result['generated_text']
    }
}

output_path = Path('./phase6_ablation_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")

# Generate comparison report
print("\n" + "="*80)
print("COMPARISON REPORT")
print("="*80)

print("\n### Logit Changes (Ablation - Baseline) ###\n")

print("Individual Feature Ablations:")
print(f"{'Condition':<30} {'Δ logit(18)':<15} {'Δ logit( 18)':<15} {'Δ logit(9)':<15} {'Δ logit( 9)':<15}")
print("-" * 90)

for condition_name, data in ablation_results.items():
    delta_18 = data['logits']['18'] - baseline_logits['18']
    delta_sp18 = data['logits'][' 18'] - baseline_logits[' 18']
    delta_9 = data['logits']['9'] - baseline_logits['9']
    delta_sp9 = data['logits'][' 9'] - baseline_logits[' 9']

    print(f"{condition_name:<30} {delta_18:<15.4f} {delta_sp18:<15.4f} {delta_9:<15.4f} {delta_sp9:<15.4f}")

print("\nCombined Ablation:")
delta_18 = combined_logits['18'] - baseline_logits['18']
delta_sp18 = combined_logits[' 18'] - baseline_logits[' 18']
delta_9 = combined_logits['9'] - baseline_logits['9']
delta_sp9 = combined_logits[' 9'] - baseline_logits[' 9']

print(f"{'All mult features':<30} {delta_18:<15.4f} {delta_sp18:<15.4f} {delta_9:<15.4f} {delta_sp9:<15.4f}")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
