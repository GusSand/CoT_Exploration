#!/usr/bin/env python3
"""
Phase 6: Simple Ablation Experiment

Simplified approach: Run CODI generation with and without multiplication feature ablation,
then compare the logits for '18' vs '9' at the final position.
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
import re

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*80)
print("PHASE 6: SIMPLE MULTIPLICATION FEATURE ABLATION")
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


# Load Phase 5 results
print("\n[1/6] Loading Phase 5 results...")
phase5_path = Path('./phase5_simple_precision_results.json')
with open(phase5_path, 'r') as f:
    phase5_data = json.load(f)

# Get top multiplication features per layer
mult_features_to_ablate = {}
for layer in ['early', 'middle', 'late']:
    layer_features = []
    for feat_id, feat_data in phase5_data['feature_reports_by_layer'][layer].items():
        if feat_data is None:
            continue
        for token_data in feat_data['top_tokens']:
            if token_data['token'] == '*' and token_data['precision'] > 0.70:
                layer_features.append(int(feat_id))
                break
    mult_features_to_ablate[layer] = layer_features

print("\nMultiplication features to ablate:")
for layer in ['early', 'middle', 'late']:
    layer_num = {'early': 4, 'middle': 8, 'late': 14}[layer]
    print(f"  {layer.upper()} (L{layer_num}): {len(mult_features_to_ablate[layer])} features")


# Load model
print("\n[2/6] Loading CODI-LLAMA...")
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
print("\n[3/6] Loading SAE...")
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
print("\n[4/6] Loading first test example...")
gsm8k_dataset = load_dataset("gsm8k", "main")
first_test_example = gsm8k_dataset['test'][0]
question = first_test_example['question']
answer = first_test_example['answer']

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")


# Ablation hook
class AblationHook:
    def __init__(self, sae, features_to_ablate):
        self.sae = sae
        self.features_to_ablate = set(features_to_ablate)
        self.enabled = True

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden_states = output[0]

        with torch.no_grad():
            last_hidden = hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)

            # Ablate selected features
            for feat_id in self.features_to_ablate:
                features[:, :, feat_id] = 0.0

            reconstructed = self.sae.decode(features)
            hidden_states[:, -1:, :] = reconstructed

        return (hidden_states,) + output[1:]


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def run_codi_generation(model, tokenizer, question, training_args, ablation_hooks=None):
    """Run full CODI generation adapted from test.py"""

    # Register hooks
    handles = []
    if ablation_hooks:
        for layer_idx, hook in ablation_hooks.items():
            handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

    try:
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

            # Latent iterations
            for i in range(training_args.inf_latent_iterations):
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

            # Add EOT
            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device=device)
                ).unsqueeze(0)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)
                ).unsqueeze(0)

            output = eot_emb

            # Generate tokens
            pred_tokens = []
            all_logits = []

            for i in range(256):  # max_new_tokens
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                all_logits.append(logits[0].cpu().float())

                if training_args.greedy:
                    next_token_id = torch.argmax(logits, dim=-1).item()
                else:
                    next_token_id = torch.multinomial(F.softmax(logits / 0.1, dim=-1), 1).item()

                pred_tokens.append(next_token_id)

                if next_token_id == tokenizer.eos_token_id:
                    break

                output = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([next_token_id], device=device)
                ).unsqueeze(0)

            decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)

            return {
                'generated_text': decoded,
                'predicted_answer': extract_answer_number(decoded),
                'all_logits': all_logits,
                'tokens': pred_tokens
            }

    finally:
        for handle in handles:
            handle.remove()


# Get answer token IDs
answer_tokens = {
    '18': tokenizer.encode('18', add_special_tokens=False)[0],
    ' 18': tokenizer.encode(' 18', add_special_tokens=False)[0],
    '9': tokenizer.encode('9', add_special_tokens=False)[0],
    ' 9': tokenizer.encode(' 9', add_special_tokens=False)[0],
}

print("\nAnswer token IDs:")
for text, token_id in answer_tokens.items():
    print(f"  '{text}': {token_id}")


# Run baseline
print("\n[5/6] Running baseline (no ablation)...")
baseline_result = run_codi_generation(model, tokenizer, question, training_args, ablation_hooks=None)

print(f"\nBaseline generated: {baseline_result['generated_text']}")
print(f"Baseline predicted answer: {baseline_result['predicted_answer']}")
print(f"True answer: 18")

# Get logits at final position
final_logits_baseline = baseline_result['all_logits'][-1]
baseline_logits = {text: final_logits_baseline[token_id].item() for text, token_id in answer_tokens.items()}

print("\nBaseline logits at final position:")
for text, logit in baseline_logits.items():
    print(f"  '{text}': {logit:.4f}")


# Run with ablation
print("\n[6/6] Running with multiplication feature ablation...")

all_hooks = {}
for layer in ['early', 'middle', 'late']:
    layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
    features_to_ablate = mult_features_to_ablate[layer]

    if features_to_ablate:
        hook = AblationHook(sae, features_to_ablate)
        all_hooks[layer_idx] = hook

ablated_result = run_codi_generation(model, tokenizer, question, training_args, ablation_hooks=all_hooks)

print(f"\nAblated generated: {ablated_result['generated_text']}")
print(f"Ablated predicted answer: {ablated_result['predicted_answer']}")
print(f"True answer: 18")

# Get logits at final position
final_logits_ablated = ablated_result['all_logits'][-1]
ablated_logits = {text: final_logits_ablated[token_id].item() for text, token_id in answer_tokens.items()}

print("\nAblated logits at final position:")
for text, logit in ablated_logits.items():
    print(f"  '{text}': {logit:.4f}")


# Compare
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print("\n### Logit Changes (Ablation - Baseline) ###\n")
print(f"{'Token':<10} {'Baseline':<15} {'Ablated':<15} {'Change':<15}")
print("-" * 55)

for text in ['18', ' 18', '9', ' 9']:
    baseline = baseline_logits[text]
    ablated = ablated_logits[text]
    change = ablated - baseline
    print(f"{text:<10} {baseline:<15.4f} {ablated:<15.4f} {change:<15.4f}")

print(f"\nBaseline answer: {baseline_result['predicted_answer']}")
print(f"Ablated answer: {ablated_result['predicted_answer']}")
print(f"True answer: 18")

# Save results
output = {
    'experiment': 'simple_multiplication_ablation',
    'test_example': {'question': question, 'answer': answer},
    'multiplication_features_ablated': mult_features_to_ablate,
    'answer_tokens': {text: int(token_id) for text, token_id in answer_tokens.items()},
    'baseline': {
        'generated_text': baseline_result['generated_text'],
        'predicted_answer': baseline_result['predicted_answer'],
        'final_logits': baseline_logits
    },
    'ablated': {
        'generated_text': ablated_result['generated_text'],
        'predicted_answer': ablated_result['predicted_answer'],
        'final_logits': ablated_logits
    },
    'logit_changes': {text: ablated_logits[text] - baseline_logits[text] for text in answer_tokens.keys()}
}

output_path = Path('./phase6_simple_ablation_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
