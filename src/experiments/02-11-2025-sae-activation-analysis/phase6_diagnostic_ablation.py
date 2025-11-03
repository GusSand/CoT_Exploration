#!/usr/bin/env python3
"""
Phase 6: Diagnostic Ablation Experiment

Three-part diagnostic:
1. Verify hooks are firing and actually modifying activations
2. Try ablation during visible token generation phase
3. Check if multiplication features activate during this problem
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
print("PHASE 6: DIAGNOSTIC ABLATION EXPERIMENT")
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
print("\n[SETUP] Loading Phase 5 results...")
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
print("\n[SETUP] Loading CODI-LLAMA...")
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
print("\n[SETUP] Loading SAE...")
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
print("\n[SETUP] Loading first test example...")
gsm8k_dataset = load_dataset("gsm8k", "main")
first_test_example = gsm8k_dataset['test'][0]
question = first_test_example['question']
answer = first_test_example['answer']

print(f"\nQuestion: {question}")
print(f"Answer: {answer}")


# Diagnostic ablation hook with logging
class DiagnosticAblationHook:
    def __init__(self, sae, features_to_ablate, layer_name):
        self.sae = sae
        self.features_to_ablate = set(features_to_ablate)
        self.layer_name = layer_name
        self.enabled = True
        self.call_count = 0
        self.activations_before = []
        self.activations_after = []
        self.feature_activations = []  # Track which features actually fire

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        self.call_count += 1
        hidden_states = output[0]

        # Only process if 3D tensor
        if len(hidden_states.shape) != 3:
            return output

        with torch.no_grad():
            # Process last position
            last_hidden = hidden_states[:, -1:, :]

            # Store before ablation
            self.activations_before.append(last_hidden[0, 0, :5].cpu().float().numpy())

            # Encode
            features = self.sae.encode(last_hidden)

            # Track which multiplication features are active (>0.1 threshold)
            active_mult_features = []
            for feat_id in self.features_to_ablate:
                feat_val = features[0, 0, feat_id].item()
                if feat_val > 0.1:
                    active_mult_features.append((feat_id, feat_val))

            if active_mult_features:
                self.feature_activations.append({
                    'call': self.call_count,
                    'active_features': active_mult_features
                })

            # Ablate selected features
            for feat_id in self.features_to_ablate:
                features[:, :, feat_id] = 0.0

            # Reconstruct
            reconstructed = self.sae.decode(features)

            # Store after ablation
            self.activations_after.append(reconstructed[0, 0, :5].cpu().float().numpy())

            # Replace
            hidden_states[:, -1:, :] = reconstructed

        return (hidden_states,) + output[1:]


# Feature tracking hook (no ablation, just observe)
class FeatureTrackingHook:
    def __init__(self, sae, features_to_track, layer_name):
        self.sae = sae
        self.features_to_track = set(features_to_track)
        self.layer_name = layer_name
        self.enabled = True
        self.activations = []

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        hidden_states = output[0]

        # Only process if 3D tensor
        if len(hidden_states.shape) != 3:
            return output

        with torch.no_grad():
            last_hidden = hidden_states[:, -1:, :]
            features = self.sae.encode(last_hidden)

            # Track which multiplication features are active
            active_features = {}
            for feat_id in self.features_to_track:
                feat_val = features[0, 0, feat_id].item()
                if feat_val > 0.1:  # Threshold
                    active_features[int(feat_id)] = float(feat_val)

            if active_features:
                self.activations.append({
                    'layer': self.layer_name,
                    'active_features': active_features
                })

        return output


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def run_codi_generation(model, tokenizer, question, training_args,
                       ablate_latent=False, ablate_visible=False,
                       track_only=False, mult_features=None, sae=None):
    """
    Run full CODI generation with different ablation modes.

    Args:
        ablate_latent: Apply ablation during latent iterations
        ablate_visible: Apply ablation during visible token generation
        track_only: Just track activations, don't ablate
    """

    # Create hooks
    hooks_dict = {}
    feature_trackers = []

    for layer in ['early', 'middle', 'late']:
        layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer]
        features = mult_features[layer] if mult_features else []

        if not features:
            continue

        if track_only:
            hooks_dict[layer_idx] = FeatureTrackingHook(sae, features, layer)
            feature_trackers.append(hooks_dict[layer_idx])
        else:
            hooks_dict[layer_idx] = DiagnosticAblationHook(sae, features, layer)

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
            # Register hooks for THIS iteration if ablate_latent
            handles = []
            if (ablate_latent or track_only) and hooks_dict:
                for layer_idx, hook in hooks_dict.items():
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

            # Remove hooks after this iteration
            for handle in handles:
                handle.remove()

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

        # Generate tokens - optionally with hooks
        pred_tokens = []
        all_logits = []

        # Register hooks for visible generation if ablate_visible
        handles = []
        if ablate_visible and hooks_dict:
            for layer_idx, hook in hooks_dict.items():
                # Reset hook counters for visible phase
                if hasattr(hook, 'call_count'):
                    hook.call_count = 0
                    hook.activations_before = []
                    hook.activations_after = []
                    hook.feature_activations = []
                handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
                handles.append(handle)

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

        # Remove hooks
        for handle in handles:
            handle.remove()

        decoded = tokenizer.decode(pred_tokens, skip_special_tokens=True)

        result = {
            'generated_text': decoded,
            'predicted_answer': extract_answer_number(decoded),
            'all_logits': all_logits,
            'tokens': pred_tokens
        }

        # Add hook diagnostics
        if not track_only:
            result['hook_diagnostics'] = {}
            for layer_idx, hook in hooks_dict.items():
                if isinstance(hook, DiagnosticAblationHook):
                    result['hook_diagnostics'][layer_idx] = {
                        'call_count': hook.call_count,
                        'activations_changed': len(hook.activations_before) > 0,
                        'sample_before': hook.activations_before[0].tolist() if hook.activations_before else None,
                        'sample_after': hook.activations_after[0].tolist() if hook.activations_after else None,
                        'active_mult_features': hook.feature_activations
                    }
        else:
            result['feature_tracking'] = []
            for tracker in feature_trackers:
                result['feature_tracking'].extend(tracker.activations)

        return result


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


# ============================================================================
# DIAGNOSTIC 1: Track feature activations (no ablation)
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 1: TRACK MULTIPLICATION FEATURE ACTIVATIONS (NO ABLATION)")
print("="*80)

tracking_result = run_codi_generation(
    model, tokenizer, question, training_args,
    ablate_latent=False, ablate_visible=False, track_only=True,
    mult_features=mult_features_to_ablate, sae=sae
)

print(f"\nGenerated: {tracking_result['generated_text'][:100]}...")
print(f"\nFeature activations during latent iterations:")
if tracking_result['feature_tracking']:
    for act in tracking_result['feature_tracking']:
        print(f"  {act['layer']}: {len(act['active_features'])} features active")
        # Show top 5
        sorted_feats = sorted(act['active_features'].items(), key=lambda x: x[1], reverse=True)[:5]
        for feat_id, val in sorted_feats:
            print(f"    Feature {feat_id}: {val:.3f}")
else:
    print("  NO multiplication features activated during latent iterations!")


# ============================================================================
# DIAGNOSTIC 2: Ablate during latent iterations WITH VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 2: ABLATE DURING LATENT ITERATIONS (WITH VERIFICATION)")
print("="*80)

ablated_latent_result = run_codi_generation(
    model, tokenizer, question, training_args,
    ablate_latent=True, ablate_visible=False, track_only=False,
    mult_features=mult_features_to_ablate, sae=sae
)

print(f"\nGenerated: {ablated_latent_result['generated_text'][:100]}...")

# Check hook diagnostics
print("\nHook diagnostics:")
for layer_idx, diag in ablated_latent_result['hook_diagnostics'].items():
    layer_name = {4: 'early', 8: 'middle', 14: 'late'}[layer_idx]
    print(f"\n  {layer_name.upper()} (L{layer_idx}):")
    print(f"    Hook called: {diag['call_count']} times")
    print(f"    Activations changed: {diag['activations_changed']}")
    if diag['sample_before'] and diag['sample_after']:
        before = np.array(diag['sample_before'])
        after = np.array(diag['sample_after'])
        diff = np.abs(before - after).mean()
        print(f"    Mean absolute change in first 5 dims: {diff:.6f}")
    if diag['active_mult_features']:
        print(f"    Active mult features: {len(diag['active_mult_features'])} timesteps")
        for act in diag['active_mult_features'][:3]:
            print(f"      Call {act['call']}: {len(act['active_features'])} features")

# Get logits
final_logits_ablated_latent = ablated_latent_result['all_logits'][-1]
logits_ablated_latent = {text: final_logits_ablated_latent[token_id].item()
                         for text, token_id in answer_tokens.items()}

print("\nFinal logits:")
for text, logit in logits_ablated_latent.items():
    print(f"  '{text}': {logit:.4f}")


# ============================================================================
# DIAGNOSTIC 3: Ablate during VISIBLE token generation
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC 3: ABLATE DURING VISIBLE TOKEN GENERATION")
print("="*80)

ablated_visible_result = run_codi_generation(
    model, tokenizer, question, training_args,
    ablate_latent=False, ablate_visible=True, track_only=False,
    mult_features=mult_features_to_ablate, sae=sae
)

print(f"\nGenerated: {ablated_visible_result['generated_text'][:100]}...")

# Check hook diagnostics
print("\nHook diagnostics:")
for layer_idx, diag in ablated_visible_result['hook_diagnostics'].items():
    layer_name = {4: 'early', 8: 'middle', 14: 'late'}[layer_idx]
    print(f"\n  {layer_name.upper()} (L{layer_idx}):")
    print(f"    Hook called: {diag['call_count']} times")
    if diag['active_mult_features']:
        print(f"    Active mult features: {len(diag['active_mult_features'])} timesteps")

# Get logits
final_logits_ablated_visible = ablated_visible_result['all_logits'][-1]
logits_ablated_visible = {text: final_logits_ablated_visible[token_id].item()
                          for text, token_id in answer_tokens.items()}

print("\nFinal logits:")
for text, logit in logits_ablated_visible.items():
    print(f"  '{text}': {logit:.4f}")


# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

output = {
    'experiment': 'diagnostic_ablation',
    'test_example': {'question': question, 'answer': answer},
    'multiplication_features': mult_features_to_ablate,
    'answer_tokens': {text: int(token_id) for text, token_id in answer_tokens.items()},

    'diagnostic_1_tracking': {
        'description': 'Track multiplication feature activations during latent iterations',
        'generated_text': tracking_result['generated_text'],
        'predicted_answer': tracking_result['predicted_answer'],
        'feature_activations': tracking_result['feature_tracking']
    },

    'diagnostic_2_ablate_latent': {
        'description': 'Ablate multiplication features during latent iterations',
        'generated_text': ablated_latent_result['generated_text'],
        'predicted_answer': ablated_latent_result['predicted_answer'],
        'final_logits': logits_ablated_latent,
        'hook_diagnostics': ablated_latent_result['hook_diagnostics']
    },

    'diagnostic_3_ablate_visible': {
        'description': 'Ablate multiplication features during visible token generation',
        'generated_text': ablated_visible_result['generated_text'],
        'predicted_answer': ablated_visible_result['predicted_answer'],
        'final_logits': logits_ablated_visible,
        'hook_diagnostics': ablated_visible_result['hook_diagnostics']
    }
}

output_path = Path('./phase6_diagnostic_ablation_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {output_path}")
print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
