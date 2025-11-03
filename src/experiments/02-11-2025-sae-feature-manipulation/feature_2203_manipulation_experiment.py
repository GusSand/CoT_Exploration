#!/usr/bin/env python3
"""
SAE Feature 2203 Manipulation Experiment

Based on phase9_feature_2203_timing.py, but MANIPULATES Feature 2203:
- Ablates (zeros) feature for original problem
- Adds feature to variant_a and variant_b
- Tests different intervention magnitudes
- Records decoded tokens at each CoT step and final answer
"""

import torch
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import numpy as np
from datetime import datetime

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
print("SAE FEATURE 2203 MANIPULATION EXPERIMENT")
print("="*80)


class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int = 2048, n_features: int = 8192, l1_coefficient: float = 0.001):
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


class FeatureManipulationHook:
    """Hook for MANIPULATING Feature 2203 during forward pass"""

    def __init__(self, sae, model, tokenizer, layer_name, feature_id=2203,
                 intervention_type="none", magnitude=1.0):
        self.sae = sae
        self.model = model
        self.tokenizer = tokenizer
        self.layer_name = layer_name
        self.feature_id = feature_id
        self.intervention_type = intervention_type  # "none", "ablate", "add"
        self.magnitude = magnitude
        self.activations = []
        self.decoded_tokens = []
        self.position_counter = 0

    def __call__(self, module, input, output):
        hidden_states = output[0]

        # Skip initial encoding (many positions) - only process single latent positions
        # Latent iterations have shape [1, 2048] (single position)
        # Initial encoding has shape [N, 2048] where N > 1 (many positions)
        if len(hidden_states.shape) == 2 and hidden_states.shape[0] != 1:
            return output

        # Also skip if it's 3D (shouldn't happen but just in case)
        if len(hidden_states.shape) == 3:
            return output

        with torch.no_grad():
            # Process the hidden state (already single position [1, 2048])
            last_hidden = hidden_states.unsqueeze(1).to(torch.bfloat16)  # [1, 1, 2048], match SAE dtype

            # Encode to SAE features
            features = self.sae.encode(last_hidden)

            # Record original activation
            original_activation = features[0, 0, self.feature_id].cpu().float().item()

            # INTERVENTION: Modify feature in feature space
            if self.intervention_type == "ablate":
                features[:, :, self.feature_id] = 0.0
            elif self.intervention_type == "add":
                features[:, :, self.feature_id] += self.magnitude

            # Decode token for tracking (need to convert back to model's original dtype for lm_head)
            # hidden_states from output[0] is in the model's original dtype
            model_hidden_for_decode = last_hidden.squeeze(1).to(output[0].dtype)
            logits = self.model.codi.lm_head(model_hidden_for_decode)
            if len(logits.shape) > 1:
                token_id = torch.argmax(logits[-1], dim=-1).item()
            else:
                token_id = torch.argmax(logits, dim=-1).item()
            token_str = self.tokenizer.decode([token_id])

            # Store data
            self.activations.append({
                'position': self.position_counter,
                'layer': self.layer_name,
                'original_activation': original_activation,
                'intervention': self.intervention_type,
                'magnitude': self.magnitude if self.intervention_type == "add" else 0.0
            })

            self.decoded_tokens.append({
                'position': self.position_counter,
                'token_id': token_id,
                'token_str': token_str
            })

            self.position_counter += 1

            # Apply intervention: decode and modify hidden states
            if self.intervention_type != "none":
                original_dtype = hidden_states.dtype
                reconstructed = self.sae.decode(features)  # [1, 1, 2048] in bfloat16
                hidden_states = reconstructed.squeeze(1).to(original_dtype)  # [1, 2048], match original dtype

        return (hidden_states,) + output[1:]


def run_experiment(model, tokenizer, question, sae, training_args, problem_name,
                   intervention_type="none", magnitude=1.0):
    """Run single experiment with intervention"""

    # Create hooks (indexed by layer number for easier access)
    hooks = {}
    for layer_name, layer_idx in [('early', 4), ('middle', 8), ('late', 14)]:
        hook = FeatureManipulationHook(
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            layer_name=layer_name,
            feature_id=2203,
            intervention_type=intervention_type,
            magnitude=magnitude
        )
        hooks[layer_idx] = hook

    # Prepare input
    batch = tokenizer(question, return_tensors="pt", padding="longest")
    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).unsqueeze(0)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).unsqueeze(0)

    batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1).to(device)
    batch["attention_mask"] = torch.cat((
        batch["attention_mask"],
        torch.ones(batch["attention_mask"].size(0), bot_tensor.size(1), dtype=torch.long)
    ), dim=1).to(device)

    # STEP 1: Process input + BOT token
    print(f"  Processing BOT...")

    # Register hooks for question encoding
    handles = []
    for layer_idx, hook in hooks.items():
        handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
        handles.append(handle)

    with torch.no_grad():
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
    original_dtype = latent_embd.dtype

    if training_args.use_prj:
        latent_embd = model.prj(latent_embd.float()).to(original_dtype)

    # STEP 2: Latent iterations
    print(f"  Processing {training_args.inf_latent_iterations} CoT iterations...")
    for iteration_idx in range(training_args.inf_latent_iterations):
        # Register hooks
        handles = []
        for layer_idx, hook in hooks.items():
            handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

        with torch.no_grad():
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
            latent_embd = model.prj(latent_embd.float()).to(original_dtype)

        # Remove hooks
        for handle in handles:
            handle.remove()

    # STEP 3: Decode final answer using lm_head on latent embedding
    print(f"  Decoding final answer from latent state...")

    with torch.no_grad():
        # Get logits directly from final latent embedding
        logits = model.codi.lm_head(latent_embd.squeeze(1))

        # Get most likely token
        predicted_token_id = torch.argmax(logits, dim=-1).item()
        final_answer = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()

    # Extract numeric answer
    try:
        numeric_answer = int(''.join(filter(str.isdigit, final_answer.split()[0])))
    except:
        numeric_answer = None

    # Collect results
    results = {
        'problem_name': problem_name,
        'intervention_type': intervention_type,
        'magnitude': magnitude,
        'final_answer_text': final_answer,
        'final_answer_numeric': numeric_answer,
        'layers': {}
    }

    # Package results by layer name
    layer_map = {4: 'early', 8: 'middle', 14: 'late'}
    for layer_idx, hook in hooks.items():
        layer_name = layer_map[layer_idx]
        results['layers'][layer_name] = {
            'activations': hook.activations,
            'decoded_tokens': hook.decoded_tokens
        }

    return results


# Main execution
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
model = model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
print("Model loaded")

# Load SAE
print("\n[2/4] Loading SAE...")
sae_path = "/workspace/1_gpt2_codi_and_sae/src/experiments/sae_pilot/results/sae_weights.pt"
checkpoint = torch.load(sae_path, map_location='cpu')
config = checkpoint['config']
sae = SparseAutoencoder(input_dim=config['input_dim'], n_features=config['n_features'], l1_coefficient=config['l1_coefficient'])
sae.load_state_dict(checkpoint['model_state_dict'])
sae = sae.to(device).to(torch.bfloat16)  # Match model precision
sae.eval()
print("SAE loaded")

# Define problems
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

# Run experiments
print("\n[3/4] Running experiments...")
all_results = []
magnitudes = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

# Baseline
print("\nBASELINE (no intervention):")
for prob_name, prob_data in problems.items():
    print(f"  {prob_name}...")
    result = run_experiment(model, tokenizer, prob_data['question'], sae, training_args,
                           prob_name, "none", 0.0)
    result['expected_answer'] = prob_data['answer']
    all_results.append(result)
    print(f"    Answer: {result['final_answer_numeric']} (expected: {prob_data['answer']})")

# Original: ablate
print("\nORIGINAL (ablate Feature 2203):")
print(f"  original...")
result = run_experiment(model, tokenizer, problems['original']['question'], sae, training_args,
                       'original', "ablate", 0.0)
result['expected_answer'] = problems['original']['answer']
all_results.append(result)
print(f"    Answer: {result['final_answer_numeric']} (expected: {problems['original']['answer']})")

# Variants: add
print("\nVARIANTS (add Feature 2203):")
for prob_name in ['variant_a', 'variant_b']:
    for mag in magnitudes:
        print(f"  {prob_name} (magnitude={mag})...")
        result = run_experiment(model, tokenizer, problems[prob_name]['question'], sae, training_args,
                               prob_name, "add", mag)
        result['expected_answer'] = problems[prob_name]['answer']
        all_results.append(result)
        print(f"    Answer: {result['final_answer_numeric']} (expected: {problems[prob_name]['answer']})")

# Save results
print("\n[4/4] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"feature_2203_manipulation_results_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump({
        'experiment_config': {
            'feature_id': 2203,
            'layers_tracked': ['early', 'middle', 'late'],
            'layer_indices': [4, 8, 14],
            'problems': list(problems.keys()),
            'intervention_magnitudes': magnitudes
        },
        'results': all_results
    }, f, indent=2)

print(f"Results saved to: {output_file}")

# Print summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Problem':<15} {'Intervention':<12} {'Magnitude':<10} {'Expected':<10} {'Got':<10} {'Correct'}")
print("-" * 80)

for result in all_results:
    correct = result['final_answer_numeric'] == result['expected_answer'] if result['final_answer_numeric'] is not None else False
    print(f"{result['problem_name']:<15} "
          f"{result['intervention_type']:<12} "
          f"{result['magnitude']:<10.1f} "
          f"{result['expected_answer']:<10} "
          f"{str(result['final_answer_numeric']):<10} "
          f"{'OK' if correct else 'X'}")

print("\n" + "="*80)
