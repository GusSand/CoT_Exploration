#!/usr/bin/env python3
"""
SAE Feature 2203 Manipulation Experiment - V6

FIXES from V5:
- Decode tokens consistently for ALL layer configurations
- Track BOTH BOT and all 6 CoT tokens (7 total)
- Use separate decoding pass to avoid hook conflicts
- Show token predictions before AND after intervention

Compare intervention effects across different layers:
- Layer 4 only (early)
- Layer 8 only (middle)
- Layer 14 only (late)
- All three layers together

Magnitude: 20
Target: CoT tokens only
"""

import torch
import json
import sys
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
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
print("SAE FEATURE 2203 MANIPULATION EXPERIMENT V6")
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
    """Hook for manipulating Feature 2203 at specific layers"""

    def __init__(self, sae, layer_name, feature_id=2203,
                 intervention_type="none", magnitude=20.0):
        self.sae = sae
        self.layer_name = layer_name
        self.feature_id = feature_id
        self.intervention_type = intervention_type
        self.magnitude = magnitude
        self.activations = []
        self.position_counter = 0
        self.bot_position = None

    def __call__(self, module, input, output):
        hidden_states = output[0]

        # Determine if this is BOT or CoT
        is_bot = False
        is_cot = False

        if len(hidden_states.shape) == 2 and hidden_states.shape[0] == 1:
            if self.bot_position is None:
                self.bot_position = self.position_counter
                is_bot = True
            else:
                is_cot = True

        # Skip multi-position tensors
        if len(hidden_states.shape) == 2 and hidden_states.shape[0] != 1:
            return output

        # Skip 3D tensors
        if len(hidden_states.shape) == 3:
            return output

        with torch.no_grad():
            # Encode to SAE features
            last_hidden = hidden_states.unsqueeze(1).to(torch.bfloat16)
            features = self.sae.encode(last_hidden)
            original_activation = features[0, 0, self.feature_id].cpu().float().item()

            # Store activation data
            self.activations.append({
                'position': self.position_counter,
                'layer': self.layer_name,
                'original_activation': original_activation,
                'is_bot': is_bot,
                'is_cot': is_cot
            })

            self.position_counter += 1

        # INTERVENTION on CoT tokens only
        if self.intervention_type == "add" and is_cot:
            with torch.no_grad():
                feature_direction = self.sae.decoder.weight[:, self.feature_id].to(hidden_states.dtype).to(hidden_states.device)
                hidden_states.add_(self.magnitude * feature_direction)

        return output


def decode_cot_tokens(model, tokenizer, question, training_args):
    """
    Decode what tokens the model predicts at each CoT step.
    This is a separate pass WITHOUT interventions to see clean token predictions.
    Returns: {'bot': token_str, 'cot': [token1, token2, ...]}
    """

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

    with torch.no_grad():
        # STEP 1: Process input + BOT token
        outputs = model.codi(
            input_ids=batch["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=batch["attention_mask"]
        )

        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
        original_dtype = latent_embd.dtype

        # Decode BOT token prediction
        bot_logits = model.codi.lm_head(outputs.hidden_states[-1][:, -1, :])
        bot_token_id = torch.argmax(bot_logits, dim=-1).item()
        bot_token_str = tokenizer.decode([bot_token_id])

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd.float()).to(original_dtype)

        # STEP 2: Latent iterations (6 CoT steps)
        cot_tokens = []
        for iteration_idx in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Decode this CoT step
            cot_logits = model.codi.lm_head(outputs.hidden_states[-1][:, -1, :])
            cot_token_id = torch.argmax(cot_logits, dim=-1).item()
            cot_token_str = tokenizer.decode([cot_token_id])
            cot_tokens.append(cot_token_str)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd.float()).to(original_dtype)

    return {'bot': bot_token_str, 'cot': cot_tokens}


def extract_first_answer(text: str) -> float:
    """Extract FIRST number from generated text"""
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if not numbers:
        return float('inf')
    return float(numbers[0])


def run_experiment(model, tokenizer, question, sae, training_args, problem_name,
                   intervention_type="none", magnitude=20.0, layer_config="none"):
    """
    Run experiment with Feature 2203 manipulation

    layer_config: "none", "layer4", "layer8", "layer14", or "all"
    """

    # Determine which layers to apply hooks to
    if layer_config == "none":
        target_layers = {}
    elif layer_config == "layer4":
        target_layers = {4: 'early'}
    elif layer_config == "layer8":
        target_layers = {8: 'middle'}
    elif layer_config == "layer14":
        target_layers = {14: 'late'}
    elif layer_config == "all":
        target_layers = {4: 'early', 8: 'middle', 14: 'late'}
    else:
        raise ValueError(f"Unknown layer_config: {layer_config}")

    # Create hooks
    hooks = {}
    for layer_idx, layer_name in target_layers.items():
        hooks[layer_idx] = FeatureManipulationHook(
            sae=sae,
            layer_name=layer_name,
            feature_id=2203,
            intervention_type=intervention_type,
            magnitude=magnitude
        )

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

    for handle in handles:
        handle.remove()

    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
    original_dtype = latent_embd.dtype

    if training_args.use_prj:
        latent_embd = model.prj(latent_embd.float()).to(original_dtype)

    # STEP 2: Latent iterations (6 CoT steps)
    for iteration_idx in range(training_args.inf_latent_iterations):
        handles = []
        for layer_idx, hook in hooks.items():
            handle = model.codi.model.model.layers[layer_idx].register_forward_hook(hook)
            handles.append(handle)

        with torch.no_grad():
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

        for handle in handles:
            handle.remove()

    # STEP 3: Generate final answer
    with torch.no_grad():
        # Add EOT token
        if training_args.remove_eos:
            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)
        else:
            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device=device)
            ).unsqueeze(0)

        output = eot_emb
        pred_tokens = []
        max_new_tokens = 256

        for i in range(max_new_tokens):
            out = model.codi(
                inputs_embeds=output,
                output_hidden_states=False,
                attention_mask=None,
                use_cache=True,
                output_attentions=False,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

            next_token_id = torch.argmax(logits, dim=-1).item()
            pred_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            output = model.get_embd(model.codi, model.model_name)(
                torch.tensor([next_token_id], dtype=torch.long, device=device)
            ).unsqueeze(1)

        final_answer_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        final_answer_numeric = extract_first_answer(final_answer_text)

    # Decode CoT tokens separately (without intervention)
    token_predictions = decode_cot_tokens(model, tokenizer, question, training_args)

    # Collect activations from all layers
    all_activations = {}
    for layer_idx, hook in hooks.items():
        layer_name = {4: 'early', 8: 'middle', 14: 'late'}[layer_idx]
        all_activations[layer_name] = hook.activations

    results = {
        'problem_name': problem_name,
        'intervention_type': intervention_type,
        'magnitude': magnitude,
        'layer_config': layer_config,
        'final_answer_text': final_answer_text,
        'final_answer_numeric': final_answer_numeric,
        'token_predictions': token_predictions,
        'activations': all_activations
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
sae = sae.to(device).to(torch.bfloat16)
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

layer_configs = ["none", "layer4", "layer8", "layer14", "all"]

for layer_config in layer_configs:
    if layer_config == "none":
        print(f"\n{'='*80}")
        print(f"BASELINE (no intervention)")
        print(f"{'='*80}")
        intervention_type = "none"
    else:
        print(f"\n{'='*80}")
        print(f"ADD FEATURE 2203 (magnitude=20, {layer_config})")
        print(f"{'='*80}")
        intervention_type = "add"

    for prob_name, prob_data in problems.items():
        print(f"\n{prob_name}...")
        result = run_experiment(model, tokenizer, prob_data['question'], sae, training_args,
                               prob_name, intervention_type, 20.0, layer_config)
        result['expected_answer'] = prob_data['answer']
        all_results.append(result)
        print(f"  Answer: {result['final_answer_numeric']} (expected: {prob_data['answer']})")
        print(f"  BOT token: {result['token_predictions']['bot']}")
        print(f"  CoT tokens: {result['token_predictions']['cot']}")

# Save results
print("\n[4/4] Saving results...")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"feature_2203_manipulation_v6_results_{timestamp}.json"

with open(output_file, 'w') as f:
    json.dump({
        'experiment_config': {
            'feature_id': 2203,
            'magnitude': 20.0,
            'intervention_target': 'cot_only',
            'layer_configs_tested': layer_configs,
            'answer_extraction': 'first_number',
            'token_decoding': 'separate_clean_pass'
        },
        'results': all_results
    }, f, indent=2)

print(f"Results saved to: {output_file}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n{'Problem':<12} {'Layer Config':<12} {'Expected':<10} {'Got':<10} {'Correct'}")
print("-" * 60)

for result in all_results:
    correct = result['final_answer_numeric'] == result['expected_answer']
    print(f"{result['problem_name']:<12} "
          f"{result['layer_config']:<12} "
          f"{result['expected_answer']:<10} "
          f"{str(result['final_answer_numeric']):<10} "
          f"{'✓' if correct else 'X'}")

print("\n" + "="*80)
