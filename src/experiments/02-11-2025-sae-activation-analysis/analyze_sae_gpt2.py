#!/usr/bin/env python3
"""
Analyze which SAE latents fire during CODI-GPT2 chain-of-thought processing.

Analyzes the first two continuous tokens of the 1st GSM8K test example.
"""

import torch
import sys
import os
import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login

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
print(f"Using device: {device}")


class SparseAutoencoder(torch.nn.Module):
    """Sparse Autoencoder matching the training code"""
    def __init__(self, input_dim: int = 768, n_features: int = 4096,
                 l1_coefficient: float = 0.0005):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.l1_coefficient = l1_coefficient

        self.encoder = torch.nn.Linear(input_dim, n_features, bias=True)
        self.decoder = torch.nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features"""
        features = torch.nn.functional.relu(self.encoder(x))
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode features back to input space"""
        return self.decoder(features)

    def forward(self, x: torch.Tensor):
        """Forward pass returning reconstruction and features"""
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features


def load_gpt2_model():
    """Load CODI-GPT2 model"""
    print("="*80)
    print("Loading CODI-GPT2 from Local Checkpoint")
    print("="*80)

    gpt2_model_args = ModelArguments(
        model_name_or_path="gpt2",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/1_gpt2_codi_and_sae/codi/models/CODI-gpt2",
        full_precision=True,
        token=None
    )

    gpt2_training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=768,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    gpt2_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=gpt2_model_args.lora_r,
        lora_alpha=gpt2_model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )

    gpt2_model = CODI(gpt2_model_args, gpt2_training_args, gpt2_lora_config)
    checkpoint_path = os.path.join(gpt2_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    gpt2_model.load_state_dict(state_dict, strict=False)
    gpt2_model.codi.tie_weights()
    gpt2_model = gpt2_model.to(device)
    gpt2_model = gpt2_model.float()  # Ensure full precision
    gpt2_model.eval()

    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

    return gpt2_model, gpt2_tokenizer, gpt2_training_args


def load_sae():
    """Load trained SAE model for GPT2 (768-dim)"""
    print("\n" + "="*80)
    print("Loading SAE Model")
    print("="*80)

    sae_path = "/workspace/1_gpt2_codi_and_sae/src/experiments/gpt2_sae_training/results/sae_model_gpt2.pt"
    checkpoint = torch.load(sae_path, map_location='cpu')

    # Check dimensions
    input_dim = checkpoint['encoder.weight'].shape[1]
    n_features = checkpoint['encoder.weight'].shape[0]

    config = {
        'input_dim': input_dim,
        'n_features': n_features,
        'l1_coefficient': 0.0005
    }
    print(f"SAE Config: input_dim={input_dim}, n_features={n_features}")

    sae = SparseAutoencoder(
        input_dim=config['input_dim'],
        n_features=config['n_features'],
        l1_coefficient=config['l1_coefficient']
    )

    sae.load_state_dict(checkpoint)
    sae = sae.to(device)
    sae.eval()

    return sae, config


def run_cot_with_hidden_states(model, tokenizer, training_args, question):
    """Run CoT and capture hidden states at each continuous token position"""
    batch_size = 1
    questions = [question]

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer(questions, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    hidden_states_list = []

    with torch.no_grad():
        # Initial encoding
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        # Store hidden state before and after projection
        hidden_states_list.append({
            'position': 0,
            'hidden_state_before_prj': latent_embd.squeeze().cpu(),
        })

        # Apply projection if enabled (this gives us 768-dim for SAE)
        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)
            hidden_states_list[-1]['hidden_state_after_prj'] = latent_embd.squeeze().cpu()

        # CoT iterations
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1:, :]

            # Store hidden state
            hidden_states_list.append({
                'position': i + 1,
                'hidden_state_before_prj': latent_embd.squeeze().cpu(),
            })

            # Apply projection
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)
                hidden_states_list[-1]['hidden_state_after_prj'] = latent_embd.squeeze().cpu()

    return hidden_states_list


def analyze_sae_activations(sae, hidden_states_list, top_k=20):
    """Analyze which SAE latents fire for each hidden state"""
    results = []

    with torch.no_grad():
        for hs_info in hidden_states_list:
            position = hs_info['position']

            # Use the projected hidden state (768-dim) for SAE
            hidden_state = hs_info['hidden_state_after_prj'].to(device).unsqueeze(0)

            # Get SAE features
            features = sae.encode(hidden_state).squeeze().cpu().numpy()

            # Find firing latents (features > 0)
            firing_mask = features > 0
            firing_indices = np.where(firing_mask)[0]
            firing_values = features[firing_mask]

            # Get top-k most active latents
            if len(firing_indices) > 0:
                sorted_idx = np.argsort(firing_values)[::-1]
                top_k_idx = sorted_idx[:min(top_k, len(sorted_idx))]
                top_latents = firing_indices[top_k_idx]
                top_values = firing_values[top_k_idx]
            else:
                top_latents = np.array([])
                top_values = np.array([])

            results.append({
                'position': position,
                'n_firing': len(firing_indices),
                'firing_indices': firing_indices.tolist(),
                'firing_values': firing_values.tolist(),
                'top_k_latents': top_latents.tolist(),
                'top_k_values': top_values.tolist(),
                'sparsity': float(len(firing_indices) / len(features))
            })

    return results


def main():
    # Load model and SAE
    model, tokenizer, training_args = load_gpt2_model()
    sae, sae_config = load_sae()

    # Load first GSM8K test example
    print("\n" + "="*80)
    print("Loading GSM8K Test Example")
    print("="*80)

    gsm8k_dataset = load_dataset("gsm8k", "main")
    first_example = gsm8k_dataset['test'][0]
    question = first_example['question']
    answer = first_example['answer']

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # Run CoT and capture hidden states
    print("\n" + "="*80)
    print("Running Chain-of-Thought Inference")
    print("="*80)

    hidden_states_list = run_cot_with_hidden_states(model, tokenizer, training_args, question)
    print(f"Captured {len(hidden_states_list)} hidden states (positions 0-{len(hidden_states_list)-1})")

    # Analyze SAE activations
    print("\n" + "="*80)
    print("Analyzing SAE Latent Activations")
    print("="*80)

    activation_results = analyze_sae_activations(sae, hidden_states_list, top_k=20)

    # Print results for first two positions
    print("\n" + "="*80)
    print("SAE Activation Analysis: First Two Continuous Tokens")
    print("="*80)

    for result in activation_results[:2]:
        print(f"\nPosition {result['position']}:")
        print(f"  Total firing latents: {result['n_firing']} / {sae_config['n_features']}")
        print(f"  Sparsity: {result['sparsity']:.4f}")
        print(f"  Top 20 firing latents:")
        for idx, (latent_idx, value) in enumerate(zip(result['top_k_latents'], result['top_k_values'])):
            print(f"    {idx+1}. Latent {latent_idx}: {value:.4f}")

    # Save full results
    output = {
        'question': question,
        'answer': answer,
        'sae_config': sae_config,
        'n_positions': len(activation_results),
        'activation_results': activation_results
    }

    output_path = Path('./sae_activation_analysis_gpt2.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved full results to {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics Across All Positions")
    print("="*80)

    sparsities = [r['sparsity'] for r in activation_results]
    n_firings = [r['n_firing'] for r in activation_results]

    print(f"Average sparsity: {np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}")
    print(f"Average firing latents: {np.mean(n_firings):.1f} ± {np.std(n_firings):.1f}")
    print(f"Min firing: {np.min(n_firings)}, Max firing: {np.max(n_firings)}")


if __name__ == '__main__':
    main()
