#!/usr/bin/env python3
"""
Phase 1: Extract SAE feature activations for entire GSM8K training set.

Extracts hidden states from L4, L8, L14 during CoT processing and passes them
through the trained SAE to identify which features fire for each example.

Saves results for Phase 2 interpretation.
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
from tqdm import tqdm
import time

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
    def __init__(self, input_dim: int = 2048, n_features: int = 8192,
                 l1_coefficient: float = 0.001):
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


def load_llama_model():
    """Load CODI-LLAMA model"""
    print("="*80)
    print("Loading CODI-LLAMA")
    print("="*80)

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
    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    return llama_model, llama_tokenizer, llama_training_args


def load_sae():
    """Load trained SAE model for LLAMA (2048-dim)"""
    print("\n" + "="*80)
    print("Loading SAE Model")
    print("="*80)

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
    sae = sae.to(device)
    sae = sae.to(torch.bfloat16)  # Match model precision
    sae.eval()

    return sae, config


def extract_cot_activations_with_sae(model, tokenizer, training_args, sae, question):
    """
    Extract CoT hidden states and SAE feature activations for a single example.
    Returns compact representation: only firing features per position/layer.
    """
    batch_size = 1

    # Layer indices to extract
    layer_indices = {'early': 4, 'middle': 8, 'late': 14}

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer([question], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    results = []

    with torch.no_grad():
        # Initial encoding
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # CoT iterations - extract from multiple layers
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            position_data = {'position': i, 'layers': {}}

            # Extract and process each layer
            for layer_name, layer_idx in layer_indices.items():
                hidden_state = outputs.hidden_states[layer_idx][:, -1, :].to(device).to(torch.bfloat16).unsqueeze(0)

                # Get SAE features
                features = sae.encode(hidden_state).squeeze().float().cpu().numpy()

                # Find firing latents (activation > 0)
                firing_mask = features > 0
                firing_indices = np.where(firing_mask)[0]
                firing_values = features[firing_mask]

                # Store only firing features (compact representation)
                position_data['layers'][layer_name] = {
                    'layer_idx': layer_idx,
                    'firing_indices': firing_indices.tolist(),
                    'firing_values': firing_values.tolist(),
                    'n_firing': len(firing_indices),
                    'sparsity': float(len(firing_indices) / len(features))
                }

            results.append(position_data)

            # Continue with projection for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1:, :]
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return results


def main():
    start_time = time.time()

    # Load models
    model, tokenizer, training_args = load_llama_model()
    sae, sae_config = load_sae()

    # Load full training set
    print("\n" + "="*80)
    print("Loading GSM8K Training Set")
    print("="*80)

    gsm8k_dataset = load_dataset("gsm8k", "main")
    train_set = gsm8k_dataset['train']
    n_total = len(train_set)
    print(f"Total training examples: {n_total}")

    # Process all examples
    print("\n" + "="*80)
    print("Extracting SAE Activations")
    print("="*80)

    all_results = []
    extraction_start = time.time()

    for idx in tqdm(range(n_total), desc="Extracting"):
        example = train_set[idx]
        question = example['question']
        answer = example['answer']

        # Extract activations
        activations = extract_cot_activations_with_sae(
            model, tokenizer, training_args, sae, question
        )

        # Store with metadata
        example_data = {
            'example_idx': idx,
            'question': question,
            'answer': answer,
            'activations': activations
        }

        all_results.append(example_data)

        # Save checkpoint every 1000 examples
        if (idx + 1) % 1000 == 0:
            checkpoint_path = Path(f'./phase1_checkpoint_{idx+1}.json')
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'sae_config': sae_config,
                    'n_examples_processed': idx + 1,
                    'results': all_results
                }, f)
            print(f"\n  Checkpoint saved: {checkpoint_path}")

    extraction_time = time.time() - extraction_start

    # Save final results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    output = {
        'sae_config': sae_config,
        'n_examples': n_total,
        'layers_analyzed': ['early (L4)', 'middle (L8)', 'late (L14)'],
        'extraction_time_seconds': extraction_time,
        'extraction_rate': n_total / extraction_time,
        'results': all_results
    }

    output_path = Path('./phase1_sae_activations_full.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Full results saved to: {output_path}")

    # Compute and save summary statistics
    print("\n" + "="*80)
    print("Computing Summary Statistics")
    print("="*80)

    # Aggregate statistics across all examples
    layer_names = ['early', 'middle', 'late']
    summary_stats = {}

    for layer_name in layer_names:
        all_sparsities = []
        all_n_firing = []
        feature_activation_counts = np.zeros(sae_config['n_features'])

        for example in all_results:
            for position_data in example['activations']:
                layer_data = position_data['layers'][layer_name]
                all_sparsities.append(layer_data['sparsity'])
                all_n_firing.append(layer_data['n_firing'])

                # Count how many times each feature fired
                for feat_idx in layer_data['firing_indices']:
                    feature_activation_counts[feat_idx] += 1

        summary_stats[layer_name] = {
            'avg_sparsity': float(np.mean(all_sparsities)),
            'std_sparsity': float(np.std(all_sparsities)),
            'avg_n_firing': float(np.mean(all_n_firing)),
            'std_n_firing': float(np.std(all_n_firing)),
            'min_firing': int(np.min(all_n_firing)),
            'max_firing': int(np.max(all_n_firing)),
            'feature_activation_counts': feature_activation_counts.tolist(),
            'n_features_never_fired': int(np.sum(feature_activation_counts == 0)),
            'n_features_fired_at_least_once': int(np.sum(feature_activation_counts > 0))
        }

        print(f"\n{layer_name.upper()} (L{4 if layer_name=='early' else 8 if layer_name=='middle' else 14}):")
        print(f"  Average sparsity: {summary_stats[layer_name]['avg_sparsity']:.4f} ± {summary_stats[layer_name]['std_sparsity']:.4f}")
        print(f"  Average firing: {summary_stats[layer_name]['avg_n_firing']:.1f} ± {summary_stats[layer_name]['std_n_firing']:.1f}")
        print(f"  Range: {summary_stats[layer_name]['min_firing']} - {summary_stats[layer_name]['max_firing']}")
        print(f"  Features that fired at least once: {summary_stats[layer_name]['n_features_fired_at_least_once']} / {sae_config['n_features']}")

    # Save summary
    summary_path = Path('./phase1_summary_statistics.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'sae_config': sae_config,
            'n_examples': n_total,
            'total_positions': n_total * training_args.inf_latent_iterations,
            'layer_stats': summary_stats
        }, f, indent=2)

    print(f"\nSummary statistics saved to: {summary_path}")

    # Final timing report
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"Total examples processed: {n_total}")
    print(f"Extraction time: {extraction_time/60:.1f} minutes")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Extraction rate: {n_total/extraction_time:.2f} examples/second")
    print(f"\nResults ready for Phase 2 interpretation!")


if __name__ == '__main__':
    main()
