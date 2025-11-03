#!/usr/bin/env python3
"""
Phase 3: Analyze Top 20 Features for First Test Example

For the first GSM8K test example:
- Extract top 20 firing features at positions 0 and 1
- Across all 3 layers (early, middle, late)
- Generate unigram correlation reports for each feature
"""

import torch
import json
import sys
import os
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

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        reconstruction = self.decoder(features)
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
    """Load trained SAE model"""
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
    sae = sae.to(torch.bfloat16)
    sae.eval()

    return sae, config


def extract_top_features_first_example(model, tokenizer, training_args, sae, question):
    """
    Extract top 20 firing features for first test example at positions 0 and 1.
    """
    batch_size = 1
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

    results = {
        'question': question,
        'positions': {}
    }

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

        # Process positions 0 and 1
        for pos in range(2):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            position_data = {'layers': {}}

            # Extract from each layer
            for layer_name, layer_idx in layer_indices.items():
                hidden_state = outputs.hidden_states[layer_idx][:, -1, :].to(device).to(torch.bfloat16).unsqueeze(0)

                # Get SAE features
                features = sae.encode(hidden_state).squeeze().float().cpu().numpy()

                # Get top 20 firing features
                firing_indices = features.argsort()[-20:][::-1]  # Top 20 in descending order
                firing_values = features[firing_indices]

                position_data['layers'][layer_name] = {
                    'top_20_indices': firing_indices.tolist(),
                    'top_20_values': firing_values.tolist()
                }

            results['positions'][pos] = position_data

            # Continue projection for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1:, :]
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return results


def main():
    print("="*80)
    print("PHASE 3: Analyze Top 20 Features for First Test Example")
    print("="*80)

    # Load GSM8K test set
    print("\nLoading GSM8K test set...")
    gsm8k_dataset = load_dataset("gsm8k", "main")
    test_set = gsm8k_dataset['test']
    first_test_example = test_set[0]

    print(f"\nFirst test example:")
    print(f"Question: {first_test_example['question']}")
    print(f"Answer: {first_test_example['answer']}")

    # Load model and SAE
    model, tokenizer, training_args = load_llama_model()
    sae, sae_config = load_sae()

    # Extract top features
    print("\n" + "="*80)
    print("Extracting Top 20 Features at Positions 0 and 1")
    print("="*80)

    results = extract_top_features_first_example(
        model, tokenizer, training_args, sae,
        first_test_example['question']
    )

    # Display results
    print("\n" + "="*80)
    print("RESULTS: Top 20 Features per Layer per Position")
    print("="*80)

    for pos in [0, 1]:
        print(f"\n{'='*80}")
        print(f"Position {pos}")
        print(f"{'='*80}")

        for layer_name in ['early', 'middle', 'late']:
            layer_data = results['positions'][pos]['layers'][layer_name]
            print(f"\n{layer_name.upper()} (L{4 if layer_name=='early' else 8 if layer_name=='middle' else 14}):")
            print(f"{'Rank':<6} {'Feature':<10} {'Activation':<12}")
            print("-" * 30)

            for rank, (feat_idx, feat_val) in enumerate(zip(
                layer_data['top_20_indices'],
                layer_data['top_20_values']
            ), 1):
                print(f"{rank:<6} {feat_idx:<10} {feat_val:<12.4f}")

    # Collect all unique features to analyze
    all_features_to_analyze = set()
    for pos in [0, 1]:
        for layer_name in ['early', 'middle', 'late']:
            layer_data = results['positions'][pos]['layers'][layer_name]
            all_features_to_analyze.update(layer_data['top_20_indices'])

    print(f"\n\nTotal unique features to analyze: {len(all_features_to_analyze)}")
    print(f"Features: {sorted(all_features_to_analyze)}")

    # Save results
    output = {
        'first_test_example': {
            'question': first_test_example['question'],
            'answer': first_test_example['answer']
        },
        'top_features_analysis': results,
        'unique_features': sorted(list(all_features_to_analyze))
    }

    output_path = Path('./phase3_first_test_example_features.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
