#!/usr/bin/env python3
"""
Analyze SAE latent activations for CODI-LLAMA across multiple layers.

Extracts hidden states from L4, L8, L14 (early, middle, late) during CoT processing
and analyzes which SAE features fire at each layer.

Also decodes top layer activations to tokens to see what the model is "thinking".
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
    print("Loading CODI-LLAMA from Local Checkpoint")
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


def decode_hidden_state_to_tokens(model, tokenizer, hidden_state, top_k=5):
    """Decode a hidden state to the top-k most likely tokens with probabilities"""
    with torch.no_grad():
        # hidden_state: (hidden_dim,) -> add batch dim -> (1, hidden_dim)
        hidden_state = hidden_state.unsqueeze(0).to(device).to(torch.bfloat16)

        # Pass through LM head to get logits
        logits = model.codi.lm_head(hidden_state)  # (1, vocab_size)

        # Get probabilities
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (vocab_size,)

        # Get top-k
        top_probs, top_indices = torch.topk(probs, k=top_k)

        # Decode tokens
        top_tokens = []
        for idx, prob in zip(top_indices.cpu().numpy(), top_probs.float().cpu().numpy()):
            token_str = tokenizer.decode([idx])
            top_tokens.append({
                'token_id': int(idx),
                'token_str': token_str,
                'probability': float(prob)
            })

        return top_tokens


def run_cot_with_multilayer_hidden_states(model, tokenizer, training_args, question):
    """Run CoT and capture hidden states from multiple layers at each position"""
    batch_size = 1
    questions = [question]

    # Layer indices to extract (matching SAE training: L4, L8, L14)
    layer_indices = {
        'early': 4,   # L4
        'middle': 8,  # L8
        'late': 14    # L14
    }

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

        # Extract from BoT (Beginning of Thought)
        # Get embeddings and pass through model to get hidden states
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # CoT iterations - extract from multiple layers
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Extract hidden states from specified layers
            position_data = {
                'position': i,
                'layers': {}
            }

            for layer_name, layer_idx in layer_indices.items():
                # outputs.hidden_states is a tuple of (layer_0, layer_1, ..., layer_N)
                # Each is (batch_size, seq_len, hidden_dim)
                # We want the last token position: [:, -1, :]
                hidden_state = outputs.hidden_states[layer_idx][:, -1, :].squeeze().cpu()
                position_data['layers'][layer_name] = {
                    'layer_idx': layer_idx,
                    'hidden_state': hidden_state
                }

            # Also get the final layer for token decoding
            final_hidden_state = outputs.hidden_states[-1][:, -1, :].squeeze().cpu()
            position_data['final_layer'] = {
                'layer_idx': len(outputs.hidden_states) - 1,
                'hidden_state': final_hidden_state
            }

            hidden_states_list.append(position_data)

            # Continue with projection for next iteration
            latent_embd = outputs.hidden_states[-1][:, -1:, :]
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return hidden_states_list


def analyze_sae_activations_multilayer(sae, hidden_states_list, model, tokenizer, top_k_features=20):
    """Analyze SAE activations for each layer separately"""
    results = []

    layer_names = ['early', 'middle', 'late']

    with torch.no_grad():
        for pos_data in hidden_states_list:
            position = pos_data['position']
            position_results = {
                'position': position,
                'layers': {}
            }

            # Analyze each layer
            for layer_name in layer_names:
                layer_data = pos_data['layers'][layer_name]
                hidden_state = layer_data['hidden_state'].to(device).to(torch.bfloat16).unsqueeze(0)

                # Get SAE features
                features = sae.encode(hidden_state).squeeze().float().cpu().numpy()

                # Find firing latents
                firing_mask = features > 0
                firing_indices = np.where(firing_mask)[0]
                firing_values = features[firing_mask]

                # Get top-k most active
                if len(firing_indices) > 0:
                    sorted_idx = np.argsort(firing_values)[::-1]
                    top_k_idx = sorted_idx[:min(top_k_features, len(sorted_idx))]
                    top_latents = firing_indices[top_k_idx]
                    top_values = firing_values[top_k_idx]
                else:
                    top_latents = np.array([])
                    top_values = np.array([])

                position_results['layers'][layer_name] = {
                    'layer_idx': layer_data['layer_idx'],
                    'n_firing': len(firing_indices),
                    'firing_indices': firing_indices.tolist(),
                    'firing_values': firing_values.tolist(),
                    'top_k_latents': top_latents.tolist(),
                    'top_k_values': top_values.tolist(),
                    'sparsity': float(len(firing_indices) / len(features))
                }

            # Decode final layer to tokens
            final_hidden = pos_data['final_layer']['hidden_state']
            top_tokens = decode_hidden_state_to_tokens(model, tokenizer, final_hidden, top_k=5)
            position_results['decoded_tokens'] = top_tokens

            results.append(position_results)

    return results


def main():
    # Load model and SAE
    model, tokenizer, training_args = load_llama_model()
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

    # Run CoT and capture hidden states from multiple layers
    print("\n" + "="*80)
    print("Running Chain-of-Thought Inference (Multi-Layer Extraction)")
    print("="*80)

    hidden_states_list = run_cot_with_multilayer_hidden_states(
        model, tokenizer, training_args, question
    )
    print(f"Captured {len(hidden_states_list)} positions × 3 layers")

    # Analyze SAE activations per layer
    print("\n" + "="*80)
    print("Analyzing SAE Latent Activations (Per Layer)")
    print("="*80)

    activation_results = analyze_sae_activations_multilayer(
        sae, hidden_states_list, model, tokenizer, top_k_features=20
    )

    # Print results for first two positions
    print("\n" + "="*80)
    print("SAE Activation Analysis: First Two Continuous Tokens")
    print("="*80)

    for result in activation_results[:2]:
        print(f"\n{'='*80}")
        print(f"Position {result['position']}:")
        print(f"{'='*80}")

        for layer_name in ['early', 'middle', 'late']:
            layer_result = result['layers'][layer_name]
            print(f"\n  Layer: {layer_name.upper()} (L{layer_result['layer_idx']})")
            print(f"    Total firing latents: {layer_result['n_firing']} / {sae_config['n_features']}")
            print(f"    Sparsity: {layer_result['sparsity']:.4f}")
            print(f"    Top 10 firing latents:")
            for idx, (latent_idx, value) in enumerate(zip(
                layer_result['top_k_latents'][:10],
                layer_result['top_k_values'][:10]
            )):
                print(f"      {idx+1}. Latent {latent_idx}: {value:.4f}")

        print(f"\n  Decoded Tokens (from final layer):")
        for idx, token_info in enumerate(result['decoded_tokens']):
            print(f"    {idx+1}. '{token_info['token_str']}' (ID: {token_info['token_id']}, p={token_info['probability']:.4f})")

    # Save full results
    output = {
        'question': question,
        'answer': answer,
        'sae_config': sae_config,
        'n_positions': len(activation_results),
        'layers_analyzed': ['early (L4)', 'middle (L8)', 'late (L14)'],
        'activation_results': activation_results
    }

    output_path = Path('./sae_activation_analysis_llama_multilayer.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nSaved full results to {output_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    for layer_name in ['early', 'middle', 'late']:
        sparsities = [r['layers'][layer_name]['sparsity'] for r in activation_results]
        n_firings = [r['layers'][layer_name]['n_firing'] for r in activation_results]

        print(f"\nLayer: {layer_name.upper()}")
        print(f"  Average sparsity: {np.mean(sparsities):.4f} ± {np.std(sparsities):.4f}")
        print(f"  Average firing latents: {np.mean(n_firings):.1f} ± {np.std(n_firings):.1f}")
        print(f"  Min firing: {np.min(n_firings)}, Max firing: {np.max(n_firings)}")


if __name__ == '__main__':
    main()
