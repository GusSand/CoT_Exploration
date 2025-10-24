"""
Extract Token 5 Layer 14 Activations for Multi-Token Intervention

Extracts activation vectors from Token 5 at Layer 14 (late layer) for each
operation type to enable multi-token intervention testing.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add CODI to path
codi_path = Path(__file__).parent.parent.parent.parent / "codi"
sys.path.insert(0, str(codi_path))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType


def extract_token5_l14_activations(
    test_set_path: str,
    model_path: str,
    output_path: str,
    device: str = 'cuda'
):
    """Extract Token 5 L14 activations for each operation type."""

    print("Loading test set...")
    test_set = json.load(open(test_set_path))

    print(f"Loading CODI LLaMA model from {model_path}...")

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=[
            '--model_name_or_path', 'meta-llama/Llama-3.2-1B-Instruct',
            '--output_dir', './tmp',
            '--num_latent', '6',
            '--use_lora', 'True',
            '--ckpt_dir', model_path,
            '--use_prj', 'True',
            '--prj_dim', '2048',
            '--lora_r', '128',
            '--lora_alpha', '32',
            '--lora_init', 'True',
        ]
    )

    model_args.train = False
    training_args.greedy = True

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        init_lora_weights=True,
    )

    # Load model
    model = CODI(model_args, training_args, lora_config)

    # Load checkpoint
    import os
    from safetensors.torch import load_file
    try:
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    model.float()
    model.to(device)
    model.eval()

    tokenizer = model.tokenizer
    num_latent = training_args.num_latent

    print("Model loaded successfully!")

    # Collect activations by operation type
    activations_by_op = defaultdict(list)

    print(f"\nExtracting Token 5 L14 activations from {len(test_set)} problems...")

    for idx, problem in enumerate(test_set):
        if (idx + 1) % 10 == 0:
            print(f"  Progress: {idx + 1}/{len(test_set)}")

        op_type = problem['operation_type']
        question = problem['question']

        with torch.no_grad():
            # Tokenize
            inputs = tokenizer(question, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # Get embeddings
            input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

            # Forward for context
            outputs = model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # BOT embedding
            bot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.bot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)

            # Process latent thoughts
            latent_embd = bot_emb

            for latent_step in range(num_latent):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Extract Token 5 (index 5) at Layer 14 (index 14)
                if latent_step == 5:
                    # Layer 14 hidden states
                    layer_14_hidden = outputs.hidden_states[14]  # Layer 14
                    activation = layer_14_hidden[:, -1, :].cpu().numpy()
                    activations_by_op[op_type].append(activation[0])

                # Continue processing
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection
                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

    print("\nComputing operation means...")

    # Compute mean vectors for each operation type
    operation_means = {}
    for op_type, activations in activations_by_op.items():
        activations = np.array(activations)
        mean_vec = np.mean(activations, axis=0)
        operation_means[op_type] = mean_vec.tolist()
        print(f"  {op_type}: {len(activations)} samples, mean shape: {mean_vec.shape}")

    # Generate random control vector
    random_vec = np.random.randn(mean_vec.shape[0]).astype(np.float32)
    random_vec = random_vec / np.linalg.norm(random_vec) * np.linalg.norm(mean_vec)

    # Save results
    results = {
        'operation_means': operation_means,
        'random_control': random_vec.tolist(),
        'metadata': {
            'token': 5,
            'layer': 14,
            'layer_name': 'late',
            'num_problems': len(test_set),
            'activation_dim': mean_vec.shape[0]
        }
    }

    json.dump(results, open(output_path, 'w'), indent=2)

    print(f"\n✅ Token 5 L14 activations saved to: {output_path}")
    print(f"   - Operation means: {list(operation_means.keys())}")
    print(f"   - Activation dimension: {mean_vec.shape[0]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='~/codi_ckpt/llama_gsm8k')
    parser.add_argument('--test_set', default='test_set_60.json')
    parser.add_argument('--output', default='token5_activation_vectors.json')
    args = parser.parse_args()

    # Resolve paths
    base_dir = Path(__file__).parent
    test_set_path = base_dir / args.test_set
    output_path = base_dir / args.output

    extract_token5_l14_activations(
        str(test_set_path),
        args.model_path,
        str(output_path)
    )


if __name__ == '__main__':
    main()
