#!/usr/bin/env python3
"""
Validation script for LLaMA CODI checkpoint.

Tests:
1. Checkpoint loading
2. Model architecture (16 layers, 2048 hidden dim)
3. Forward pass on sample problem
4. Activation caching
"""

import sys
import os
import torch
import json

# Add codi to path
sys.path.insert(0, os.path.join(os.getcwd(), 'codi'))

from cache_activations import ActivationCacher


def validate_checkpoint(checkpoint_path: str):
    """Validate LLaMA checkpoint exists and has correct structure."""
    print("=" * 60)
    print("STEP 1: Validate Checkpoint Structure")
    print("=" * 60)

    # Check files exist
    required_files = ['pytorch_model.bin', 'tokenizer.json']
    for file in required_files:
        file_path = os.path.join(checkpoint_path, file)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024**2)
            print(f"✓ {file}: {size_mb:.1f} MB")
        else:
            print(f"✗ {file}: NOT FOUND")
            return False

    print(f"\n✓ Checkpoint structure valid: {checkpoint_path}\n")
    return True


def validate_model_loading(checkpoint_path: str):
    """Test loading LLaMA model with ActivationCacher."""
    print("=" * 60)
    print("STEP 2: Load LLaMA Model")
    print("=" * 60)

    # Need to modify ActivationCacher for LLaMA - let's test direct loading first
    print(f"Loading from: {checkpoint_path}")
    print("Model: Llama-3.2-1B-Instruct")
    print("Expected: 16 layers, 2048 hidden dim, 6 latent tokens")

    try:
        # This will fail - we need to create llama version
        print("\n⚠️  Need to create LLaMA-specific loader...")
        print("   GPT-2 ActivationCacher uses:")
        print("   - model_name: 'gpt2'")
        print("   - prj_dim: 768")
        print("   - lora_target: ['c_attn', 'c_proj', 'c_fc']")
        print("\n   LLaMA needs:")
        print("   - model_name: 'meta-llama/Llama-3.2-1B-Instruct'")
        print("   - prj_dim: 2048")
        print("   - lora_target: ['q_proj', 'v_proj', 'k_proj', 'o_proj']")

        return False
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return False


def validate_forward_pass(checkpoint_path: str):
    """Test forward pass on sample problem."""
    print("=" * 60)
    print("STEP 3: Test Forward Pass")
    print("=" * 60)

    # Load a sample problem
    pairs_file = 'src/experiments/activation_patching/data/problem_pairs.json'
    with open(pairs_file, 'r') as f:
        pairs = json.load(f)

    sample = pairs[0]['clean']
    print(f"Sample problem: {sample['question'][:80]}...")
    print(f"Expected answer: {sample['answer']}")

    print("\n⚠️  Skipping - need LLaMA-compatible loader first")
    return False


def validate_layer_config():
    """Validate layer configuration for LLaMA."""
    print("=" * 60)
    print("STEP 4: Layer Configuration")
    print("=" * 60)

    total_layers = 16

    layer_config = {
        'early': int(total_layers * 0.25),    # L4
        'middle': int(total_layers * 0.5),     # L8
        'late': int(total_layers * 0.875)      # L14 (14/16 = 87.5%)
    }

    print(f"Total layers: {total_layers}")
    print(f"\nLayer mapping (LLaMA):")
    for name, idx in layer_config.items():
        pct = (idx / total_layers) * 100
        print(f"  {name:8s}: L{idx:2d} ({pct:5.1f}%)")

    print(f"\nComparison to GPT-2 (12 layers):")
    gpt2_config = {'early': 3, 'middle': 6, 'late': 11}
    for name in ['early', 'middle', 'late']:
        gpt2_pct = (gpt2_config[name] / 12) * 100
        llama_pct = (layer_config[name] / total_layers) * 100
        print(f"  {name:8s}: GPT-2 L{gpt2_config[name]:2d} ({gpt2_pct:5.1f}%) | LLaMA L{layer_config[name]:2d} ({llama_pct:5.1f}%)")

    print("\n✓ Layer configuration validated\n")
    return layer_config


def main():
    checkpoint_path = os.path.expanduser(
        "~/codi_ckpt/llama_commonsense/gsm8k_llama1b_latent_baseline/"
        "Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/"
    )

    print("\n" + "=" * 60)
    print("LLaMA CODI Checkpoint Validation")
    print("=" * 60 + "\n")

    # Run validation steps
    results = []

    # Step 1: Checkpoint structure
    results.append(("Checkpoint structure", validate_checkpoint(checkpoint_path)))

    # Step 2: Model loading
    results.append(("Model loading", validate_model_loading(checkpoint_path)))

    # Step 3: Forward pass
    results.append(("Forward pass", validate_forward_pass(checkpoint_path)))

    # Step 4: Layer config
    layer_config = validate_layer_config()
    results.append(("Layer configuration", True))

    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for step, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {step}")

    all_passed = all(r[1] for r in results)

    if not all_passed:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("Need to create LLaMA-compatible ActivationCacher:")
        print("1. Create cache_activations_llama.py")
        print("2. Update model_name_or_path to 'meta-llama/Llama-3.2-1B-Instruct'")
        print("3. Update prj_dim to 2048")
        print("4. Update LoRA target_modules for LLaMA")
        print("5. Update layer indices: L4, L8, L14")
        print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
