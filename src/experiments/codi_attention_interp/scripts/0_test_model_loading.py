#!/usr/bin/env python3
"""
Quick test: Verify model loads and can run inference on one problem.
"""
import json
import sys
import torch
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'scripts' / 'experiments'))

from cache_activations_llama import ActivationCacherLLaMA
from run_ablation_N_tokens_llama import NTokenPatcher, extract_answer_number

def test_model_loading():
    """Test model loading and basic inference."""
    print("=" * 80)
    print("MODEL LOADING TEST")
    print("=" * 80)

    # Load model
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    print(f"\nStep 1: Loading model from {model_path}...")

    try:
        cacher = ActivationCacherLLaMA(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load test dataset
    dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
    print(f"\nStep 2: Loading test problem from {dataset_file}...")

    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    test_problem = dataset[0]
    print(f"✓ Loaded test problem: {test_problem['gsm8k_id']}")
    print(f"  Question: {test_problem['question'][:100]}...")
    print(f"  Expected answer: {test_problem['answer']}")

    # Run baseline inference
    print(f"\nStep 3: Running baseline inference...")

    try:
        patcher = NTokenPatcher(cacher, num_tokens=6)
        output = patcher._generate_with_patching(test_problem['question'], max_new_tokens=200)
        predicted = extract_answer_number(output)

        print(f"✓ Inference complete!")
        print(f"  Generated output: {output[:200]}...")
        print(f"  Extracted answer: {predicted}")
        print(f"  Expected answer: {test_problem['answer']}")
        print(f"  Correct: {predicted == test_problem['answer']}")

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test token ablation
    print(f"\nStep 4: Testing token ablation (zero out token 0)...")

    try:
        # Cache all 6 tokens
        all_acts = patcher.cache_N_token_activations(test_problem['question'], 'middle')
        print(f"✓ Cached {len(all_acts)} token activations")
        print(f"  Activation shape: {all_acts[0].shape}")

        # Zero out token 0
        patched_acts = [torch.zeros_like(all_acts[0])] + all_acts[1:]

        # Run with token 0 ablated
        ablated_output = patcher.run_with_N_tokens_patched(
            problem_text=test_problem['question'],
            patch_activations=patched_acts,
            layer_name='middle',
            max_new_tokens=200
        )

        ablated_pred = extract_answer_number(ablated_output)
        print(f"✓ Ablation complete!")
        print(f"  Ablated prediction: {ablated_pred}")
        print(f"  Baseline prediction: {predicted}")
        print(f"  Ablation changed answer: {ablated_pred != predicted}")

    except Exception as e:
        print(f"✗ Ablation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("Ready to run full experiments.")

    return True


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
