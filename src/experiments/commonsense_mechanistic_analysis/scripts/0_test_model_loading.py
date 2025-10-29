#!/usr/bin/env python3
"""
Story 1: Environment Setup & Data Validation
Test that CommonsenseQA CODI model loads and can run inference.
"""
import json
import sys
import torch
from pathlib import Path
from datasets import load_dataset

# Add paths
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))

from cache_activations_llama import ActivationCacherLLaMA

# Import NTokenPatcher for proper CODI inference
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'scripts' / 'experiments'))
from run_ablation_N_tokens_llama import NTokenPatcher

def format_commonsense_question(example):
    """Format CommonsenseQA example as expected by CODI model."""
    question = example['question']
    choices = example['choices']

    formatted = f"Question: {question}\nChoices:\n"
    for label, text in zip(choices['label'], choices['text']):
        formatted += f"{label}: {text}\n"

    return formatted.strip()

def extract_answer_letter(output_text):
    """
    Extract answer letter (A-E) from model output.
    Expected format: "The answer is: A"
    """
    output_text = output_text.strip().upper()

    # Look for "THE ANSWER IS: X" pattern
    if "THE ANSWER IS:" in output_text:
        parts = output_text.split("THE ANSWER IS:")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            # Extract first letter
            for char in answer_part:
                if char in ['A', 'B', 'C', 'D', 'E']:
                    return char

    # Fallback: look for first A-E in output
    for char in output_text:
        if char in ['A', 'B', 'C', 'D', 'E']:
            return char

    return "INVALID"

def test_model_loading():
    """Test model loading and basic inference."""
    print("=" * 80)
    print("COMMONSENSE QA MODEL LOADING TEST")
    print("=" * 80)

    # Load model
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_commonsense' / 'gsm8k_llama1b_latent_baseline' / 'Llama-3.2-1B-Instruct' / 'ep_3' / 'lr_0.0008' / 'seed_11')
    print(f"\nStep 1: Loading model from {model_path}...")

    try:
        cacher = ActivationCacherLLaMA(model_path)
        print("✓ Model loaded successfully!")
        print(f"  Model type: {type(cacher.model).__name__}")
        print(f"  Device: {cacher.device}")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load validation dataset
    print(f"\nStep 2: Loading CommonsenseQA validation dataset...")

    try:
        dataset = load_dataset('tau/commonsense_qa', split='validation')
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total examples: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test formatting
    test_example = dataset[0]
    print(f"\nStep 3: Testing data formatting...")
    print(f"  ID: {test_example['id']}")
    print(f"  Concept: {test_example['question_concept']}")
    print(f"  Answer Key: {test_example['answerKey']}")

    formatted_q = format_commonsense_question(test_example)
    print(f"\n  Formatted question:")
    print("  " + "\n  ".join(formatted_q.split("\n")))

    # Run baseline inference
    print(f"\nStep 4: Running baseline inference...")

    try:
        # Use NTokenPatcher for proper CODI inference
        patcher = NTokenPatcher(cacher, num_tokens=6)

        # Generate with CODI model
        output_text = patcher._generate_with_patching(formatted_q, max_new_tokens=200)
        predicted = extract_answer_letter(output_text)

        print(f"✓ Inference complete!")
        print(f"  Generated output: {output_text[:150]}...")
        print(f"  Extracted answer: {predicted}")
        print(f"  Expected answer: {test_example['answerKey']}")
        print(f"  Correct: {predicted == test_example['answerKey']}")

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test on a few more examples
    print(f"\nStep 5: Testing on 5 random examples...")

    correct = 0
    test_indices = [0, 10, 100, 500, 1000]

    for idx in test_indices:
        example = dataset[idx]
        formatted_q = format_commonsense_question(example)

        output_text = patcher._generate_with_patching(formatted_q, max_new_tokens=200)
        predicted = extract_answer_letter(output_text)
        expected = example['answerKey']

        is_correct = predicted == expected
        correct += is_correct

        print(f"  Example {idx}: {predicted} vs {expected} {'✓' if is_correct else '✗'}")

    accuracy = correct / len(test_indices)
    print(f"\n  Accuracy: {correct}/{len(test_indices)} = {accuracy:.1%}")

    # Verify GSM8K comparison data exists
    print(f"\nStep 6: Verifying GSM8K comparison data...")

    gsm8k_attention_path = project_root / 'src' / 'experiments' / 'codi_attention_flow' / 'results' / 'llama'
    gsm8k_token_importance_path = project_root / 'src' / 'experiments' / 'codi_attention_interp' / 'results'

    if gsm8k_attention_path.exists():
        files = list(gsm8k_attention_path.glob('*.json'))
        print(f"✓ GSM8K attention data found: {len(files)} JSON files")
    else:
        print(f"✗ GSM8K attention data not found at {gsm8k_attention_path}")
        return False

    if gsm8k_token_importance_path.exists():
        files = list(gsm8k_token_importance_path.glob('*_100.json'))
        print(f"✓ GSM8K token importance data found: {len(files)} JSON files")
    else:
        print(f"✗ GSM8K token importance data not found at {gsm8k_token_importance_path}")
        return False

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print(f"Model configuration:")
    print(f"  - Checkpoint: {model_path}")
    print(f"  - Dataset: tau/commonsense_qa (validation split)")
    print(f"  - Total examples: {len(dataset)}")
    print(f"  - Test accuracy: {accuracy:.1%}")
    print(f"\nReady to proceed with Story 2 (Attention Flow) and Story 3 (Token Importance).")

    return True


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
