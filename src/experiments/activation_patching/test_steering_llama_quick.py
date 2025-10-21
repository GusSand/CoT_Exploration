#!/usr/bin/env python3
"""Quick test of LLaMA steering to verify it works"""

import json
import sys
import torch
from pathlib import Path

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG
from run_steering_experiment_llama import generate_with_steering, extract_answer_number, answers_match

# Load one test problem
dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_llama.json'
with open(dataset_file) as f:
    dataset = json.load(f)

pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'
with open(pairs_file) as f:
    all_problems = json.load(f)

problem_lookup = {p['pair_id']: p for p in all_problems}

# Get first test problem
test_item = dataset['test_correct'][0]
pair_id = test_item['pair_id']
problem = problem_lookup[pair_id]
question = problem['clean']['question']
expected = test_item['expected']

print(f"Testing pair {pair_id}")
print(f"Question: {question[:100]}...")
print(f"Expected: {expected}")

# Load model
model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
print(f"\nLoading LLaMA...")
cacher = ActivationCacherLLaMA(model_path, device='cuda')

# Load steering direction
direction_file = Path(__file__).parent / 'results' / 'steering_activations_llama' / 'middle' / 'steering_direction.pt'
direction_data = torch.load(direction_file)
direction = direction_data['direction']

print(f"\nDirection norm: {direction_data['direction_norm']:.4f}")

# Test with alpha=0 (no steering)
print(f"\n--- Testing alpha=0.0 (baseline) ---")
try:
    output = generate_with_steering(cacher, question, 'middle', 0.0, direction)
    predicted = extract_answer_number(output)
    correct = answers_match(predicted, expected)

    print(f"Output: {output[:200]}...")
    print(f"Predicted: {predicted}")
    print(f"Correct: {correct}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test with alpha=1.0 (positive steering)
print(f"\n--- Testing alpha=1.0 (amplify) ---")
try:
    output = generate_with_steering(cacher, question, 'middle', 1.0, direction)
    predicted = extract_answer_number(output)
    correct = answers_match(predicted, expected)

    print(f"Output: {output[:200]}...")
    print(f"Predicted: {predicted}")
    print(f"Correct: {correct}")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n✓ Test complete!")
