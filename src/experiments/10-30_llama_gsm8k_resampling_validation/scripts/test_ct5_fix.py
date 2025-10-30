#!/usr/bin/env python3
"""Quick test to verify CT5 swapping now works"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_model, extract_answer

# Import FIXED swapping function
import importlib.util
spec = importlib.util.spec_from_file_location("swapping", Path(__file__).parent / "2_implement_swapping.py")
swapping_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(swapping_module)
generate_with_swapped_ct = swapping_module.generate_with_swapped_ct

# Load cache
cache_path = Path(__file__).parent / '../data/ct_hidden_states_cache_pilot.pkl'
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

# Load model
model, tokenizer = load_model()

# Test CT5 swap on first 5 problems
print('Testing FIXED CT5 swapping:')
print('='*60)

changes = 0
for i in range(5):
    problem_A = cache[i]
    problem_B = cache[(i+1) % len(cache)]

    baseline = problem_A['baseline_prediction']
    baseline_num = extract_answer(baseline)

    # Swap CT5
    swapped = generate_with_swapped_ct(model, tokenizer, problem_A, problem_B, swap_position=5)
    swapped_num = extract_answer(swapped)

    changed = (swapped_num != baseline_num)
    if changed:
        changes += 1

    print(f'Problem {i}: Baseline={baseline_num}, Swapped={swapped_num}, Changed={changed}')

print('='*60)
print(f'CT5 swap changed {changes}/5 outputs')
if changes > 0:
    print('✓ Fix worked! CT5 swapping now has an effect')
else:
    print('✗ Still broken - CT5 swaps have no effect')
