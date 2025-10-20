#!/usr/bin/env python3
"""
Quick baseline check for LLaMA on problem pairs.
"""

import json
import sys
import re
from tqdm import tqdm

sys.path.insert(0, '.')
from cache_activations_llama import ActivationCacherLLaMA
from patch_and_eval import ActivationPatcher

def extract_answer_number(text: str):
    """Extract the numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
        r'\$?\s*(-?\d+(?:\.\d+)?)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                return int(num) if num.is_integer() else num
            except ValueError:
                continue

    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            num = float(numbers[-1])
            return int(num) if num.is_integer() else num
        except ValueError:
            pass

    return None

def answers_match(predicted, expected):
    """Check if predicted answer matches expected."""
    if predicted is None or expected is None:
        return False

    try:
        pred_float = float(predicted)
        exp_float = float(expected)
        return abs(pred_float - exp_float) < 0.01
    except (ValueError, TypeError):
        return False

# Load model
print("Loading LLaMA model...")
model_path = "/home/paperspace/codi_ckpt/llama_gsm8k/"
cacher = ActivationCacherLLaMA(model_path)
patcher = ActivationPatcher(cacher)

# Load pairs
with open('data/problem_pairs.json', 'r') as f:
    pairs = json.load(f)

print(f"\nTesting {len(pairs)} pairs...")

clean_correct = 0
corrupted_correct = 0
both_correct = 0
clean_correct_corrupted_wrong = 0

for pair in tqdm(pairs, desc="Testing"):
    # Test clean
    clean_output = patcher.run_without_patch(
        problem_text=pair['clean']['question'],
        max_new_tokens=200
    )
    clean_pred = extract_answer_number(clean_output)
    clean_is_correct = answers_match(clean_pred, pair['clean']['answer'])

    # Test corrupted
    corrupted_output = patcher.run_without_patch(
        problem_text=pair['corrupted']['question'],
        max_new_tokens=200
    )
    corrupted_pred = extract_answer_number(corrupted_output)
    corrupted_is_correct = answers_match(corrupted_pred, pair['corrupted']['answer'])

    if clean_is_correct:
        clean_correct += 1
    if corrupted_is_correct:
        corrupted_correct += 1
    if clean_is_correct and corrupted_is_correct:
        both_correct += 1
    if clean_is_correct and not corrupted_is_correct:
        clean_correct_corrupted_wrong += 1

print("\n" + "=" * 60)
print("LLaMA BASELINE PERFORMANCE")
print("=" * 60)
print(f"Clean correct:                    {clean_correct}/{len(pairs)} ({100*clean_correct/len(pairs):.1f}%)")
print(f"Corrupted correct:                {corrupted_correct}/{len(pairs)} ({100*corrupted_correct/len(pairs):.1f}%)")
print(f"Both correct:                     {both_correct}/{len(pairs)} ({100*both_correct/len(pairs):.1f}%)")
print(f"Clean✓ + Corrupted✗ (usable):     {clean_correct_corrupted_wrong}/{len(pairs)} ({100*clean_correct_corrupted_wrong/len(pairs):.1f}%)")
print("=" * 60)

if clean_correct_corrupted_wrong > 0:
    print(f"\n✓ Have {clean_correct_corrupted_wrong} usable pairs for activation patching!")
else:
    print("\n⚠️  No usable pairs for activation patching.")
