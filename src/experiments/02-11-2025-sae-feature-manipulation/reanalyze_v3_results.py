#!/usr/bin/env python3
"""
Re-analyze V3 results using FIRST number instead of LAST
"""

import json
import re

results_file = "/workspace/CoT_Exploration/src/experiments/02-11-2025-sae-feature-manipulation/results/feature_2203_manipulation_v3_results_20251103_093724.json"

with open(results_file, 'r') as f:
    data = json.load(f)

def extract_first_answer(text):
    """Extract FIRST number from text"""
    text = text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if not numbers:
        return float('inf')
    return float(numbers[0])

print("="*80)
print("V3 RESULTS RE-ANALYSIS (Using FIRST number)")
print("="*80)

print(f"\n{'Problem':<12} {'Interv':<8} {'Target':<6} {'Mag':<6} {'Expected':<10} {'First':<10} {'Last':<10} {'✓First'}")
print("-" * 95)

for result in data['results']:
    first_answer = extract_first_answer(result['final_answer_text'])
    last_answer = result['final_answer_numeric']
    expected = result['expected_answer']

    correct_first = (first_answer == expected)

    print(f"{result['problem_name']:<12} "
          f"{result['intervention_type']:<8} "
          f"{result['intervention_target']:<6} "
          f"{result['magnitude']:<6.1f} "
          f"{expected:<10} "
          f"{str(first_answer):<10} "
          f"{str(last_answer):<10} "
          f"{'✓' if correct_first else 'X'}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY BY INTERVENTION TYPE")
print("="*80)

baseline = [r for r in data['results'] if r['intervention_type'] == 'none']
ablate = [r for r in data['results'] if r['intervention_type'] == 'ablate']
add = [r for r in data['results'] if r['intervention_type'] == 'add']

print("\nBASELINE:")
for r in baseline:
    first = extract_first_answer(r['final_answer_text'])
    print(f"  {r['problem_name']:>10s}: {first} (expected {r['expected_answer']})")

print("\nABLATION (by target and coefficient):")
for target in ['cot', 'bot', 'both']:
    print(f"\n  Target={target}:")
    target_results = [r for r in ablate if r['intervention_target'] == target]
    for r in target_results:
        first = extract_first_answer(r['final_answer_text'])
        print(f"    coef={r['magnitude']}: {first} (expected {r['expected_answer']})")

print("\nADDITION (by problem, target, and magnitude):")
for problem in ['variant_a', 'variant_b']:
    print(f"\n  {problem.upper()}:")
    for target in ['cot', 'bot', 'both']:
        print(f"    Target={target}:")
        target_results = [r for r in add if r['problem_name'] == problem and r['intervention_target'] == target]
        for r in target_results:
            first = extract_first_answer(r['final_answer_text'])
            print(f"      mag={r['magnitude']:>3.0f}: {first} (expected {r['expected_answer']})")

print("\n" + "="*80)
