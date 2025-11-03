#!/usr/bin/env python3
"""
Analyze Feature 2203 Manipulation Results
"""

import json
import sys

# Load results
results_file = sys.argv[1] if len(sys.argv) > 1 else "results/feature_2203_manipulation_results_20251103_083652.json"

with open(results_file, 'r') as f:
    data = json.load(f)

print("="*80)
print("FEATURE 2203 MANIPULATION ANALYSIS")
print("="*80)

# Extract results
results = data['results']

# Group by condition
baseline = [r for r in results if r['intervention_type'] == 'none']
ablate = [r for r in results if r['intervention_type'] == 'ablate']
add = [r for r in results if r['intervention_type'] == 'add']

print("\n1. BASELINE (No Intervention)")
print("-" * 80)
for r in baseline:
    print(f"\n{r['problem_name'].upper()}")
    for layer in ['early', 'middle', 'late']:
        acts = [a['original_activation'] for a in r['layers'][layer]['activations']]
        print(f"  {layer:>6s}: {[round(a, 3) for a in acts]}")

print("\n\n2. ABLATION (Original Problem)")
print("-" * 80)
for r in ablate:
    print(f"\n{r['problem_name'].upper()}")
    for layer in ['early', 'middle', 'late']:
        acts = [a['original_activation'] for a in r['layers'][layer]['activations']]
        print(f"  {layer:>6s}: {[round(a, 3) for a in acts]}")

print("\n\n3. ADDITION (Variants)")
print("-" * 80)
# Group by problem and magnitude
for problem in ['variant_a', 'variant_b']:
    print(f"\n{problem.upper()}:")
    for mag in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
        r = [x for x in add if x['problem_name'] == problem and x['magnitude'] == mag][0]
        print(f"\n  Magnitude {mag}:")
        for layer in ['early', 'middle']:
            acts = [a['original_activation'] for a in r['layers'][layer]['activations']]
            print(f"    {layer:>6s}: {[round(a, 3) for a in acts]}")

print("\n\n4. DECODED TOKENS")
print("-" * 80)
print("\nBaseline Original (early layer):")
tokens = baseline[0]['layers']['early']['decoded_tokens']
for t in tokens:
    print(f"  Position {t['position']}: {t['token_str']!r} (id: {t['token_id']})")

print("\nAblated Original (early layer):")
tokens = ablate[0]['layers']['early']['decoded_tokens']
for t in tokens:
    print(f"  Position {t['position']}: {t['token_str']!r} (id: {t['token_id']})")

print("\n" + "="*80)
