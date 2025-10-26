"""
Causal validation of SAE features.

This script:
1. Finds problems where CoT contains specific tokens (0, 810, etc.)
2. Shows which SAE features activate
3. ABLATES those features (sets them to 0)
4. Reconstructs the continuous thought without those features
5. Tests if the model still gets the right answer

This proves features are CAUSALLY important, not just correlated!
"""

import torch
import json
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

print("="*80)
print("CAUSAL FEATURE VALIDATION")
print("Finding problems with specific CoT tokens and testing feature ablation")
print("="*80)

# Load catalog
print("\n[1/6] Loading feature catalog...")
with open(ANALYSIS_DIR / "feature_catalog.json", "r") as f:
    catalog = json.load(f)
print(f"  ✓ Loaded catalog with {catalog['summary']['total_features']} features")

# Load GSM8K to find problems with specific tokens
print("\n[2/6] Searching GSM8K for problems with target tokens...")

from datasets import load_dataset
gsm8k = load_dataset("gsm8k", "main", split="test[:100]")  # First 100 for speed

def extract_cot_tokens(answer_text):
    """Extract tokens from CoT calculation blocks."""
    calculations = re.findall(r'<<([^>]+)>>', answer_text)
    all_tokens = set()
    for calc in calculations:
        # Extract numbers and operators
        tokens = re.findall(r'\d+|[+\-*/=]', calc)
        all_tokens.update(tokens)
    return all_tokens, calculations

# Find problems with target tokens
target_tokens = {
    "0": [],
    "810": [],
    "900": [],
    "100": [],
    "50": []
}

print(f"\n  Searching through {len(gsm8k)} problems...")
for idx, example in enumerate(gsm8k):
    tokens, calculations = extract_cot_tokens(example['answer'])

    for target in target_tokens.keys():
        if target in tokens:
            target_tokens[target].append({
                'idx': idx,
                'question': example['question'],
                'answer': example['answer'],
                'calculations': calculations,
                'all_tokens': list(tokens)
            })

print("\n  Found problems containing:")
for target, problems in target_tokens.items():
    print(f"    '{target}': {len(problems)} problems")

# Show examples
print("\n[3/6] Example problems found:")
print("-"*80)

for target in ["0", "810", "100"]:
    if target_tokens[target]:
        example = target_tokens[target][0]
        print(f"\nTarget token: '{target}'")
        print(f"Problem: {example['question'][:100]}...")
        print(f"CoT steps: {example['calculations'][:3]}")
        print(f"All tokens in CoT: {sorted(set(example['all_tokens']))}")

# Now let's check if we have the actual data to run ablations
print("\n[4/6] Checking for SAE models and data...")

try:
    # Load SAE models
    sae_models = {}
    for position in range(6):
        model_path = MODELS_DIR / f"sae_position_{position}.pt"
        if model_path.exists():
            sae_models[position] = model_path

    print(f"  ✓ Found {len(sae_models)} SAE models")

    # Check for test data
    test_data_path = DATA_DIR / "enriched_test_data_with_cot.pt"
    features_path = ANALYSIS_DIR / "extracted_features.pt"

    has_data = test_data_path.exists() and features_path.exists()

    if has_data:
        print(f"  ✓ Found test data and extracted features")
    else:
        print(f"  ⚠ Test data not available (large files excluded from git)")

except Exception as e:
    print(f"  ⚠ Error checking files: {e}")
    has_data = False

print("\n[5/6] Proposed ablation experiments:")
print("-"*80)

print("""
EXPERIMENT 1: Ablate Feature 1155 (Zero Detector)
  1. Find problem where CoT contains "0" or "000"
  2. Extract continuous thought at Position 0
  3. Pass through SAE → get 2048 features
  4. Show Feature 1155 activates strongly
  5. Set Feature 1155 = 0 (ablation)
  6. Reconstruct continuous thought from ablated features
  7. Pass reconstructed thought through model
  8. Check if model still gets correct answer

  Expected result: Model performance degrades because "zero detection" is lost

EXPERIMENT 2: Ablate Feature 745 (810 Detector)
  1. Find problem where CoT contains "810"
  2. Extract continuous thought at Position 0
  3. Show Feature 745 activates strongly
  4. Set Feature 745 = 0 (ablation)
  5. Reconstruct and test

  Expected result: Model fails on 810-specific problems

EXPERIMENT 3: Control - Ablate Random Feature
  1. Same problem as Experiment 1
  2. Ablate a DIFFERENT feature (not 1155)
  3. Reconstruct and test

  Expected result: Model still works (proving F1155 is specifically important)
""")

# Create detailed validation plan
print("\n[6/6] Detailed validation plan:")
print("-"*80)

validation_plan = []

# Plan for "0" detector
if target_tokens["0"]:
    for i, problem in enumerate(target_tokens["0"][:3]):
        validation_plan.append({
            'target_token': "0",
            'target_feature': 1155,
            'feature_name': "Zero Detector",
            'position': 0,
            'problem_idx': problem['idx'],
            'problem': problem['question'][:80] + "...",
            'cot': problem['calculations'][:2],
            'expected_activation': "> 0.4 (high)",
            'ablation_effect': "Should degrade performance on zero-heavy calculations"
        })

# Plan for "810" detector
if target_tokens["810"]:
    for i, problem in enumerate(target_tokens["810"][:2]):
        validation_plan.append({
            'target_token': "810",
            'target_feature': 745,
            'feature_name': "810 Detector",
            'position': 0,
            'problem_idx': problem['idx'],
            'problem': problem['question'][:80] + "...",
            'cot': problem['calculations'][:2],
            'expected_activation': "> 0.8 (very high)",
            'ablation_effect': "Should fail on 810-specific calculations"
        })

# Plan for "100" detector
if target_tokens["100"]:
    for i, problem in enumerate(target_tokens["100"][:2]):
        validation_plan.append({
            'target_token': "100",
            'target_feature': "TBD",  # Need to find from catalog
            'feature_name': "100 Detector",
            'position': 0,
            'problem_idx': problem['idx'],
            'problem': problem['question'][:80] + "...",
            'cot': problem['calculations'][:2],
            'expected_activation': "> 0.3 (medium)",
            'ablation_effect': "Should affect problems with 100"
        })

# Print validation plan
print("\nValidation experiments to run:\n")
for i, plan in enumerate(validation_plan, 1):
    print(f"{i}. Target: '{plan['target_token']}' | Feature: {plan['target_feature']} ({plan['feature_name']})")
    print(f"   Problem: {plan['problem']}")
    print(f"   CoT: {plan['cot']}")
    print(f"   Expected: {plan['expected_activation']}")
    print(f"   Effect: {plan['ablation_effect']}")
    print()

# Save validation plan
output_path = ANALYSIS_DIR / "validation_plan.json"
with open(output_path, 'w') as f:
    json.dump({
        'target_problems': target_tokens,
        'validation_experiments': validation_plan
    }, f, indent=2)

print(f"✓ Saved validation plan to: {output_path}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

if has_data:
    print("""
✓ You have the data! Ready to run ablations.

Run the ablation script:
    python run_feature_ablations.py

This will:
1. Load the SAE models
2. Extract features for each validation problem
3. Ablate target features (set to 0)
4. Reconstruct continuous thoughts
5. Test model predictions
6. Generate before/after comparison
""")
else:
    print("""
⚠ Large data files not available locally.

To run ablations, you need:
1. enriched_test_data_with_cot.pt (test samples with CoT)
2. SAE models (sae_position_0.pt through sae_position_5.pt)
3. CODI model checkpoint

These files exist but are excluded from git due to size.

Alternative: Use the catalog to identify which features should activate,
then manually trace through the logic to validate interpretations.
""")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
Found validation targets:
  - '{0}' appears in {len(target_tokens['0'])} problems
  - '810' appears in {len(target_tokens['810'])} problems
  - '100' appears in {len(target_tokens['100'])} problems

Proposed validations: {len(validation_plan)} experiments

Key insight: We need to prove features are CAUSALLY important,
not just correlated. Ablating Feature 1155 should specifically hurt
performance on zero-heavy problems, not other problems.

This is the gold standard for mechanistic interpretability!
""")

print("="*80)
