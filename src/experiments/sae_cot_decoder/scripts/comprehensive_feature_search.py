"""
Comprehensive search for validation examples across all discovered features.

This searches for problems containing:
- Specific numbers (0, 100, 200, 300, 810, 900)
- Operations (*, /, +, -)
- Special patterns (round numbers, decimals, large numbers)
"""

import json
import re
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
ANALYSIS_DIR = BASE_DIR / "analysis"

print("="*80)
print("COMPREHENSIVE FEATURE VALIDATION SEARCH")
print("="*80)

# Load catalog to see what features we found
with open(ANALYSIS_DIR / "feature_catalog.json", "r") as f:
    catalog = json.load(f)

# Extract all unique tokens that features detect
print("\n[1/4] Analyzing discovered features...")
all_detected_tokens = defaultdict(list)

for position in range(6):
    pos_features = catalog["positions"][str(position)]["top_100_features"]
    for feature in pos_features[:50]:  # Top 50 per position
        fid = feature["feature_id"]
        for token_info in feature["enriched_tokens"][:5]:  # Top 5 tokens per feature
            token = token_info["token_str"]
            enrichment = token_info["enrichment"]
            p_value = token_info["p_value"]

            all_detected_tokens[token].append({
                'position': position,
                'feature_id': fid,
                'enrichment': enrichment,
                'p_value': p_value
            })

# Group tokens by category
number_tokens = {}
operation_tokens = {}
special_tokens = {}

for token, features in all_detected_tokens.items():
    # Calculate average enrichment
    avg_enrichment = sum(f['enrichment'] for f in features) / len(features)
    min_pvalue = min(f['p_value'] for f in features)

    entry = {
        'num_features': len(features),
        'avg_enrichment': avg_enrichment,
        'min_pvalue': min_pvalue,
        'example_features': features[:3]
    }

    if token.isdigit() or (token.replace('.', '').isdigit()):
        number_tokens[token] = entry
    elif token in ['*', '/', '+', '-', '=']:
        operation_tokens[token] = entry
    else:
        special_tokens[token] = entry

print(f"\nDiscovered feature categories:")
print(f"  Number tokens: {len(number_tokens)}")
print(f"  Operation tokens: {len(operation_tokens)}")
print(f"  Special tokens: {len(special_tokens)}")

# Show top tokens by enrichment
print("\n[2/4] Top tokens by feature enrichment:")
print("-"*80)

all_tokens_sorted = sorted(
    [(k, v) for k, v in {**number_tokens, **operation_tokens, **special_tokens}.items()],
    key=lambda x: x[1]['avg_enrichment'],
    reverse=True
)

print("\nToken  | Num Features | Avg Enrichment | Min p-value    | Category")
print("-"*75)
for token, info in all_tokens_sorted[:30]:
    category = "Number" if token in number_tokens else ("Op" if token in operation_tokens else "Special")
    pval_str = f"<10^{int(-np.log10(max(info['min_pvalue'], 1e-300)))}" if info['min_pvalue'] < 0.01 else f"{info['min_pvalue']:.2e}"
    print(f"{token:6s} | {info['num_features']:12d} | {info['avg_enrichment']:14.1%} | {pval_str:14s} | {category}")

# Search GSM8K for these patterns
print("\n[3/4] Searching GSM8K for validation problems...")

gsm8k = load_dataset("gsm8k", "main", split="test[:500]")  # Search more problems

def extract_cot_info(answer_text):
    """Extract detailed info from CoT."""
    calculations = re.findall(r'<<([^>]+)>>', answer_text)

    all_tokens = set()
    numbers = set()
    operations = set()

    for calc in calculations:
        # Extract all components
        tokens = re.findall(r'\d+\.?\d*|[+\-*/=]', calc)
        for t in tokens:
            all_tokens.add(t)
            if t.replace('.', '').isdigit():
                numbers.add(t)
            elif t in ['+', '-', '*', '/', '=']:
                operations.add(t)

    return {
        'calculations': calculations,
        'all_tokens': all_tokens,
        'numbers': numbers,
        'operations': operations
    }

# Search for HIGH-VALUE targets (tokens with strong features)
high_value_targets = {
    # Numbers with specific detectors
    "0": {"type": "number", "feature": 1155, "name": "Zero detector"},
    "00": {"type": "pattern", "feature": 1155, "name": "Double zero"},
    "000": {"type": "pattern", "feature": 1155, "name": "Triple zero"},
    "810": {"type": "number", "feature": 745, "name": "810 detector"},
    "900": {"type": "number", "feature": 1450, "name": "900 detector"},
    "100": {"type": "number", "feature": "TBD", "name": "100 detector"},
    "200": {"type": "number", "feature": "TBD", "name": "200 detector"},
    "300": {"type": "number", "feature": "TBD", "name": "300 detector"},

    # Operations
    "*": {"type": "operation", "feature": "TBD", "name": "Multiplication"},
    "/": {"type": "operation", "feature": "TBD", "name": "Division"},
    "=": {"type": "operation", "feature": "TBD", "name": "Equality"},

    # Special patterns
    ".": {"type": "pattern", "feature": "TBD", "name": "Decimal point"},
}

# Search for each target
found_problems = defaultdict(list)

print(f"  Searching {len(gsm8k)} problems for {len(high_value_targets)} targets...")

for idx, example in enumerate(gsm8k):
    cot_info = extract_cot_info(example['answer'])

    for target, target_info in high_value_targets.items():
        # Check if target appears
        if target in cot_info['all_tokens'] or any(target in calc for calc in cot_info['calculations']):
            found_problems[target].append({
                'idx': idx,
                'question': example['question'],
                'answer': example['answer'],
                'cot': cot_info['calculations'],
                'numbers': list(cot_info['numbers']),
                'operations': list(cot_info['operations']),
                'feature_id': target_info['feature'],
                'feature_name': target_info['name']
            })

print("\n[4/4] Results - Problems found for each target:")
print("-"*80)

results = []
for target, problems in sorted(found_problems.items(), key=lambda x: len(x[1]), reverse=True):
    target_info = high_value_targets[target]
    print(f"\n'{target}' ({target_info['name']}, Feature {target_info['feature']}): {len(problems)} problems")

    if problems:
        # Show first example
        ex = problems[0]
        print(f"  Example: {ex['question'][:70]}...")
        print(f"  CoT: {ex['cot'][:3]}")
        print(f"  Numbers in CoT: {sorted(ex['numbers'])[:10]}")

        results.append({
            'target': target,
            'feature_id': target_info['feature'],
            'feature_name': target_info['name'],
            'num_problems': len(problems),
            'examples': problems[:5]  # Top 5 examples
        })

# Save comprehensive results
output = {
    'discovered_tokens': {
        'numbers': {k: {
            'num_features': v['num_features'],
            'avg_enrichment': v['avg_enrichment'],
            'min_pvalue': v['min_pvalue']
        } for k, v in number_tokens.items()},
        'operations': {k: {
            'num_features': v['num_features'],
            'avg_enrichment': v['avg_enrichment'],
            'min_pvalue': v['min_pvalue']
        } for k, v in operation_tokens.items()}
    },
    'validation_targets': results,
    'total_problems_searched': len(gsm8k)
}

output_path = ANALYSIS_DIR / "comprehensive_validation_targets.json"
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Saved comprehensive results to: {output_path}")

# Print prioritized validation plan
print("\n" + "="*80)
print("PRIORITIZED VALIDATION EXPERIMENTS")
print("="*80)

print("\nHIGH PRIORITY (Specific number detectors):")
priority_targets = ["0", "000", "810", "900", "100"]
for target in priority_targets:
    if target in found_problems and found_problems[target]:
        count = len(found_problems[target])
        feature = high_value_targets[target]['feature']
        print(f"  ✓ '{target}': {count} problems (Feature {feature})")
    else:
        print(f"  ✗ '{target}': 0 problems found")

print("\nMEDIUM PRIORITY (Operations):")
for target in ["*", "/", "="]:
    if target in found_problems and found_problems[target]:
        count = len(found_problems[target])
        print(f"  ✓ '{target}': {count} problems")

print("\nLOW PRIORITY (Special patterns):")
for target in [".", "00"]:
    if target in found_problems and found_problems[target]:
        count = len(found_problems[target])
        print(f"  ✓ '{target}': {count} problems")

print("\n" + "="*80)
print("RECOMMENDED ABLATION EXPERIMENTS")
print("="*80)

print("""
EXPERIMENT SET A: Number Detectors (Highly Specific)

1. Feature 1155 (Zero Detector)
   - Target: Problems with "0" or "000" in CoT
   - Found: {num_zero} problems
   - Ablation: Set F1155 = 0, test if model fails on zero-heavy math
   - Control: Ablate random feature, verify model still works

2. Feature 745 (810 Detector)
   - Target: Problems with "810" in CoT
   - Found: {num_810} problems
   - Ablation: Set F745 = 0, test if model fails on 810 specifically
   - Hypothesis: Very targeted - should only affect 810, not other numbers

3. Feature 1450 (900 Detector)
   - Target: Problems with "900" in CoT
   - Found: {num_900} problems
   - Ablation: Set F1450 = 0, test impact
   - Comparison: Compare to 810 ablation - are they independent?

EXPERIMENT SET B: Operation Detectors (Broader)

4. Multiplication Feature
   - Target: Problems with "*" in CoT
   - Found: {num_mult} problems
   - Ablation: Should affect ALL multiplication, not specific numbers
   - Expected: Broader impact than number-specific features

EXPERIMENT SET C: Cross-Feature Interactions

5. Combined Ablation
   - Target: Problem with "900" AND "*"
   - Ablate: Both F1450 AND multiplication feature
   - Expected: Additive degradation
   - Tests: Whether features are independent or interact

CONTROL EXPERIMENTS:

6. Random Feature Ablation
   - Target: Same problems as Experiment 1
   - Ablate: Random inactive feature (activation < 0.01)
   - Expected: NO impact
   - Proves: Specific features matter, not just any perturbation

7. Position Comparison
   - Target: Same problem, different positions
   - Ablate: F1155 at Position 0 vs Position 3
   - Expected: Different impacts (positions encode different things)
""".format(
    num_zero=len(found_problems.get("0", [])),
    num_810=len(found_problems.get("810", [])),
    num_900=len(found_problems.get("900", [])),
    num_mult=len(found_problems.get("*", []))
))

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print(f"""
1. Review validation targets in: {output_path}

2. Implement ablation script:
   - Load SAE models
   - For each target problem:
     * Extract continuous thoughts
     * Pass through SAE → get features
     * Ablate specific feature (set to 0)
     * Reconstruct from ablated features
     * Test model prediction

3. Run ablations in order of priority:
   Priority 1: Feature 1155 (Zero) - {len(found_problems.get("0", []))} problems
   Priority 2: Feature 745 (810) - {len(found_problems.get("810", []))} problems
   Priority 3: Feature 1450 (900) - {len(found_problems.get("900", []))} problems

4. Compare results:
   - Specific features → specific impacts
   - Proves features are CAUSALLY important
   - Not just correlations!

This validates our SAE interpretations with causal evidence.
""")

print("="*80)
