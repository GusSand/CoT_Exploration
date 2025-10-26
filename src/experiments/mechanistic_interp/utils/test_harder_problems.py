"""Test on harder GSM8K problems to see if we get importance differences."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from codi_interface import CODIInterface, StepImportanceMeasurer

# Load CODI
model_path = str(Path.home() / 'codi_ckpt/llama_gsm8k')
print("Loading CODI model...")
interface = CODIInterface(model_path)

# Load actual GSM8K problems
data_path = Path(__file__).parent.parent / 'data' / 'stratified_test_problems.json'
with open(data_path, 'r') as f:
    problems = json.load(f)

# Test on a few problems of varying difficulty
test_indices = [0, 250, 500, 750]  # One from each difficulty level

measurer = StepImportanceMeasurer(interface, layer_idx=8, debug=False)

print(f"\n{'='*80}")
print("Testing Step Importance on Real GSM8K Problems")
print(f"{'='*80}")

for idx in test_indices:
    problem = problems[idx]
    question = problem['question']
    expected = problem['answer']
    difficulty = problem['reasoning_steps']

    print(f"\n{'='*80}")
    print(f"Problem {idx} ({difficulty}-step problem)")
    print(f"Q: {question[:100]}...")
    print(f"Expected answer: {expected}")
    print(f"{'='*80}")

    # Test just position 3 (zero 0,1,2) to save time
    print("\nMeasuring position 3 (ablating positions 0, 1, 2)...")

    result = measurer.measure_position_importance(question, position=3)

    print(f"\nResults:")
    print(f"  Baseline answer: {result['baseline_answer'][:80]}")
    print(f"  Ablated answer:  {result['ablated_answer'][:80]}")
    print(f"  Answers match: {result['answers_match']}")
    print(f"  Importance: {result['importance_score']:.3f}")

    # Check if correct
    expected_str = str(expected)
    baseline_correct = expected_str in result['baseline_answer'] or result['baseline_answer'] in expected_str
    ablated_correct = expected_str in result['ablated_answer'] or result['ablated_answer'] in expected_str

    print(f"\n  Baseline correct: {baseline_correct}")
    print(f"  Ablated correct:  {ablated_correct}")
    print(f"  Accuracy delta: {1.0 if baseline_correct else 0.0} → {1.0 if ablated_correct else 0.0}")

print(f"\n{'='*80}")
print("Test complete!")
print(f"{'='*80}")
