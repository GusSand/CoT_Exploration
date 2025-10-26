"""Quick debug test to see if zeroing hook is working."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from codi_interface import CODIInterface, StepImportanceMeasurer

# Load CODI
model_path = str(Path.home() / 'codi_ckpt/llama_gsm8k')
print("Loading CODI model...")
interface = CODIInterface(model_path)

# Create measurer with debug enabled
measurer = StepImportanceMeasurer(interface, layer_idx=8, debug=True)

# Test problem
problem = "John has 3 bags with 7 apples each. How many apples does he have in total?"

print("\n" + "="*60)
print("Testing position 3 (zero positions 0, 1, 2)")
print("="*60)

result = measurer.measure_position_importance(problem, position=3)

print(f"\nBaseline: {result['baseline_answer']}")
print(f"Ablated:  {result['ablated_answer']}")
print(f"Match: {result['answers_match']}")
print(f"Importance: {result['importance_score']}")
