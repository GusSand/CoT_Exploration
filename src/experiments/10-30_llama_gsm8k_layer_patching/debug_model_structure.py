"""
Debug script to inspect CODI model structure
"""

import sys
import torch
from pathlib import Path

# Add paths
exp_dir = Path(__file__).parent
sys.path.insert(0, str(exp_dir))

import config
from core.model_loader import load_codi_model

# Load model
print("Loading model...")
model, tokenizer = load_codi_model(
    config.CHECKPOINT_PATH,
    config.MODEL_NAME,
    num_latent=config.NUM_LATENT,
    device="cpu"  # Use CPU for inspection
)

print("\n" + "="*80)
print("Model Structure Inspection")
print("="*80 + "\n")

# Check if model has base_model (LoRA wrapper)
print("Checking model structure...")
print(f"  has base_model: {hasattr(model.codi, 'base_model')}")

if hasattr(model.codi, 'base_model'):
    print(f"  base_model type: {type(model.codi.base_model)}")
    print(f"  has model: {hasattr(model.codi.base_model, 'model')}")

    if hasattr(model.codi.base_model, 'model'):
        print(f"  model type: {type(model.codi.base_model.model)}")
        print(f"  has model.model: {hasattr(model.codi.base_model.model, 'model')}")

        if hasattr(model.codi.base_model.model, 'model'):
            base = model.codi.base_model.model.model
            print(f"  model.model type: {type(base)}")
            print(f"  has layers: {hasattr(base, 'layers')}")

            if hasattr(base, 'layers'):
                print(f"  Number of layers: {len(base.layers)}")
else:
    print(f"  direct model type: {type(model.codi)}")
    print(f"  has model: {hasattr(model.codi, 'model')}")

    if hasattr(model.codi, 'model'):
        print(f"  model type: {type(model.codi.model)}")
        print(f"  has layers: {hasattr(model.codi.model, 'layers')}")

        if hasattr(model.codi.model, 'layers'):
            print(f"  Number of layers: {len(model.codi.model.layers)}")

# Try different access patterns
print("\nTrying different layer access patterns...")

try:
    layers1 = model.codi.base_model.model.model.layers
    print(f"✓ model.codi.base_model.model.model.layers: {len(layers1)} layers")
except Exception as e:
    print(f"❌ model.codi.base_model.model.model.layers: {e}")

try:
    layers2 = model.codi.model.layers
    print(f"✓ model.codi.model.layers: {len(layers2)} layers")
except Exception as e:
    print(f"❌ model.codi.model.layers: {e}")

try:
    layers3 = model.codi.base_model.layers
    print(f"✓ model.codi.base_model.layers: {len(layers3)} layers")
except Exception as e:
    print(f"❌ model.codi.base_model.layers: {e}")

print("\nInspecting full module hierarchy...")
for name, module in model.codi.named_modules():
    if 'layers' in name and not name.endswith('layers'):
        continue
    if 'layers' in name:
        print(f"  {name}: {type(module)}")
        if hasattr(module, '__len__'):
            print(f"    Length: {len(module)}")
        break
