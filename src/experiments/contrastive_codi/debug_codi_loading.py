#!/usr/bin/env python3
"""Debug CODI model loading to identify the issue."""

import torch
import sys
from pathlib import Path

# Add CODI path for imports
sys.path.append('/home/paperspace/dev/CoT_Exploration/codi')

try:
    from src.model import CODI, ModelArguments, TrainingArguments
    print("✅ Successfully imported CODI modules")
except ImportError as e:
    print(f"❌ Failed to import CODI modules: {e}")
    sys.exit(1)

def find_checkpoint_bin(ckpt_root):
    """Find pytorch_model.bin in checkpoint directory."""
    candidates = list(ckpt_root.rglob("pytorch_model.bin"))
    if not candidates:
        raise FileNotFoundError(f"No pytorch_model.bin found in {ckpt_root}")
    print(f"Found {len(candidates)} checkpoint files:")
    for c in candidates:
        print(f"  - {c}")
    # Take the deepest path (most nested checkpoint)
    candidates.sort(key=lambda p: len(p.parts))
    selected = candidates[-1]
    print(f"Selected: {selected}")
    return selected

def test_codi_loading():
    """Test CODI model loading from checkpoint."""

    ckpt_dir = "/home/paperspace/codi_ckpt/contrastive_liars_llama1b_smoke_test"
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔍 Testing CODI loading from: {ckpt_dir}")
    print(f"📱 Device: {device}")

    try:
        # Find checkpoint file
        ckpt_root = Path(ckpt_dir)
        if not ckpt_root.exists():
            print(f"❌ Checkpoint directory does not exist: {ckpt_dir}")
            return False

        weights_path = find_checkpoint_bin(ckpt_root)
        print(f"📁 Using weights: {weights_path}")

        # Test weight loading first
        print("🔄 Loading state dict...")
        state_dict = torch.load(weights_path, map_location=device)
        print(f"✅ State dict loaded. Keys: {len(state_dict.keys())}")
        print("📋 First 5 keys:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {i+1}. {key}")

        # Test CODI instantiation
        print("🔄 Creating CODI model...")
        model_args = ModelArguments(model_name_or_path=base_model, full_precision=False, train=False)
        training_args = TrainingArguments(output_dir=str(ckpt_root), bf16=True, num_latent=6, use_lora=False)
        model = CODI(model_args=model_args, training_args=training_args, lora_config=None)
        print("✅ CODI model created successfully")

        # Test moving to device
        print(f"🔄 Moving model to {device}...")
        model = model.to(device)
        print("✅ Model moved to device")

        # Test loading weights
        print("🔄 Loading weights into model...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✅ Weights loaded!")
        print(f"📊 Missing keys: {len(missing_keys)}")
        print(f"📊 Unexpected keys: {len(unexpected_keys)}")

        if missing_keys:
            print("⚠️  Missing keys (first 5):")
            for key in missing_keys[:5]:
                print(f"    - {key}")

        if unexpected_keys:
            print("⚠️  Unexpected keys (first 5):")
            for key in unexpected_keys[:5]:
                print(f"    - {key}")

        # Test that model has continuous thought components
        print("🔍 Checking for continuous thought components...")
        if hasattr(model, 'num_latent'):
            print(f"✅ Model has num_latent: {model.num_latent}")
        else:
            print("❌ Model missing num_latent attribute")

        # Check for continuous thought layers
        ct_components = []
        for name, module in model.named_modules():
            if 'latent' in name.lower() or 'continuous' in name.lower() or 'ct' in name.lower():
                ct_components.append(name)

        if ct_components:
            print(f"✅ Found {len(ct_components)} continuous thought components:")
            for comp in ct_components[:10]:  # Show first 10
                print(f"    - {comp}")
        else:
            print("❌ No continuous thought components found")

        print("✅ CODI loading test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ CODI loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_codi_loading()
    if success:
        print("\n🎉 CODI model can be loaded successfully!")
    else:
        print("\n💥 CODI model loading failed!")
        sys.exit(1)