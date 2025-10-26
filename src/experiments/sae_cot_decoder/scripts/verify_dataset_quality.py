"""
Quick verification of generated full dataset files.

Checks:
1. Files exist and have correct shapes
2. Data types are correct
3. Metadata is consistent
4. No NaN or inf values
"""

import torch
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

print("="*80)
print("DATASET QUALITY VERIFICATION")
print("="*80)

# Load training data
print("\n[1/2] Loading training data...")
train_path = DATA_DIR / "full_train_activations.pt"
train_data = torch.load(train_path)

print(f"  File: {train_path}")
print(f"  Size: {train_path.stat().st_size / (1024**3):.2f} GB")

# Check structure
print("\n  Structure:")
print(f"    Keys: {list(train_data.keys())}")
print(f"    Config: {train_data['config']}")

# Check activations
activations = train_data['activations']
print(f"\n  Activations:")
print(f"    Shape: {activations.shape}")
print(f"    Dtype: {activations.dtype}")
print(f"    Min: {activations.min():.4f}")
print(f"    Max: {activations.max():.4f}")
print(f"    Mean: {activations.mean():.4f}")
print(f"    Std: {activations.std():.4f}")
print(f"    Has NaN: {torch.isnan(activations).any()}")
print(f"    Has Inf: {torch.isinf(activations).any()}")

# Check metadata
metadata = train_data['metadata']
print(f"\n  Metadata:")
print(f"    problem_ids: {len(metadata['problem_ids'])} entries")
print(f"    layers: {len(metadata['layers'])} entries (min={min(metadata['layers'])}, max={max(metadata['layers'])})")
print(f"    positions: {len(metadata['positions'])} entries (min={min(metadata['positions'])}, max={max(metadata['positions'])})")
print(f"    cot_sequences: {len(metadata['cot_sequences'])} entries")

# Verify expected counts
expected_samples = train_data['config']['num_problems'] * train_data['config']['num_layers'] * train_data['config']['num_ct_tokens']
actual_samples = len(activations)
print(f"\n  Sample count verification:")
print(f"    Expected: {expected_samples:,}")
print(f"    Actual: {actual_samples:,}")
print(f"    Match: {expected_samples == actual_samples}")

# Load validation data
print("\n[2/2] Loading validation data...")
val_path = DATA_DIR / "full_val_activations.pt"
val_data = torch.load(val_path)

print(f"  File: {val_path}")
print(f"  Size: {val_path.stat().st_size / (1024**3):.2f} GB")

# Check structure
print(f"\n  Structure:")
print(f"    Config: {val_data['config']}")

# Check activations
val_activations = val_data['activations']
print(f"\n  Activations:")
print(f"    Shape: {val_activations.shape}")
print(f"    Dtype: {val_activations.dtype}")
print(f"    Min: {val_activations.min():.4f}")
print(f"    Max: {val_activations.max():.4f}")
print(f"    Mean: {val_activations.mean():.4f}")
print(f"    Std: {val_activations.std():.4f}")
print(f"    Has NaN: {torch.isnan(val_activations).any()}")
print(f"    Has Inf: {torch.isinf(val_activations).any()}")

# Verify expected counts
expected_val_samples = val_data['config']['num_problems'] * val_data['config']['num_layers'] * val_data['config']['num_ct_tokens']
actual_val_samples = len(val_activations)
print(f"\n  Sample count verification:")
print(f"    Expected: {expected_val_samples:,}")
print(f"    Actual: {actual_val_samples:,}")
print(f"    Match: {expected_val_samples == actual_val_samples}")

# Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

all_checks_passed = True

# Check 1: No NaN or Inf
check1 = not (torch.isnan(activations).any() or torch.isinf(activations).any() or
              torch.isnan(val_activations).any() or torch.isinf(val_activations).any())
print(f"✓ No NaN/Inf values: {check1}")
all_checks_passed &= check1

# Check 2: Sample counts match
check2 = (expected_samples == actual_samples) and (expected_val_samples == actual_val_samples)
print(f"✓ Sample counts match: {check2}")
all_checks_passed &= check2

# Check 3: Shapes are correct
check3 = (activations.shape[1] == 2048) and (val_activations.shape[1] == 2048)
print(f"✓ Hidden dimension is 2048: {check3}")
all_checks_passed &= check3

# Check 4: Total problem count
total_problems = train_data['config']['num_problems'] + val_data['config']['num_problems']
expected_total = 7473  # Full GSM8K train set
check4 = abs(total_problems - expected_total) <= 1  # Allow for rounding
print(f"✓ Total problems close to 7473: {total_problems} (expected ~{expected_total})")
all_checks_passed &= check4

print("\n" + "="*80)
if all_checks_passed:
    print("✅ ALL QUALITY CHECKS PASSED!")
    print("\nDataset is ready for SAE training.")
else:
    print("⚠️  SOME CHECKS FAILED - Review data before training")
print("="*80)
