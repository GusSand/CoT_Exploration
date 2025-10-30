#!/usr/bin/env python3
"""
Test loading zen-E/CommonsenseQA-GPT4omini dataset
Using the same approach as CODI training script (train.py line 381)
"""
import sys
from datasets import load_dataset

print("Attempting to load zen-E/CommonsenseQA-GPT4omini...")
print("Using approach from codi/train.py line 381\n")

try:
    # Load the full dataset first (train.py approach)
    print("Step 1: Load full dataset...")
    dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
    print(f"✓ Dataset loaded successfully!")
    print(f"  Available splits: {list(dataset.keys())}")

    # Access validation split
    print("\nStep 2: Access validation split...")
    validation = dataset['validation']
    print(f"✓ Validation split accessed!")
    print(f"  Validation size: {len(validation)}")

    # Show first example
    print("\nStep 3: Inspect first example...")
    example = validation[0]
    print(f"  Keys: {list(example.keys())}")
    print(f"\nFirst example:")
    for key, value in example.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        else:
            print(f"  {key}: {value}")

    # Verify format matches what model expects
    print("\n" + "="*80)
    print("SUCCESS: Dataset loaded correctly!")
    print("="*80)

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()

    print("\n" + "="*80)
    print("ALTERNATIVE: Try finding cached data...")
    print("="*80)

    import os
    from pathlib import Path

    # Check common cache locations
    cache_dirs = [
        Path.home() / ".cache" / "huggingface" / "datasets",
        Path("/home/paperspace/.cache/huggingface/datasets"),
        Path("/home/ubuntu/.cache/huggingface/datasets"),
    ]

    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"\nSearching {cache_dir}...")
            for item in cache_dir.rglob("*CommonsenseQA*"):
                print(f"  Found: {item}")
