"""
Story 1.1: Dataset Access & Download

Download the liars-bench Instructed Deception (ID) dataset from Hugging Face.
Requires HF token for Cadenza-Labs/liars-bench access.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from collections import Counter

def download_liars_bench(hf_token: str, output_dir: str = "../data/raw"):
    """
    Download liars-bench dataset and save to disk.

    Args:
        hf_token: HuggingFace authentication token
        output_dir: Directory to save raw dataset
    """
    print("=" * 80)
    print("STORY 1.1: Downloading Liars-Bench Instructed Deception Dataset")
    print("=" * 80)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Loading dataset from Cadenza-Labs/liars-bench...")
    print("      (This may take a few minutes on first download)")

    try:
        # Load the Instructed Deception (ID) config specifically
        dataset = load_dataset(
            "Cadenza-Labs/liars-bench",
            "instructed-deception",  # Specify the ID config
            token=hf_token,
            trust_remote_code=True
        )

        print(f"✅ Dataset loaded successfully!")
        print(f"\n[2/4] Analyzing dataset structure...")

        # Print dataset info
        print(f"\nDataset splits available: {list(dataset.keys())}")

        for split_name, split_data in dataset.items():
            print(f"\n--- Split: {split_name} ---")
            print(f"  Number of examples: {len(split_data)}")
            print(f"  Features: {split_data.features}")

            # Show first example
            if len(split_data) > 0:
                print(f"\n  First example:")
                first_example = split_data[0]
                for key, value in first_example.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"    {key}: {value[:100]}...")
                    else:
                        print(f"    {key}: {value}")

        print(f"\n[3/4] Looking for Instructed Deception (ID) subset...")

        # Try to identify ID subset
        # Common patterns: 'instructed_deception', 'ID', 'id', or check columns
        id_subset = None

        # Check if there's a specific subset or if we need to filter
        for split_name, split_data in dataset.items():
            if 'instructed' in split_name.lower() or split_name.lower() == 'id':
                id_subset = split_data
                print(f"✅ Found ID subset in split: {split_name}")
                break

            # Check if there's a column indicating subset type
            if len(split_data) > 0:
                first_example = split_data[0]
                if 'subset' in first_example or 'task_type' in first_example or 'category' in first_example:
                    # Analyze the subset distribution
                    subset_key = 'subset' if 'subset' in first_example else ('task_type' if 'task_type' in first_example else 'category')
                    subset_values = [ex[subset_key] for ex in split_data]
                    subset_counts = Counter(subset_values)

                    print(f"\n  Found subset indicator column: '{subset_key}'")
                    print(f"  Subset distribution:")
                    for subset_name, count in subset_counts.most_common():
                        print(f"    - {subset_name}: {count} examples")
                        if 'instructed' in str(subset_name).lower() or str(subset_name).lower() == 'id':
                            id_subset = split_data.filter(lambda x: x[subset_key] == subset_name)
                            print(f"\n✅ Found ID subset: {subset_name} ({count} examples)")
                            break

        # Save the dataset(s)
        print(f"\n[4/4] Saving dataset to {output_path}...")

        for split_name, split_data in dataset.items():
            output_file = output_path / f"liars_bench_{split_name}.json"

            # Convert to list of dicts for JSON serialization
            data_list = [dict(example) for example in split_data]

            with open(output_file, 'w') as f:
                json.dump(data_list, f, indent=2)

            print(f"  ✅ Saved {split_name}: {output_file} ({len(data_list)} examples)")

        # If we found an ID subset specifically, save it separately
        if id_subset is not None:
            id_output_file = output_path / "liars_bench_id_only.json"
            id_data_list = [dict(example) for example in id_subset]

            with open(id_output_file, 'w') as f:
                json.dump(id_data_list, f, indent=2)

            print(f"\n  ✅ Saved ID subset: {id_output_file} ({len(id_data_list)} examples)")

        # Save metadata
        metadata = {
            "dataset_name": "Cadenza-Labs/liars-bench",
            "splits": {split: len(data) for split, data in dataset.items()},
            "features": {split: str(data.features) for split, data in dataset.items()},
            "id_subset_found": id_subset is not None,
            "id_subset_size": len(id_subset) if id_subset is not None else None
        }

        metadata_file = output_path / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n  ✅ Saved metadata: {metadata_file}")

        print("\n" + "=" * 80)
        print("✅ STORY 1.1 COMPLETE: Dataset downloaded successfully!")
        print("=" * 80)
        print(f"\nNext steps:")
        print(f"  1. Review the downloaded files in {output_path}")
        print(f"  2. Identify the correct ID subset if not automatically detected")
        print(f"  3. Run Story 1.2: Preprocessing script")

        return metadata

    except Exception as e:
        print(f"\n❌ ERROR: Failed to download dataset")
        print(f"   Error message: {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"  1. Verify HF token has access to Cadenza-Labs/liars-bench")
        print(f"  2. Check if dataset requires special access approval")
        print(f"  3. Verify internet connection")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download liars-bench dataset")
    parser.add_argument("--hf-token", type=str, required=True,
                       help="HuggingFace authentication token")
    parser.add_argument("--output-dir", type=str,
                       default="../data/raw",
                       help="Output directory for raw data")

    args = parser.parse_args()

    # Convert to absolute path
    script_dir = Path(__file__).parent
    output_dir = (script_dir / args.output_dir).resolve()

    metadata = download_liars_bench(args.hf_token, str(output_dir))

    print(f"\n📊 Dataset Metadata Summary:")
    print(json.dumps(metadata, indent=2))
