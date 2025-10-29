#!/usr/bin/env python3
"""
Convert Personal Relations data to CODI training format.

CODI expects:
{
  "question": "Alice's parent's friend",
  "cot": "1. Alice's parent = Bob 2. Alice's parent's friend = Charlie",
  "answer": "Charlie"
}
"""

import json


def convert_to_codi_format(input_file: str, output_file: str):
    """Convert generated data to CODI format."""
    with open(input_file, 'r') as f:
        data = json.load(f)

    codi_data = []
    for item in data:
        # Join CoT steps with space (as CODI expects for icot format)
        cot_string = " ".join(item['cot_steps'])

        codi_item = {
            'question': item['question'],
            'cot': cot_string,
            'answer': item['answer']
        }

        codi_data.append(codi_item)

    with open(output_file, 'w') as f:
        json.dump(codi_data, f, indent=2)

    print(f"Converted {len(codi_data)} examples")
    print(f"Output: {output_file}")

    # Print example
    if codi_data:
        print(f"\nExample:")
        print(json.dumps(codi_data[0], indent=2))


def main():
    print("Converting Personal Relations data to CODI format...")
    print("=" * 80)

    # Convert train
    print("\nConverting train...")
    convert_to_codi_format('generated_train.json', 'personal_relations_train_codi.json')

    # Convert val
    print("\nConverting val...")
    convert_to_codi_format('generated_val.json', 'personal_relations_val_codi.json')

    # Convert test
    print("\nConverting test...")
    convert_to_codi_format('generated_test.json', 'personal_relations_test_codi.json')

    print(f"\n{'=' * 80}")
    print("✅ Conversion complete!")


if __name__ == "__main__":
    main()
