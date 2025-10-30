#!/usr/bin/env python3
"""
Debug CommonsenseQA Model Performance

Investigate why the model has severe answer bias (47% E, 33% A, 4.7% D).
"""
import sys
sys.path.insert(0, '/home/paperspace/dev/CoT_Exploration/codi')

import torch
import json
from pathlib import Path
from model_loader import CODIModelLoader
from datasets import load_dataset
import random

def test_commonsense_model():
    """Test CommonsenseQA model with manual examples."""
    print("="*80)
    print("COMMONSENSEQA MODEL DEBUG")
    print("="*80)

    # Load model
    loader = CODIModelLoader()
    model, tokenizer, metadata = loader.load_model('commonsense')

    # Load a few test examples
    dataset = load_dataset('commonsense_qa', split='validation')

    # Test with examples that should have different answers
    test_examples = []

    # Find one example for each answer A-E
    for answer_key in ['A', 'B', 'C', 'D', 'E']:
        for ex in dataset:
            if ex['answerKey'] == answer_key:
                test_examples.append(ex)
                break

    print(f"\nTesting {len(test_examples)} examples (one per answer A-E)")
    print("="*80)

    results = []

    for i, example in enumerate(test_examples, 1):
        question = example['question']
        choices = example['choices']
        correct_answer = example['answerKey']

        # Format input
        formatted_input = loader.format_input('commonsense', example)

        print(f"\n{'='*80}")
        print(f"Example {i}: Correct Answer = {correct_answer}")
        print(f"{'='*80}")
        print(f"Question: {question[:100]}...")
        print(f"\nChoices:")
        for label, text in zip(choices['label'], choices['text']):
            print(f"  {label}: {text}")

        # Generate
        input_ids = tokenizer(formatted_input, return_tensors="pt").input_ids.to('cuda:0')

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,  # Deterministic
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip input to get only generated text
        if full_output.startswith(formatted_input):
            generated_text = full_output[len(formatted_input):].strip()
        else:
            generated_text = full_output

        # Extract answer
        predicted = loader.extract_answer('commonsense', generated_text, example)

        # Show output
        print(f"\nGenerated Output:")
        print(f"{generated_text[:300]}...")
        print(f"\nExtracted Answer: {predicted}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Result: {'✓ CORRECT' if predicted == correct_answer else '✗ WRONG'}")

        results.append({
            'correct_answer': correct_answer,
            'predicted': predicted,
            'generated_text': generated_text,
            'is_correct': predicted == correct_answer
        })

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    correct_count = sum(r['is_correct'] for r in results)
    print(f"Accuracy: {correct_count}/{len(results)} = {correct_count/len(results):.1%}")

    print("\nPrediction Distribution:")
    from collections import Counter
    pred_dist = Counter([r['predicted'] for r in results])
    for answer in ['A', 'B', 'C', 'D', 'E']:
        count = pred_dist.get(answer, 0)
        print(f"  {answer}: {count}/{len(results)} ({count/len(results):.0%})")

    # Check for patterns
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)

    # Check if all outputs contain "THE ANSWER IS:"
    has_answer_pattern = sum('THE ANSWER IS:' in r['generated_text'].upper() for r in results)
    print(f"Outputs with 'THE ANSWER IS:' pattern: {has_answer_pattern}/{len(results)}")

    # Check first letter extracted
    print("\nFirst A-E letter in each output:")
    for i, r in enumerate(results, 1):
        first_letter = None
        for char in r['generated_text'].upper():
            if char in ['A', 'B', 'C', 'D', 'E']:
                first_letter = char
                break
        print(f"  Ex{i} (true={r['correct_answer']}): First letter = {first_letter}, Predicted = {r['predicted']}")

    # Save detailed outputs
    output_path = Path('results/commonsense_debug.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved detailed outputs to {output_path}")

    # Test with completely random prompts to check for default behavior
    print("\n" + "="*80)
    print("BASELINE TEST: Empty/Minimal Prompts")
    print("="*80)

    minimal_tests = [
        "Question: What is the answer?\nChoices:\nA: apple\nB: banana\nC: carrot\nD: dog\nE: elephant\nAnswer:",
        "A B C D E?",
        "Choose A, B, C, D, or E:"
    ]

    for i, prompt in enumerate(minimal_tests, 1):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda:0')

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if full_output.startswith(prompt):
            generated = full_output[len(prompt):].strip()
        else:
            generated = full_output

        # Extract first A-E
        extracted = None
        for char in generated.upper():
            if char in ['A', 'B', 'C', 'D', 'E']:
                extracted = char
                break

        print(f"\nTest {i}: {prompt[:50]}...")
        print(f"  Generated: {generated[:100]}...")
        print(f"  Extracted: {extracted}")

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)

    loader.unload_model()

if __name__ == '__main__':
    test_commonsense_model()
