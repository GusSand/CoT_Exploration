#!/usr/bin/env python3
"""
Simple CommonsenseQA Inference Debug
Just run inference on 10 examples and show raw outputs.
"""
import sys
sys.path.insert(0, '/home/paperspace/dev/CoT_Exploration/codi')

import torch
from model_loader import CODIModelLoader
from datasets import load_dataset

# Load model
print("Loading CommonsenseQA model...")
loader = CODIModelLoader()
model, tokenizer, metadata = loader.load_model('commonsense')

# Load 10 test examples - one for each answer
dataset = load_dataset('commonsense_qa', split='validation')

# Get diverse examples
test_examples = []
for answer_key in ['A', 'B', 'C', 'D', 'E']:
    for ex in dataset:
        if ex['answerKey'] == answer_key:
            test_examples.append(ex)
            break
    if len(test_examples) >= 5:
        break

print(f"\n{'='*80}")
print(f"Testing {len(test_examples)} examples")
print(f"{'='*80}\n")

results = []

for i, example in enumerate(test_examples, 1):
    question = example['question']
    choices = example['choices']
    correct_answer = example['answerKey']

    # Format input using loader's method
    formatted_input = loader.format_input('commonsense', example)

    print(f"{'='*80}")
    print(f"Example {i}: True Answer = {correct_answer}")
    print(f"{'='*80}")
    print(f"Question: {question[:100]}...")
    print(f"\nFormatted Input:")
    print(formatted_input[:200] + "...")

    # Run inference
    try:
        inputs = tokenizer(formatted_input, return_tensors="pt").to(loader.device)

        with torch.no_grad():
            outputs = model.codi.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Strip input
        if full_output.startswith(formatted_input):
            generated_text = full_output[len(formatted_input):].strip()
        else:
            generated_text = full_output

        # Extract answer
        predicted = loader.extract_answer('commonsense', generated_text, example)

        is_correct = (predicted == correct_answer)

        print(f"\nGenerated Output:")
        print(f'"{generated_text}"')
        print(f"\nExtracted Answer: {predicted}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

        results.append({
            'correct_answer': correct_answer,
            'predicted': predicted,
            'generated': generated_text,
            'is_correct': is_correct
        })

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
correct = sum(r['is_correct'] for r in results)
print(f"Accuracy: {correct}/{len(results)} = {correct/len(results):.1%}")

from collections import Counter
pred_dist = Counter([r['predicted'] for r in results])
print(f"\nPrediction Distribution:")
for ans in ['A', 'B', 'C', 'D', 'E']:
    count = pred_dist.get(ans, 0)
    print(f"  {ans}: {count}/{len(results)}")

print(f"\n{'='*80}")
print("Check if all outputs start with same letter...")
print(f"{'='*80}")
for i, r in enumerate(results, 1):
    first_letter = None
    for char in r['generated'].upper():
        if char in ['A', 'B', 'C', 'D', 'E']:
            first_letter = char
            break
    print(f"Ex{i}: First A-E in output = {first_letter}, Predicted = {r['predicted']}, True = {r['correct_answer']}")

loader.unload_model()
print("\nDone!")
