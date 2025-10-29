#!/usr/bin/env python3
"""
Few-shot evaluation of LLaMA on Personal Relations Task.

Tests whether LLaMA-1B can solve compositional reasoning with few-shot prompting
before investing in CODI training.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import random
from tqdm import tqdm


def load_data(split: str = "test") -> List[Dict]:
    """Load data split."""
    with open(f'{split}.json', 'r') as f:
        return json.load(f)


def create_universe_context(universe_str: str) -> str:
    """Format universe relationships for the prompt."""
    # The universe from the original CSV
    # Parse and format nicely
    relationships = universe_str.replace(';;', '\n').strip()
    return relationships


def create_few_shot_prompt(test_example: Dict,
                           few_shot_examples: List[Dict],
                           include_cot: bool = True) -> str:
    """
    Create few-shot prompt.

    Args:
        test_example: The example to evaluate
        few_shot_examples: Examples to use as demonstrations
        include_cot: Whether to include Chain-of-Thought reasoning
    """
    prompt = "You are solving a personal relations reasoning task.\n\n"

    # Add few-shot examples
    if few_shot_examples:
        prompt += "Here are some examples:\n\n"

        for i, ex in enumerate(few_shot_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {ex['question']}\n"

            if include_cot and ex['cot_steps']:
                prompt += "Reasoning:\n"
                for step in ex['cot_steps']:
                    prompt += f"  {step}\n"

            prompt += f"Answer: {ex['answer']}\n\n"

    # Add test question
    prompt += "Now answer this question:\n"
    prompt += f"Question: {test_example['question']}\n"

    if include_cot:
        prompt += "Reasoning:\n"
    else:
        prompt += "Answer:"

    return prompt


def extract_answer(generated_text: str, question: str) -> str:
    """Extract the answer from generated text."""
    # Remove the prompt
    if "Answer:" in generated_text:
        answer_part = generated_text.split("Answer:")[-1].strip()
    else:
        answer_part = generated_text.strip()

    # Take first line/word as answer (person name)
    answer = answer_part.split('\n')[0].strip()

    # Remove punctuation
    answer = answer.replace('.', '').replace(',', '').strip()

    return answer


def evaluate_few_shot(model, tokenizer,
                     test_examples: List[Dict],
                     train_examples: List[Dict],
                     n_shot: int = 3,
                     include_cot: bool = True) -> Dict:
    """
    Run few-shot evaluation.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        test_examples: Test set
        train_examples: Training set (for few-shot examples)
        n_shot: Number of few-shot examples (0, 3, or 5)
        include_cot: Whether to include Chain-of-Thought
    """
    results = []
    correct = 0

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {n_shot}-shot" + (" with CoT" if include_cot else " (direct)"))
    print(f"{'=' * 80}")

    for test_ex in tqdm(test_examples, desc="Evaluating"):
        # Sample few-shot examples (same complexity preferred)
        if n_shot > 0:
            # Get examples of same complexity
            same_complexity = [ex for ex in train_examples
                             if ex['complexity'] == test_ex['complexity']]

            if len(same_complexity) >= n_shot:
                few_shot = random.sample(same_complexity, n_shot)
            else:
                # Fall back to any examples
                few_shot = random.sample(train_examples, min(n_shot, len(train_examples)))
        else:
            few_shot = []

        # Create prompt
        prompt = create_few_shot_prompt(test_ex, few_shot, include_cot)

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract answer
        predicted = extract_answer(generated[len(prompt):], test_ex['question'])
        correct_answer = test_ex['answer']

        is_correct = predicted.lower() == correct_answer.lower()
        if is_correct:
            correct += 1

        results.append({
            'question': test_ex['question'],
            'complexity': test_ex['complexity'],
            'correct_answer': correct_answer,
            'predicted': predicted,
            'correct': is_correct,
            'prompt': prompt,
            'generated': generated[len(prompt):]
        })

    accuracy = correct / len(test_examples) * 100
    print(f"\nAccuracy: {correct}/{len(test_examples)} = {accuracy:.1f}%")

    # Breakdown by complexity
    print(f"\nBy complexity:")
    for comp in sorted(set(ex['complexity'] for ex in test_examples)):
        comp_results = [r for r in results if r['complexity'] == comp]
        comp_correct = sum(r['correct'] for r in comp_results)
        comp_total = len(comp_results)
        print(f"  Complexity {comp}: {comp_correct}/{comp_total} = {comp_correct/comp_total*100:.1f}%")

    return {
        'n_shot': n_shot,
        'include_cot': include_cot,
        'accuracy': accuracy,
        'results': results
    }


def main():
    print("=" * 80)
    print("PERSONAL RELATIONS - FEW-SHOT EVALUATION")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    train_data = load_data('train')
    test_data = load_data('test')

    print(f"Train: {len(train_data)} examples")
    print(f"Test: {len(test_data)} examples")

    # Load model
    print("\nLoading LLaMA-3.2-1B-Instruct...")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"Model loaded on: {model.device}")

    # Run evaluations
    all_results = []

    # 0-shot (no examples)
    result_0shot = evaluate_few_shot(
        model, tokenizer, test_data, train_data,
        n_shot=0, include_cot=False
    )
    all_results.append(result_0shot)

    # 3-shot with CoT
    result_3shot_cot = evaluate_few_shot(
        model, tokenizer, test_data, train_data,
        n_shot=3, include_cot=True
    )
    all_results.append(result_3shot_cot)

    # 5-shot with CoT
    result_5shot_cot = evaluate_few_shot(
        model, tokenizer, test_data, train_data,
        n_shot=5, include_cot=True
    )
    all_results.append(result_5shot_cot)

    # Save results
    output_file = 'few_shot_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Setting':<20} {'Accuracy':<10}")
    print("-" * 30)
    for result in all_results:
        setting = f"{result['n_shot']}-shot"
        if result['include_cot'] and result['n_shot'] > 0:
            setting += " + CoT"
        print(f"{setting:<20} {result['accuracy']:.1f}%")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"Paper baseline (LLaMA Extensional): ~80%")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
