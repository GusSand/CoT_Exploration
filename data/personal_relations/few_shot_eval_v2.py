#!/usr/bin/env python3
"""
Few-shot evaluation of LLaMA on Personal Relations Task (CORRECTED VERSION).

This version includes universe relationships in prompts.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import random
from tqdm import tqdm


def format_universe(universe_str: str) -> str:
    """Format universe relationships for display."""
    # Clean up the universe string
    formatted = universe_str.replace(';;', '\n').replace('  ', '\n')
    # Remove extra whitespace
    lines = [line.strip() for line in formatted.split('\n') if line.strip()]
    return '\n'.join(lines)


def create_few_shot_prompt(test_example: Dict,
                           few_shot_examples: List[Dict],
                           include_cot: bool = True) -> str:
    """
    Create few-shot prompt WITH universe context.
    """
    prompt = "You are solving a personal relations reasoning task.\n\n"

    # Add test universe context FIRST
    prompt += "Given the following relationships:\n\n"
    prompt += format_universe(test_example['universe'])
    prompt += "\n\n"

    # Add few-shot examples
    if few_shot_examples:
        prompt += "Here are some example questions and answers:\n\n"

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


def extract_answer(generated_text: str) -> str:
    """Extract the answer from generated text."""
    # Look for "Answer:" in the text
    if "Answer:" in generated_text:
        answer_part = generated_text.split("Answer:")[-1].strip()
    else:
        # Take the last meaningful line
        lines = [l.strip() for l in generated_text.strip().split('\n') if l.strip()]
        if lines:
            answer_part = lines[-1]
        else:
            answer_part = generated_text.strip()

    # Extract just the name (first word usually)
    # Remove common prefixes
    answer_part = answer_part.replace("The answer is", "").strip()
    answer_part = answer_part.replace("Answer:", "").strip()

    # Take first word (the person's name)
    answer = answer_part.split()[0] if answer_part.split() else answer_part

    # Remove punctuation
    answer = answer.replace('.', '').replace(',', '').replace('!', '').strip()

    return answer


def evaluate_few_shot(model, tokenizer,
                     test_examples: List[Dict],
                     train_examples: List[Dict],
                     n_shot: int = 3,
                     include_cot: bool = True) -> Dict:
    """
    Run few-shot evaluation with universe context.
    """
    results = []
    correct = 0

    print(f"\n{'=' * 80}")
    print(f"EVALUATING: {n_shot}-shot" + (" with CoT" if include_cot else " (direct)"))
    print(f"{'=' * 80}")

    for test_ex in tqdm(test_examples, desc="Evaluating"):
        # Sample few-shot examples from SAME universe
        same_universe = [ex for ex in train_examples
                        if ex['group'] == test_ex['group']]

        if n_shot > 0:
            if len(same_universe) >= n_shot:
                few_shot = random.sample(same_universe, n_shot)
            else:
                # Fall back to different complexity but similar
                same_complexity = [ex for ex in train_examples
                                 if ex['complexity'] == test_ex['complexity']]
                few_shot = random.sample(same_complexity, min(n_shot, len(same_complexity)))
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
        predicted = extract_answer(generated[len(prompt):])
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
            'prompt_length': len(prompt),
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
        if comp_total > 0:
            print(f"  Complexity {comp}: {comp_correct}/{comp_total} = {comp_correct/comp_total*100:.1f}%")

    return {
        'n_shot': n_shot,
        'include_cot': include_cot,
        'accuracy': accuracy,
        'results': results
    }


def main():
    print("=" * 80)
    print("PERSONAL RELATIONS - FEW-SHOT EVALUATION (V2 - WITH UNIVERSE CONTEXT)")
    print("=" * 80)

    # Load data with universe context
    print("\nLoading data...")
    with open('train_with_universe.json', 'r') as f:
        train_data = json.load(f)
    with open('test_with_universe.json', 'r') as f:
        test_data = json.load(f)

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

    # 0-shot with universe
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
    output_file = 'few_shot_results_v2.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Setting':<25} {'Accuracy':<10} {'vs V1':<10}")
    print("-" * 45)

    # Load v1 results for comparison
    try:
        with open('few_shot_results.json', 'r') as f:
            v1_results = json.load(f)
        v1_accs = {(r['n_shot'], r['include_cot']): r['accuracy'] for r in v1_results}
    except:
        v1_accs = {}

    for result in all_results:
        setting = f"{result['n_shot']}-shot"
        if result['include_cot'] and result['n_shot'] > 0:
            setting += " + CoT"

        key = (result['n_shot'], result['include_cot'])
        v1_acc = v1_accs.get(key, 0)
        improvement = result['accuracy'] - v1_acc

        print(f"{setting:<25} {result['accuracy']:.1f}%{'':<6} +{improvement:.1f}%")

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_file}")
    print(f"Paper baseline (LLaMA Extensional): ~80%")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
