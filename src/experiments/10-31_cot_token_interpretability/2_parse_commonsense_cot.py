"""
Parse CoT tokens from CommonsenseQA CODI baseline model predictions.

Analyzes whether explicit CoT reasoning is interpretable for commonsense reasoning tasks.
Extracts reasoning patterns from generated CoT sequences.

Output: commonsense_cot_analysis.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def parse_eval_log(log_path):
    """
    Parse evaluation log to extract predictions with generated CoT.

    Returns:
        List of dicts with question, generated, predicted, ground_truth
    """
    with open(log_path, 'r') as f:
        content = f.read()

    # Parse question blocks
    question_pattern = r'Question (\d+): Question: (.*?)\n.*?Generated: (.*?)\nPredicted: ([A-E]), Ground Truth: ([A-E])'
    matches = re.findall(question_pattern, content, re.DOTALL)

    results = []
    for match in matches:
        q_id, question, generated, predicted, ground_truth = match
        results.append({
            'question_id': int(q_id),
            'question': question.strip(),
            'generated': generated.strip(),
            'predicted': predicted.strip(),
            'ground_truth': ground_truth.strip(),
            'correct': predicted == ground_truth
        })

    return results


def analyze_reasoning_patterns(generated_text):
    """
    Analyze the reasoning patterns in generated CoT.

    Returns:
        dict with reasoning characteristics
    """
    # Count sentences (rough approximation)
    sentences = [s.strip() for s in generated_text.split('.') if s.strip()]
    num_sentences = len(sentences)

    # Look for reasoning indicators
    reasoning_words = {
        'causal': ['because', 'therefore', 'thus', 'so', 'as a result', 'consequently'],
        'comparative': ['while', 'whereas', 'compared to', 'unlike', 'however', 'although'],
        'logical': ['if', 'then', 'must', 'should', 'would', 'typically', 'generally'],
        'evidential': ['among', 'provided', 'as', 'since', 'given that'],
    }

    pattern_counts = Counter()
    for category, words in reasoning_words.items():
        for word in words:
            if word.lower() in generated_text.lower():
                pattern_counts[category] += 1

    # Check for explicit step-by-step reasoning
    has_multi_step = num_sentences >= 2
    has_reasoning_words = sum(pattern_counts.values()) > 0

    return {
        'num_sentences': num_sentences,
        'reasoning_patterns': dict(pattern_counts),
        'has_multi_step': has_multi_step,
        'has_reasoning_words': has_reasoning_words,
        'total_reasoning_indicators': sum(pattern_counts.values()),
        'length_chars': len(generated_text),
    }


def categorize_question_type(question_text):
    """
    Categorize the type of commonsense reasoning.

    Returns:
        Category string
    """
    question_lower = question_text.lower()

    if any(word in question_lower for word in ['where', 'place', 'location']):
        return 'spatial'
    elif any(word in question_lower for word in ['why', 'reason', 'because']):
        return 'causal'
    elif any(word in question_lower for word in ['what', 'which']):
        return 'factual'
    elif any(word in question_lower for word in ['when', 'time']):
        return 'temporal'
    elif any(word in question_lower for word in ['how']):
        return 'procedural'
    else:
        return 'other'


def analyze_commonsense_cot():
    """Analyze CoT token interpretability for CommonsenseQA dataset."""
    print("="*80)
    print("COMMONSENSEQA COT TOKEN ANALYSIS")
    print("="*80)

    # Parse evaluation log
    log_path = Path('codi/baseline_commonsense_eval.log')
    print(f"\nParsing evaluation log: {log_path}")

    results = parse_eval_log(log_path)
    print(f"Loaded {len(results)} predictions")

    # Calculate accuracy
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / len(results)) * 100
    print(f"Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")

    # Analysis structures
    reasoning_analyses = []
    question_types = Counter()
    avg_reasoning_length = 0
    total_reasoning_indicators = 0

    # Separate correct and incorrect
    correct_examples = []
    incorrect_examples = []

    print(f"\nAnalyzing CoT reasoning patterns...")
    for result in results:
        generated = result['generated']

        # Analyze reasoning
        analysis = analyze_reasoning_patterns(generated)

        # Categorize question
        q_type = categorize_question_type(result['question'])
        question_types[q_type] += 1

        # Store full analysis
        full_analysis = {
            **result,
            **analysis,
            'question_type': q_type,
        }
        reasoning_analyses.append(full_analysis)

        avg_reasoning_length += analysis['length_chars']
        total_reasoning_indicators += analysis['total_reasoning_indicators']

        if result['correct']:
            correct_examples.append(full_analysis)
        else:
            incorrect_examples.append(full_analysis)

    # Compute statistics
    avg_reasoning_length /= len(results)
    avg_reasoning_indicators = total_reasoning_indicators / len(results)
    avg_sentences = sum(a['num_sentences'] for a in reasoning_analyses) / len(results)

    # Count how many have multi-step reasoning
    multi_step_count = sum(1 for a in reasoning_analyses if a['has_multi_step'])
    reasoning_words_count = sum(1 for a in reasoning_analyses if a['has_reasoning_words'])

    print(f"\n✓ Analysis complete")
    print(f"\n{'='*80}")
    print("INTERPRETABILITY ANALYSIS")
    print(f"{'='*80}")

    print(f"\n1. Reasoning Structure:")
    print(f"   Average CoT length: {avg_reasoning_length:.1f} characters")
    print(f"   Average sentences per CoT: {avg_sentences:.1f}")
    print(f"   Examples with multi-step reasoning: {multi_step_count}/{len(results)} ({100*multi_step_count/len(results):.1f}%)")
    print(f"   Examples with reasoning words: {reasoning_words_count}/{len(results)} ({100*reasoning_words_count/len(results):.1f}%)")

    print(f"\n2. Reasoning Pattern Distribution:")
    # Aggregate reasoning patterns
    all_patterns = Counter()
    for analysis in reasoning_analyses:
        for pattern, count in analysis['reasoning_patterns'].items():
            all_patterns[pattern] += count

    total_patterns = sum(all_patterns.values())
    for pattern, count in all_patterns.most_common():
        pct = (count / total_patterns) * 100 if total_patterns > 0 else 0
        print(f"   {pattern}: {count} ({pct:.1f}%)")

    print(f"\n   Average reasoning indicators per example: {avg_reasoning_indicators:.2f}")

    print(f"\n3. Question Type Distribution:")
    for q_type, count in question_types.most_common():
        pct = (count / len(results)) * 100
        print(f"   {q_type}: {count} ({pct:.1f}%)")

    print(f"\n4. Interpretability Assessment:")

    # Interpretability score based on presence of explicit reasoning
    interpretability_score = (reasoning_words_count / len(results)) * 100

    print(f"   Examples with explicit reasoning: {reasoning_words_count}/{len(results)} ({interpretability_score:.1f}%)")
    print(f"   Average reasoning depth: {avg_reasoning_indicators:.2f} indicators per example")

    print(f"\n   Interpretability Score: {interpretability_score:.1f}%")
    if interpretability_score > 90:
        print(f"   Assessment: HIGHLY INTERPRETABLE - CoT shows rich, explicit reasoning")
    elif interpretability_score > 70:
        print(f"   Assessment: MODERATELY INTERPRETABLE - Most examples show reasoning")
    else:
        print(f"   Assessment: LOW INTERPRETABILITY - Limited reasoning shown")

    print(f"\n5. Sample Reasoning Patterns:")

    # Show examples of different types
    print(f"\n   Spatial reasoning:")
    spatial_examples = [a for a in reasoning_analyses if a['question_type'] == 'spatial'][:3]
    for ex in spatial_examples:
        status = "✓" if ex['correct'] else "✗"
        print(f"     {status} Q: {ex['question'][:60]}...")
        print(f"        CoT: {ex['generated'][:100]}...")

    print(f"\n   Factual reasoning:")
    factual_examples = [a for a in reasoning_analyses if a['question_type'] == 'factual'][:3]
    for ex in factual_examples:
        status = "✓" if ex['correct'] else "✗"
        print(f"     {status} Q: {ex['question'][:60]}...")
        print(f"        CoT: {ex['generated'][:100]}...")

    print(f"\n   Causal reasoning:")
    causal_examples = [a for a in reasoning_analyses if a['question_type'] == 'causal'][:3]
    for ex in causal_examples:
        status = "✓" if ex['correct'] else "✗"
        print(f"     {status} Q: {ex['question'][:60]}...")
        print(f"        CoT: {ex['generated'][:100]}...")

    # Save detailed results
    output_dir = Path('src/experiments/10-31_cot_token_interpretability')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'commonsense_cot_analysis.json'

    print(f"\n{'='*80}")
    print(f"Saving detailed results to: {output_path}")

    output_data = {
        'summary': {
            'total_examples': len(results),
            'accuracy': accuracy,
            'avg_cot_length': avg_reasoning_length,
            'avg_sentences': avg_sentences,
            'avg_reasoning_indicators': avg_reasoning_indicators,
            'interpretability_score': interpretability_score,
            'multi_step_count': multi_step_count,
            'reasoning_words_count': reasoning_words_count,
        },
        'reasoning_patterns': dict(all_patterns),
        'question_types': dict(question_types),
        'reasoning_analyses': reasoning_analyses[:100],  # First 100 for space
        'correct_examples': correct_examples[:20],
        'incorrect_examples': incorrect_examples[:20],
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Saved ({size_kb:.1f} KB)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    return output_data


if __name__ == '__main__':
    analyze_commonsense_cot()
