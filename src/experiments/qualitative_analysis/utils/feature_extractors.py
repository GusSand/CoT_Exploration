#!/usr/bin/env python3
"""
Problem Feature Extraction

Extract features from GSM8K problems to analyze which problem characteristics
correlate with intervention impact.

Features:
- n_tokens: Number of tokens in question
- n_operations: Number of arithmetic operations in solution
- operation_types: Types of operations (add, subtract, multiply, divide)
- number_magnitude: Maximum number in problem
- multi_step: Whether problem requires multiple steps
- has_division: Whether solution involves division
- has_fractions: Whether problem mentions fractions
"""
import re
from typing import Dict, List


def count_operations(solution_text: str) -> int:
    """
    Count number of arithmetic operations in GSM8K solution.

    GSM8K solutions follow pattern like:
    "Janet sells 16 - 3 - 4 = 9 eggs per day...
     She makes 9 * 2 = $18 every day...
     #### 18"

    Args:
        solution_text: Full solution text from GSM8K

    Returns:
        Number of arithmetic operations
    """
    # Look for patterns like "= 9" where there's a calculation
    # Each "=" after numbers/operators indicates a step
    calculation_steps = re.findall(r'[0-9\s\+\-\*\/\(\)]+\s*=\s*[\$]?[0-9]+', solution_text)
    return len(calculation_steps)


def identify_operations(solution_text: str) -> List[str]:
    """
    Identify types of operations used in solution.

    Returns:
        List of operation types: ['add', 'subtract', 'multiply', 'divide']
    """
    operations = set()

    if '+' in solution_text:
        operations.add('add')
    if '-' in solution_text:
        operations.add('subtract')
    if '*' in solution_text or '×' in solution_text:
        operations.add('multiply')
    if '/' in solution_text or '÷' in solution_text:
        operations.add('divide')

    return sorted(list(operations))


def extract_numbers(text: str) -> List[int]:
    """Extract all numbers from text."""
    # Find all numbers (including decimals)
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(n) for n in numbers if n]


def max_number_magnitude(question: str) -> int:
    """Get maximum number mentioned in question."""
    numbers = extract_numbers(question)
    return int(max(numbers)) if numbers else 0


def has_fractions_or_decimals(question: str) -> bool:
    """Check if question mentions fractions or decimals."""
    fraction_keywords = ['half', 'third', 'quarter', 'fraction', 'decimal', 'percent']
    question_lower = question.lower()

    for keyword in fraction_keywords:
        if keyword in question_lower:
            return True

    # Check for decimal numbers
    if re.search(r'\d+\.\d+', question):
        return True

    return False


def extract_features(question: str, solution: str) -> Dict:
    """
    Extract all features from a GSM8K problem.

    Args:
        question: Problem question text
        solution: Full solution text (including steps and #### answer)

    Returns:
        Dict of features
    """
    n_operations = count_operations(solution)
    operation_types = identify_operations(solution)

    features = {
        'question_length': len(question),
        'n_tokens': len(question.split()),  # Simple word count
        'n_operations': n_operations,
        'operation_types': operation_types,
        'n_operation_types': len(operation_types),
        'max_number': max_number_magnitude(question),
        'multi_step': n_operations > 1,
        'has_division': 'divide' in operation_types,
        'has_multiplication': 'multiply' in operation_types,
        'has_fractions': has_fractions_or_decimals(question),
    }

    return features


def extract_features_batch(problems: List[Dict]) -> List[Dict]:
    """
    Extract features for a batch of problems.

    Args:
        problems: List of dicts with 'question' and 'answer' keys
                  (answer includes solution steps)

    Returns:
        List of feature dicts
    """
    features_list = []

    for problem in problems:
        question = problem['question']
        solution = problem.get('answer', '')

        features = extract_features(question, solution)
        features['question'] = question[:100] + '...'  # Truncate for storage

        features_list.append(features)

    return features_list


def print_feature_summary(features_list: List[Dict]):
    """Print summary statistics of features."""
    import numpy as np

    n_problems = len(features_list)

    print(f"\n{'='*60}")
    print(f"Feature Summary ({n_problems} problems)")
    print(f"{'='*60}")

    # Numeric features
    for feature in ['n_tokens', 'n_operations', 'n_operation_types', 'max_number']:
        values = [f[feature] for f in features_list]
        print(f"\n{feature}:")
        print(f"  Mean: {np.mean(values):.1f}")
        print(f"  Median: {np.median(values):.1f}")
        print(f"  Range: [{min(values)}, {max(values)}]")

    # Boolean features
    for feature in ['multi_step', 'has_division', 'has_multiplication', 'has_fractions']:
        count = sum(1 for f in features_list if f[feature])
        percentage = (count / n_problems) * 100
        print(f"\n{feature}: {count} ({percentage:.1f}%)")

    print(f"{'='*60}")


if __name__ == "__main__":
    # Test feature extraction
    test_problem = {
        'question': "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        'answer': "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer's market.\n#### 18"
    }

    features = extract_features(test_problem['question'], test_problem['answer'])

    print("Extracted features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
