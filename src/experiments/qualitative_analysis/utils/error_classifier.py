#!/usr/bin/env python3
"""
Error Taxonomy Classifier

Classifies prediction errors into categories to understand qualitative
behavioral changes from attention interventions.

Error Categories:
- CORRECT: Prediction matches gold answer
- CALC_ERROR: Wrong arithmetic operation or calculation mistake
- LOGIC_ERROR: Correct operations but wrong logical reasoning
- OFF_BY_FACTOR: Answer off by multiplication/division factor (2x, 10x, etc)
- SIGN_ERROR: Wrong sign (positive/negative)
- OPERATION_REVERSAL: Used addition instead of subtraction (or vice versa)
- PARTIAL_ANSWER: Correct intermediate step but incomplete
- NONSENSE: Incoherent or unparseable output
- NONE: Failed to generate answer
"""
import re
from typing import Dict, Optional, Tuple


ERROR_TAXONOMY = {
    "CORRECT": "Prediction matches gold answer",
    "CALC_ERROR": "Wrong arithmetic operation or calculation mistake",
    "LOGIC_ERROR": "Correct operations but wrong logical reasoning",
    "OFF_BY_FACTOR": "Answer off by multiplication/division factor (2x, 10x, etc)",
    "SIGN_ERROR": "Wrong sign (positive/negative)",
    "OPERATION_REVERSAL": "Used addition instead of subtraction (or vice versa)",
    "PARTIAL_ANSWER": "Correct intermediate step but incomplete",
    "NONSENSE": "Incoherent or unparseable output",
    "NONE": "Failed to generate answer"
}


def classify_error(question: str, gold_answer: int, pred_answer: Optional[int],
                   correct: bool) -> Tuple[str, float]:
    """
    Classify error type using rule-based heuristics.

    Args:
        question: The problem question
        gold_answer: Correct answer
        pred_answer: Predicted answer (or None if failed)
        correct: Whether prediction is correct

    Returns:
        (error_type, confidence) tuple
        error_type: One of ERROR_TAXONOMY keys
        confidence: 0.0-1.0, how confident we are in this classification
    """
    # Rule 1: Correct answer
    if correct:
        return "CORRECT", 1.0

    # Rule 2: No prediction
    if pred_answer is None:
        return "NONE", 1.0

    # Rule 3: Sign error (exact opposite)
    if pred_answer == -gold_answer:
        return "SIGN_ERROR", 1.0

    # Rule 4: Off by factor (2x, 10x, 0.5x, 0.1x, etc)
    if gold_answer != 0:
        ratio = abs(pred_answer / gold_answer)

        # Check common factors
        common_factors = [2, 3, 4, 5, 10, 100, 0.5, 0.25, 0.1, 0.01]
        for factor in common_factors:
            if abs(ratio - factor) < 0.01:  # Allow small floating point error
                return "OFF_BY_FACTOR", 0.9

    # Rule 5: Very close (within 10%) - likely calculation error
    if gold_answer != 0:
        percent_diff = abs((pred_answer - gold_answer) / gold_answer)
        if percent_diff < 0.1:
            return "CALC_ERROR", 0.7

    # Rule 6: Partial answer (pred is a divisor or multiple but not exact factor)
    if gold_answer != 0 and pred_answer != 0:
        if gold_answer % pred_answer == 0 or pred_answer % gold_answer == 0:
            # One divides the other (might be partial computation)
            return "PARTIAL_ANSWER", 0.6

    # Rule 7: Operation reversal heuristics
    # Check if pred = -gold (addition instead of subtraction)
    # or pred = 2*gold (didn't subtract, just added)
    # This is domain-specific to GSM8K

    # Default: General calculation error (medium confidence)
    # In future, could use LLM for more nuanced classification
    return "CALC_ERROR", 0.5


def classify_error_batch(results: list) -> list:
    """
    Classify a batch of results.

    Args:
        results: List of dicts with keys: question, gold_answer, pred_answer, correct

    Returns:
        Same list with added keys: error_type, error_confidence
    """
    classified_results = []

    for result in results:
        error_type, confidence = classify_error(
            result['question'],
            result['gold_answer'],
            result.get('pred_answer'),
            result.get('correct', False)
        )

        result_with_classification = result.copy()
        result_with_classification['error_type'] = error_type
        result_with_classification['error_confidence'] = confidence

        classified_results.append(result_with_classification)

    return classified_results


def get_error_distribution(classified_results: list) -> Dict[str, int]:
    """
    Get distribution of error types.

    Args:
        classified_results: Results with error_type field

    Returns:
        Dict mapping error_type to count
    """
    distribution = {error_type: 0 for error_type in ERROR_TAXONOMY.keys()}

    for result in classified_results:
        error_type = result.get('error_type', 'CALC_ERROR')
        distribution[error_type] += 1

    return distribution


def print_error_summary(distribution: Dict[str, int], total: int, condition_name: str):
    """Print formatted error distribution summary."""
    print(f"\n{'='*60}")
    print(f"Error Distribution - {condition_name}")
    print(f"{'='*60}")

    for error_type, count in sorted(distribution.items(), key=lambda x: -x[1]):
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{error_type:20s}: {count:4d} ({percentage:5.1f}%)")

    print(f"{'='*60}")
    print(f"Total: {total}")


if __name__ == "__main__":
    # Test the classifier
    test_cases = [
        {"question": "Test", "gold_answer": 18, "pred_answer": 18, "correct": True},
        {"question": "Test", "gold_answer": 18, "pred_answer": 26, "correct": False},
        {"question": "Test", "gold_answer": 18, "pred_answer": None, "correct": False},
        {"question": "Test", "gold_answer": 100, "pred_answer": 200, "correct": False},
        {"question": "Test", "gold_answer": 50, "pred_answer": -50, "correct": False},
    ]

    classified = classify_error_batch(test_cases)

    for result in classified:
        print(f"Gold: {result['gold_answer']}, Pred: {result['pred_answer']} "
              f"-> {result['error_type']} (conf: {result['error_confidence']:.2f})")
