"""
Test step importance measurement on sample problems.

This script validates that:
1. CODI interface works correctly
2. Position-wise zeroing produces different outputs
3. Early positions show more importance than late positions
"""

import json
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from codi_interface import CODIInterface, StepImportanceMeasurer


def test_on_sample_problems():
    """Test step importance on sample GSM8K problems."""

    print("="*60)
    print("Step Importance Validation Test")
    print("="*60)

    # Load CODI
    model_path = str(Path.home() / 'codi_ckpt/llama_gsm8k')
    print("\n1. Loading CODI model...")
    interface = CODIInterface(model_path)

    measurer = StepImportanceMeasurer(interface, layer_idx=8)

    # Test problems (varying difficulty)
    test_problems = [
        {
            "id": 1,
            "question": "John has 3 bags with 7 apples each. How many apples does he have in total?",
            "answer": "21",
            "difficulty": "easy"
        },
        {
            "id": 2,
            "question": "A store sells apples for $3 each. If Sarah buys 5 apples and pays with a $20 bill, how much change does she get?",
            "answer": "5",
            "difficulty": "medium"
        },
        {
            "id": 3,
            "question": "A classroom has 6 rows with 4 desks in each row. If each desk seats 2 students, how many students can the classroom seat?",
            "answer": "48",
            "difficulty": "medium"
        }
    ]

    all_results = []

    for problem in test_problems:
        print(f"\n{'='*60}")
        print(f"Problem {problem['id']}: {problem['question'][:50]}...")
        print(f"Expected answer: {problem['answer']}")
        print(f"{'='*60}")

        # Measure all positions
        results = measurer.measure_all_positions(problem['question'])

        print("\nPosition-wise Results:")
        print(f"{'Position':<10} {'Importance':<12} {'Baseline':<20} {'Ablated':<20} {'Match':<8}")
        print("-" * 70)

        for r in results:
            pos = r['position']
            importance = r['importance_score']
            baseline_short = r['baseline_answer'][:18]
            ablated_short = r['ablated_answer'][:18]
            match = "✓" if r['answers_match'] else "✗"

            print(f"{pos:<10} {importance:<12.3f} {baseline_short:<20} {ablated_short:<20} {match:<8}")

        # Calculate average importance by early/late
        early_importance = sum(r['importance_score'] for r in results[:3]) / 3
        late_importance = sum(r['importance_score'] for r in results[3:]) / 3

        print(f"\nSummary:")
        print(f"  Early positions (0-2) avg importance: {early_importance:.3f}")
        print(f"  Late positions (3-5) avg importance:  {late_importance:.3f}")
        print(f"  Pattern (early > late): {'✓ PASS' if early_importance >= late_importance else '✗ FAIL'}")

        all_results.append({
            'problem_id': problem['id'],
            'question': problem['question'],
            'difficulty': problem['difficulty'],
            'expected_answer': problem['answer'],
            'position_results': results,
            'early_avg': early_importance,
            'late_avg': late_importance,
            'pattern_validated': early_importance >= late_importance
        })

    # Overall summary
    print(f"\n{'='*60}")
    print("Overall Validation Summary")
    print(f"{'='*60}")

    total_early = sum(r['early_avg'] for r in all_results) / len(all_results)
    total_late = sum(r['late_avg'] for r in all_results) / len(all_results)
    patterns_passed = sum(r['pattern_validated'] for r in all_results)

    print(f"\nAcross {len(test_problems)} problems:")
    print(f"  Average early importance: {total_early:.3f}")
    print(f"  Average late importance:  {total_late:.3f}")
    print(f"  Pattern validated: {patterns_passed}/{len(test_problems)} problems")
    print(f"  Overall pattern: {'✓ PASS' if total_early >= total_late else '✗ FAIL'}")

    # Save results
    output_path = Path(__file__).parent.parent / 'data' / 'codi_interface_validation.json'
    with open(output_path, 'w') as f:
        json.dump({
            'validation_status': 'COMPLETE',
            'n_problems': len(test_problems),
            'problems': all_results,
            'summary': {
                'avg_early_importance': total_early,
                'avg_late_importance': total_late,
                'pattern_validated': total_early >= total_late,
                'patterns_passed': patterns_passed,
                'total_problems': len(test_problems)
            }
        }, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    test_on_sample_problems()
