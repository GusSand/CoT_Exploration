#!/usr/bin/env python3
"""
End-to-End GSM8K CoT Dataset Expansion

Orchestrates the full pipeline:
1. Load original GSM8K problems (test + train)
2. Exclude already-tested 532 problems
3. Test CoT necessity on new problems
4. Calculate difficulty metrics
5. Combine with existing 132 problems
6. Stratify to target distribution: 2-step (≥150), 3-step (≥150), 4-step (≥100), 5+ (≥50)

Usage:
    python expand_gsm8k_cot_dataset.py --num_samples 5000 --checkpoint_every 100
    python expand_gsm8k_cot_dataset.py --resume  # Resume from checkpoint
"""

import json
import sys
import torch
import argparse
import re
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from datasets import load_dataset

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'experiments'))
sys.path.insert(0, str(project_root / 'codi'))

# Import existing code
from cache_activations_llama import ActivationCacherLLaMA
from run_ablation_N_tokens_llama import NTokenPatcher, extract_answer_number, answers_match


# ============================================================================
# STEP 1: Load GSM8K Original Problems
# ============================================================================

def extract_answer_from_gsm8k(answer_text):
    """Extract numerical answer from GSM8K answer field."""
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', answer_text)
    if match:
        answer_str = match.group(1).replace(',', '')
        try:
            if '.' in answer_str:
                return float(answer_str)
            return int(answer_str)
        except ValueError:
            pass

    # Fallback: find last number
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_text)
    if numbers:
        answer_str = numbers[-1].replace(',', '')
        try:
            if '.' in answer_str:
                return float(answer_str)
            return int(answer_str)
        except ValueError:
            pass
    return None


def load_already_tested():
    """Load questions from the 532 already-tested pairs."""
    pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'

    if not pairs_file.exists():
        print(f"Warning: {pairs_file} not found. No problems will be excluded.")
        return set()

    with open(pairs_file) as f:
        pairs = json.load(f)

    tested_questions = set(pair['clean']['question'].strip() for pair in pairs)
    return tested_questions


def load_gsm8k_candidates(num_samples=5000):
    """Load original GSM8K problems, excluding already-tested ones."""
    print("\n" + "="*80)
    print("STEP 1: LOAD ORIGINAL GSM8K PROBLEMS")
    print("="*80)

    tested_questions = load_already_tested()
    print(f"Excluding {len(tested_questions)} already-tested problems")

    # Load datasets
    print("Loading GSM8K test set...")
    test_dataset = load_dataset('gsm8k', 'main', split='test')
    print("Loading GSM8K train set...")
    train_dataset = load_dataset('gsm8k', 'main', split='train')

    candidates = []

    # Process test set
    for idx, problem in enumerate(test_dataset):
        if len(candidates) >= num_samples:
            break

        question = problem['question'].strip()
        if question in tested_questions:
            continue

        answer = extract_answer_from_gsm8k(problem['answer'])
        if answer is None:
            continue

        candidates.append({
            'gsm8k_id': f'test_{idx}',
            'question': question,
            'answer': answer,
            'full_solution': problem['answer'],
            'source': 'test'
        })

    # Process train set if needed
    if len(candidates) < num_samples:
        for idx, problem in enumerate(train_dataset):
            if len(candidates) >= num_samples:
                break

            question = problem['question'].strip()
            if question in tested_questions:
                continue

            answer = extract_answer_from_gsm8k(problem['answer'])
            if answer is None:
                continue

            candidates.append({
                'gsm8k_id': f'train_{idx}',
                'question': question,
                'answer': answer,
                'full_solution': problem['answer'],
                'source': 'train'
            })

    print(f"✓ Loaded {len(candidates)} new GSM8K problems")
    return candidates


# ============================================================================
# STEP 2: Test CoT Necessity
# ============================================================================

def test_cot_necessity(candidates, model_path, checkpoint_file, resume=False):
    """Test which problems need CoT tokens."""
    print("\n" + "="*80)
    print("STEP 2: TEST COT NECESSITY")
    print("="*80)

    # Resume from checkpoint
    start_idx = 0
    results = []
    if resume and Path(checkpoint_file).exists():
        print(f"Resuming from checkpoint: {checkpoint_file}")
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
        results = checkpoint_data.get('results', [])
        start_idx = len(results)
        print(f"Resuming from problem {start_idx}/{len(candidates)}")

    # Load model
    print(f"Loading LLaMA model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)
    patcher = NTokenPatcher(cacher, num_tokens=6)

    # Test each problem
    print(f"Testing {len(candidates) - start_idx} problems...")

    for i in tqdm(range(start_idx, len(candidates)), desc="Testing CoT necessity"):
        problem = candidates[i]
        question = problem['question']
        expected = problem['answer']

        try:
            # Baseline (with CoT) - use patcher's run_without_patch
            baseline_output = patcher.run_without_patch(question, max_new_tokens=200)
            baseline_pred = extract_answer_number(baseline_output)
            baseline_correct = answers_match(baseline_pred, expected)

            # Ablated (without CoT)
            sample_act = patcher.cache_N_token_activations(question, 'middle')[0]
            zero_activations = [torch.zeros_like(sample_act) for _ in range(6)]

            ablated_output = patcher.run_with_N_tokens_patched(
                problem_text=question,
                patch_activations=zero_activations,
                layer_name='middle',
                max_new_tokens=200
            )

            ablated_pred = extract_answer_number(ablated_output)
            ablated_correct = answers_match(ablated_pred, expected)

            # Needs CoT if baseline correct but ablated wrong
            needs_cot = baseline_correct and not ablated_correct

            result = {
                **problem,
                'baseline_correct': baseline_correct,
                'baseline_prediction': baseline_pred,
                'ablated_correct': ablated_correct,
                'ablated_prediction': ablated_pred,
                'needs_cot': needs_cot,
                'success': True
            }

        except Exception as e:
            print(f"\nError on {problem['gsm8k_id']}: {e}")
            result = {**problem, 'success': False, 'error': str(e), 'needs_cot': False}

        results.append(result)

        # Checkpoint every 100
        if (i + 1) % 100 == 0:
            save_checkpoint(results, checkpoint_file)

    # Final save
    save_checkpoint(results, checkpoint_file)

    # Filter to CoT-needed
    cot_needed = [r for r in results if r.get('success') and r.get('needs_cot')]

    stats = {
        'tested': len(results),
        'successful': sum(1 for r in results if r.get('success')),
        'baseline_correct': sum(1 for r in results if r.get('baseline_correct')),
        'needs_cot': len(cot_needed)
    }

    print(f"\n✓ Testing complete:")
    print(f"  Tested: {stats['tested']}")
    if stats['successful'] > 0:
        print(f"  Baseline correct: {stats['baseline_correct']} ({100*stats['baseline_correct']/stats['successful']:.1f}%)")
        print(f"  Needs CoT: {stats['needs_cot']} ({100*stats['needs_cot']/stats['successful']:.1f}%)")
    else:
        print(f"  Baseline correct: {stats['baseline_correct']}")
        print(f"  Needs CoT: {stats['needs_cot']}")

    return cot_needed


def save_checkpoint(results, checkpoint_file):
    """Save checkpoint."""
    checkpoint_data = {
        'timestamp': datetime.now().isoformat(),
        'problems_tested': len(results),
        'needs_cot_count': sum(1 for r in results if r.get('needs_cot')),
        'results': results
    }

    Path(checkpoint_file).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)


# ============================================================================
# STEP 3: Calculate Difficulty
# ============================================================================

def count_reasoning_steps(solution_text):
    """Count reasoning steps from GSM8K solution."""
    calc_blocks = re.findall(r'<<[^>]+>>', solution_text)
    if len(calc_blocks) == 0:
        calc_blocks = re.findall(r'=\s*\d+', solution_text)
    return len(calc_blocks)


def classify_difficulty(num_steps):
    """Classify by difficulty."""
    if num_steps <= 2:
        return '2-step'
    elif num_steps == 3:
        return '3-step'
    elif num_steps == 4:
        return '4-step'
    else:
        return '5+step'


def add_difficulty_metrics(problems):
    """Add difficulty metrics to problems."""
    print("\n" + "="*80)
    print("STEP 3: CALCULATE DIFFICULTY METRICS")
    print("="*80)

    for problem in problems:
        solution_text = problem.get('full_solution', '')
        reasoning_steps = count_reasoning_steps(solution_text)
        difficulty = classify_difficulty(reasoning_steps)

        problem['reasoning_steps'] = reasoning_steps
        problem['difficulty'] = difficulty

    # Print distribution
    difficulty_counts = Counter(p['difficulty'] for p in problems)
    print("\nDifficulty distribution:")
    for difficulty in ['2-step', '3-step', '4-step', '5+step']:
        count = difficulty_counts.get(difficulty, 0)
        print(f"  {difficulty:10s}: {count:4d}")

    return problems


# ============================================================================
# STEP 4: Load Existing Problems
# ============================================================================

def load_existing_cot_problems():
    """Load the existing 132 original GSM8K problems."""
    print("\n" + "="*80)
    print("STEP 4: LOAD EXISTING 132 PROBLEMS")
    print("="*80)

    existing_file = Path(__file__).parent / 'data' / 'llama_cot_all.json'

    if not existing_file.exists():
        print(f"Warning: {existing_file} not found. Starting fresh.")
        return []

    with open(existing_file) as f:
        all_problems = json.load(f)

    # Filter to clean (original) variants only
    existing = [p for p in all_problems if p['variant'] == 'clean']

    # Mark as existing and add difficulty classification
    for p in existing:
        p['is_existing'] = True
        p['gsm8k_id'] = f'existing_{p["pair_id"]}'
        # Add difficulty if not present
        if 'difficulty' not in p:
            p['difficulty'] = classify_difficulty(p['reasoning_steps'])

    print(f"✓ Loaded {len(existing)} existing problems")

    # Print difficulty distribution
    difficulty_counts = Counter(classify_difficulty(p['reasoning_steps']) for p in existing)
    print("\nExisting difficulty distribution:")
    for difficulty in ['2-step', '3-step', '4-step', '5+step']:
        count = difficulty_counts.get(difficulty, 0)
        print(f"  {difficulty:10s}: {count:4d}")

    return existing


# ============================================================================
# STEP 5: Stratify and Filter
# ============================================================================

def stratify_and_filter(problems, targets):
    """Stratify problems and sample to meet targets.

    Args:
        problems: List of problems with difficulty metrics
        targets: Dict like {'2-step': 150, '3-step': 150, '4-step': 100, '5+step': 50}

    Returns:
        Final stratified dataset
    """
    print("\n" + "="*80)
    print("STEP 5: STRATIFY AND FILTER TO TARGETS")
    print("="*80)

    # Group by difficulty
    by_difficulty = {
        '2-step': [],
        '3-step': [],
        '4-step': [],
        '5+step': []
    }

    for p in problems:
        difficulty = p['difficulty']
        by_difficulty[difficulty].append(p)

    # Sample from each bucket
    import random
    random.seed(42)

    final_dataset = []

    for difficulty, target in targets.items():
        available = by_difficulty[difficulty]
        needed = target

        # Separate existing from new
        existing = [p for p in available if p.get('is_existing')]
        new = [p for p in available if not p.get('is_existing')]

        # Take all existing first
        selected = existing.copy()

        # Sample from new to reach target
        if len(selected) < needed:
            remaining_needed = needed - len(selected)
            if len(new) >= remaining_needed:
                selected.extend(random.sample(new, remaining_needed))
            else:
                selected.extend(new)  # Take all available

        final_dataset.extend(selected)

        print(f"\n{difficulty}:")
        print(f"  Target: {target}")
        print(f"  Available: {len(available)} ({len(existing)} existing + {len(new)} new)")
        print(f"  Selected: {len(selected)} ({len([p for p in selected if p.get('is_existing')])} existing + {len([p for p in selected if not p.get('is_existing')])} new)")

        if len(selected) < target:
            print(f"  ⚠️ WARNING: Only {len(selected)}/{target} available!")

    print(f"\n✓ Final dataset: {len(final_dataset)} problems")

    return final_dataset


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Expand GSM8K CoT dataset")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='Number of new problems to test (default: 5000)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to LLaMA model (default: ~/codi_ckpt/llama_gsm8k)')
    parser.add_argument('--checkpoint_file', type=str, default='data/gsm8k_expansion_checkpoint.json',
                        help='Checkpoint file for resuming')
    parser.add_argument('--output', type=str, default='data/llama_cot_original_stratified_final.json',
                        help='Output file for final dataset')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--skip_testing', action='store_true',
                        help='Skip CoT testing (use existing checkpoint)')

    args = parser.parse_args()

    # Set model path
    if args.model_path is None:
        args.model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    print("="*80)
    print("GSM8K COT DATASET EXPANSION - END-TO-END PIPELINE")
    print("="*80)
    print(f"Target distribution:")
    print(f"  2-step: ≥150")
    print(f"  3-step: ≥150")
    print(f"  4-step: ≥100")
    print(f"  5+step: ≥50")
    print(f"  Total:  450-1500")

    # Target distribution
    targets = {
        '2-step': 150,
        '3-step': 150,
        '4-step': 100,
        '5+step': 50
    }

    # Step 1: Load candidates
    if args.skip_testing and Path(args.checkpoint_file).exists():
        print("\nSkipping candidate loading (using checkpoint)")
        candidates = []
    else:
        candidates = load_gsm8k_candidates(args.num_samples)

    # Step 2: Test CoT necessity
    if args.skip_testing:
        print("\nSkipping CoT testing (loading from checkpoint)")
        with open(args.checkpoint_file) as f:
            checkpoint = json.load(f)
        new_cot_problems = [r for r in checkpoint['results'] if r.get('needs_cot')]
    else:
        new_cot_problems = test_cot_necessity(
            candidates, args.model_path, args.checkpoint_file, args.resume
        )

    # Step 3: Calculate difficulty
    new_cot_problems = add_difficulty_metrics(new_cot_problems)

    # Step 4: Load existing problems
    existing_problems = load_existing_cot_problems()

    # Combine
    all_problems = existing_problems + new_cot_problems
    print(f"\nTotal problems available: {len(all_problems)}")

    # Step 5: Stratify and filter
    final_dataset = stratify_and_filter(all_problems, targets)

    # Save final dataset
    output_path = Path(__file__).parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(final_dataset, f, indent=2)

    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"✓ Final dataset saved to: {output_path}")
    print(f"✓ Total problems: {len(final_dataset)}")

    # Final distribution
    final_counts = Counter(p['difficulty'] for p in final_dataset)
    print(f"\nFinal distribution:")
    for difficulty in ['2-step', '3-step', '4-step', '5+step']:
        count = final_counts.get(difficulty, 0)
        target = targets[difficulty]
        status = "✓" if count >= target else "⚠️"
        print(f"  {status} {difficulty:10s}: {count:4d} / {target}")


if __name__ == "__main__":
    main()
