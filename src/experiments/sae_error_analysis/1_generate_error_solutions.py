"""
Generate solutions with both correct and incorrect answers for error analysis.

Strategy:
1. Temperature sampling (0.7, 0.9, 1.0) for natural errors
2. Truncated CoT (1, 2, 3 tokens) for insufficient reasoning errors
3. Extract continuous thoughts for all solutions
4. Label based on answer correctness

Target: 500+ incorrect solutions from 200 base problems
Output: ~1800 total solutions (correct + incorrect)
"""

import json
import torch
import re
from pathlib import Path
from tqdm import tqdm
import random
from typing import Dict, List, Tuple
import sys

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig, TaskType
from safetensors.torch import load_file
import os


def load_base_problems(n_problems: int = 200, stratify: bool = True) -> List[Dict]:
    """
    Load base problems for generation.

    Args:
        n_problems: Number of problems to load
        stratify: Whether to stratify by difficulty

    Returns:
        List of problem dicts with question, answer, difficulty
    """
    print(f"\nLoading {n_problems} base problems...")

    # Load from stratified dataset
    data_path = Path('/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json')

    with open(data_path, 'r') as f:
        all_problems = json.load(f)

    if stratify:
        # Sample equally from each difficulty
        by_difficulty = {}
        for prob in all_problems:
            diff = prob['difficulty']
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(prob)

        # Sample n_problems / 4 from each difficulty
        per_difficulty = n_problems // 4
        selected = []
        for diff, probs in by_difficulty.items():
            sampled = random.sample(probs, min(per_difficulty, len(probs)))
            selected.extend(sampled)

        # Fill remaining if needed
        if len(selected) < n_problems:
            remaining_probs = [p for p in all_problems if p not in selected]
            selected.extend(random.sample(remaining_probs, n_problems - len(selected)))

        problems = selected[:n_problems]
    else:
        problems = random.sample(all_problems, n_problems)

    print(f"  Loaded {len(problems)} problems")
    print(f"  Difficulty distribution:")
    diff_counts = {}
    for p in problems:
        diff = p['difficulty']
        diff_counts[diff] = diff_counts.get(diff, 0) + 1
    for diff, count in sorted(diff_counts.items()):
        print(f"    {diff}: {count}")

    return problems


def extract_answer_from_solution(solution: str) -> str:
    """
    Extract numerical answer from solution string.

    Looks for #### N format (GSM8K standard).
    """
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', solution)
    if match:
        return match.group(1)

    # Fallback: look for last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', solution)
    if numbers:
        return numbers[-1]

    return None


def generate_solution_variants(
    model,
    tokenizer,
    device,
    problem: Dict,
    config: Dict
) -> Dict:
    """
    Generate a single solution variant and extract continuous thoughts.

    Args:
        model: LLaMA model
        tokenizer: Tokenizer
        device: Device
        problem: Problem dict
        config: Generation config {temp, n_cot_tokens, config_name}

    Returns:
        Solution dict with continuous thoughts and correctness label
    """
    question = problem['question']
    ground_truth = str(problem['answer'])

    # Generate with continuous thoughts
    result = generate_with_continuous_thoughts(
        model=model,
        tokenizer=tokenizer,
        prompt=question,
        device=device,
        temperature=config['temp'],
        n_continuous_tokens=config['n_cot_tokens'],
        max_new_tokens=256,
        return_continuous_thoughts=True
    )

    generated_text = result['generated_text']
    continuous_thoughts = result['continuous_thoughts']  # [n_tokens, hidden_dim]

    # Extract answer
    predicted_answer = extract_answer_from_solution(generated_text)

    # Check correctness
    is_correct = (predicted_answer == ground_truth) if predicted_answer else False

    return {
        'problem_id': problem.get('gsm8k_id', problem.get('pair_id')),
        'question': question,
        'ground_truth': ground_truth,
        'predicted_answer': predicted_answer,
        'is_correct': is_correct,
        'generated_solution': generated_text,
        'continuous_thoughts': continuous_thoughts.cpu().numpy().tolist(),
        'config': config['config_name'],
        'temperature': config['temp'],
        'n_cot_tokens': config['n_cot_tokens'],
        'difficulty': problem.get('difficulty', 'unknown'),
        'reasoning_steps': problem.get('reasoning_steps', -1)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_problems', type=int, default=200,
                        help='Number of base problems')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/data')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                        help='Save checkpoint every N problems')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint file')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(42)
    torch.manual_seed(42)

    # Load model
    print("\n" + "="*80)
    print("LOADING LLAMA MODEL")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/home/paperspace/codi_ckpt/llama_gsm8k'

    model, tokenizer = load_model_and_tokenizer(model_path, device)
    model.eval()

    print(f"  Model loaded: {model_path}")
    print(f"  Device: {device}")

    # Load problems
    problems = load_base_problems(n_problems=args.n_problems, stratify=True)

    # Define generation configs
    configs = [
        # Temperature sampling (natural errors)
        {'temp': 0.7, 'n_cot_tokens': 6, 'config_name': 'temp_0.7_sample1'},
        {'temp': 0.7, 'n_cot_tokens': 6, 'config_name': 'temp_0.7_sample2'},
        {'temp': 0.9, 'n_cot_tokens': 6, 'config_name': 'temp_0.9_sample1'},
        {'temp': 0.9, 'n_cot_tokens': 6, 'config_name': 'temp_0.9_sample2'},
        {'temp': 1.0, 'n_cot_tokens': 6, 'config_name': 'temp_1.0_sample1'},

        # Truncated CoT (insufficient reasoning errors)
        {'temp': 0.0, 'n_cot_tokens': 1, 'config_name': 'truncated_1token'},
        {'temp': 0.0, 'n_cot_tokens': 2, 'config_name': 'truncated_2token'},
        {'temp': 0.0, 'n_cot_tokens': 3, 'config_name': 'truncated_3token'},

        # Baseline (correct, for comparison)
        {'temp': 0.0, 'n_cot_tokens': 6, 'config_name': 'baseline_greedy'},
    ]

    print(f"\n" + "="*80)
    print(f"GENERATION PLAN")
    print("="*80)
    print(f"  Base problems: {len(problems)}")
    print(f"  Configs per problem: {len(configs)}")
    print(f"  Total solutions: {len(problems) * len(configs)}")
    print(f"\nConfigs:")
    for cfg in configs:
        print(f"  - {cfg['config_name']}: temp={cfg['temp']}, n_cot={cfg['n_cot_tokens']}")

    # Resume from checkpoint if specified
    all_solutions = []
    start_idx = 0

    if args.resume_from:
        print(f"\nResuming from checkpoint: {args.resume_from}")
        with open(args.resume_from, 'r') as f:
            checkpoint = json.load(f)
            all_solutions = checkpoint['solutions']
            start_idx = checkpoint['next_problem_idx']
        print(f"  Loaded {len(all_solutions)} solutions")
        print(f"  Starting from problem {start_idx}/{len(problems)}")

    # Generate solutions
    print(f"\n" + "="*80)
    print(f"GENERATING SOLUTIONS")
    print("="*80)

    correct_count = 0
    incorrect_count = 0

    try:
        for prob_idx in tqdm(range(start_idx, len(problems)), desc="Problems"):
            problem = problems[prob_idx]

            for config in configs:
                try:
                    solution = generate_solution_variants(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        problem=problem,
                        config=config
                    )

                    all_solutions.append(solution)

                    if solution['is_correct']:
                        correct_count += 1
                    else:
                        incorrect_count += 1

                except Exception as e:
                    print(f"\n  Error on problem {prob_idx}, config {config['config_name']}: {e}")
                    continue

            # Checkpoint every N problems
            if (prob_idx + 1) % args.checkpoint_every == 0:
                checkpoint_path = output_dir / f'checkpoint_p{prob_idx+1}.json'
                checkpoint = {
                    'solutions': all_solutions,
                    'next_problem_idx': prob_idx + 1,
                    'stats': {
                        'total_solutions': len(all_solutions),
                        'correct': correct_count,
                        'incorrect': incorrect_count,
                        'problems_processed': prob_idx + 1
                    }
                }

                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint, f, indent=2)

                print(f"\n  Checkpoint saved: {checkpoint_path}")
                print(f"  Progress: {correct_count} correct, {incorrect_count} incorrect")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        checkpoint_path = output_dir / f'checkpoint_interrupted_p{prob_idx}.json'
        checkpoint = {
            'solutions': all_solutions,
            'next_problem_idx': prob_idx,
            'stats': {
                'total_solutions': len(all_solutions),
                'correct': correct_count,
                'incorrect': incorrect_count,
                'problems_processed': prob_idx
            }
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"  Saved: {checkpoint_path}")
        return

    # Save final results
    print(f"\n" + "="*80)
    print(f"GENERATION COMPLETE")
    print("="*80)
    print(f"  Total solutions: {len(all_solutions)}")
    print(f"  Correct: {correct_count} ({100*correct_count/len(all_solutions):.1f}%)")
    print(f"  Incorrect: {incorrect_count} ({100*incorrect_count/len(all_solutions):.1f}%)")

    # Save full dataset
    output_path = output_dir / 'error_solutions_full.json'

    final_data = {
        'metadata': {
            'n_base_problems': len(problems),
            'n_configs': len(configs),
            'total_solutions': len(all_solutions),
            'correct_count': correct_count,
            'incorrect_count': incorrect_count,
            'configs': configs
        },
        'solutions': all_solutions
    }

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\n✅ Saved full dataset: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Save config breakdown
    print(f"\nBreakdown by config:")
    config_stats = {}
    for sol in all_solutions:
        cfg = sol['config']
        if cfg not in config_stats:
            config_stats[cfg] = {'correct': 0, 'incorrect': 0}

        if sol['is_correct']:
            config_stats[cfg]['correct'] += 1
        else:
            config_stats[cfg]['incorrect'] += 1

    for cfg, stats in sorted(config_stats.items()):
        total = stats['correct'] + stats['incorrect']
        error_rate = 100 * stats['incorrect'] / total if total > 0 else 0
        print(f"  {cfg:25s}: {stats['incorrect']:3d} incorrect / {total:3d} total ({error_rate:5.1f}% error)")

    print(f"\n🎯 Target met: {incorrect_count} >= 500 incorrect? {'✅ YES' if incorrect_count >= 500 else '❌ NO'}")


if __name__ == "__main__":
    main()
