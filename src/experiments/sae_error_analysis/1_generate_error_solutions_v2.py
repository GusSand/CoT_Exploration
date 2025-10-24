"""
Generate solutions with errors by adapting extract_continuous_thoughts.py

Generates multiple solutions per problem using temperature and truncation.
Simpler approach: reuse proven CODI extraction code.
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm
import random
import re

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType
import os
from safetensors.torch import load_file


# Layer configuration (same as operation_circuits)
LAYER_CONFIG = {
    'early': 4,
    'middle': 8,
    'late': 14
}


def load_codi_model(model_path: str, device: str = 'cuda'):
    """Load CODI model - adapted from extract_continuous_thoughts.py"""
    print(f"Loading CODI LLaMA model from {model_path}...")

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=[
            '--model_name_or_path', 'meta-llama/Llama-3.2-1B-Instruct',
            '--output_dir', './tmp',
            '--num_latent', '6',
            '--use_lora', 'True',
            '--ckpt_dir', model_path,
            '--use_prj', 'True',
            '--prj_dim', '2048',
            '--lora_r', '128',
            '--lora_alpha', '32',
            '--lora_init', 'True',
        ]
    )

    model_args.train = False
    training_args.greedy = True

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)

    try:
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    model.float()
    model.to(device)
    model.eval()

    print("Model loaded successfully!")
    return model, training_args.num_latent


def extract_answer(solution: str) -> str:
    """Extract numerical answer from solution"""
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', solution)
    if match:
        return match.group(1)

    numbers = re.findall(r'-?\d+(?:\.\d+)?', solution)
    if numbers:
        return numbers[-1]

    return None


def generate_one_solution(
    model,
    question: str,
    device: str,
    temperature: float = 0.0,
    n_latent: int = 6,
    max_new_tokens: int = 256
) -> Dict:
    """
    Generate one solution and extract continuous thoughts.
    Adapted from extract_continuous_thoughts.py
    """
    tokenizer = model.tokenizer

    # Tokenize input
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)

    # Generate with CODI
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
            num_latent=n_latent,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

    # Decode solution
    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    solution_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract continuous thoughts from hidden states
    # CODI stores [THINK] tokens at specific positions
    hidden_states_all = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim) per layer

    continuous_thoughts = {}

    for layer_name, layer_idx in LAYER_CONFIG.items():
        # Get hidden states for this layer
        # hidden_states_all[step][layer][batch, 1, hidden]
        layer_thoughts = []

        # Extract all n_latent [THINK] tokens
        for token_idx in range(n_latent):
            # Access the hidden state at the appropriate generation step
            if token_idx < len(hidden_states_all):
                step_hidden = hidden_states_all[token_idx]
                if layer_idx < len(step_hidden):
                    layer_hidden = step_hidden[layer_idx]
                    # layer_hidden is [batch, 1, hidden_dim]
                    thought_vector = layer_hidden[0, -1, :].cpu().numpy().tolist()
                    layer_thoughts.append(thought_vector)

        continuous_thoughts[layer_name] = layer_thoughts

    return {
        'solution_text': solution_text,
        'continuous_thoughts': continuous_thoughts
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_problems', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default='src/experiments/sae_error_analysis/data')
    parser.add_argument('--model_path', type=str, default='/home/paperspace/codi_ckpt/llama_gsm8k')
    parser.add_argument('--checkpoint_every', type=int, default=25)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print("="*80)
    print("LOADING CODI MODEL")
    print("="*80)
    model, n_latent = load_codi_model(args.model_path, device)

    # Load problems
    print("\n" + "="*80)
    print("LOADING PROBLEMS")
    print("="*80)

    data_path = '/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json'
    with open(data_path, 'r') as f:
        all_problems = json.load(f)

    # Sample stratified
    by_difficulty = {}
    for prob in all_problems:
        diff = prob['difficulty']
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(prob)

    per_difficulty = args.n_problems // 4
    selected = []
    for diff, probs in by_difficulty.items():
        selected.extend(random.sample(probs, min(per_difficulty, len(probs))))

    problems = selected[:args.n_problems]

    print(f"  Loaded {len(problems)} problems")

    # Generation configs
    configs = [
        {'temp': 0.7, 'n_tokens': 6, 'name': 'temp0.7_s1'},
        {'temp': 0.7, 'n_tokens': 6, 'name': 'temp0.7_s2'},
        {'temp': 0.9, 'n_tokens': 6, 'name': 'temp0.9_s1'},
        {'temp': 0.9, 'n_tokens': 6, 'name': 'temp0.9_s2'},
        {'temp': 1.0, 'n_tokens': 6, 'name': 'temp1.0'},
        {'temp': 0.0, 'n_tokens': 1, 'name': 'trunc_1tok'},
        {'temp': 0.0, 'n_tokens': 2, 'name': 'trunc_2tok'},
        {'temp': 0.0, 'n_tokens': 3, 'name': 'trunc_3tok'},
        {'temp': 0.0, 'n_tokens': 6, 'name': 'baseline'},
    ]

    print(f"\nGeneration plan:")
    print(f"  Problems: {len(problems)}")
    print(f"  Configs: {len(configs)}")
    print(f"  Total: {len(problems) * len(configs)}")

    # Generate
    print("\n" + "="*80)
    print("GENERATING SOLUTIONS")
    print("="*80)

    all_solutions = []
    correct_count = 0
    incorrect_count = 0

    for prob_idx in tqdm(range(len(problems)), desc="Problems"):
        problem = problems[prob_idx]
        question = problem['question']
        ground_truth = str(problem['answer'])

        for config in configs:
            try:
                result = generate_one_solution(
                    model=model,
                    question=question,
                    device=device,
                    temperature=config['temp'],
                    n_latent=config['n_tokens'],
                    max_new_tokens=256
                )

                predicted = extract_answer(result['solution_text'])
                is_correct = (predicted == ground_truth) if predicted else False

                all_solutions.append({
                    'problem_id': problem.get('gsm8k_id', f'prob_{prob_idx}'),
                    'question': question,
                    'ground_truth': ground_truth,
                    'predicted_answer': predicted,
                    'is_correct': is_correct,
                    'solution': result['solution_text'],
                    'continuous_thoughts': result['continuous_thoughts'],
                    'config': config['name'],
                    'temperature': config['temp'],
                    'n_cot_tokens': config['n_tokens'],
                    'difficulty': problem.get('difficulty', 'unknown')
                })

                if is_correct:
                    correct_count += 1
                else:
                    incorrect_count += 1

            except Exception as e:
                print(f"\nError on problem {prob_idx}, config {config['name']}: {e}")
                continue

        # Checkpoint
        if (prob_idx + 1) % args.checkpoint_every == 0:
            checkpoint_path = output_dir / f'checkpoint_p{prob_idx+1}.json'
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'solutions': all_solutions,
                    'stats': {
                        'correct': correct_count,
                        'incorrect': incorrect_count,
                        'total': len(all_solutions)
                    }
                }, f, indent=2)

            print(f"\n  Checkpoint: {checkpoint_path}")
            print(f"  Progress: {correct_count} correct, {incorrect_count} incorrect")

    # Save final
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"  Total: {len(all_solutions)}")
    print(f"  Correct: {correct_count} ({100*correct_count/len(all_solutions):.1f}%)")
    print(f"  Incorrect: {incorrect_count} ({100*incorrect_count/len(all_solutions):.1f}%)")

    output_path = output_dir / 'error_solutions_full.json'
    with open(output_path, 'w') as f:
        json.dump({
            'metadata': {
                'n_problems': len(problems),
                'total_solutions': len(all_solutions),
                'correct': correct_count,
                'incorrect': incorrect_count,
                'configs': configs
            },
            'solutions': all_solutions
        }, f, indent=2)

    print(f"\n✅ Saved: {output_path}")
    print(f"🎯 Target: {incorrect_count} >= 500? {'✅' if incorrect_count >= 500 else '❌'}")


if __name__ == "__main__":
    main()
