#!/usr/bin/env python3
"""
Run LLaMA Steering Experiment - Full Dataset

Tests activation steering on LLaMA using full 532-pair dataset.
Tests multiple alpha values and all three layers.

Usage:
    python run_steering_experiment_llama_full.py
"""

import json
import sys
import re
import torch
from pathlib import Path
from tqdm import tqdm

# Add paths
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir / 'core'))
sys.path.insert(0, str(project_root / 'codi'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG


def extract_answer_number(text: str):
    """Extract numerical answer from generated text."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
        r'The answer is:\s*(-?\d+(?:\.\d+)?)',
        r'\$?\s*(-?\d+(?:\.\d+)?)\s*$',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                num = float(match.group(1))
                return int(num) if num.is_integer() else num
            except ValueError:
                continue

    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        try:
            num = float(numbers[-1])
            return int(num) if num.is_integer() else num
        except ValueError:
            pass

    return None


def answers_match(predicted, expected):
    """Check if predicted matches expected."""
    if predicted is None or expected is None:
        return False

    try:
        pred_float = float(predicted)
        exp_float = float(expected)
        return abs(pred_float - exp_float) < 0.01
    except (ValueError, TypeError):
        return False


def generate_with_steering(cacher, question: str, layer_name: str, alpha: float, direction: torch.Tensor):
    """Generate answer with steering applied to continuous thoughts.

    Args:
        cacher: ActivationCacherLLaMA instance
        question: Problem text
        layer_name: 'early', 'middle', or 'late'
        alpha: Steering strength
        direction: [6, 2048] steering direction

    Returns:
        str: Generated answer
    """
    device = cacher.device
    layer_idx = LAYER_CONFIG[layer_name]
    direction = direction.to(device)  # [6, 2048]

    with torch.no_grad():
        # Tokenize input
        inputs = cacher.tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Get initial embeddings
        input_embd = cacher.model.get_embd(cacher.model.codi, cacher.model.model_name)(input_ids).to(device)

        # Forward through model
        outputs = cacher.model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )

        past_key_values = outputs.past_key_values

        # Get BOT embedding
        bot_emb = cacher.model.get_embd(cacher.model.codi, cacher.model.model_name)(
            torch.tensor([cacher.model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        # Process latent tokens with steering
        latent_embd = bot_emb

        for latent_step in range(cacher.num_latent):
            outputs = cacher.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )

            past_key_values = outputs.past_key_values

            # Get final layer hidden state
            final_hidden = outputs.hidden_states[-1][:, -1, :]  # [1, 2048]

            # APPLY STEERING: Add steering vector to final hidden state
            # This affects what gets projected for the next latent step
            steered_hidden = final_hidden + alpha * direction[latent_step].unsqueeze(0)  # [1, 2048]

            # Prepare for next iteration
            latent_embd = steered_hidden.unsqueeze(1)  # [1, 1, 2048]

            # Apply projection if used
            if cacher.model.use_prj:
                latent_embd = cacher.model.prj(latent_embd)

        # Get EOT embedding (will be processed in first iteration of generation loop)
        eot_emb = cacher.model.get_embd(cacher.model.codi, cacher.model.model_name)(
            torch.tensor([cacher.model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        # Generate answer tokens using MANUAL LOOP
        # LLaMA's .generate() doesn't work properly with past_key_values alone
        pred_tokens = []
        output_emb = eot_emb  # First iteration will process EOT token

        for _ in range(200):  # max_new_tokens
            out = cacher.model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :cacher.model.codi.config.vocab_size-1]

            # Greedy decoding
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == cacher.tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = cacher.model.get_embd(cacher.model.codi, cacher.model.model_name)(
                next_token_id
            ).unsqueeze(1)

        # Decode answer
        output_text = cacher.tokenizer.decode(pred_tokens, skip_special_tokens=True)

    return output_text


def run_steering_test(cacher, problems, layer_name: str, alpha: float, direction: torch.Tensor):
    """Test steering on a set of problems.

    Returns:
        dict: Results including accuracy
    """
    correct = 0
    total = len(problems)
    results = []

    for item in tqdm(problems, desc=f"α={alpha:+.1f}"):
        question = item['question']
        expected = item['expected']

        try:
            output = generate_with_steering(cacher, question, layer_name, alpha, direction)
            predicted = extract_answer_number(output)
            is_correct = answers_match(predicted, expected)

            if is_correct:
                correct += 1

            results.append({
                'pair_id': item['pair_id'],
                'predicted': predicted,
                'expected': expected,
                'correct': is_correct
            })

        except Exception as e:
            results.append({
                'pair_id': item['pair_id'],
                'error': str(e)
            })

    accuracy = 100 * correct / total if total > 0 else 0

    return {
        'alpha': alpha,
        'correct': correct,
        'total': total,
        'accuracy': accuracy,
        'results': results
    }


def main():
    """Run steering experiments on all layers with multiple alpha values."""
    print("="*80)
    print("LLAMA STEERING EXPERIMENT - FULL DATASET (532 PAIRS)")
    print("="*80)

    # Load steering dataset (FULL VERSION)
    dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_llama_full.json'
    with open(dataset_file) as f:
        dataset = json.load(f)

    # Load problem pairs to get questions
    pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'
    with open(pairs_file) as f:
        all_problems = json.load(f)

    problem_lookup = {p['pair_id']: p for p in all_problems}

    # Prepare test set
    test_problems = []
    for item in dataset['test_correct'] + dataset['test_wrong']:
        pair_id = item['pair_id']
        problem = problem_lookup[pair_id]
        test_problems.append({
            'pair_id': pair_id,
            'question': problem['clean']['question'],
            'expected': item['expected']
        })

    print(f"\nTest set: {len(test_problems)} problems")
    print(f"Training set size: {len(dataset['train_correct']) + len(dataset['train_wrong'])} samples")

    # Initialize LLaMA
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    print(f"\nLoading LLaMA model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path, device='cuda')

    # Test alpha values (same as GPT-2 for comparison)
    alphas = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # Test all three layers
    for layer_name in ['early', 'middle', 'late']:
        print(f"\n{'='*80}")
        print(f"TESTING LAYER: {layer_name.upper()} (L{LAYER_CONFIG[layer_name]})")
        print("="*80)

        # Load steering direction (FROM FULL DATASET)
        direction_file = Path(__file__).parent / 'results' / 'steering_activations_llama_full' / layer_name / 'steering_direction.pt'
        direction_data = torch.load(direction_file)
        direction = direction_data['direction']

        print(f"\nSteering direction computed from:")
        print(f"  Correct: {direction_data['n_correct']} samples")
        print(f"  Wrong:   {direction_data['n_wrong']} samples")
        print(f"  Direction norm: {direction_data['direction_norm']:.4f}")
        print(f"\nTesting {len(alphas)} alpha values on {len(test_problems)} problems")

        # Run experiments
        all_results = []

        for alpha in alphas:
            print(f"\n--- Alpha = {alpha:+.1f} ---")
            result = run_steering_test(cacher, test_problems, layer_name, alpha, direction)
            all_results.append(result)
            print(f"Accuracy: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

        # Save detailed results
        output_dir = Path(__file__).parent / 'results' / 'steering_experiments_llama_full'
        output_dir.mkdir(parents=True, exist_ok=True)

        detailed_file = output_dir / f'detailed_{layer_name}.json'
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Saved detailed results: {detailed_file}")

        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY: {layer_name.upper()}")
        print("="*80)
        print(f"{'Alpha':>8s}  {'Accuracy':>8s}  {'Correct':>7s}")
        print("-" * 30)
        for result in all_results:
            print(f"{result['alpha']:>+8.1f}  {result['accuracy']:>7.1f}%  {result['correct']:>3d}/{result['total']:<3d}")

    print("\n" + "="*80)
    print("✅ FULL DATASET STEERING EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nDataset size comparison:")
    print("  Small dataset:  6 test, 16 train")
    print("  Full dataset:  107 test, 425 train")
    print("  Increase:      17.8× test, 26.6× train")


if __name__ == "__main__":
    main()
