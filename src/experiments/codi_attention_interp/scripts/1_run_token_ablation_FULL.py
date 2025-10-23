#!/usr/bin/env python3
"""
FULL CCTA: Corruption-based Continuous Thought Attribution

Implements ALL corruption methods and measurements:

Corruptions:
1. Zero ablation
2. Gaussian noise (σ = 0.1, 0.5, 1.0, 2.0)
3. Random replacement
4. Shuffling

Measurements:
1. Answer accuracy drop
2. KL divergence from baseline
3. Attention pattern disruption

Usage:
    python 1_run_token_ablation_FULL.py [--test_mode]
"""
import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'scripts' / 'experiments'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG
from run_ablation_N_tokens_llama import NTokenPatcher, extract_answer_number, answers_match


def compute_kl_divergence(logits_baseline, logits_corrupted):
    """Compute KL divergence between baseline and corrupted output distributions."""
    # Convert logits to probabilities
    probs_baseline = torch.softmax(logits_baseline, dim=-1)
    probs_corrupted = torch.softmax(logits_corrupted, dim=-1)

    # Compute KL divergence: KL(baseline || corrupted)
    kl_div = torch.nn.functional.kl_div(
        probs_corrupted.log(),
        probs_baseline,
        reduction='batchmean'
    )

    return float(kl_div.item())


def compute_attention_disruption(attn_baseline, attn_corrupted):
    """Compute L2 distance between baseline and corrupted attention patterns."""
    # Flatten attention matrices and compute L2 distance
    diff = attn_baseline - attn_corrupted
    l2_dist = torch.norm(diff, p=2).item()

    return float(l2_dist)


def run_full_ccta(test_mode=False):
    """Run full CCTA with all corruption methods and measurements."""

    # Paths
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    if test_mode:
        dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
        output_file = Path(__file__).parent.parent / 'results' / 'ccta_full_results_test.json'
        print("=" * 80)
        print("FULL CCTA - TEST MODE (10 problems)")
        print("=" * 80)
    else:
        dataset_file = Path(__file__).parent.parent / 'results' / 'full_dataset_100.json'
        output_file = Path(__file__).parent.parent / 'results' / 'ccta_full_results_100.json'
        print("=" * 80)
        print("FULL CCTA - PRODUCTION MODE (100 problems)")
        print("=" * 80)

    print(f"\nLoading LLaMA CODI model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)
    model = cacher.model
    device = cacher.device

    test_layer = 'middle'
    print(f"Testing at layer: {test_layer} (L{LAYER_CONFIG[test_layer]})")

    print(f"\nLoading dataset from {dataset_file}...")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} problems")

    # Define corruption methods
    corruption_methods = {
        'zero': 'Zero ablation',
        'gauss_0.1': 'Gaussian σ=0.1',
        'gauss_0.5': 'Gaussian σ=0.5',
        'gauss_1.0': 'Gaussian σ=1.0',
        'gauss_2.0': 'Gaussian σ=2.0',
        'random': 'Random replacement',
        'shuffle': 'Position shuffle'
    }

    print(f"\nCorruption methods: {len(corruption_methods)}")
    for name, desc in corruption_methods.items():
        print(f"  - {name}: {desc}")

    # Results storage
    results = []

    # Cache random replacements pool (for random corruption)
    print("\nPre-caching activations for random replacement...")
    random_pool = {}
    for token_pos in range(6):
        random_pool[token_pos] = []

    # Cache activations from first 20 problems for random pool
    pool_size = min(20, len(dataset))
    for i in range(pool_size):
        problem = dataset[i]
        patcher = NTokenPatcher(cacher, num_tokens=6)
        acts = patcher.cache_N_token_activations(problem['question'], test_layer)
        for token_pos in range(6):
            random_pool[token_pos].append(acts[token_pos])

    print(f"✓ Cached {pool_size} problems × 6 tokens for random replacement")

    # Calculate total experiments
    total_experiments = len(dataset) * (1 + 6 * len(corruption_methods))  # baseline + 6 tokens × 7 corruptions
    pbar = tqdm(total=total_experiments, desc="Running CCTA")

    for problem_idx, problem in enumerate(dataset):
        problem_id = problem['gsm8k_id']
        question = problem['question']
        expected_answer = problem['answer']
        difficulty = problem['difficulty']

        problem_result = {
            'problem_id': problem_id,
            'difficulty': difficulty,
            'expected_answer': expected_answer,
            'baseline': {},
            'corruptions': {}
        }

        try:
            # ========================================
            # BASELINE: Run without any corruption
            # ========================================
            patcher = NTokenPatcher(cacher, num_tokens=6)

            # Get baseline output with attention
            baseline_output, baseline_logits, baseline_attention = run_with_measurements(
                patcher, question, test_layer, model, device
            )

            baseline_pred = extract_answer_number(baseline_output)
            baseline_correct = answers_match(baseline_pred, expected_answer)

            problem_result['baseline'] = {
                'output': baseline_output,
                'predicted_answer': baseline_pred,
                'correct': baseline_correct
            }

            pbar.update(1)
            pbar.set_postfix({'problem': problem_id, 'baseline': 'OK'})

            # Cache baseline activations
            baseline_acts = patcher.cache_N_token_activations(question, test_layer)

            # ========================================
            # CORRUPT EACH TOKEN WITH EACH METHOD
            # ========================================
            for token_pos in range(6):
                problem_result['corruptions'][f'token_{token_pos}'] = {}

                for corruption_name in corruption_methods.keys():
                    # Create corrupted activations
                    corrupted_acts = create_corrupted_activations(
                        baseline_acts=baseline_acts,
                        token_pos=token_pos,
                        corruption_type=corruption_name,
                        random_pool=random_pool,
                        problem_idx=problem_idx,
                        pool_size=pool_size
                    )

                    # Run with corrupted token
                    corrupted_output, corrupted_logits, corrupted_attention = run_with_measurements(
                        patcher, question, test_layer, model, device,
                        patch_activations=corrupted_acts
                    )

                    corrupted_pred = extract_answer_number(corrupted_output)
                    corrupted_correct = answers_match(corrupted_pred, expected_answer)

                    # Compute measurements
                    importance = baseline_correct and not corrupted_correct
                    kl_div = compute_kl_divergence(baseline_logits, corrupted_logits)
                    attn_disruption = compute_attention_disruption(baseline_attention, corrupted_attention)

                    problem_result['corruptions'][f'token_{token_pos}'][corruption_name] = {
                        'still_correct': corrupted_correct,
                        'importance': 1 if importance else 0,
                        'kl_divergence': kl_div,
                        'attention_disruption': attn_disruption
                    }

                    pbar.update(1)
                    pbar.set_postfix({
                        'problem': problem_id,
                        'token': token_pos,
                        'method': corruption_name,
                        'fail': importance
                    })

            # Compute aggregate statistics
            problem_result['summary'] = compute_problem_summary(problem_result)

            results.append(problem_result)

        except Exception as e:
            print(f"\nError on problem {problem_id}: {e}")
            import traceback
            traceback.print_exc()

            problem_result['error'] = str(e)
            results.append(problem_result)

            # Skip remaining experiments for this problem
            remaining = 1 + 6 * len(corruption_methods) - 1  # Already did baseline
            pbar.update(remaining)

    pbar.close()

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print_summary(results, corruption_methods)

    print(f"\n✓ Results saved to {output_file}")
    print("=" * 80)

    return results


def create_corrupted_activations(baseline_acts, token_pos, corruption_type,
                                 random_pool=None, problem_idx=0, pool_size=20):
    """Create corrupted version of activations based on corruption type."""
    corrupted_acts = [act.clone() for act in baseline_acts]

    if corruption_type == 'zero':
        # Zero ablation
        corrupted_acts[token_pos] = torch.zeros_like(baseline_acts[token_pos])

    elif corruption_type.startswith('gauss_'):
        # Gaussian noise
        sigma = float(corruption_type.split('_')[1])
        noise = torch.randn_like(baseline_acts[token_pos]) * sigma
        corrupted_acts[token_pos] = baseline_acts[token_pos] + noise

    elif corruption_type == 'random':
        # Random replacement from pool
        # Pick a random activation from the pool (not from same problem)
        pool = random_pool[token_pos]
        random_idx = (problem_idx + 1) % len(pool)  # Avoid same problem
        corrupted_acts[token_pos] = pool[random_idx]

    elif corruption_type == 'shuffle':
        # Shuffle all token positions
        indices = torch.randperm(6)
        shuffled_acts = [baseline_acts[i] for i in indices]
        return shuffled_acts  # Return shuffled entire sequence

    return corrupted_acts


def generate_with_measurements(patcher, question, test_layer, model, device, patch_activations=None):
    """Generate text and capture logits/attention during generation.

    Returns:
        output (str): Generated text
        logits (torch.Tensor): Logits from first answer token
        attention (torch.Tensor): Attention from first answer token
    """
    with torch.no_grad():
        # Setup
        tokenizer = patcher.tokenizer
        inputs = tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Process input
        input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)
        outputs = model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = outputs.past_key_values

        # BOT token
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)
        latent_embd = bot_emb

        # Setup patching if needed
        hook_handle = None
        current_step = [0]  # Use list to allow modification in closure

        if patch_activations is not None:
            layer_idx = LAYER_CONFIG[test_layer]

            def patch_hook(module, input, output):
                if current_step[0] < len(patch_activations):
                    activation_to_patch = patch_activations[current_step[0]]
                    if isinstance(output, tuple):
                        hidden_states = output[0].clone()
                        hidden_states[:, -1, :] = activation_to_patch.to(device)
                        return (hidden_states,) + output[1:]
                    else:
                        hidden_states = output.clone()
                        hidden_states[:, -1, :] = activation_to_patch.to(device)
                        return hidden_states
                return output

            # Get layer and attach hook
            try:
                target_layer = model.codi.base_model.model.model.layers[layer_idx]
            except AttributeError:
                target_layer = model.codi.model.layers[layer_idx]
            hook_handle = target_layer.register_forward_hook(patch_hook)

        try:
            # Generate continuous thoughts (with or without patching)
            for latent_step in range(patcher.num_latent):
                current_step[0] = latent_step

                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Disable patching for answer generation
            current_step[0] = 999

            # EOT token
            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)
            output_emb = eot_emb

            # Generate answer tokens and capture measurements on FIRST token
            pred_tokens = []
            captured_logits = None
            captured_attention = None

            for token_idx in range(200):
                out = model.codi(
                    inputs_embeds=output_emb,
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_attentions=True  # ← KEY: Enable attention capture
                )

                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # CAPTURE on first answer token
                if token_idx == 0:
                    captured_logits = logits[0].cpu()  # Shape: [vocab_size]

                    # Average attention across all heads and layers
                    # out.attentions is tuple of (num_layers,) each shape [batch, heads, seq, seq]
                    last_layer_attn = out.attentions[-1][0]  # Shape: [heads, seq_len, seq_len]
                    captured_attention = last_layer_attn.mean(dim=0).cpu()  # Average heads → [seq_len, seq_len]

                # Greedy decoding
                next_token_id = torch.argmax(logits, dim=-1)

                if next_token_id.item() == tokenizer.eos_token_id:
                    break

                pred_tokens.append(next_token_id.item())
                output_emb = model.get_embd(model.codi, model.model_name)(
                    next_token_id
                ).unsqueeze(1)

        finally:
            if hook_handle is not None:
                hook_handle.remove()

        # Decode output
        output = tokenizer.decode(pred_tokens, skip_special_tokens=True)

        return output, captured_logits, captured_attention


def run_with_measurements(patcher, question, test_layer, model, device, patch_activations=None):
    """Run inference and extract output, logits, and attention."""
    return generate_with_measurements(patcher, question, test_layer, model, device, patch_activations)


def compute_problem_summary(problem_result):
    """Compute summary statistics for a single problem."""
    summary = {
        'by_token': {},
        'by_corruption': {}
    }

    # Aggregate by token
    for token_key in problem_result['corruptions'].keys():
        token_pos = int(token_key.split('_')[1])
        corruptions = problem_result['corruptions'][token_key]

        summary['by_token'][f'token_{token_pos}'] = {
            'mean_importance': np.mean([c['importance'] for c in corruptions.values()]),
            'mean_kl_div': np.mean([c['kl_divergence'] for c in corruptions.values()]),
            'mean_attn_disruption': np.mean([c['attention_disruption'] for c in corruptions.values()])
        }

    # Aggregate by corruption type
    corruption_types = list(problem_result['corruptions']['token_0'].keys())
    for corr_type in corruption_types:
        importance_scores = []
        kl_divs = []
        attn_disruptions = []

        for token_key in problem_result['corruptions'].keys():
            data = problem_result['corruptions'][token_key][corr_type]
            importance_scores.append(data['importance'])
            kl_divs.append(data['kl_divergence'])
            attn_disruptions.append(data['attention_disruption'])

        summary['by_corruption'][corr_type] = {
            'mean_importance': np.mean(importance_scores),
            'mean_kl_div': np.mean(kl_divs),
            'mean_attn_disruption': np.mean(attn_disruptions)
        }

    return summary


def print_summary(results, corruption_methods):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("FULL CCTA COMPLETE")
    print("=" * 80)

    successful = [r for r in results if 'error' not in r]
    print(f"Problems processed: {len(successful)}/{len(results)}")

    baseline_correct = sum(1 for r in successful if r['baseline']['correct'])
    print(f"Baseline correct: {baseline_correct}/{len(successful)}")

    print(f"\nCorruption methods tested: {len(corruption_methods)}")
    print(f"Total experiments: {len(results) * 6 * len(corruption_methods)}")

    # Token-level importance by corruption method
    print(f"\nImportance by Token Position (across all corruptions):")
    for token_pos in range(6):
        importance_scores = []
        for result in successful:
            if result['baseline']['correct']:
                token_key = f'token_{token_pos}'
                for corr_type in corruption_methods.keys():
                    importance_scores.append(
                        result['corruptions'][token_key][corr_type]['importance']
                    )

        if importance_scores:
            mean_imp = np.mean(importance_scores)
            print(f"  Token {token_pos}: {mean_imp:.3f} ({100*mean_imp:.1f}% critical)")

    # Corruption method comparison
    print(f"\nImportance by Corruption Method (averaged across tokens):")
    for corr_name, corr_desc in corruption_methods.items():
        importance_scores = []
        for result in successful:
            if result['baseline']['correct']:
                for token_pos in range(6):
                    token_key = f'token_{token_pos}'
                    importance_scores.append(
                        result['corruptions'][token_key][corr_name]['importance']
                    )

        if importance_scores:
            mean_imp = np.mean(importance_scores)
            print(f"  {corr_desc:20s}: {mean_imp:.3f} ({100*mean_imp:.1f}% failure rate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true', help='Run on 10-problem test set')
    args = parser.parse_args()

    run_full_ccta(test_mode=args.test_mode)
