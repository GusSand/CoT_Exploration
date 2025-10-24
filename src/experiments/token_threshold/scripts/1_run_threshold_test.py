#!/usr/bin/env python3
"""
Threshold Degradation Experiment

Tests how accuracy degrades as we corrupt 1→6 continuous thought tokens.

Key questions:
- What is the degradation curve?
- Does 4/6 corruption (67%) cause catastrophic failure?
- Which single tokens are most critical (from skip tests)?

Usage:
    python 1_run_threshold_test.py
"""
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(project_root / 'src' / 'experiments' / 'activation_patching' / 'core'))
sys.path.insert(0, str(project_root / 'src' / 'experiments' / 'activation_patching' / 'scripts' / 'experiments'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG
from run_ablation_N_tokens_llama import NTokenPatcher, extract_answer_number, answers_match
from corruption_utils import corrupt_n_tokens, get_all_corruption_configs
from utils import WandBLogger, DEFAULT_CORRUPTION_METHODS


def generate_with_patching(patcher, question, test_layer, model, device, patch_activations=None):
    """
    Generate answer with optional activation patching.

    Args:
        patcher: NTokenPatcher instance
        question: Question text
        test_layer: Layer name ('early', 'middle', 'late')
        model: CODI model
        device: torch device
        patch_activations: Optional list of activations to patch

    Returns:
        output: Generated text
    """
    with torch.no_grad():
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
        current_step = [0]

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

            # Generate answer tokens
            pred_tokens = []
            for token_idx in range(200):
                out = model.codi(
                    inputs_embeds=output_emb,
                    use_cache=True,
                    past_key_values=past_key_values
                )

                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

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

        return output


def run_threshold_experiment():
    """Run threshold degradation experiment on 10 test problems."""

    # Paths
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
    output_file = Path(__file__).parent.parent / 'results' / 'threshold_test_10.json'

    print("=" * 80)
    print("THRESHOLD DEGRADATION EXPERIMENT")
    print("=" * 80)

    # Initialize WandB
    logger = WandBLogger(
        project="codi-token-threshold",
        experiment_name="threshold_test_pilot",
        config={
            'dataset_size': 10,
            'corruption_methods': DEFAULT_CORRUPTION_METHODS,
            'test_layer': 'middle'
        },
        tags=['threshold', 'pilot', 'degradation']
    )

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

    # Get all corruption configurations
    corruption_configs = get_all_corruption_configs(n_tokens=6)
    print(f"\nCorruption configurations: {len(corruption_configs)}")
    print(f"Corruption methods: {DEFAULT_CORRUPTION_METHODS}")
    print(f"Total experiments per problem: {len(corruption_configs) * len(DEFAULT_CORRUPTION_METHODS)}")

    # Results storage
    results = []

    # Calculate total experiments
    total_experiments = len(dataset) * (1 + len(corruption_configs) * len(DEFAULT_CORRUPTION_METHODS))
    pbar = tqdm(total=total_experiments, desc="Running threshold test")

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
            'corruptions': []
        }

        try:
            # BASELINE: Run without any corruption
            patcher = NTokenPatcher(cacher, num_tokens=6)

            baseline_output = generate_with_patching(
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
            pbar.set_postfix({'problem': problem_id, 'baseline': baseline_correct})

            # Cache baseline activations
            baseline_acts = patcher.cache_N_token_activations(question, test_layer)

            # Run all corruption experiments
            for config in corruption_configs:
                level = config['level']
                positions = config['positions']
                label = config['label']

                for corruption_method in DEFAULT_CORRUPTION_METHODS:
                    # Create corrupted activations
                    corrupted_acts = corrupt_n_tokens(
                        baseline_acts=baseline_acts,
                        positions=positions,
                        corruption_method=corruption_method
                    )

                    # Generate with corrupted activations
                    corrupted_output = generate_with_patching(
                        patcher, question, test_layer, model, device,
                        patch_activations=corrupted_acts
                    )

                    corrupted_pred = extract_answer_number(corrupted_output)
                    corrupted_correct = answers_match(corrupted_pred, expected_answer)

                    # Record result
                    corruption_result = {
                        'corruption_level': level,
                        'positions': positions,
                        'position_label': label,
                        'corruption_method': corruption_method,
                        'output': corrupted_output,
                        'predicted_answer': corrupted_pred,
                        'correct': corrupted_correct,
                        'importance': baseline_correct and not corrupted_correct
                    }

                    problem_result['corruptions'].append(corruption_result)

                    # Log to WandB
                    logger.log_experiment(
                        problem_id=problem_id,
                        experiment_type='threshold',
                        result={
                            'corruption_level': level,
                            'position_label': label,
                            'corruption_method': corruption_method,
                            'baseline_correct': baseline_correct,
                            'corrupted_correct': corrupted_correct,
                            'accuracy_drop': baseline_correct and not corrupted_correct,
                            'difficulty': difficulty
                        }
                    )

                    pbar.update(1)
                    pbar.set_postfix({
                        'problem': problem_id,
                        'level': level,
                        'method': corruption_method[:4],
                        'fail': corruption_result['importance']
                    })

            results.append(problem_result)

        except Exception as e:
            print(f"\nError on problem {problem_id}: {e}")
            import traceback
            traceback.print_exc()

            problem_result['error'] = str(e)
            results.append(problem_result)

            # Skip remaining experiments for this problem
            remaining = 1 + len(corruption_configs) * len(DEFAULT_CORRUPTION_METHODS) - 1
            pbar.update(remaining)

    pbar.close()

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Print summary
    print_summary(results)

    logger.finish()

    print("=" * 80)

    return results


def print_summary(results):
    """Print summary statistics."""
    import numpy as np

    print("\n" + "=" * 80)
    print("THRESHOLD TEST COMPLETE")
    print("=" * 80)

    successful = [r for r in results if 'error' not in r]
    print(f"Problems processed: {len(successful)}/{len(results)}")

    baseline_correct = sum(1 for r in successful if r['baseline']['correct'])
    print(f"Baseline correct: {baseline_correct}/{len(successful)}")

    # Summary by corruption level
    print(f"\nAccuracy by Corruption Level (# tokens corrupted):")
    for level in range(1, 7):
        level_results = []
        for problem in successful:
            if problem['baseline']['correct']:
                for corruption in problem['corruptions']:
                    if corruption['corruption_level'] == level:
                        level_results.append(corruption['correct'])

        if level_results:
            accuracy = np.mean(level_results)
            count = len(level_results)
            print(f"  Level {level}: {accuracy:.1%} accuracy ({count} experiments)")


if __name__ == "__main__":
    run_threshold_experiment()
