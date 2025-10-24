#!/usr/bin/env python3
"""
Token Enhancement Experiment

Tests if amplifying specific token activations improves reasoning performance.

Key questions:
- Which token positions are most enhancement-responsive?
- What is the optimal enhancement multiplier?
- Are any tokens significantly more important than others?

Usage:
    python 3_run_enhancement_test.py
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
from corruption_utils import enhance_token
from utils import WandBLogger, DEFAULT_ENHANCEMENT_MULTIPLIERS


def generate_with_patching(patcher, question, test_layer, model, device, patch_activations=None):
    """Generate answer with optional activation patching (same as threshold test)."""
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
            # Generate continuous thoughts
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


def run_enhancement_experiment():
    """Run token enhancement experiment on 10 test problems."""

    # Paths
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
    output_file = Path(__file__).parent.parent / 'results' / 'enhancement_test_10.json'

    print("=" * 80)
    print("TOKEN ENHANCEMENT EXPERIMENT")
    print("=" * 80)

    # Initialize WandB
    logger = WandBLogger(
        project="codi-token-threshold",
        experiment_name="enhancement_test_pilot",
        config={
            'dataset_size': 10,
            'enhancement_multipliers': DEFAULT_ENHANCEMENT_MULTIPLIERS,
            'test_layer': 'middle'
        },
        tags=['enhancement', 'pilot', 'amplification']
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

    print(f"\nEnhancement multipliers: {DEFAULT_ENHANCEMENT_MULTIPLIERS}")
    print(f"Token positions: 0-5 (all 6 positions)")
    print(f"Total experiments per problem: {6 * len(DEFAULT_ENHANCEMENT_MULTIPLIERS)}")

    # Results storage
    results = []

    # Calculate total experiments
    total_experiments = len(dataset) * (1 + 6 * len(DEFAULT_ENHANCEMENT_MULTIPLIERS))
    pbar = tqdm(total=total_experiments, desc="Running enhancement test")

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
            'enhancements': []
        }

        try:
            # BASELINE: Run without any enhancement
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

            # Test all position × multiplier combinations
            for position in range(6):
                for multiplier in DEFAULT_ENHANCEMENT_MULTIPLIERS:
                    # Create enhanced activations
                    enhanced_acts = enhance_token(
                        baseline_acts=baseline_acts,
                        position=position,
                        multiplier=multiplier
                    )

                    # Generate with enhanced activations
                    enhanced_output = generate_with_patching(
                        patcher, question, test_layer, model, device,
                        patch_activations=enhanced_acts
                    )

                    enhanced_pred = extract_answer_number(enhanced_output)
                    enhanced_correct = answers_match(enhanced_pred, expected_answer)

                    # Record result
                    enhancement_result = {
                        'position': position,
                        'multiplier': multiplier,
                        'output': enhanced_output,
                        'predicted_answer': enhanced_pred,
                        'correct': enhanced_correct,
                        'improvement': not baseline_correct and enhanced_correct,
                        'degradation': baseline_correct and not enhanced_correct
                    }

                    problem_result['enhancements'].append(enhancement_result)

                    # Log to WandB
                    logger.log_experiment(
                        problem_id=problem_id,
                        experiment_type='enhancement',
                        result={
                            'position': position,
                            'multiplier': multiplier,
                            'baseline_correct': baseline_correct,
                            'enhanced_correct': enhanced_correct,
                            'improvement': enhancement_result['improvement'],
                            'degradation': enhancement_result['degradation'],
                            'difficulty': difficulty
                        }
                    )

                    pbar.update(1)
                    pbar.set_postfix({
                        'problem': problem_id,
                        'pos': position,
                        'mult': f'{multiplier:.1f}x',
                        'improve': enhancement_result['improvement']
                    })

            results.append(problem_result)

        except Exception as e:
            print(f"\nError on problem {problem_id}: {e}")
            import traceback
            traceback.print_exc()

            problem_result['error'] = str(e)
            results.append(problem_result)

            # Skip remaining experiments for this problem
            remaining = 1 + 6 * len(DEFAULT_ENHANCEMENT_MULTIPLIERS) - 1
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
    print("ENHANCEMENT TEST COMPLETE")
    print("=" * 80)

    successful = [r for r in results if 'error' not in r]
    print(f"Problems processed: {len(successful)}/{len(results)}")

    baseline_correct = sum(1 for r in successful if r['baseline']['correct'])
    print(f"Baseline correct: {baseline_correct}/{len(successful)}")

    # Summary by position
    print(f"\nAccuracy by Token Position (averaged across multipliers):")
    for position in range(6):
        position_results = []
        for problem in successful:
            for enhancement in problem['enhancements']:
                if enhancement['position'] == position:
                    position_results.append(enhancement['correct'])

        if position_results:
            accuracy = np.mean(position_results)
            count = len(position_results)
            print(f"  Position {position}: {accuracy:.1%} accuracy ({count} experiments)")

    # Count improvements and degradations
    total_improvements = sum(
        1 for p in successful for e in p['enhancements'] if e['improvement']
    )
    total_degradations = sum(
        1 for p in successful for e in p['enhancements'] if e['degradation']
    )

    print(f"\nOverall Enhancement Effects:")
    print(f"  Improvements: {total_improvements}")
    print(f"  Degradations: {total_degradations}")


if __name__ == "__main__":
    run_enhancement_experiment()
