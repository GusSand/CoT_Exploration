"""
Positional Patching Experiment

Tests if specific token positions are more causally important than others.

Usage:
    python run_positional_patching.py --model_path ~/codi_ckpt/gpt2_gsm8k/ \
                                       --problem_pairs data/problem_pairs.json \
                                       --positions 0 1 4 5 \
                                       --config_name "endpoints" \
                                       --output_dir results_positional_endpoints/
"""

import json
import os
import re
import argparse
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
import wandb
import torch

from cache_activations import ActivationCacher, LAYER_CONFIG
from patch_and_eval import ActivationPatcher

SCRIPT_DIR = Path(__file__).parent.absolute()


def extract_answer_number(text: str):
    """Extract numerical answer."""
    patterns = [
        r'####\s*(-?\d+(?:\.\d+)?)',
        r'(?:answer|total|result)(?:\s+is)?\s*[:=]?\s*\$?\s*(-?\d+(?:\.\d+)?)',
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


def answers_match(predicted, expected) -> bool:
    """Check if answers match."""
    if predicted is None or expected is None:
        return False
    try:
        return abs(float(predicted) - float(expected)) < 0.01
    except (ValueError, TypeError):
        return False


def classify_output(generated_text: str, clean_answer, corrupted_answer) -> str:
    """Classify output."""
    extracted = extract_answer_number(generated_text)
    if extracted is None:
        if len(generated_text) < 10 or not any(c.isalpha() for c in generated_text):
            return "gibberish"
        return "other_coherent"
    if answers_match(extracted, clean_answer):
        return "clean_answer"
    elif answers_match(extracted, corrupted_answer):
        return "corrupted_answer"
    else:
        return "other_coherent"


class PositionalPatcher:
    """Patches specific token positions (not just first N)."""

    def __init__(self, cacher: ActivationCacher, positions: List[int]):
        self.cacher = cacher
        self.model = cacher.model
        self.tokenizer = cacher.tokenizer
        self.device = cacher.device
        self.num_latent = cacher.num_latent
        self.positions = sorted(positions)  # e.g., [0, 1, 4, 5]

        assert all(0 <= p < 6 for p in positions), f"Positions must be 0-5, got {positions}"
        assert len(positions) > 0, "Must specify at least one position"

        self.baseline_patcher = ActivationPatcher(cacher)

        # State for patching
        self.patch_activations = None  # Dict: {position: activation}
        self.patch_layer_idx = None
        self.current_step = 0
        self.hook_handle = None

    def cache_positional_activations(self, problem_text: str, layer_name: str) -> Dict[int, torch.Tensor]:
        """Cache activations for specific positions."""
        layer_idx = LAYER_CONFIG[layer_name]
        activations = {}

        with torch.no_grad():
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Cache ALL 6 tokens, then filter to requested positions
            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Only cache if this position is in our list
                if latent_step in self.positions:
                    activation = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                    activations[latent_step] = activation

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

        return activations

    def _get_layer_module(self, layer_idx: int):
        """Get transformer layer module."""
        try:
            return self.model.codi.base_model.model.transformer.h[layer_idx]
        except AttributeError:
            return self.model.codi.transformer.h[layer_idx]

    def _create_patch_hook(self):
        """Create hook that patches only specified positions."""
        def patch_hook(module, input, output):
            # Patch if current step is in our position list
            if self.patch_activations is not None and self.current_step in self.patch_activations:
                activation_to_patch = self.patch_activations[self.current_step]

                if isinstance(output, tuple):
                    hidden_states = output[0].clone()
                    hidden_states[:, -1, :] = activation_to_patch.to(self.device)
                    return (hidden_states,) + output[1:]
                else:
                    hidden_states = output.clone()
                    hidden_states[:, -1, :] = activation_to_patch.to(self.device)
                    return hidden_states

            return output

        return patch_hook

    def run_with_positional_patching(
        self,
        problem_text: str,
        patch_activations: Dict[int, torch.Tensor],
        layer_name: str,
        max_new_tokens: int = 200
    ) -> str:
        """Run with specific positions patched."""
        self.patch_activations = patch_activations
        self.patch_layer_idx = LAYER_CONFIG[layer_name]
        self.current_step = 0

        target_layer = self._get_layer_module(self.patch_layer_idx)
        hook = self._create_patch_hook()
        self.hook_handle = target_layer.register_forward_hook(hook)

        try:
            answer = self._generate_with_patching(problem_text, max_new_tokens)
        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None

        self.patch_activations = None
        self.current_step = 0

        return answer

    def _generate_with_patching(self, problem_text: str, max_new_tokens: int) -> str:
        """Generate with patching active."""
        with torch.no_grad():
            inputs = self.tokenizer(problem_text, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = outputs.past_key_values

            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Generate latent thoughts WITH PATCHING at specified positions
            for latent_step in range(self.num_latent):
                self.current_step = latent_step  # Hook will check this

                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            # EOT token
            eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            outputs = self.model.codi(
                inputs_embeds=eot_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Generate answer
            generated_ids = self.model.codi.generate(
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return answer

    def run_without_patch(self, problem_text: str, max_new_tokens: int = 200) -> str:
        """Run without patching."""
        return self.baseline_patcher.run_without_patch(problem_text, max_new_tokens)


class PositionalExperimentRunner:
    """Runs positional patching experiment."""

    def __init__(self, model_path: str, positions: List[int], config_name: str, output_dir: str, wandb_project: str = None):
        self.positions = positions
        self.config_name = config_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading CODI model...")
        cacher = ActivationCacher(model_path)
        self.patcher = PositionalPatcher(cacher, positions)
        print("✓ Model loaded")

        self.use_wandb = wandb_project is not None
        if self.use_wandb:
            print("Initializing WandB...")
            wandb.init(
                project=wandb_project,
                name=f"positional-{config_name}",
                dir=str(SCRIPT_DIR / "wandb_runs"),
                config={
                    "experiment": "positional_patching",
                    "model": model_path,
                    "positions": positions,
                    "config_name": config_name,
                    "num_positions": len(positions),
                    "hypothesis": "Test if early/middle/late positions have different causal importance"
                }
            )
            print(f"✓ WandB initialized: {wandb.run.url}")

    def run_pair(self, pair: Dict) -> Dict:
        """Run experiment on single pair."""
        pair_id = pair['pair_id']
        clean_q = pair['clean']['question']
        corrupted_q = pair['corrupted']['question']
        clean_ans = pair['clean']['answer']
        corrupted_ans = pair['corrupted']['answer']

        # Baselines
        clean_generated = self.patcher.run_without_patch(clean_q, max_new_tokens=200)
        clean_predicted = extract_answer_number(clean_generated)
        clean_correct = answers_match(clean_predicted, clean_ans)

        corrupted_generated = self.patcher.run_without_patch(corrupted_q, max_new_tokens=200)
        corrupted_predicted = extract_answer_number(corrupted_generated)
        corrupted_correct = answers_match(corrupted_predicted, corrupted_ans)

        if not (clean_correct and corrupted_correct):
            return {
                'pair_id': pair_id,
                'valid': False,
                'reason': f'clean_correct={clean_correct}, corrupted_correct={corrupted_correct}'
            }

        # Patch specified positions
        patched_results = {}
        for layer_name in LAYER_CONFIG.keys():
            clean_activations = self.patcher.cache_positional_activations(clean_q, layer_name)

            patched_generated = self.patcher.run_with_positional_patching(
                corrupted_q,
                clean_activations,
                layer_name,
                max_new_tokens=200
            )
            patched_predicted = extract_answer_number(patched_generated)
            classification = classify_output(patched_generated, clean_ans, corrupted_ans)

            patched_results[layer_name] = {
                'classification': classification,
                'extracted': patched_predicted,
                'generated': patched_generated,
                'matches_clean': answers_match(patched_predicted, clean_ans),
                'matches_corrupted': answers_match(patched_predicted, corrupted_ans)
            }

        return {
            'pair_id': pair_id,
            'valid': True,
            'clean': {'correct': clean_correct, 'expected': clean_ans, 'extracted': clean_predicted, 'generated': clean_generated},
            'corrupted': {'correct': corrupted_correct, 'expected': corrupted_ans, 'extracted': corrupted_predicted, 'generated': corrupted_generated},
            'patched': patched_results
        }

    def run_experiment(self, problem_pairs: List[Dict]) -> Dict:
        """Run full experiment."""
        print(f"\n{'='*60}")
        print(f"POSITIONAL PATCHING: {self.config_name}")
        print(f"{'='*60}")
        print(f"Positions: {self.positions}")
        print(f"Total pairs: {len(problem_pairs)}")
        print(f"{'='*60}\n")

        valid_results = []
        invalid_results = []

        for pair in tqdm(problem_pairs, desc=f"Processing ({self.config_name})"):
            try:
                result = self.run_pair(pair)
                if result['valid']:
                    valid_results.append(result)
                    if self.use_wandb:
                        log_data = {'pair_id': result['pair_id']}
                        for layer_name in LAYER_CONFIG.keys():
                            classification = result['patched'][layer_name]['classification']
                            log_data[f'{layer_name}_classification'] = classification
                            log_data[f'{layer_name}_matches_clean'] = int(result['patched'][layer_name]['matches_clean'])
                        wandb.log(log_data)
                else:
                    invalid_results.append(result)
            except Exception as e:
                print(f"\nERROR on pair {pair['pair_id']}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if len(valid_results) == 0:
            return {'error': 'no_valid_results'}

        metrics = self._calculate_metrics(valid_results)

        results_data = {
            'summary': metrics,
            'valid_results': valid_results,
            'invalid_results': invalid_results,
            'config': {
                'total_pairs': len(problem_pairs),
                'valid_pairs': len(valid_results),
                'positions': self.positions,
                'config_name': self.config_name
            }
        }

        results_path = os.path.join(self.output_dir, f'experiment_results_{self.config_name}.json')
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")

        if self.use_wandb:
            summary_log = {'summary/valid_pairs': len(valid_results)}
            for layer_name in LAYER_CONFIG.keys():
                for cls in ['clean_answer', 'corrupted_answer', 'other_coherent', 'gibberish']:
                    pct = metrics[layer_name]['classification_pcts'][cls]
                    summary_log[f'summary/{layer_name}_{cls}_pct'] = pct
            wandb.log(summary_log)

        return results_data

    def _calculate_metrics(self, valid_results: List[Dict]) -> Dict:
        """Calculate metrics."""
        total_valid = len(valid_results)

        layer_metrics = {}
        for layer_name in LAYER_CONFIG.keys():
            classifications = {
                'clean_answer': 0,
                'corrupted_answer': 0,
                'other_coherent': 0,
                'gibberish': 0
            }

            for r in valid_results:
                classification = r['patched'][layer_name]['classification']
                classifications[classification] += 1

            classification_pcts = {k: v / total_valid for k, v in classifications.items()}

            layer_metrics[layer_name] = {
                'classifications': classifications,
                'classification_pcts': classification_pcts
            }

        return {
            'total_valid': total_valid,
            **layer_metrics
        }

    def print_summary(self, results: Dict):
        """Print summary."""
        if 'error' in results:
            print(f"\n⚠️  Experiment failed: {results['error']}")
            return

        metrics = results['summary']

        print(f"\n\n{'='*60}")
        print(f"POSITIONAL RESULTS: {self.config_name}")
        print(f"{'='*60}")
        print(f"Positions: {self.positions}")
        print(f"Valid pairs: {metrics['total_valid']}")
        print(f"{'='*60}\n")

        print(f"{'Layer':<10} {'Clean %':<10} {'Corrupt %':<12} {'Other %':<10} {'Gibberish %':<12}")
        print(f"{'-'*60}")

        for layer_name in LAYER_CONFIG.keys():
            m = metrics[layer_name]
            clean_pct = 100 * m['classification_pcts']['clean_answer']
            corrupt_pct = 100 * m['classification_pcts']['corrupted_answer']
            other_pct = 100 * m['classification_pcts']['other_coherent']
            gibberish_pct = 100 * m['classification_pcts']['gibberish']

            layer_idx = LAYER_CONFIG[layer_name]
            print(f"{layer_name.capitalize()} (L{layer_idx:2d})  "
                  f"{clean_pct:5.1f}%     "
                  f"{corrupt_pct:5.1f}%       "
                  f"{other_pct:5.1f}%     "
                  f"{gibberish_pct:5.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--problem_pairs', type=str, required=True)
    parser.add_argument('--positions', nargs='+', type=int, required=True,
                       help="Token positions to patch (0-5)")
    parser.add_argument('--config_name', type=str, required=True,
                       help="Name for this configuration (e.g., 'endpoints', 'middle')")
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--wandb_project', type=str, default='codi-activation-patching')
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    with open(args.problem_pairs, 'r') as f:
        pairs = json.load(f)
    print(f"✓ Loaded {len(pairs)} problem pairs")

    wandb_project = None if args.no_wandb else args.wandb_project
    runner = PositionalExperimentRunner(
        args.model_path,
        args.positions,
        args.config_name,
        args.output_dir,
        wandb_project
    )

    results = runner.run_experiment(pairs)
    runner.print_summary(results)

    if runner.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
