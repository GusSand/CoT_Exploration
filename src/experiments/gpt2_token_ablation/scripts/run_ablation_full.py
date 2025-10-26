"""
GPT-2 Experiment 4: Token Ablation (FULL IMPLEMENTATION)

Tests impact of removing each continuous thought token on model performance.

Ablation strategy: Replace token activation with zeros at each position.
Tests all 6 tokens on a subset of samples (100 for speed).
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType


class GPT2TokenAblator:
    """Ablates continuous thought tokens in GPT-2 CODI."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize GPT-2 CODI model."""
        self.device = device
        print(f"Loading GPT-2 CODI model from {model_path}...")

        parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_args_into_dataclasses(
            args=[
                '--model_name_or_path', 'gpt2',
                '--output_dir', './tmp',
                '--num_latent', '6',
                '--use_lora', 'True',
                '--ckpt_dir', model_path,
                '--use_prj', 'True',
                '--prj_dim', '768',
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
            target_modules=['c_attn', 'c_proj', 'c_fc'],
            init_lora_weights=True,
        )

        self.model = CODI(model_args, training_args, lora_config)

        import os
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.codi.tie_weights()
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = training_args.num_latent

        print("Model loaded successfully!")

    def extract_answer(self, text: str) -> int:
        """Extract numerical answer."""
        try:
            if '####' in text:
                answer_str = text.split('####')[-1].strip()
            else:
                import re
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
                if numbers:
                    answer_str = numbers[-1]
                else:
                    return None

            answer_str = answer_str.replace(',', '')
            return int(float(answer_str))
        except:
            return None

    def predict_with_ablation(self, question: str, ablate_token: int = None):
        """
        Generate prediction with optional token ablation.

        Args:
            question: Input question
            ablate_token: Token index to ablate (0-5), or None for baseline

        Returns:
            Predicted answer (int or None)
        """
        with torch.no_grad():
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
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

            for latent_step in range(self.num_latent):
                # Ablate if this is the target token
                if ablate_token is not None and latent_step == ablate_token:
                    # Zero out the embedding
                    latent_embd = torch.zeros_like(latent_embd)

                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Update for next iteration (unless ablated)
                if ablate_token is None or latent_step != ablate_token:
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                    if self.model.use_prj:
                        latent_embd = self.model.prj(latent_embd)

            # EOT and generate
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
            max_new_tokens = 256
            generated_ids = []

            for _ in range(max_new_tokens):
                if len(generated_ids) == 0:
                    next_token_logits = outputs.logits[:, -1, :]
                else:
                    last_token_id = torch.tensor([[generated_ids[-1]]], device=self.device)
                    last_token_embd = self.model.get_embd(self.model.codi, self.model.model_name)(last_token_id)

                    outputs = self.model.codi(
                        inputs_embeds=last_token_embd,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

                next_token_id = torch.argmax(next_token_logits, dim=-1).item()

                if next_token_id == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)

            prediction_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            prediction = self.extract_answer(prediction_text)

            return prediction

    def run_ablation_study(self, samples, output_dir: Path):
        """Run ablation on all tokens for given samples."""

        print(f"\nRunning ablation study on {len(samples)} samples...")

        results = []

        for i, sample in enumerate(samples):
            print(f"[{i+1}/{len(samples)}] {sample['question'][:50]}...")

            ground_truth = sample['ground_truth']

            # Baseline (no ablation)
            baseline_pred = self.predict_with_ablation(sample['question'], ablate_token=None)
            baseline_correct = (baseline_pred == ground_truth)

            ablation_results = {
                'baseline': baseline_correct
            }

            # Ablate each token
            for token_idx in range(6):
                ablated_pred = self.predict_with_ablation(sample['question'], ablate_token=token_idx)
                ablated_correct = (ablated_pred == ground_truth)
                ablation_results[f'ablate_token_{token_idx}'] = ablated_correct

            results.append({
                'id': sample['id'],
                'question': sample['question'],
                'ground_truth': ground_truth,
                'baseline_correct': baseline_correct,
                'ablations': ablation_results
            })

            # Checkpoint every 20
            if (i + 1) % 20 == 0:
                checkpoint_path = output_dir / f"ablation_checkpoint_{i+1}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  Checkpoint saved")

        return results


def main():
    """Run GPT-2 token ablation experiment."""

    print("="*60)
    print("GPT-2 EXPERIMENT 4: TOKEN ABLATION")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "src/experiments/gpt2_shared_data/gpt2_predictions_1000.json"
    output_dir = project_root / "src/experiments/gpt2_token_ablation/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path.home() / "codi_ckpt" / "gpt2_gsm8k"

    # Load data (use subset for speed)
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Use first 100 samples
    samples = data['samples'][:100]
    print(f"  Using {len(samples)} samples")

    # Initialize ablator
    ablator = GPT2TokenAblator(str(model_path))

    # Run ablation
    results = ablator.run_ablation_study(samples, output_dir)

    # Save results
    results_path = output_dir / "ablation_results_gpt2.json"
    with open(results_path, 'w') as f:
        json.dump({
            'metadata': {
                'model': 'GPT-2 CODI',
                'n_samples': len(samples),
                'n_tokens': 6,
                'date': datetime.now().isoformat()
            },
            'results': results
        }, f, indent=2)

    print(f"\n  Results saved: {results_path}")

    # Compute statistics
    baseline_acc = sum(1 for r in results if r['baseline_correct']) / len(results)
    print(f"\n  Baseline accuracy: {baseline_acc:.4f}")

    for token_idx in range(6):
        token_acc = sum(1 for r in results if r['ablations'][f'ablate_token_{token_idx}']) / len(results)
        impact = baseline_acc - token_acc
        print(f"  Ablate Token {token_idx}: {token_acc:.4f} (impact: {impact:+.4f})")

    print("\n" + "="*60)
    print("✅ GPT-2 TOKEN ABLATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
