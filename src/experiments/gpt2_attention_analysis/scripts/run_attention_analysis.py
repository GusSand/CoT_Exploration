"""
GPT-2 Experiment 1: Attention Analysis (PRIORITY)

Analyzes attention patterns in GPT-2 CODI across all 12 layers and 6 tokens.

Input: gpt2_predictions_1000.json (shared data)
Output:
  - attention_weights_gpt2.json
  - attention_heatmaps/
  - token_importance_scores.json
"""

import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType


class GPT2AttentionAnalyzer:
    """Analyzes attention patterns in GPT-2 CODI."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize GPT-2 CODI model."""
        self.device = device
        print(f"Loading GPT-2 CODI model from {model_path}...")

        # Parse arguments
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
        print(f"  Architecture: GPT-2 (12 layers, 12 heads, 768 dim)")

    def extract_attention(self, question: str) -> Dict:
        """Extract attention weights for all layers and tokens."""

        with torch.no_grad():
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Storage for attention weights [layers × tokens × heads × seq_len × seq_len]
            attention_by_token = {f'token_{i}': [] for i in range(6)}

            latent_embd = bot_emb

            for token_idx in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_attentions=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Extract attention from all layers
                token_attentions = []
                for layer_idx in range(12):
                    attn = outputs.attentions[layer_idx]  # [batch, heads, seq, seq]
                    # Take last query position (the current token)
                    attn_last = attn[0, :, -1, :].cpu().numpy()  # [heads, seq]
                    token_attentions.append(attn_last.tolist())

                attention_by_token[f'token_{token_idx}'].append(token_attentions)

                # Update latent embedding
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            return attention_by_token

    def analyze_dataset(self, data_path: str, output_dir: Path):
        """Analyze attention for all samples."""

        print(f"\nLoading shared data from {data_path}...")
        with open(data_path, 'r') as f:
            data = json.load(f)

        samples = data['samples']
        n_samples = len(samples)

        print(f"  Loaded {n_samples} samples")
        print(f"\nStarting attention extraction...")

        results = []

        for i, sample in enumerate(samples):
            print(f"[{i+1}/{n_samples}] Processing...")

            # Extract attention
            attention = self.extract_attention(sample['question'])

            results.append({
                'id': sample['id'],
                'is_correct': sample['is_correct'],
                'attention': attention
            })

            # Save checkpoint every 100
            if (i + 1) % 100 == 0:
                checkpoint_path = output_dir / f"attention_checkpoint_{i+1}.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(results, f)
                print(f"  Checkpoint saved: {checkpoint_path}")

        # Save final
        output_path = output_dir / "attention_weights_gpt2.json"
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'model': 'GPT-2 CODI',
                    'n_samples': n_samples,
                    'n_layers': 12,
                    'n_tokens': 6,
                    'n_heads': 12,
                    'extraction_date': datetime.now().isoformat()
                },
                'results': results
            }, f)

        print(f"\n✅ Attention extraction complete!")
        print(f"   Output: {output_path}")

        return results


def main():
    """Run GPT-2 attention analysis."""

    print("="*60)
    print("GPT-2 EXPERIMENT 1: ATTENTION ANALYSIS")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "src/experiments/gpt2_shared_data/gpt2_predictions_1000.json"
    output_dir = project_root / "src/experiments/gpt2_attention_analysis/results"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path.home() / "codi_ckpt" / "gpt2_gsm8k"

    # Initialize analyzer
    analyzer = GPT2AttentionAnalyzer(str(model_path))

    # Run analysis
    results = analyzer.analyze_dataset(str(data_path), output_dir)

    print(f"\n" + "="*60)
    print("ATTENTION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Samples processed: {len(results)}")
    print(f"Results: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
