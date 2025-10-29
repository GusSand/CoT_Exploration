#!/usr/bin/env python3
"""
Attention Analysis Data Collection

Collects comprehensive data for 3 follow-up analyses:
1. A2: Attention Redistribution - WHERE does attention flow when CT0 blocked?
2. D1: Layer Divergence - WHEN do reasoning paths diverge (baseline vs CT0-blocked)?
3. F1: Hidden State Patterns - CAN we detect errors from CT activations?

This script runs intervention experiments while saving:
- Full attention weights at every layer/head/position
- Per-layer hidden states at CT positions
- Problem metadata (complexity, correctness, error magnitude)

Time estimate: 4-6 hours for full dataset (1,319 problems × 2 conditions)
Outputs: Large HDF5 files with attention weights + activations

Usage:
    python 8_collect_attention_analysis_data.py --model llama --n_problems 100 --output_dir results/attention_data
"""

import json
import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_model, extract_answer, validate_model_architecture


class AttentionDataCollector:
    """
    Collect comprehensive attention and activation data during CODI generation.

    For each problem, saves:
    - Attention weights: [n_layers, n_heads, seq_len, seq_len] at each CT generation step
    - Hidden states: [n_layers, seq_len, hidden_dim] at each CT generation step
    - CT activations: [6, hidden_dim] for the 6 continuous thought positions
    - Metadata: problem_id, condition (baseline/ct0_blocked), correctness, answers
    """

    def __init__(self, model, model_name: str, tokenizer, device: str = 'cuda'):
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device

        # Storage for collected data
        self.attention_storage = []
        self.hidden_state_storage = []
        self.ct_activation_storage = []

        # Hooks for capturing data
        self.attention_hooks = []
        self.hidden_state_hooks = []

    def _register_attention_hooks(self):
        """Register forward hooks to capture attention weights."""
        self.captured_attentions = []

        def attention_hook(module, input, output):
            """Capture attention weights from attention module output."""
            # Output format: (hidden_states, attention_weights, ...)
            if len(output) > 1 and output[1] is not None:
                # attention_weights: [batch, n_heads, seq_len, seq_len]
                self.captured_attentions.append(output[1].detach().cpu())

        # Register hooks on all attention modules
        if self.model_name == 'llama':
            layers = self.model.codi.model.model.layers
        elif self.model_name == 'gpt2':
            layers = self.model.codi.transformer.h

        for layer_idx, layer in enumerate(layers):
            if self.model_name == 'llama':
                attn_module = layer.self_attn
            elif self.model_name == 'gpt2':
                attn_module = layer.attn

            hook = attn_module.register_forward_hook(attention_hook)
            self.attention_hooks.append(hook)

    def _register_hidden_state_hooks(self):
        """Register forward hooks to capture per-layer hidden states."""
        self.captured_hidden_states = []

        def hidden_state_hook(module, input, output):
            """Capture hidden states from layer output."""
            # output[0]: hidden_states [batch, seq_len, hidden_dim]
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            self.captured_hidden_states.append(hidden_states.detach().cpu())

        # Register hooks on all transformer layers
        if self.model_name == 'llama':
            layers = self.model.codi.model.model.layers
        elif self.model_name == 'gpt2':
            layers = self.model.codi.transformer.h

        for layer in layers:
            hook = layer.register_forward_hook(hidden_state_hook)
            self.hidden_state_hooks.append(hook)

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.attention_hooks + self.hidden_state_hooks:
            hook.remove()
        self.attention_hooks = []
        self.hidden_state_hooks = []

    def generate_with_data_collection(
        self,
        question: str,
        condition: str = 'baseline',
        max_new_tokens: int = 100
    ) -> Dict:
        """
        Generate answer while collecting attention weights and hidden states.

        Args:
            question: Problem question text
            condition: 'baseline' or 'ct0_blocked'
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dict with:
            - generated_text: Answer text
            - attention_data: List of attention weight tensors per CT generation step
            - hidden_state_data: List of hidden state tensors per CT generation step
            - ct_activations: [6, hidden_dim] tensor of CT position activations
        """

        # Register hooks
        self._register_attention_hooks()
        self._register_hidden_state_hooks()

        # Tokenize question
        inputs = self.tokenizer(question, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']

        # Question forward pass
        self.captured_attentions = []
        self.captured_hidden_states = []

        question_outputs = self.model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True
        )
        past_key_values = question_outputs.past_key_values
        question_len = input_ids.shape[1]

        # Generate BOT token
        bot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        latent_embd = bot_emb
        ct_start_pos = question_len + 1  # After question + BOT

        # Storage for CT generation data
        ct_attention_data = []
        ct_hidden_state_data = []
        ct_activations = []

        # Generate 6 continuous thought tokens
        for step in range(6):
            current_pos = ct_start_pos + step

            # Clear captured data
            self.captured_attentions = []
            self.captured_hidden_states = []

            # Apply CT0 blocking if condition is 'ct0_blocked'
            attention_mask = None
            if condition == 'ct0_blocked' and step > 0:  # Block CT0 after it's generated
                full_seq_len = current_pos + 1
                attention_mask = torch.zeros(1, full_seq_len, device=self.device)
                attention_mask[0, ct_start_pos] = -10000.0  # Block CT0 position

            outputs = self.model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                past_key_values=past_key_values,
                attention_mask=attention_mask
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Save CT activation
            ct_activations.append(latent_embd.squeeze(1).detach().cpu())

            if self.model.use_prj:
                latent_embd = self.model.prj(latent_embd)

            # Store captured attention and hidden states
            ct_attention_data.append({
                'step': step,
                'attentions': self.captured_attentions.copy() if self.captured_attentions else None
            })
            ct_hidden_state_data.append({
                'step': step,
                'hidden_states': self.captured_hidden_states.copy() if self.captured_hidden_states else None
            })

        # Generate EOT token
        eot_emb = self.model.get_embd(self.model.codi, self.model_name)(
            torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
        ).unsqueeze(0)

        output_emb = eot_emb
        pred_tokens = []

        for _ in range(max_new_tokens):
            out = self.model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :self.model.codi.config.vocab_size-1]
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = self.model.get_embd(self.model.codi, self.model_name)(
                next_token_id
            ).unsqueeze(1)

        # Decode answer
        generated_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        # Clear hooks
        self._clear_hooks()

        return {
            'generated_text': generated_text,
            'attention_data': ct_attention_data,
            'hidden_state_data': ct_hidden_state_data,
            'ct_activations': torch.stack(ct_activations) if ct_activations else None
        }


def run_data_collection(
    model_name: str = 'llama',
    n_problems: int = 100,
    output_dir: str = 'results/attention_data',
    save_frequency: int = 10
):
    """
    Run comprehensive data collection for attention analysis.

    For each problem, runs TWO conditions:
    1. Baseline (normal generation)
    2. CT0-blocked (CT0 attention masked)

    Saves data incrementally to HDF5 files to avoid memory issues.
    """

    print(f"\n{'='*60}")
    print(f"Attention Analysis Data Collection")
    print(f"Model: {model_name}")
    print(f"Problems: {n_problems}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Validate architecture
    print("\nValidating model architecture...")
    validate_model_architecture(model, model_name, tokenizer)
    print("✓ Architecture validated")

    # Load dataset
    print("\nLoading GSM8K test set...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = list(dataset.select(range(min(n_problems, len(dataset)))))
    print(f"Loaded {len(test_problems)} problems")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize collector
    collector = AttentionDataCollector(model, model_name, tokenizer, device)

    # Results storage
    results = []

    # Process each problem
    for problem_idx, problem in enumerate(tqdm(test_problems, desc="Collecting data")):
        question = problem['question']
        gold_answer = extract_answer(problem['answer'])

        problem_data = {
            'problem_id': problem_idx,
            'question': question,
            'gold_answer': gold_answer
        }

        # Baseline condition
        try:
            baseline_result = collector.generate_with_data_collection(
                question=question,
                condition='baseline'
            )
            pred_answer_baseline = extract_answer(baseline_result['generated_text'])

            problem_data['baseline'] = {
                'pred_answer': pred_answer_baseline,
                'correct': (pred_answer_baseline == gold_answer),
                'generated_text': baseline_result['generated_text'],
                'has_attention_data': baseline_result['attention_data'] is not None,
                'has_hidden_state_data': baseline_result['hidden_state_data'] is not None,
                'ct_activations_shape': baseline_result['ct_activations'].shape if baseline_result['ct_activations'] is not None else None
            }

            # Save attention data to HDF5
            if baseline_result['attention_data'] is not None:
                h5_file = output_path / f'{model_name}_problem_{problem_idx:04d}_baseline_attention.h5'
                with h5py.File(h5_file, 'w') as f:
                    for step_data in baseline_result['attention_data']:
                        step = step_data['step']
                        if step_data['attentions']:
                            for layer_idx, attn in enumerate(step_data['attentions']):
                                f.create_dataset(
                                    f'step_{step}/layer_{layer_idx}/attention',
                                    data=attn.numpy(),
                                    compression='gzip'
                                )

            # Save hidden states to HDF5
            if baseline_result['hidden_state_data'] is not None:
                h5_file = output_path / f'{model_name}_problem_{problem_idx:04d}_baseline_hidden.h5'
                with h5py.File(h5_file, 'w') as f:
                    for step_data in baseline_result['hidden_state_data']:
                        step = step_data['step']
                        if step_data['hidden_states']:
                            for layer_idx, hidden in enumerate(step_data['hidden_states']):
                                f.create_dataset(
                                    f'step_{step}/layer_{layer_idx}/hidden_state',
                                    data=hidden.numpy(),
                                    compression='gzip'
                                )

            # Save CT activations
            if baseline_result['ct_activations'] is not None:
                np_file = output_path / f'{model_name}_problem_{problem_idx:04d}_baseline_ct_activations.npy'
                np.save(np_file, baseline_result['ct_activations'].numpy())

        except Exception as e:
            print(f"\nFailed on baseline problem {problem_idx}: {e}")
            problem_data['baseline'] = {'error': str(e)}

        # CT0-blocked condition
        try:
            ct0_result = collector.generate_with_data_collection(
                question=question,
                condition='ct0_blocked'
            )
            pred_answer_ct0 = extract_answer(ct0_result['generated_text'])

            problem_data['ct0_blocked'] = {
                'pred_answer': pred_answer_ct0,
                'correct': (pred_answer_ct0 == gold_answer),
                'generated_text': ct0_result['generated_text'],
                'has_attention_data': ct0_result['attention_data'] is not None,
                'has_hidden_state_data': ct0_result['hidden_state_data'] is not None,
                'ct_activations_shape': ct0_result['ct_activations'].shape if ct0_result['ct_activations'] is not None else None
            }

            # Save attention data to HDF5
            if ct0_result['attention_data'] is not None:
                h5_file = output_path / f'{model_name}_problem_{problem_idx:04d}_ct0blocked_attention.h5'
                with h5py.File(h5_file, 'w') as f:
                    for step_data in ct0_result['attention_data']:
                        step = step_data['step']
                        if step_data['attentions']:
                            for layer_idx, attn in enumerate(step_data['attentions']):
                                f.create_dataset(
                                    f'step_{step}/layer_{layer_idx}/attention',
                                    data=attn.numpy(),
                                    compression='gzip'
                                )

            # Save hidden states to HDF5
            if ct0_result['hidden_state_data'] is not None:
                h5_file = output_path / f'{model_name}_problem_{problem_idx:04d}_ct0blocked_hidden.h5'
                with h5py.File(h5_file, 'w') as f:
                    for step_data in ct0_result['hidden_state_data']:
                        step = step_data['step']
                        if step_data['hidden_states']:
                            for layer_idx, hidden in enumerate(step_data['hidden_states']):
                                f.create_dataset(
                                    f'step_{step}/layer_{layer_idx}/hidden_state',
                                    data=hidden.numpy(),
                                    compression='gzip'
                                )

            # Save CT activations
            if ct0_result['ct_activations'] is not None:
                np_file = output_path / f'{model_name}_problem_{problem_idx:04d}_ct0blocked_ct_activations.npy'
                np.save(np_file, ct0_result['ct_activations'].numpy())

        except Exception as e:
            print(f"\nFailed on CT0-blocked problem {problem_idx}: {e}")
            problem_data['ct0_blocked'] = {'error': str(e)}

        # Compute impact
        if 'baseline' in problem_data and 'ct0_blocked' in problem_data:
            baseline_correct = problem_data['baseline'].get('correct', False)
            ct0_correct = problem_data['ct0_blocked'].get('correct', False)

            if baseline_correct and not ct0_correct:
                problem_data['impact'] = 'degradation'
            elif not baseline_correct and ct0_correct:
                problem_data['impact'] = 'improvement'
            else:
                problem_data['impact'] = 'no_change'

        results.append(problem_data)

        # Save metadata periodically
        if (problem_idx + 1) % save_frequency == 0:
            metadata_file = output_path / f'{model_name}_metadata_checkpoint_{problem_idx+1}.json'
            with open(metadata_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n  ✓ Saved checkpoint at {problem_idx+1} problems")

    # Final save
    metadata_file = output_path / f'{model_name}_metadata_final.json'
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Data collection complete!")
    print(f"  Total problems: {len(results)}")
    print(f"  Output directory: {output_path}")
    print(f"  Files generated:")
    print(f"    - Attention HDF5: {len(results) * 2} files")
    print(f"    - Hidden state HDF5: {len(results) * 2} files")
    print(f"    - CT activations NPY: {len(results) * 2} files")
    print(f"    - Metadata JSON: 1 file")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Collect attention analysis data')
    parser.add_argument('--model', type=str, default='llama', choices=['llama', 'gpt2'])
    parser.add_argument('--n_problems', type=int, default=100,
                       help='Number of problems to process')
    parser.add_argument('--output_dir', type=str, default='results/attention_data',
                       help='Output directory for data files')
    parser.add_argument('--save_frequency', type=int, default=10,
                       help='Save metadata checkpoint every N problems')
    args = parser.parse_args()

    results = run_data_collection(
        model_name=args.model,
        n_problems=args.n_problems,
        output_dir=args.output_dir,
        save_frequency=args.save_frequency
    )


if __name__ == '__main__':
    main()
