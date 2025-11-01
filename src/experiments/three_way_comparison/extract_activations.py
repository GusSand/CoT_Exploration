#!/usr/bin/env python3
"""
Story 2: Activation Extraction Pipeline
Extract hidden states and attention patterns from all 3 CODI models.
"""
import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from model_loader import CODIModelLoader
from utils.data_loading import load_task_data


class ActivationExtractor:
    """Extract activations from CODI models."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / 'config.json'

        with open(config_path) as f:
            self.config = json.load(f)

        self.loader = CODIModelLoader(config_path)
        self.activations = {}

    def extract_ct_hidden_states_and_generate(self, model, input_text, tokenizer):
        """
        Extract CT hidden states AND generate answer using proper CODI flow.

        UPDATED: Matches original CODI evaluation (codi/test.py:226-267)
        - Maintains KV cache throughout: question → BOT → CT tokens → EOT → answer
        - Extracts CT hidden states during proper generation
        - Uses manual token-by-token generation (not .generate())
        - Uses max_new_tokens=256 (like original)

        Returns:
            tuple: (ct_hidden_states, generated_text)
            - ct_hidden_states: np.array of shape [n_layers, n_ct_tokens, hidden_dim]
            - generated_text: str (generated answer text)
        """
        device = model.codi.device
        num_latent = 6  # Number of CT tokens
        num_layers = 16  # LLaMA-1B layers
        hidden_dim = 2048  # LLaMA-1B hidden dimension

        # Storage for CT hidden states
        ct_hidden_states = np.zeros((num_layers, num_latent, hidden_dim), dtype=np.float32)

        with torch.no_grad():
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # Add BOT token (like original)
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long, device=device).unsqueeze(0)
            input_ids = torch.cat((input_ids, bot_tensor), dim=1)

            # Forward through input + BOT (question encoding)
            outputs = model.codi(
                input_ids=input_ids,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Apply projection
            if model.use_prj:
                latent_embd = model.prj(latent_embd)

            # Generate CT tokens (6 iterations) and capture hidden states
            for ct_idx in range(num_latent):
                # Forward through current CT token
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                # Extract hidden states from all layers for this CT token
                hidden_states = outputs.hidden_states  # Tuple of [batch, seq, hidden] per layer
                for layer_idx in range(num_layers):
                    # Get the last token's hidden state (the CT token we just generated)
                    # Convert BFloat16 to Float32 for numpy storage
                    ct_hidden_states[layer_idx, ct_idx] = hidden_states[layer_idx][0, -1].cpu().float().numpy()

                # Prepare for next iteration
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection
                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Add EOT token (like original)
            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)

            output = eot_emb

            # Generate answer tokens manually (like original, max_new_tokens=256)
            pred_tokens = []
            for i in range(256):
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                # Greedy decoding
                next_token_id = torch.argmax(logits, dim=-1).squeeze(-1)

                # Stop at EOS
                if next_token_id == tokenizer.eos_token_id:
                    break

                pred_tokens.append(next_token_id.item())

                # Get next embedding - get_embd returns [batch, hidden], then unsqueeze(1) adds seq dim
                # Shape: [1, hidden] -> [1, 1, hidden]
                output = model.get_embd(model.codi, model.model_name)(next_token_id.reshape(1)).unsqueeze(1)

            # Decode generated text
            generated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)

        return ct_hidden_states, generated_text

    def extract_ct_tokens(self, hidden_states, ct_positions):
        """
        Extract hidden states for CT token positions.

        Args:
            hidden_states: Dict of layer_idx -> hidden states [batch, seq, hidden]
            ct_positions: List of CT token positions in sequence

        Returns:
            np.array of shape [n_layers, n_ct_tokens, hidden_dim]
        """
        n_layers = len(hidden_states)
        n_ct_tokens = len(ct_positions)
        hidden_dim = hidden_states[0].shape[-1]

        ct_hidden = np.zeros((n_layers, n_ct_tokens, hidden_dim), dtype=np.float32)

        for layer_idx in range(n_layers):
            hidden = hidden_states[layer_idx][0]  # Remove batch dim
            for ct_idx, pos in enumerate(ct_positions):
                if pos < hidden.shape[0]:
                    ct_hidden[layer_idx, ct_idx] = hidden[pos].numpy()

        return ct_hidden

    def extract_activations_for_task(self, task: str, n_examples: int = 100):
        """
        Extract activations for a single task.

        Args:
            task: Task name
            n_examples: Number of examples to process
        """
        print(f"\n{'='*80}")
        print(f"EXTRACTING ACTIVATIONS: {task.upper()}")
        print('='*80)

        # Load model
        model, tokenizer, metadata = self.loader.load_model(task)

        # Load data
        data = load_task_data(task, self.config, n_examples, self.config['random_seed'])

        # Extract activations
        results = []

        for idx, example in enumerate(tqdm(data, desc=f"Processing {task}")):
            try:
                # Format input
                input_text = self.loader.format_input(task, example)

                # Extract CT hidden states AND generate answer using proper CODI flow
                # This maintains KV cache throughout: question → CT tokens → answer
                ct_hidden, output_text = self.extract_ct_hidden_states_and_generate(
                    model, input_text, tokenizer
                )

                # Extract predicted answer from generated text
                predicted = self.loader.extract_answer(task, output_text, example)

                # Get ground truth
                if task == 'personal_relations':
                    ground_truth = example['answer']
                elif task == 'gsm8k':
                    # Extract number from answer
                    import re
                    answer_str = example['answer']
                    numbers = re.findall(r'####\s*(-?\d+)', answer_str)
                    ground_truth = numbers[0] if numbers else "UNKNOWN"
                elif task == 'commonsense':
                    # Support both old and new formats
                    ground_truth = example.get('answer', example.get('answerKey', 'UNKNOWN'))

                correct = (str(predicted).strip().upper() == str(ground_truth).strip().upper())

                # Store result
                result = {
                    'example_id': idx,
                    'question': input_text[:200],  # Truncate for storage
                    'answer': ground_truth,
                    'predicted': predicted,
                    'correct': correct,
                    'hidden_states': ct_hidden,  # [16 layers, 6 tokens, 2048 dims]
                }

                results.append(result)

            except Exception as e:
                print(f"\n  Error processing example {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Checkpoint every 25 examples
            if (idx + 1) % self.config['checkpoint_interval'] == 0:
                self._save_checkpoint(task, results, idx + 1)

        # Save final results
        output_path = Path(self.config['output_dir']) / f'activations_{task}.npz'
        self._save_results(task, results, output_path, metadata)

        # Unload model
        self.loader.unload_model()

        return results

    def _save_checkpoint(self, task: str, results: List[Dict], n_processed: int):
        """Save intermediate checkpoint."""
        checkpoint_path = Path(self.config['output_dir']) / f'activations_{task}_checkpoint_{n_processed}.npz'
        print(f"\n  Saving checkpoint: {n_processed} examples processed")

        np.savez_compressed(
            checkpoint_path,
            n_examples=n_processed,
            results=results
        )

    def _save_results(self, task: str, results: List[Dict], output_path: Path, metadata: Dict):
        """Save final results to NPZ format."""
        print(f"\nSaving results to {output_path}...")

        # Prepare arrays
        n_examples = len(results)
        hidden_states_array = np.array([r['hidden_states'] for r in results])  # [N, 16, 6, 2048]

        # Calculate statistics
        n_correct = sum(r['correct'] for r in results)
        accuracy = n_correct / n_examples if n_examples > 0 else 0.0

        print(f"  Examples: {n_examples}")
        print(f"  Accuracy: {n_correct}/{n_examples} = {accuracy:.1%}")
        print(f"  Hidden states shape: {hidden_states_array.shape}")

        # Save (avoid keyword conflict with metadata)
        save_dict = {
            # Metadata
            'task_name': task,
            'n_examples': n_examples,
            'n_correct': n_correct,
            'accuracy': accuracy,
            # Data
            'hidden_states': hidden_states_array,
            'example_ids': np.array([r['example_id'] for r in results]),
            'correct': np.array([r['correct'] for r in results]),
            # Store questions/answers as JSON string (NPZ doesn't handle strings well)
            'examples_json': json.dumps([{
                'question': r['question'],
                'answer': r['answer'],
                'predicted': r['predicted']
            } for r in results])
        }

        # Add metadata without conflicts
        for key, value in metadata.items():
            if key not in save_dict:
                save_dict[f'meta_{key}'] = value

        np.savez_compressed(output_path, **save_dict)

        print(f"  ✓ Saved to {output_path}")

    def extract_all_tasks(self, n_examples: int = 100):
        """Extract activations for all three tasks."""
        print("\n" + "="*80)
        print("THREE-WAY CODI ACTIVATION EXTRACTION")
        print("="*80)
        print(f"Extracting {n_examples} examples per task")
        print(f"Random seed: {self.config['random_seed']}")
        print(f"Device: {self.config['device']}")
        print("="*80)

        all_results = {}

        for task in self.config['tasks']:
            results = self.extract_activations_for_task(task, n_examples)
            all_results[task] = results

        print("\n" + "="*80)
        print("EXTRACTION COMPLETE!")
        print("="*80)

        for task, results in all_results.items():
            n_correct = sum(r['correct'] for r in results)
            accuracy = n_correct / len(results) if results else 0.0
            print(f"{task:20s}: {n_correct:3d}/{len(results):3d} = {accuracy:5.1%}")

        print("="*80)

        return all_results


def main():
    """Main extraction script."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract CODI activations")
    parser.add_argument('--n_examples', type=int, default=100, help='Examples per task')
    parser.add_argument('--task', type=str, default='all', help='Specific task or "all"')
    args = parser.parse_args()

    extractor = ActivationExtractor()

    if args.task == 'all':
        extractor.extract_all_tasks(args.n_examples)
    else:
        extractor.extract_activations_for_task(args.task, args.n_examples)


if __name__ == '__main__':
    main()
