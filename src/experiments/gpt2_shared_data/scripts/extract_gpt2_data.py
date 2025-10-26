"""
Phase 1: Extract Shared GPT-2 Data

Extracts 1000 GPT-2 CODI predictions with:
- Questions & ground truth answers
- Model predictions & correctness
- Continuous thoughts (all 12 layers × 6 tokens × 768 dims)

Output: gpt2_predictions_1000.json (shared by all 4 experiments)
"""

import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from datasets import load_dataset

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType


class GPT2DataExtractor:
    """Extracts predictions and continuous thoughts from GPT-2 CODI model."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Initialize GPT-2 CODI model."""
        self.device = device
        print(f"Loading GPT-2 CODI model from {model_path}...")

        # Parse arguments for CODI model
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

        # Modify for inference
        model_args.train = False
        training_args.greedy = True

        # Create LoRA config for GPT-2
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=['c_attn', 'c_proj', 'c_fc'],
            init_lora_weights=True,
        )

        # Load model
        self.model = CODI(model_args, training_args, lora_config)

        # Load checkpoint weights
        import os
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        self.model.load_state_dict(state_dict, strict=False)
        self.model.codi.tie_weights()

        # Convert to float32
        self.model.float()
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.tokenizer
        self.num_latent = training_args.num_latent

        print("Model loaded successfully!")
        print(f"  Architecture: GPT-2 (124M params)")
        print(f"  Layers: 12")
        print(f"  Hidden dim: 768")
        print(f"  Latent tokens: {self.num_latent}")

    def extract_answer(self, text: str) -> int:
        """Extract numerical answer from CoT text."""
        try:
            # Look for #### marker (GSM8k format)
            if '####' in text:
                answer_str = text.split('####')[-1].strip()
            else:
                # Try to find last number
                import re
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
                if numbers:
                    answer_str = numbers[-1]
                else:
                    return None

            # Remove commas and convert
            answer_str = answer_str.replace(',', '')
            return int(float(answer_str))
        except:
            return None

    def predict_with_thoughts(self, question: str) -> Dict:
        """
        Generate prediction and extract continuous thoughts from all layers.

        Returns:
            Dict with 'prediction', 'prediction_text', and 'thoughts'
            thoughts: Dict[str, List[List[float]]] mapping layer names to token activations
        """
        with torch.no_grad():
            # Tokenize input
            inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
            input_ids = inputs["input_ids"]

            # Get initial embeddings
            input_embd = self.model.get_embd(self.model.codi, self.model.model_name)(input_ids).to(self.device)

            # Forward through model
            outputs = self.model.codi(
                inputs_embeds=input_embd,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            # Get BOT embedding
            bot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.bot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Storage for thoughts from all layers
            thoughts = {f'layer_{i}': [] for i in range(12)}  # GPT-2 has 12 layers

            # Process latent thoughts
            latent_embd = bot_emb

            for latent_step in range(self.num_latent):
                outputs = self.model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )

                past_key_values = outputs.past_key_values

                # Extract thoughts from all 12 layers
                for layer_idx in range(12):
                    thought = outputs.hidden_states[layer_idx][:, -1, :].cpu()
                    thoughts[f'layer_{layer_idx}'].append(thought.squeeze(0).tolist())

                # Update latent embedding for next iteration
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Apply projection
                if self.model.use_prj:
                    latent_embd = self.model.prj(latent_embd)

            # Get EOT embedding
            eot_emb = self.model.get_embd(self.model.codi, self.model.model_name)(
                torch.tensor([self.model.eot_id], dtype=torch.long, device=self.device)
            ).unsqueeze(0)

            # Final forward pass after EOT
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
                    # First token after EOT
                    next_token_logits = outputs.logits[:, -1, :]
                else:
                    # Subsequent tokens
                    last_token_id = torch.tensor([[generated_ids[-1]]], device=self.device)
                    last_token_embd = self.model.get_embd(self.model.codi, self.model.model_name)(last_token_id)

                    outputs = self.model.codi(
                        inputs_embeds=last_token_embd,
                        use_cache=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]

                # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()

                # Check for EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token_id)

            # Decode prediction
            prediction_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            prediction = self.extract_answer(prediction_text)

            return {
                'prediction': prediction,
                'prediction_text': prediction_text,
                'thoughts': thoughts
            }

    def extract_dataset(self, n_samples: int = 1000, output_path: str = None):
        """
        Extract predictions for n_samples from GSM8k validation set.

        Args:
            n_samples: Number of samples to process
            output_path: Path to save results
        """
        print(f"\nLoading GSM8k validation set...")
        dataset = load_dataset("gsm8k", "main", split="test")

        if n_samples > len(dataset):
            n_samples = len(dataset)
            print(f"  Requested {n_samples} samples, but only {len(dataset)} available")

        print(f"  Processing {n_samples} samples...")

        results = []
        correct_count = 0

        for i in range(n_samples):
            sample = dataset[i]
            question = sample['question']
            ground_truth_text = sample['answer']
            ground_truth = self.extract_answer(ground_truth_text)

            print(f"\n[{i+1}/{n_samples}] Processing...")
            print(f"  Question: {question[:100]}...")

            # Get prediction and thoughts
            result = self.predict_with_thoughts(question)

            # Check correctness
            is_correct = (result['prediction'] == ground_truth)
            if is_correct:
                correct_count += 1

            # Store result
            results.append({
                'id': i,
                'question': question,
                'ground_truth': ground_truth,
                'ground_truth_text': ground_truth_text,
                'prediction': result['prediction'],
                'prediction_text': result['prediction_text'],
                'is_correct': is_correct,
                'thoughts': result['thoughts']
            })

            # Progress update
            accuracy = (correct_count / (i + 1)) * 100
            print(f"  Prediction: {result['prediction']}")
            print(f"  Ground truth: {ground_truth}")
            print(f"  Correct: {is_correct}")
            print(f"  Running accuracy: {accuracy:.2f}% ({correct_count}/{i+1})")

            # Save checkpoint every 100 samples
            if (i + 1) % 100 == 0:
                checkpoint_path = output_path.replace('.json', f'_checkpoint_{i+1}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump({
                        'metadata': self._create_metadata(i + 1, correct_count),
                        'samples': results
                    }, f, indent=2)
                print(f"\n  Checkpoint saved: {checkpoint_path}")

        # Save final results
        final_data = {
            'metadata': self._create_metadata(n_samples, correct_count),
            'samples': results
        }

        with open(output_path, 'w') as f:
            json.dump(final_data, f, indent=2)

        print("\n" + "="*60)
        print("EXTRACTION COMPLETE")
        print("="*60)
        print(f"Total samples: {n_samples}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {(correct_count/n_samples)*100:.2f}%")
        print(f"Output: {output_path}")
        print("="*60)

        return final_data

    def _create_metadata(self, n_samples, correct_count):
        """Create metadata dict."""
        return {
            'model': 'GPT-2 CODI',
            'checkpoint': 'gpt2_gsm8k',
            'n_samples': n_samples,
            'n_correct': correct_count,
            'n_incorrect': n_samples - correct_count,
            'accuracy': (correct_count / n_samples) * 100,
            'layers': list(range(12)),
            'n_layers': 12,
            'n_tokens': 6,
            'hidden_dim': 768,
            'extraction_date': datetime.now().isoformat(),
            'dataset': 'GSM8k test set'
        }


def main():
    """Extract 1000 GPT-2 predictions with continuous thoughts."""
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent.parent
    model_path = Path.home() / "codi_ckpt" / "gpt2_gsm8k"
    output_path = project_root / "src/experiments/gpt2_shared_data/gpt2_predictions_1000.json"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    extractor = GPT2DataExtractor(str(model_path))

    # Extract data
    data = extractor.extract_dataset(n_samples=1000, output_path=str(output_path))

    print(f"\n✅ Success! Shared GPT-2 data ready at:")
    print(f"   {output_path}")
    print(f"\nAll 4 experiments can now use this shared data.")

    return data


if __name__ == "__main__":
    main()
