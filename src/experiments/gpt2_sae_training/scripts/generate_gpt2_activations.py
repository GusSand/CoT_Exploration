"""
Generate GPT-2 continuous thought activations for TopK SAE training.

Extracts activations from 1,000 stratified GSM8K problems:
- 12 layers × 6 positions = 72 samples per problem
- 80/20 train/test split
- Output: ~57,600 train samples, ~14,400 test samples

Runtime: ~30-45 minutes
"""

import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType


class GPT2ThoughtExtractor:
    """Extracts continuous thought activations from GPT-2 CODI model."""

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
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu', weights_only=False)
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


def load_stratified_1000_problems(dataset_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load 1000 stratified GSM8K problems and split into train/test.

    Args:
        dataset_path: Path to llama_cot_original_stratified_1000.json

    Returns:
        Tuple of (train_problems, test_problems)
    """
    with open(dataset_path, 'r') as f:
        all_problems = json.load(f)

    print(f"Loaded {len(all_problems)} problems from stratified dataset")

    # Split into train (80%) and test (20%)
    train_set, test_set = train_test_split(
        all_problems,
        train_size=0.8,
        random_state=42,
        shuffle=True
    )

    print(f"Split: {len(train_set)} train, {len(test_set)} test")

    return train_set, test_set


def extract_activations_batch(problems: List[Dict[str, Any]],
                              extractor: ContinuousThoughtExtractor,
                              num_ct_tokens: int = 6,
                              num_layers: int = 12) -> Dict[str, Any]:
    """Extract continuous thought activations for GPT-2.

    Args:
        problems: List of GSM8K problem dictionaries
        extractor: ContinuousThoughtExtractor instance
        num_ct_tokens: Number of continuous thought tokens (6 for CODI)
        num_layers: Number of layers (12 for GPT-2)

    Returns:
        Dictionary containing activations and metadata
    """
    all_activations = []
    all_metadata = {
        'problem_ids': [],
        'layers': [],
        'positions': [],
    }

    print(f"\nExtracting activations for {len(problems)} problems...")
    print(f"  Layers: 0-{num_layers-1}")
    print(f"  Continuous thought tokens: {num_ct_tokens}")
    print(f"  Expected samples: {len(problems)} × {num_layers} × {num_ct_tokens} = {len(problems) * num_layers * num_ct_tokens:,}")

    skipped = 0

    for problem_idx, problem in enumerate(tqdm(problems, desc="Processing")):
        try:
            question = problem['question']
            problem_id = problem.get('gsm8k_id', f'problem_{problem_idx}')

            with torch.no_grad():
                # Tokenize question
                inputs = extractor.tokenizer(question, return_tensors="pt").to(extractor.device)
                input_ids = inputs["input_ids"]

                # Get input embeddings
                input_embd = extractor.model.get_embd(extractor.model.codi, extractor.model.model_name)(input_ids)

                # Forward through model to get context
                outputs = extractor.model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True
                )

                past_key_values = outputs.past_key_values

                # Get BOT (Beginning of Thought) embedding
                bot_emb = extractor.model.get_embd(extractor.model.codi, extractor.model.model_name)(
                    torch.tensor([extractor.model.bot_id], dtype=torch.long, device=extractor.device)
                ).unsqueeze(0)

                # Process continuous thought tokens
                latent_embd = bot_emb

                for position in range(num_ct_tokens):
                    # Forward through model for this continuous thought token
                    outputs = extractor.model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )

                    past_key_values = outputs.past_key_values

                    # Extract hidden states from all layers
                    # outputs.hidden_states is tuple of (num_layers+1) tensors
                    # Index 0 is embedding layer, 1-12 are transformer layers for GPT-2
                    for layer in range(num_layers):
                        # Get hidden state from this layer (last token)
                        hidden_state = outputs.hidden_states[layer + 1][:, -1, :]

                        # Store activation (768-dimensional vector for GPT-2)
                        all_activations.append(hidden_state.cpu().squeeze(0))

                        # Store metadata
                        all_metadata['problem_ids'].append(problem_id)
                        all_metadata['layers'].append(layer)
                        all_metadata['positions'].append(position)

                    # Update latent embedding for next position
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if hasattr(extractor.model, 'prj') and extractor.model.use_prj:
                        latent_embd = extractor.model.prj(latent_embd)

            # Clear cache every 10 problems to prevent memory buildup
            if (problem_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n⚠️  Failed on problem {problem_idx} ({problem.get('gsm8k_id', 'unknown')}): {e}")
            skipped += 1
            continue

    # Convert to tensors
    activations_tensor = torch.stack(all_activations)

    print(f"\n✓ Extraction complete!")
    print(f"  Total samples: {len(activations_tensor):,}")
    print(f"  Shape: {activations_tensor.shape}")
    print(f"  Skipped: {skipped} problems")

    return {
        'activations': activations_tensor,
        'metadata': all_metadata,
        'config': {
            'model': 'gpt2',
            'num_problems': len(problems),
            'num_layers': num_layers,
            'num_ct_tokens': num_ct_tokens,
            'hidden_size': 768
        }
    }


def main():
    print("="*80)
    print("GPT-2 ACTIVATION GENERATION FOR TOPK SAE TRAINING")
    print("="*80)

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    dataset_path = "/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json"
    codi_checkpoint = "/home/paperspace/codi_ckpt/gpt2_gsm8k/"
    output_dir = BASE_DIR / "data"
    output_dir.mkdir(exist_ok=True)

    # Load stratified 1000-problem dataset
    print("\n[1/4] Loading stratified 1000-problem dataset...")
    train_problems, test_problems = load_stratified_1000_problems(dataset_path)

    # Initialize extractor
    print("\n[2/4] Loading GPT-2 CODI model...")
    print(f"  Checkpoint: {codi_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    extractor = ContinuousThoughtExtractor(
        model_path=codi_checkpoint,
        device=device
    )

    print("  ✓ GPT-2 model loaded")

    # Extract training set activations
    print("\n[3/4] Extracting TRAINING SET activations...")
    print(f"  Processing {len(train_problems)} problems...")
    print(f"  Estimated time: ~{len(train_problems) * 2 / 60:.1f} minutes")

    train_data = extract_activations_batch(
        train_problems,
        extractor,
        num_ct_tokens=6,
        num_layers=12  # GPT-2 has 12 layers
    )

    # Save training data
    train_path = output_dir / "gpt2_full_train_activations.pt"
    print(f"\n  Saving training data to: {train_path}")
    torch.save(train_data, train_path)
    print(f"  ✓ Saved ({train_path.stat().st_size / 1e9:.2f} GB)")

    # Extract test set activations
    print("\n[4/4] Extracting TEST SET activations...")
    print(f"  Processing {len(test_problems)} problems...")
    print(f"  Estimated time: ~{len(test_problems) * 2 / 60:.1f} minutes")

    test_data = extract_activations_batch(
        test_problems,
        extractor,
        num_ct_tokens=6,
        num_layers=12
    )

    # Save test data
    test_path = output_dir / "gpt2_full_val_activations.pt"
    print(f"\n  Saving test data to: {test_path}")
    torch.save(test_data, test_path)
    print(f"  ✓ Saved ({test_path.stat().st_size / 1e9:.2f} GB)")

    # Summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"  Train: {train_data['activations'].shape}")
    print(f"  Test: {test_data['activations'].shape}")
    print(f"  Output directory: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
