"""
Generate activations for FULL GSM8K dataset (7,473 training problems).

This script extracts continuous thought activations from CODI LLaMA model
for all 7,473 GSM8K training problems to improve SAE training diversity.

Expected output:
- ~600K training samples (7,473 problems × 16 layers × 6 positions × 0.8 split)
- ~150K test samples (0.2 split)
- Files: ~4.5GB train, ~1.1GB test

Runtime: ~3-4 hours for full dataset
"""

import sys
import json
import torch
import re
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "operation_circuits"))

from extract_continuous_thoughts import ContinuousThoughtExtractor


def load_full_gsm8k(gsm8k_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load full GSM8K dataset and split into train/test.

    Args:
        gsm8k_path: Path to gsm8k_full.json

    Returns:
        Tuple of (train_problems, test_problems_for_validation)
    """
    with open(gsm8k_path, 'r') as f:
        all_data = json.load(f)

    # Filter by split
    train_problems = [p for p in all_data if p['split'] == 'train']

    print(f"Loaded {len(train_problems)} training problems from GSM8K")

    # Split training set into train (80%) and validation (20%) for SAE training
    train_set, val_set = train_test_split(
        train_problems,
        train_size=0.8,
        random_state=42,
        shuffle=True
    )

    print(f"Split: {len(train_set)} train, {len(val_set)} validation")

    return train_set, val_set


def extract_cot_from_answer(answer: str) -> List[str]:
    """Extract chain-of-thought calculation steps from GSM8K answer.

    GSM8K format:
    "He makes $60,000/2=$<<60000/2=30000>>30,000 a year..."

    Extracts: ["60000/2=30000", ...]

    Args:
        answer: GSM8K answer string

    Returns:
        List of calculation strings
    """
    calculations = re.findall(r'<<([^>]+)>>', answer)
    return calculations


def extract_activations_batch(problems: List[Dict[str, Any]],
                              extractor: ContinuousThoughtExtractor,
                              num_ct_tokens: int = 6,
                              num_layers: int = 16,
                              batch_size: int = 1) -> Dict[str, Any]:
    """Extract continuous thought activations for a batch of problems.

    Args:
        problems: List of GSM8K problem dictionaries
        extractor: ContinuousThoughtExtractor instance
        num_ct_tokens: Number of continuous thought tokens (6 for CODI)
        num_layers: Number of layers (16 for LLaMA-3.2-1B)
        batch_size: Batch size (1 recommended to avoid OOM)

    Returns:
        Dictionary containing activations, metadata, and CoT sequences
    """
    all_activations = []
    all_metadata = {
        'problem_ids': [],
        'layers': [],
        'positions': [],
        'cot_sequences': []
    }

    print(f"\nExtracting activations for {len(problems)} problems...")
    print(f"  Layers: 0-{num_layers-1}")
    print(f"  Continuous thought tokens: {num_ct_tokens}")
    print(f"  Expected samples: {len(problems)} × {num_layers} × {num_ct_tokens} = {len(problems) * num_layers * num_ct_tokens:,}")

    skipped = 0

    for problem_idx, problem in enumerate(tqdm(problems, desc="Processing")):
        try:
            question = problem['question']
            answer = problem['answer']
            problem_id = problem.get('id', f'problem_{problem_idx}')

            # Extract CoT sequence
            cot_sequence = extract_cot_from_answer(answer)

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
                    # Index 0 is embedding layer, 1-16 are transformer layers
                    for layer in range(num_layers):
                        # Get hidden state from this layer (last token)
                        hidden_state = outputs.hidden_states[layer + 1][:, -1, :]

                        # Store activation (2048-dimensional vector)
                        all_activations.append(hidden_state.cpu().squeeze(0))

                        # Store metadata
                        all_metadata['problem_ids'].append(problem_id)
                        all_metadata['layers'].append(layer)
                        all_metadata['positions'].append(position)
                        all_metadata['cot_sequences'].append(cot_sequence)

                    # Update latent embedding for next position
                    # Apply MLP projection if model uses it
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if hasattr(extractor.model, 'prj') and extractor.model.use_prj:
                        latent_embd = extractor.model.prj(latent_embd)

            # Clear cache every 10 problems to prevent memory buildup
            if (problem_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n⚠️  Failed on problem {problem_idx} ({problem.get('id', 'unknown')}): {e}")
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
            'num_problems': len(problems),
            'num_layers': num_layers,
            'num_ct_tokens': num_ct_tokens,
            'hidden_size': 2048
        }
    }


def main():
    print("="*80)
    print("FULL GSM8K DATASET ACTIVATION GENERATION")
    print("="*80)

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    gsm8k_path = "/home/paperspace/dev/CoT_Exploration/src/experiments/operation_circuits/gsm8k_full.json"
    codi_checkpoint = "/home/paperspace/codi_ckpt/llama_gsm8k/"
    output_dir = BASE_DIR / "data"
    output_dir.mkdir(exist_ok=True)

    # Load full GSM8K dataset
    print("\n[1/4] Loading full GSM8K dataset...")
    train_problems, val_problems = load_full_gsm8k(gsm8k_path)

    # Initialize extractor
    print("\n[2/4] Loading CODI model...")
    print(f"  Checkpoint: {codi_checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    extractor = ContinuousThoughtExtractor(
        model_path=codi_checkpoint,
        device=device
    )

    print("  ✓ Model loaded")

    # Extract training set activations
    print("\n[3/4] Extracting TRAINING SET activations...")
    print(f"  Processing {len(train_problems)} problems...")
    print(f"  Estimated time: ~{len(train_problems) * 2 / 60:.1f} minutes")

    train_data = extract_activations_batch(
        train_problems,
        extractor,
        num_ct_tokens=6,
        num_layers=16
    )

    # Save training data
    train_path = output_dir / "full_train_activations.pt"
    torch.save(train_data, train_path)
    size_mb = train_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✓ Saved: {train_path}")
    print(f"    Size: {size_mb:.1f} MB")

    # Extract validation set activations
    print("\n[4/4] Extracting VALIDATION SET activations...")
    print(f"  Processing {len(val_problems)} problems...")

    val_data = extract_activations_batch(
        val_problems,
        extractor,
        num_ct_tokens=6,
        num_layers=16
    )

    # Save validation data
    val_path = output_dir / "full_val_activations.pt"
    torch.save(val_data, val_path)
    size_mb = val_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✓ Saved: {val_path}")
    print(f"    Size: {size_mb:.1f} MB")

    # Summary
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)

    print(f"\nTraining set:")
    print(f"  Problems: {len(train_problems)}")
    print(f"  Samples: {len(train_data['activations']):,}")
    print(f"  File: {train_path}")

    print(f"\nValidation set:")
    print(f"  Problems: {len(val_problems)}")
    print(f"  Samples: {len(val_data['activations']):,}")
    print(f"  File: {val_path}")

    print(f"\nNext step: Train SAEs using this full dataset!")
    print(f"  python train_saes_full_data.py")
    print("="*80)


if __name__ == '__main__':
    main()
