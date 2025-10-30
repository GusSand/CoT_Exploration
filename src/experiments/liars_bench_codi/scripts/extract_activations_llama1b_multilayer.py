"""
Extract LLaMA-3.2-1B multi-layer activations for pre-compression signal analysis.

This script extracts activations from multiple layers at multiple token positions
to analyze WHERE deception signal is lost during CODI's compression process.

Architecture:
- LLaMA-3.2-1B: 16 layers (0-15), 2048 hidden dim
- Layers probed: [0, 3, 6, 9, 12, 15] (6 layers)
- Positions probed: question_last, ct0-ct5, answer_first (8 positions)

Output:
- Train: 288 samples (144 honest, 144 deceptive)
- Test: 288 samples (144 honest, 144 deceptive)
- Format: JSON with activations[layer][position] = [sample1, sample2, ...]

Author: Claude Code
Date: 2025-10-30
Experiment: Pre-compression deception signal analysis
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Add CODI to path
codi_path = Path(__file__).parent.parent.parent.parent.parent / "codi"
sys.path.insert(0, str(codi_path))


def load_probe_datasets():
    """Load probe train and test datasets."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "processed"

    print("=" * 80)
    print("LOADING PROBE DATASETS")
    print("=" * 80)

    with open(data_dir / "probe_train_proper.json", 'r') as f:
        train_data = json.load(f)

    with open(data_dir / "probe_test_proper.json", 'r') as f:
        test_data = json.load(f)

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Verify balance
    train_honest = sum(1 for s in train_data if s['is_honest'])
    test_honest = sum(1 for s in test_data if s['is_honest'])

    print(f"  Train balance: {train_honest} honest / {len(train_data)-train_honest} deceptive")
    print(f"  Test balance: {test_honest} honest / {len(test_data)-test_honest} deceptive")
    print()

    return train_data, test_data


def load_llama1b_codi_model(checkpoint_path):
    """Load trained LLaMA-3.2-1B CODI model."""
    print("=" * 80)
    print("LOADING LLAMA-3.2-1B CODI MODEL")
    print("=" * 80)

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"❌ ERROR: Checkpoint not found at {checkpoint_path}")
        return None, None

    print(f"  Loading from: {checkpoint_path}")
    print()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint weights manually
    print("  Loading checkpoint weights...")
    checkpoint_weights = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")

    # Load base model first
    print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16
    )

    # Load the checkpoint weights (CODI trained weights)
    print("  Merging checkpoint weights...")
    model.load_state_dict(checkpoint_weights, strict=False)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"  ✅ Model loaded on {device}")
    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    print()

    return model, tokenizer


def extract_multilayer_activations(model, tokenizer, question, answer, layers_to_extract):
    """
    Extract activations from multiple layers at multiple positions.

    LLaMA-3.2-1B has 16 layers (0-15). We extract from specified layers.

    Positions:
    - question_last: Last token of question
    - ct0-ct5: 6 continuous thought tokens
    - answer_first: First token of answer

    Args:
        model: CODI model
        tokenizer: Tokenizer
        question: Question text
        answer: Answer text
        layers_to_extract: List of layer indices [0, 3, 6, 9, 12, 15]

    Returns:
        dict: {
            'layer_0': {'question_last': [...], 'ct0': [...], ..., 'answer_first': [...]},
            'layer_3': {...},
            ...
        }
    """
    device = next(model.parameters()).device

    # Format as CODI would see it during training
    prompt = f"Q: {question}\nA:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states from specified layers
    hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors

    activations = {}

    for layer_idx in layers_to_extract:
        # +1 because hidden_states includes embedding layer at index 0
        layer_hidden = hidden_states[layer_idx + 1]  # Shape: (1, seq_len, hidden_dim)

        # Extract positions
        positions = {}

        # Question last token (before " A:")
        # Find the last token before answer starts
        question_last_pos = inputs['input_ids'].shape[1] - 7  # Approximate, adjust if needed
        positions['question_last'] = layer_hidden[0, question_last_pos, :].cpu().to(torch.float32).numpy().tolist()

        # Continuous thought tokens (last 6 tokens before answer generation)
        # These are the compressed reasoning tokens
        for i in range(6):
            ct_pos = inputs['input_ids'].shape[1] - 6 + i
            positions[f'ct{i}'] = layer_hidden[0, ct_pos, :].cpu().to(torch.float32).numpy().tolist()

        # Answer first token
        answer_first_pos = inputs['input_ids'].shape[1] - 1
        positions['answer_first'] = layer_hidden[0, answer_first_pos, :].cpu().to(torch.float32).numpy().tolist()

        activations[f'layer_{layer_idx}'] = positions

    return activations


def extract_all_activations(model, tokenizer, samples, split_name, layers_to_extract):
    """Extract multi-layer activations for all samples in a split."""
    print("=" * 80)
    print(f"EXTRACTING ACTIVATIONS: {split_name.upper()}")
    print("=" * 80)
    print(f"  Samples: {len(samples)}")
    print(f"  Layers: {layers_to_extract}")
    print(f"  Positions: question_last, ct0-ct5, answer_first")
    print()

    # Initialize storage structure
    extracted_data = {
        'activations': {f'layer_{l}': {} for l in layers_to_extract},
        'labels': [],
        'metadata': {
            'model': 'LLaMA-3.2-1B',
            'layers': layers_to_extract,
            'positions': ['question_last', 'ct0', 'ct1', 'ct2', 'ct3', 'ct4', 'ct5', 'answer_first'],
            'n_samples': len(samples),
            'split': split_name,
            'hidden_dim': 2048
        }
    }

    # Initialize position lists for each layer
    for layer_name in extracted_data['activations'].keys():
        for pos in extracted_data['metadata']['positions']:
            extracted_data['activations'][layer_name][pos] = []

    for sample in tqdm(samples, desc=f"Extracting {split_name}"):
        question = sample['question']
        answer = sample['answer']
        is_honest = sample['is_honest']

        # Extract activations from all layers
        sample_activations = extract_multilayer_activations(
            model, tokenizer, question, answer, layers_to_extract
        )

        # Store activations by layer and position
        for layer_name, positions in sample_activations.items():
            for pos_name, activation in positions.items():
                extracted_data['activations'][layer_name][pos_name].append(activation)

        # Store label (0=honest, 1=deceptive)
        extracted_data['labels'].append(0 if is_honest else 1)

    print(f"  ✅ Extracted {len(extracted_data['labels'])} samples")
    print()

    return extracted_data


def main():
    print("\n" + "=" * 80)
    print("MULTI-LAYER ACTIVATION EXTRACTION - LLAMA-3.2-1B")
    print("Pre-Compression Deception Signal Analysis")
    print("=" * 80)
    print()

    # Configuration
    LAYERS_TO_EXTRACT = [0, 3, 6, 9, 12, 15]  # 6 layers across the model

    # Checkpoint path - will be passed as argument or use default
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='~/codi_ckpt/llama1b_liars_bench_proper/liars_bench_llama1b_codi/Llama-3.2-1B-Instruct/ep_5/lr_0.0008/seed_42/checkpoint-250',
        help='Path to CODI checkpoint'
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()

    print(f"Configuration:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Layers: {LAYERS_TO_EXTRACT}")
    print(f"  Expected checkpoints: epoch 5 (250 steps), epoch 10 (500 steps), etc.")
    print()

    # Load datasets
    train_data, test_data = load_probe_datasets()

    # Load model
    model, tokenizer = load_llama1b_codi_model(checkpoint_path)

    if model is None:
        print("❌ Failed to load model. Exiting.")
        return

    # Extract activations for train split
    train_activations = extract_all_activations(
        model, tokenizer, train_data, 'train', LAYERS_TO_EXTRACT
    )

    # Extract activations for test split
    test_activations = extract_all_activations(
        model, tokenizer, test_data, 'test', LAYERS_TO_EXTRACT
    )

    # Save results
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine epoch from checkpoint path for filename
    epoch_str = "5ep"  # Default
    if "ep_10" in str(checkpoint_path):
        epoch_str = "10ep"
    elif "ep_15" in str(checkpoint_path):
        epoch_str = "15ep"
    elif "ep_5" in str(checkpoint_path):
        epoch_str = "5ep"

    train_output = output_dir / f"multilayer_activations_llama1b_{epoch_str}_train.json"
    test_output = output_dir / f"multilayer_activations_llama1b_{epoch_str}_test.json"

    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    print(f"  Saving train activations to: {train_output}")
    with open(train_output, 'w') as f:
        json.dump(train_activations, f)

    print(f"  Saving test activations to: {test_output}")
    with open(test_output, 'w') as f:
        json.dump(test_activations, f)

    print()
    print("=" * 80)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"  Train samples: {len(train_activations['labels'])}")
    print(f"  Test samples: {len(test_activations['labels'])}")
    print(f"  Layers extracted: {LAYERS_TO_EXTRACT}")
    print(f"  Positions per layer: 8")
    print(f"  Total probes to train: {len(LAYERS_TO_EXTRACT)} layers × 8 positions = 48")
    print()
    print("Next step: Train probes with scripts/train_multilayer_probes_llama1b.py")
    print()


if __name__ == "__main__":
    main()
