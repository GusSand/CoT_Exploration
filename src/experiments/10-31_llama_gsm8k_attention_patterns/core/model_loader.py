"""
Model loading utilities for CODI model with attention extraction
"""

import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
from peft import LoraConfig

def load_codi_model(checkpoint_path, model_name, num_latent=5, device="cuda"):
    """
    Load CODI model from checkpoint

    Args:
        checkpoint_path: Path to pytorch_model.bin
        model_name: Base model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        num_latent: Number of continuous thought tokens
        device: Device to load model on

    Returns:
        model: Loaded CODI model
        tokenizer: Tokenizer for the model
    """
    # Add codi to path
    codi_path = Path(__file__).parent.parent.parent.parent.parent / "codi" / "src"
    sys.path.insert(0, str(codi_path))

    from model import CODI, ModelArguments, TrainingArguments

    print(f"\nLoading CODI model...")
    print(f"  Base model: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Num latent tokens: {num_latent}")

    # Create model arguments
    model_args = ModelArguments(
        model_name_or_path=model_name,
        train=False,  # Inference mode
        full_precision=True,
        lora_r=128,
        lora_dropout=0.05
    )

    # Create training arguments (num_latent goes here!)
    training_args = TrainingArguments(
        output_dir="/tmp/codi_dummy",
        num_latent=num_latent,
        use_lora=True,
        bf16=True,
        per_device_eval_batch_size=1
    )

    # Create LoRA config
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Initialize model
    model = CODI(model_args, training_args, lora_config)
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)

    # Move to device
    model = model.to(device)

    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Get tokenizer
    tokenizer = model.tokenizer

    return model, tokenizer
