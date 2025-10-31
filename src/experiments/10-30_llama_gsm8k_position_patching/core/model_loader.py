"""
Model loader for CODI LLaMA-1B with activation extraction capabilities
"""

import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add codi to path
codi_path = Path(__file__).parent.parent.parent.parent.parent / "codi" / "src"
sys.path.insert(0, str(codi_path))

from model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig


def load_codi_model(checkpoint_path, model_name, num_latent=5, device="cuda"):
    """
    Load CODI model with checkpoint

    Args:
        checkpoint_path: Path to pytorch_model.bin checkpoint
        model_name: Base model name (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        num_latent: Number of continuous thought tokens
        device: Device to load model on

    Returns:
        model: Loaded CODI model
        tokenizer: Tokenizer
    """
    print(f"\nLoading CODI model...")
    print(f"  Base model: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Num latent tokens: {num_latent}")
    print(f"  Device: {device}")

    # Create model arguments
    model_args = ModelArguments(
        model_name_or_path=model_name,
        train=False,  # Inference mode
        full_precision=True,
        lora_r=128,
        lora_dropout=0.05
    )

    # Create training arguments (needed for model initialization)
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
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # Get tokenizer
    tokenizer = model.tokenizer

    return model, tokenizer


def prepare_codi_input(question, tokenizer, bot_id, eot_id, num_latent, device="cuda"):
    """
    Prepare input for CODI model with continuous thought tokens

    Args:
        question: Question string
        tokenizer: Tokenizer
        bot_id: Beginning of thought token ID
        eot_id: End of thought token ID
        num_latent: Number of continuous thought tokens
        device: Device

    Returns:
        input_ids: Tensor of input token IDs including BoT/EoT markers
        cot_positions: List of positions for continuous thought tokens
    """
    # Tokenize question
    question_tokens = tokenizer.encode(question, add_special_tokens=False)

    # Create input with BoT, latent placeholders, and EoT
    # Format: [question tokens] [BoT] [CT0] [CT1] ... [CTn] [EoT]
    input_ids = question_tokens + [bot_id] + [tokenizer.pad_token_id] * num_latent + [eot_id]

    # Positions of continuous thought tokens (after BoT, before EoT)
    bot_pos = len(question_tokens)
    cot_positions = list(range(bot_pos + 1, bot_pos + 1 + num_latent))

    # Convert to tensor
    input_ids = torch.tensor([input_ids], device=device)

    return input_ids, cot_positions


def get_layer_module(model, layer_idx):
    """
    Get the layer module for a specific layer index

    Args:
        model: CODI model
        layer_idx: Layer index (0 to num_layers-1)

    Returns:
        layer_module: The transformer layer module
    """
    # Navigate to the actual transformer layers
    # For CODI wrapping LLaMA: model.codi.base_model.model.model.layers[layer_idx]
    if hasattr(model.codi, 'base_model'):
        # LoRA wrapped model
        return model.codi.base_model.model.model.layers[layer_idx]
    else:
        # Direct model
        return model.codi.model.layers[layer_idx]


def extract_activations_at_layer(model, input_ids, attention_mask, layer_idx, positions=None):
    """
    Extract activations at a specific layer for specific positions

    Args:
        model: CODI model
        input_ids: Input token IDs
        attention_mask: Attention mask
        layer_idx: Layer to extract from
        positions: Token positions to extract (None = all positions)

    Returns:
        activations: Tensor of activations [batch, positions, hidden_dim]
    """
    activations = {}

    def hook_fn(module, input, output):
        # output is a tuple, first element is the hidden states
        hidden_states = output[0] if isinstance(output, tuple) else output
        if positions is not None:
            activations['hidden'] = hidden_states[:, positions, :].detach().cpu()
        else:
            activations['hidden'] = hidden_states.detach().cpu()

    # Register hook
    layer = get_layer_module(model, layer_idx)
    handle = layer.register_forward_hook(hook_fn)

    # Forward pass
    with torch.no_grad():
        _ = model.codi(input_ids=input_ids, attention_mask=attention_mask)

    # Remove hook
    handle.remove()

    return activations['hidden']
