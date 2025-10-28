"""
Utilities for ablation experiments.

Includes model loading, architecture info, and shared helper functions.
"""
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))

from cache_activations_llama import ActivationCacherLLaMA
from cache_activations import ActivationCacher


def get_model_architecture_info(model_name):
    """
    Get model-specific architecture details.

    Args:
        model_name: 'llama' or 'gpt2'

    Returns:
        dict with architecture info
    """
    if model_name == 'llama':
        return {
            'layer_attr': 'model.model.layers',  # PeftModel.model (LlamaForCausalLM) .model (LlamaModel) .layers
            'attn_output_attr': 'self_attn.o_proj',
            'n_layers': 16,
            'n_heads': 32,
            'hidden_dim': 2048,
            'uses_gqa': True  # Grouped Query Attention
        }
    elif model_name == 'gpt2':
        return {
            'layer_attr': 'transformer.h',
            'attn_output_attr': 'attn.c_proj',
            'n_layers': 12,
            'n_heads': 12,
            'hidden_dim': 768,
            'uses_gqa': False
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_attention_output_layer(model, layer_idx, model_name):
    """
    Get attention output projection layer (model-specific).

    Args:
        model: CODI model
        layer_idx: Layer index (0-based)
        model_name: 'llama' or 'gpt2'

    Returns:
        Attention output projection layer
    """
    info = get_model_architecture_info(model_name)

    # Navigate to layer container
    layer_container = model
    for attr in info['layer_attr'].split('.'):
        layer_container = getattr(layer_container, attr)
    layer = layer_container[layer_idx]

    # Navigate to attention output
    attn_output = layer
    for attr in info['attn_output_attr'].split('.'):
        attn_output = getattr(attn_output, attr)

    return attn_output


def load_model(model_name):
    """
    Load CODI model (LLaMA or GPT-2).

    Args:
        model_name: 'llama' or 'gpt2'

    Returns:
        Loaded CODI model
    """
    model_paths = {
        'llama': str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
        'gpt2': str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    }

    if model_name == 'llama':
        cacher = ActivationCacherLLaMA(model_paths[model_name])
    else:  # gpt2
        cacher = ActivationCacher(model_paths[model_name])

    return cacher.model, cacher.tokenizer


def extract_answer(answer_str):
    """
    Extract numeric answer from GSM8K format.

    Handles two formats:
    1. Gold answer format: "...#### 42"
    2. Generated answer format: "The answer is: 42" or "...42..." (any number)

    Args:
        answer_str: Answer string

    Returns:
        int: Numeric answer
    """
    import re
    try:
        # Try gold answer format first: "#### 42"
        if '####' in answer_str:
            answer = answer_str.split('####')[1].strip()
            answer = answer.replace(',', '')
            return int(answer)

        # Try generated answer format: extract numbers
        # Remove commas first
        answer_str = answer_str.replace(',', '')
        # Find all numbers (including negative)
        pred = [s for s in re.findall(r'-?\d+\.?\d*', answer_str)]
        if pred:
            # Return the last number found (usually the final answer)
            return int(float(pred[-1]))

        return None
    except (IndexError, ValueError) as e:
        print(f"Warning: Failed to parse answer '{answer_str}': {e}")
        return None


def validate_model_architecture(model, model_name, tokenizer):
    """
    Validate model architecture before ablation.

    Args:
        model: CODI model
        model_name: 'llama' or 'gpt2'
        tokenizer: Model tokenizer

    Raises:
        ValueError: If architecture doesn't match expectations
    """
    info = get_model_architecture_info(model_name)

    # Test forward pass with attention output
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_input = tokenizer("Test", return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.codi(**test_input, output_attentions=True)

    # Check attention shape
    attn_shape = outputs.attentions[0].shape
    expected_heads = info['n_heads']
    actual_heads = attn_shape[1]

    if actual_heads != expected_heads:
        raise ValueError(
            f"Attention head mismatch: expected {expected_heads}, got {actual_heads}. "
            f"Check GQA configuration for {model_name}"
        )

    print(f"✓ {model_name.upper()} architecture validated")
    print(f"  Attention shape: {attn_shape}")
    print(f"  Heads: {actual_heads}")
    print(f"  Uses GQA: {info['uses_gqa']}")
    print(f"  Hidden dim: {info['hidden_dim']}")
