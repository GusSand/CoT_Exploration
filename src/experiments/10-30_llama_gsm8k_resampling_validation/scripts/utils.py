"""
Utility functions for Resampling Experiment.

Reuses existing patterns from:
- src/experiments/codi_attention_flow/ablation/utils.py
- src/experiments/activation_patching/core/cache_activations_llama.py
"""
import torch
import numpy as np
import random
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "codi"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "activation_patching" / "core"))

from cache_activations_llama import ActivationCacherLLaMA


def set_seed(seed: int = 42):
    """
    Set all random seeds for perfect reproducibility.

    Side effects:
    - Sets Python, NumPy, PyTorch random seeds
    - Configures CUDNN for deterministic operations (~5-10% slower)
    - Sets PYTHONHASHSEED environment variable

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed (for dict/set ordering)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ All random seeds set to {seed}")


def load_model(model_path: str = None, device: str = 'cuda'):
    """
    Load CODI LLaMA model.

    Args:
        model_path: Path to checkpoint (default: ~/codi_ckpt/llama_gsm8k)
        device: Device to run on

    Returns:
        Tuple of (model, tokenizer)
    """
    if model_path is None:
        model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    print(f"Loading LLaMA-1B CODI model from {model_path}...")

    cacher = ActivationCacherLLaMA(model_path, device=device)
    model = cacher.model
    tokenizer = cacher.tokenizer

    model.eval()  # CRITICAL: Set to eval mode

    print("✓ Model loaded successfully")
    print(f"  Architecture: Llama-3.2-1B-Instruct")
    print(f"  Layers: 16")
    print(f"  Hidden dim: 2048")
    print(f"  Device: {device}")

    return model, tokenizer


def extract_answer(answer_str: str) -> int:
    """
    Extract numeric answer from GSM8K format.

    Handles:
    1. Gold format: "...#### 42"
    2. Generated format: "The answer is 42" (extracts last number)

    Args:
        answer_str: Answer string

    Returns:
        int: Numeric answer or None if parsing fails
    """
    try:
        # Try gold answer format first
        if '####' in answer_str:
            answer = answer_str.split('####')[1].strip()
            answer = answer.replace(',', '')
            return int(answer)

        # Extract last number from generated text
        answer_str = answer_str.replace(',', '')
        pred = [s for s in re.findall(r'-?\d+\.?\d*', answer_str)]
        if pred:
            return int(float(pred[-1]))

        return None
    except (IndexError, ValueError) as e:
        # print(f"Warning: Failed to parse '{answer_str}': {e}")
        return None


def load_test_problems(n_problems: int = 100, seed: int = 42) -> List[Dict]:
    """
    Load GSM8K test problems with deterministic sampling.

    Architecture:
    - Load full HuggingFace GSM8K test set (1,319 problems)
    - Sample n_problems with fixed seed
    - Return sorted by index for consistency

    Args:
        n_problems: Number of problems to sample
        seed: Random seed for reproducibility

    Returns:
        List of dicts with keys: idx, question, answer, gold_numeric
    """
    from datasets import load_dataset

    # Load full test set
    dataset = load_dataset('gsm8k', 'main', split='test')

    # Deterministic sampling
    random.seed(seed)
    all_indices = list(range(len(dataset)))
    sampled_indices = sorted(random.sample(all_indices, n_problems))

    # Extract problems
    problems = []
    for idx in sampled_indices:
        item = dataset[idx]

        # Extract numeric answer from "#### 42" format
        gold_numeric = extract_answer(item['answer'])

        problems.append({
            'idx': idx,
            'question': item['question'],
            'answer': item['answer'],  # Full solution with #### marker
            'gold_numeric': gold_numeric
        })

    print(f"✓ Loaded {len(problems)} GSM8K test problems (seed={seed})")

    return problems


def validate_cache(cache: Dict, expected_n_problems: int):
    """
    Validate cached CT hidden states.

    Args:
        cache: Loaded cache dict
        expected_n_problems: Expected number of problems

    Raises:
        AssertionError: If validation fails
    """
    assert len(cache) == expected_n_problems, \
        f"Expected {expected_n_problems} problems, got {len(cache)}"

    # Check first problem
    first_problem = cache[0]

    # Check required keys
    required_keys = ['idx', 'question', 'answer', 'gold_numeric',
                     'ct_hidden_states', 'baseline_prediction', 'baseline_correct']
    for key in required_keys:
        assert key in first_problem, f"Missing key: {key}"

    # Check tensor shape
    ct_states = first_problem['ct_hidden_states']
    assert ct_states.shape == (6, 2048), \
        f"Expected CT hidden states shape (6, 2048), got {ct_states.shape}"

    # Check for NaN/Inf
    assert not torch.isnan(ct_states).any(), "NaN detected in hidden states"
    assert not torch.isinf(ct_states).any(), "Inf detected in hidden states"

    # Check value range
    max_val = ct_states.abs().max().item()
    assert max_val < 100, f"Abnormally large values: {max_val}"

    print(f"✓ Cache validation passed ({expected_n_problems} problems)")
    print(f"  - CT hidden states shape: {ct_states.shape}")
    print(f"  - Value range: [{ct_states.min():.3f}, {ct_states.max():.3f}]")
    print(f"  - Max abs value: {max_val:.3f}")
