#!/usr/bin/env python3
"""
Multi-token corruption utilities for threshold experiments.

Extends CCTA single-token corruption to handle multiple tokens simultaneously.
"""
import torch
from typing import List, Tuple, Optional


def corrupt_n_tokens(
    baseline_acts: List[torch.Tensor],
    positions: Tuple[int, ...],
    corruption_method: str,
    random_pool: Optional[dict] = None,
    problem_idx: int = 0
) -> List[torch.Tensor]:
    """
    Corrupt N tokens simultaneously with specified method.

    Args:
        baseline_acts: List of 6 activation tensors (one per token position)
        positions: Tuple of positions to corrupt (e.g., (0, 1, 3))
        corruption_method: 'zero' or 'gauss_X.X' where X.X is sigma
        random_pool: Optional dict mapping position -> list of cached activations
        problem_idx: Problem index (for random replacement sampling)

    Returns:
        List of 6 activation tensors with specified positions corrupted
    """
    # Clone all activations
    corrupted_acts = [act.clone() for act in baseline_acts]

    # Apply corruption to specified positions
    for pos in positions:
        if corruption_method == 'zero':
            # Zero ablation
            corrupted_acts[pos] = torch.zeros_like(baseline_acts[pos])

        elif corruption_method.startswith('gauss_'):
            # Gaussian noise
            sigma = float(corruption_method.split('_')[1])
            noise = torch.randn_like(baseline_acts[pos]) * sigma
            corrupted_acts[pos] = baseline_acts[pos] + noise

        elif corruption_method == 'random' and random_pool is not None:
            # Random replacement from pool
            pool = random_pool[pos]
            random_idx = (problem_idx + 1) % len(pool)  # Avoid same problem
            corrupted_acts[pos] = pool[random_idx]

        else:
            raise ValueError(f"Unknown corruption method: {corruption_method}")

    return corrupted_acts


def shuffle_all_tokens(baseline_acts: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Shuffle all token positions randomly.

    Args:
        baseline_acts: List of 6 activation tensors

    Returns:
        List of 6 activation tensors in shuffled order
    """
    indices = torch.randperm(6)
    shuffled_acts = [baseline_acts[i] for i in indices]
    return shuffled_acts


def enhance_token(
    baseline_acts: List[torch.Tensor],
    position: int,
    multiplier: float
) -> List[torch.Tensor]:
    """
    Enhance (amplify) a single token activation.

    Args:
        baseline_acts: List of 6 activation tensors
        position: Token position to enhance (0-5)
        multiplier: Multiplication factor (e.g., 1.5 for 50% increase)

    Returns:
        List of 6 activation tensors with specified position enhanced
    """
    enhanced_acts = [act.clone() for act in baseline_acts]
    enhanced_acts[position] = baseline_acts[position] * multiplier
    return enhanced_acts


def validate_corruption(
    baseline_acts: List[torch.Tensor],
    corrupted_acts: List[torch.Tensor],
    positions: Tuple[int, ...]
) -> bool:
    """
    Validate that corruption was applied correctly.

    Args:
        baseline_acts: Original activations
        corrupted_acts: Corrupted activations
        positions: Positions that should be corrupted

    Returns:
        True if corruption is valid
    """
    # Check that corrupted positions are different
    for pos in positions:
        if torch.allclose(baseline_acts[pos], corrupted_acts[pos]):
            return False

    # Check that non-corrupted positions are identical
    all_positions = set(range(6))
    uncorrupted = all_positions - set(positions)
    for pos in uncorrupted:
        if not torch.allclose(baseline_acts[pos], corrupted_acts[pos]):
            return False

    return True


def get_all_corruption_configs(n_tokens: int = 6) -> List[dict]:
    """
    Get all corruption configurations for threshold experiment.

    Args:
        n_tokens: Total number of tokens (default 6 for CODI)

    Returns:
        List of dicts with 'level', 'positions', 'label'
    """
    from utils import get_corruption_positions, get_position_label

    configs = []

    for level in range(1, n_tokens + 1):
        positions_list = get_corruption_positions(n_tokens, level)

        for positions in positions_list:
            configs.append({
                'level': level,
                'positions': positions,
                'label': get_position_label(positions)
            })

    return configs


# Statistics for understanding corruption patterns
def compute_corruption_statistics(results: List[dict]) -> dict:
    """
    Compute summary statistics from corruption experiment results.

    Args:
        results: List of result dicts from threshold experiment

    Returns:
        Dict with statistics by corruption level and position
    """
    import numpy as np

    stats = {
        'by_level': {},
        'by_position': {},
        'by_method': {}
    }

    # Group by corruption level
    for level in range(1, 7):
        level_results = [r for r in results if r.get('corruption_level') == level]
        if level_results:
            accuracies = [r['correct'] for r in level_results]
            stats['by_level'][level] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'count': len(level_results)
            }

    # Group by individual position (for level=1 and level=5)
    for pos in range(6):
        # Level 1: corrupt only this position
        pos_corrupt_results = [
            r for r in results
            if r.get('corruption_level') == 1 and r.get('positions') == (pos,)
        ]
        # Level 5: keep only this position
        pos_keep_results = [
            r for r in results
            if r.get('corruption_level') == 5 and r.get('positions') == (pos,)
        ]

        if pos_corrupt_results:
            stats['by_position'][f'corrupt_{pos}'] = {
                'mean_accuracy': np.mean([r['correct'] for r in pos_corrupt_results]),
                'count': len(pos_corrupt_results)
            }

        if pos_keep_results:
            stats['by_position'][f'keep_{pos}'] = {
                'mean_accuracy': np.mean([r['correct'] for r in pos_keep_results]),
                'count': len(pos_keep_results)
            }

    return stats
