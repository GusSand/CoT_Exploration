"""
Metrics for analyzing CoT attention patterns
"""

import numpy as np
from scipy.stats import entropy

def compute_sequential_score(attention_matrix):
    """
    Compute sequential attention score: how much each position attends to the previous one

    Args:
        attention_matrix: [num_latent, num_latent] attention matrix
                         attention_matrix[i, j] = attention from position i TO position j

    Returns:
        sequential_score: Mean attention from position N to position N-1
        sequential_weights: List of individual sequential attention weights
    """
    num_latent = attention_matrix.shape[0]
    sequential_weights = []

    # For each position (except the first), get attention to previous position
    for i in range(1, num_latent):
        # Attention from position i TO position i-1
        attn_to_prev = attention_matrix[i, i-1]
        sequential_weights.append(float(attn_to_prev))

    sequential_score = float(np.mean(sequential_weights))
    return sequential_score, sequential_weights


def compute_self_attention_score(attention_matrix):
    """
    Compute self-attention: how much each position attends to itself

    Args:
        attention_matrix: [num_latent, num_latent] attention matrix

    Returns:
        self_attn_score: Mean self-attention across all positions
        self_attn_weights: List of individual self-attention weights
    """
    num_latent = attention_matrix.shape[0]
    self_attn_weights = []

    # For each position, get self-attention
    for i in range(num_latent):
        self_attn = attention_matrix[i, i]
        self_attn_weights.append(float(self_attn))

    self_attn_score = float(np.mean(self_attn_weights))
    return self_attn_score, self_attn_weights


def compute_attention_entropy(attention_matrix):
    """
    Compute entropy of attention distribution for each position

    High entropy = distributed attention (parallel processing)
    Low entropy = focused attention (sequential processing)

    Args:
        attention_matrix: [num_latent, num_latent] attention matrix

    Returns:
        mean_entropy: Mean entropy across all positions
        entropy_per_position: List of entropy values for each position
    """
    num_latent = attention_matrix.shape[0]
    entropy_per_position = []

    for i in range(num_latent):
        # Get attention distribution from position i
        attn_dist = attention_matrix[i, :]
        # Compute entropy (using base 2 for bits)
        ent = entropy(attn_dist, base=2)
        entropy_per_position.append(float(ent))

    mean_entropy = float(np.mean(entropy_per_position))
    return mean_entropy, entropy_per_position


def compute_forward_vs_backward_attention(attention_matrix):
    """
    Compare forward attention (to later positions) vs backward (to earlier positions)

    Args:
        attention_matrix: [num_latent, num_latent] attention matrix

    Returns:
        forward_score: Mean attention to later positions
        backward_score: Mean attention to earlier positions
        ratio: forward / backward ratio
    """
    num_latent = attention_matrix.shape[0]
    forward_weights = []
    backward_weights = []

    for i in range(num_latent):
        # Forward: attention to positions > i
        if i < num_latent - 1:
            forward_attn = np.sum(attention_matrix[i, i+1:])
            forward_weights.append(float(forward_attn))

        # Backward: attention to positions < i
        if i > 0:
            backward_attn = np.sum(attention_matrix[i, :i])
            backward_weights.append(float(backward_attn))

    forward_score = float(np.mean(forward_weights)) if forward_weights else 0.0
    backward_score = float(np.mean(backward_weights)) if backward_weights else 0.0
    ratio = forward_score / backward_score if backward_score > 0 else float('inf')

    return forward_score, backward_score, ratio


def compute_all_metrics(attention_matrix):
    """
    Compute all attention metrics for a single attention matrix

    Args:
        attention_matrix: [num_latent, num_latent] attention matrix

    Returns:
        metrics: Dict containing all computed metrics
    """
    seq_score, seq_weights = compute_sequential_score(attention_matrix)
    self_score, self_weights = compute_self_attention_score(attention_matrix)
    mean_ent, ent_per_pos = compute_attention_entropy(attention_matrix)
    fwd, bwd, ratio = compute_forward_vs_backward_attention(attention_matrix)

    return {
        'sequential_score': seq_score,
        'sequential_weights': seq_weights,
        'self_attention_score': self_score,
        'self_attention_weights': self_weights,
        'mean_entropy': mean_ent,
        'entropy_per_position': ent_per_pos,
        'forward_attention': fwd,
        'backward_attention': bwd,
        'forward_backward_ratio': ratio
    }


def aggregate_metrics_across_examples(metrics_list):
    """
    Aggregate metrics across multiple examples

    Args:
        metrics_list: List of metrics dicts

    Returns:
        aggregated: Dict with mean and std for each metric
    """
    # Extract scalar metrics
    sequential_scores = [m['sequential_score'] for m in metrics_list]
    self_scores = [m['self_attention_score'] for m in metrics_list]
    entropies = [m['mean_entropy'] for m in metrics_list]
    forward_scores = [m['forward_attention'] for m in metrics_list]
    backward_scores = [m['backward_attention'] for m in metrics_list]
    ratios = [m['forward_backward_ratio'] for m in metrics_list if np.isfinite(m['forward_backward_ratio'])]

    return {
        'sequential_score_mean': float(np.mean(sequential_scores)),
        'sequential_score_std': float(np.std(sequential_scores)),
        'self_attention_mean': float(np.mean(self_scores)),
        'self_attention_std': float(np.std(self_scores)),
        'entropy_mean': float(np.mean(entropies)),
        'entropy_std': float(np.std(entropies)),
        'forward_attention_mean': float(np.mean(forward_scores)),
        'backward_attention_mean': float(np.mean(backward_scores)),
        'forward_backward_ratio_mean': float(np.mean(ratios)) if ratios else None
    }
