"""
Metrics computation for activation patching analysis
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_kl_divergence(patched_logits, baseline_logits, temperature=1.0):
    """
    Compute KL divergence between patched and baseline output distributions

    KL(P || Q) where:
    - P = patched distribution
    - Q = baseline distribution

    Args:
        patched_logits: Logits from patched model [batch, seq_len, vocab_size]
        baseline_logits: Logits from baseline model [batch, seq_len, vocab_size]
        temperature: Temperature for softmax (default=1.0)

    Returns:
        kl_div: KL divergence averaged over positions (scalar)
        kl_per_position: KL divergence for each position [seq_len]
    """
    # Convert logits to log probabilities
    log_probs_patched = F.log_softmax(patched_logits / temperature, dim=-1)
    log_probs_baseline = F.log_softmax(baseline_logits / temperature, dim=-1)

    # Compute KL divergence: KL(P || Q) = sum(P * log(P/Q))
    # = sum(P * (log(P) - log(Q)))
    probs_patched = torch.exp(log_probs_patched)
    kl_per_token = (probs_patched * (log_probs_patched - log_probs_baseline)).sum(dim=-1)

    # Average over batch and sequence
    kl_per_position = kl_per_token.mean(dim=0).cpu().numpy()
    kl_div = kl_per_position.mean()

    return float(kl_div), kl_per_position


def compute_js_divergence(patched_logits, baseline_logits, temperature=1.0):
    """
    Compute Jensen-Shannon divergence (symmetric version of KL)

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        patched_logits: Logits from patched model
        baseline_logits: Logits from baseline model
        temperature: Temperature for softmax

    Returns:
        js_div: JS divergence (scalar)
    """
    # Convert to probabilities
    probs_patched = F.softmax(patched_logits / temperature, dim=-1)
    probs_baseline = F.softmax(baseline_logits / temperature, dim=-1)

    # Compute mixture
    probs_mixture = 0.5 * (probs_patched + probs_baseline)

    # Convert to log probabilities
    log_probs_patched = torch.log(probs_patched + 1e-10)
    log_probs_baseline = torch.log(probs_baseline + 1e-10)
    log_probs_mixture = torch.log(probs_mixture + 1e-10)

    # Compute KL divergences
    kl_pm = (probs_patched * (log_probs_patched - log_probs_mixture)).sum(dim=-1)
    kl_bm = (probs_baseline * (log_probs_baseline - log_probs_mixture)).sum(dim=-1)

    # JS divergence
    js_div = 0.5 * (kl_pm + kl_bm).mean()

    return float(js_div.cpu())


def compute_logit_difference(patched_logits, baseline_logits):
    """
    Compute L2 difference between patched and baseline logits

    Args:
        patched_logits: Logits from patched model
        baseline_logits: Logits from baseline model

    Returns:
        l2_diff: L2 distance between logits (scalar)
    """
    diff = patched_logits - baseline_logits
    l2_diff = torch.sqrt((diff ** 2).mean())
    return float(l2_diff.cpu())


def compute_top_k_overlap(patched_logits, baseline_logits, k=10):
    """
    Compute overlap in top-k predictions

    Args:
        patched_logits: Logits from patched model [batch, seq_len, vocab_size]
        baseline_logits: Logits from baseline model
        k: Number of top predictions to consider

    Returns:
        overlap: Proportion of top-k tokens that match (0 to 1)
    """
    # Get top-k indices
    _, patched_topk = torch.topk(patched_logits, k=k, dim=-1)
    _, baseline_topk = torch.topk(baseline_logits, k=k, dim=-1)

    # Compute overlap
    overlaps = []
    for i in range(patched_topk.shape[1]):  # For each position
        patched_set = set(patched_topk[0, i, :].cpu().numpy())
        baseline_set = set(baseline_topk[0, i, :].cpu().numpy())
        overlap = len(patched_set & baseline_set) / k
        overlaps.append(overlap)

    return np.mean(overlaps)


def compute_prediction_change(patched_logits, baseline_logits):
    """
    Check if top-1 prediction changes after patching

    Args:
        patched_logits: Logits from patched model [batch, seq_len, vocab_size]
        baseline_logits: Logits from baseline model

    Returns:
        changed: Boolean array indicating position where top-1 changed
        change_rate: Proportion of positions where prediction changed
    """
    # Get top-1 predictions
    patched_pred = patched_logits.argmax(dim=-1)
    baseline_pred = baseline_logits.argmax(dim=-1)

    # Check where they differ
    changed = (patched_pred != baseline_pred).cpu().numpy()[0]
    change_rate = changed.mean()

    return changed, float(change_rate)
