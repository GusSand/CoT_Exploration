"""
Metrics computation for activation patching analysis
Extended with answer_logit_difference metric
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


def compute_logit_difference(patched_logits, baseline_logits):
    """
    Compute L2 difference between patched and baseline logits

    Args:
        patched_logits: Logits from patched model
        baseline_logits: Logits from baseline logits

    Returns:
        l2_diff: L2 distance between logits (scalar)
    """
    diff = patched_logits - baseline_logits
    l2_diff = torch.sqrt((diff ** 2).mean())
    return float(l2_diff.cpu())


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


def compute_answer_logit_difference(patched_logits, clean_answer_tokens,
                                    corrupted_answer_tokens, answer_start_pos):
    """
    Measure how much patching shifts logits toward clean vs corrupted answer

    This metric directly measures whether patching recovers the correct (clean) answer
    by comparing the mean logits for clean answer tokens vs corrupted answer tokens.

    Args:
        patched_logits: Logits from patched model [batch, seq_len, vocab_size]
        clean_answer_tokens: List of token IDs for clean answer
        corrupted_answer_tokens: List of token IDs for corrupted answer
        answer_start_pos: Position where answer generation begins (after EoT)

    Returns:
        mean_diff: Mean(logits[clean_tokens]) - Mean(logits[corrupted_tokens])
                   Positive = patching shifts toward clean (recovery)
                   Negative = patching shifts toward corrupted (worsening)
        clean_score: Mean logit for clean answer tokens
        corrupted_score: Mean logit for corrupted answer tokens
    """
    # Extract logits at answer positions
    # Take first position after answer start as representative
    answer_logits = patched_logits[:, answer_start_pos, :]  # [batch, vocab_size]

    # Get logits for clean answer tokens and average
    clean_logits_list = []
    for token_id in clean_answer_tokens:
        clean_logits_list.append(answer_logits[0, token_id].item())
    clean_score = np.mean(clean_logits_list) if clean_logits_list else 0.0

    # Get logits for corrupted answer tokens and average
    corrupted_logits_list = []
    for token_id in corrupted_answer_tokens:
        corrupted_logits_list.append(answer_logits[0, token_id].item())
    corrupted_score = np.mean(corrupted_logits_list) if corrupted_logits_list else 0.0

    # Compute difference: positive = recovery toward clean
    mean_diff = clean_score - corrupted_score

    return float(mean_diff), float(clean_score), float(corrupted_score)


def tokenize_answer(answer, tokenizer):
    """
    Tokenize a numeric answer to get token IDs

    Args:
        answer: Numeric answer (int or float)
        tokenizer: Tokenizer

    Returns:
        token_ids: List of token IDs for the answer
    """
    # Convert to string and tokenize
    answer_str = str(answer)
    tokens = tokenizer.encode(answer_str, add_special_tokens=False)
    return tokens
