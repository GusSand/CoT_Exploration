#!/usr/bin/env python3
"""
Evaluate probe with softer criteria:
- Top-1 exact match
- Top-5 overlap (how many tokens appear in both top-5 lists)
- Top-5 rank correlation
- Jensen-Shannon divergence (distribution similarity)
- KL divergence
- Cosine similarity between probability distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
import argparse
import sys
import numpy as np

# Add CODI src to path
sys.path.insert(0, '/workspace/CoT_Exploration/codi/src')
from model import CODI, ModelArguments, TrainingArguments

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x):
        return self.linear(x)


def load_codi_model_correctly(args):
    """Load CODI model using the official CODI class."""
    print("="*100)
    print("LOADING CODI MODEL")
    print("="*100)

    lora_config = LoraConfig(
        r=128,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir=args.ckpt_dir,
        full_precision=True,
        token=args.token
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=args.inf_latent_iterations,
        use_prj=True,
        prj_dim=768,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    model = CODI(model_args, training_args, lora_config)
    checkpoint_path = f"{args.ckpt_dir}/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.codi.tie_weights()
    model = model.float().to(device)
    model.eval()

    print(f"Model loaded: n_layer={model.codi.config.n_layer}, has_prj={model.use_prj}")
    print("="*100)

    return model


def jensen_shannon_divergence(p, q):
    """Compute Jensen-Shannon divergence between two probability distributions."""
    p = p + 1e-10  # Avoid log(0)
    q = q + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') +
                  F.kl_div(m.log(), q, reduction='batchmean'))


def compute_top5_overlap(probe_indices, target_indices):
    """Compute how many tokens appear in both top-5 lists."""
    probe_set = set(probe_indices.cpu().tolist())
    target_set = set(target_indices.cpu().tolist())
    return len(probe_set & target_set)


def compute_rank_correlation(probe_probs, target_probs, top_k_indices):
    """Compute Spearman rank correlation for top-k tokens using numpy."""
    # Get probabilities for the union of top-k indices
    all_indices = torch.unique(top_k_indices)

    probe_vals = probe_probs[all_indices].cpu().numpy()
    target_vals = target_probs[all_indices].cpu().numpy()

    if len(probe_vals) < 2:
        return 0.0

    # Compute Spearman correlation using numpy
    # Rank the values
    probe_ranks = np.argsort(np.argsort(probe_vals))
    target_ranks = np.argsort(np.argsort(target_vals))

    # Compute correlation between ranks
    n = len(probe_ranks)
    d = probe_ranks - target_ranks
    corr = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))

    return corr if not np.isnan(corr) else 0.0


def evaluate_probe_soft(model, tokenizer, probe, dataset, args):
    """Evaluate probe with soft metrics."""
    print("\n" + "="*100)
    print(f"SOFT METRICS EVALUATION")
    print("="*100)
    print(f"Probe: {args.probe_path}")
    print(f"Test examples: {args.num_test_examples}")
    print("="*100)

    model.eval()
    lm_head = model.codi.lm_head

    # Aggregate metrics
    total_top1_matches = 0
    total_top5_overlaps = []
    total_js_divergences = []
    total_kl_divergences = []
    total_prob_cosine_sims = []
    total_hidden_cosine_sims = []
    total_rank_correlations = []

    # Per-iteration tracking
    iter_top1_matches = [0] * args.inf_latent_iterations
    iter_top5_overlaps = [[] for _ in range(args.inf_latent_iterations)]

    total_samples = 0

    num_examples = min(args.num_test_examples, len(dataset['test']))

    with torch.no_grad():
        for ex_idx in range(num_examples):
            question = dataset['test'][ex_idx]['question'].strip()

            # Tokenize
            inputs = tokenizer([question], return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Initial forward pass
            outputs = model.codi(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd_projected = model.prj(latent_embd_pre_proj)
            else:
                latent_embd_projected = latent_embd_pre_proj

            # Run continuous thought iterations
            for iteration in range(args.inf_latent_iterations):
                iter_outputs = model.codi(
                    inputs_embeds=latent_embd_projected,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values
                )

                past_key_values = iter_outputs.past_key_values
                hidden_states = iter_outputs.hidden_states

                # Extract activations
                source_hidden = hidden_states[args.source_layer + 1][0, 0, :]
                target_hidden = hidden_states[args.target_layer + 1][0, 0, :]

                # Apply probe
                probed_hidden = probe(source_hidden.unsqueeze(0)).squeeze(0)

                # Get logits and probabilities
                probed_logits = lm_head(probed_hidden)
                target_logits = lm_head(target_hidden)

                probed_probs = F.softmax(probed_logits, dim=-1)
                target_probs = F.softmax(target_logits, dim=-1)

                # Top-1 match
                probed_top1 = probed_logits.argmax()
                target_top1 = target_logits.argmax()
                top1_match = (probed_top1 == target_top1).item()

                total_top1_matches += top1_match
                iter_top1_matches[iteration] += top1_match

                # Top-5 overlap
                probe_top5_probs, probe_top5_indices = probed_probs.topk(5)
                target_top5_probs, target_top5_indices = target_probs.topk(5)

                top5_overlap = compute_top5_overlap(probe_top5_indices, target_top5_indices)
                total_top5_overlaps.append(top5_overlap)
                iter_top5_overlaps[iteration].append(top5_overlap)

                # Jensen-Shannon divergence
                js_div = jensen_shannon_divergence(probed_probs, target_probs).item()
                total_js_divergences.append(js_div)

                # KL divergence (probe || target)
                kl_div = F.kl_div(
                    probed_probs.log(),
                    target_probs,
                    reduction='batchmean'
                ).item()
                total_kl_divergences.append(kl_div)

                # Cosine similarity (probability distributions)
                prob_cos_sim = F.cosine_similarity(
                    probed_probs.unsqueeze(0),
                    target_probs.unsqueeze(0)
                ).item()
                total_prob_cosine_sims.append(prob_cos_sim)

                # Cosine similarity (hidden states)
                hidden_cos_sim = F.cosine_similarity(
                    probed_hidden.unsqueeze(0),
                    target_hidden.unsqueeze(0)
                ).item()
                total_hidden_cosine_sims.append(hidden_cos_sim)

                # Rank correlation
                all_top5_indices = torch.cat([probe_top5_indices, target_top5_indices]).unique()
                rank_corr = compute_rank_correlation(probed_probs, target_probs, all_top5_indices)
                total_rank_correlations.append(rank_corr)

                total_samples += 1

                # Detailed logging for first 10 examples
                if ex_idx < 10:
                    if iteration == 0:
                        print(f"\n{'-'*100}")
                        print(f"Example {ex_idx + 1}: {question[:70]}...")

                    match_str = "✓" if top1_match else "✗"

                    print(f"\n  Iteration {iteration + 1}:")
                    print(f"    Top-1 Match: {match_str}")
                    print(f"    Top-5 Overlap: {top5_overlap}/5 tokens")
                    print(f"    JS Divergence: {js_div:.4f}")
                    print(f"    KL Divergence: {kl_div:.4f}")
                    print(f"    Prob Cosine Sim: {prob_cos_sim:.4f}")
                    print(f"    Hidden Cosine Sim: {hidden_cos_sim:.4f}")
                    print(f"    Rank Correlation: {rank_corr:.4f}")

                    # Show top-5 tokens
                    print(f"\n    Probed top-5:")
                    for idx, prob in zip(probe_top5_indices, probe_top5_probs):
                        tok = tokenizer.decode([idx])
                        marker = "★" if idx in target_top5_indices else " "
                        print(f"      {marker} '{tok}': {prob.item()*100:.2f}%")

                    print(f"\n    Target top-5:")
                    for idx, prob in zip(target_top5_indices, target_top5_probs):
                        tok = tokenizer.decode([idx])
                        marker = "★" if idx == target_top1 else " "
                        print(f"      {marker} '{tok}': {prob.item()*100:.2f}%")

                # Prepare next iteration
                latent_embd_pre_proj = iter_outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd_projected = model.prj(latent_embd_pre_proj)
                else:
                    latent_embd_projected = latent_embd_pre_proj

            if (ex_idx + 1) % 50 == 0:
                current_top1_acc = 100.0 * total_top1_matches / total_samples
                current_avg_overlap = np.mean(total_top5_overlaps)
                print(f"\n{'-'*100}")
                print(f"Processed {ex_idx + 1}/{num_examples} examples...")
                print(f"  Running Top-1 Accuracy: {current_top1_acc:.2f}%")
                print(f"  Running Avg Top-5 Overlap: {current_avg_overlap:.2f}/5")

    # Compute final statistics
    print(f"\n{'='*100}")
    print("FINAL RESULTS")
    print(f"{'='*100}")

    top1_accuracy = 100.0 * total_top1_matches / total_samples
    avg_top5_overlap = np.mean(total_top5_overlaps)
    avg_js_div = np.mean(total_js_divergences)
    avg_kl_div = np.mean(total_kl_divergences)
    avg_prob_cos_sim = np.mean(total_prob_cosine_sims)
    avg_hidden_cos_sim = np.mean(total_hidden_cosine_sims)
    avg_rank_corr = np.mean(total_rank_correlations)

    print(f"\nOverall Metrics ({total_samples} samples):")
    print(f"  Top-1 Exact Match: {total_top1_matches}/{total_samples} = {top1_accuracy:.2f}%")
    print(f"  Top-5 Overlap: {avg_top5_overlap:.2f}/5 tokens ({avg_top5_overlap/5*100:.1f}%)")
    print(f"  Jensen-Shannon Divergence: {avg_js_div:.4f}")
    print(f"  KL Divergence: {avg_kl_div:.4f}")
    print(f"  Probability Cosine Similarity: {avg_prob_cos_sim:.4f}")
    print(f"  Hidden State Cosine Similarity: {avg_hidden_cos_sim:.4f}")
    print(f"  Rank Correlation (Spearman): {avg_rank_corr:.4f}")

    print(f"\nPer-Iteration Top-1 Accuracy:")
    for i in range(args.inf_latent_iterations):
        iter_acc = 100.0 * iter_top1_matches[i] / num_examples
        print(f"  Iteration {i+1}: {iter_top1_matches[i]}/{num_examples} = {iter_acc:.2f}%")

    print(f"\nPer-Iteration Top-5 Overlap:")
    for i in range(args.inf_latent_iterations):
        iter_overlap = np.mean(iter_top5_overlaps[i])
        print(f"  Iteration {i+1}: {iter_overlap:.2f}/5 ({iter_overlap/5*100:.1f}%)")

    # Distribution analysis
    print(f"\nTop-5 Overlap Distribution:")
    overlap_counts = {i: total_top5_overlaps.count(i) for i in range(6)}
    for overlap, count in sorted(overlap_counts.items()):
        pct = 100.0 * count / len(total_top5_overlaps)
        print(f"  {overlap}/5 overlap: {count} samples ({pct:.1f}%)")

    print(f"{'='*100}")

    return {
        'top1_accuracy': top1_accuracy,
        'avg_top5_overlap': avg_top5_overlap,
        'avg_js_divergence': avg_js_div,
        'avg_kl_divergence': avg_kl_div,
        'avg_prob_cosine_sim': avg_prob_cos_sim,
        'avg_hidden_cosine_sim': avg_hidden_cos_sim,
        'avg_rank_correlation': avg_rank_corr
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="openai-community/gpt2")
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="zen-E/GSM8k-Aug")
    parser.add_argument("--inf_latent_iterations", type=int, default=6)
    parser.add_argument("--num_test_examples", type=int, default=100)
    parser.add_argument("--source_layer", type=int, default=10)
    parser.add_argument("--target_layer", type=int, default=11)

    args = parser.parse_args()

    print("="*100)
    print("SOFT METRICS PROBE EVALUATION")
    print("="*100)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token)
    tokenizer.pad_token = tokenizer.eos_token

    # Load CODI model
    model = load_codi_model_correctly(args)

    # Load probe
    probe = LinearProbe(768, 768).to(device)
    probe.load_state_dict(torch.load(args.probe_path, map_location=device))
    probe.eval()
    print(f"\nProbe loaded from: {args.probe_path}")

    # Load dataset
    dataset = load_dataset(args.data_name)

    # Evaluate probe
    evaluate_probe_soft(model, tokenizer, probe, dataset, args)


if __name__ == "__main__":
    main()
