#!/usr/bin/env python3
"""
Head Metrics Computer - Stories 2.1-2.4

Compute flow, hub, and skip scores for each attention head.
Identify top critical heads by composite score.

Usage:
    python 6_compute_head_metrics.py [--model MODEL]

Output:
    ../results/{model}/heads_ranked_by_flow.csv
    ../results/{model}/heads_ranked_by_hub.csv
    ../results/{model}/heads_ranked_by_skip.csv
    ../results/{model}/ranked_heads.csv
    ../results/{model}/critical_heads_findings.txt
"""
import json
import numpy as np
import pandas as pd
import argparse
from pathlib import Path


def compute_flow_score(attention_matrix: np.ndarray) -> float:
    """
    Compute forward information flow score.

    Measures how much attention flows from later positions to earlier positions.

    Args:
        attention_matrix: [6, 6] attention weights

    Returns:
        flow_score: 0-1 (higher = more forward flow)
    """
    forward_flow = 0.0
    total_attention = 0.0

    for i in range(6):
        # Forward flow: position i attending to earlier positions (j < i)
        for j in range(i):
            forward_flow += attention_matrix[i, j]

        # Total attention from position i (to all valid positions 0..i)
        total_attention += attention_matrix[i, :i+1].sum()

    if total_attention > 0:
        return float(forward_flow / total_attention)
    return 0.0


def compute_hub_score(attention_matrix: np.ndarray) -> tuple[float, int]:
    """
    Compute hub connectivity score.

    Measures concentration of attention (variance in incoming attention).
    High variance = some positions act as hubs.

    Args:
        attention_matrix: [6, 6] attention weights

    Returns:
        hub_score: variance of incoming attention
        hub_position: which position is the hub (0-5)
    """
    # Sum incoming attention to each position (column sums)
    incoming_attention = attention_matrix.sum(axis=0)  # [6]

    hub_score = float(np.var(incoming_attention))
    hub_position = int(np.argmax(incoming_attention))

    return hub_score, hub_position


def compute_skip_score(attention_matrix: np.ndarray) -> float:
    """
    Compute skip connection score.

    Measures long-range dependencies (position 5 → early positions 0-2).

    Args:
        attention_matrix: [6, 6] attention weights

    Returns:
        skip_score: average attention from pos 5 to pos 0-2
    """
    # Attention from position 5 to positions 0, 1, 2
    skip_attention = attention_matrix[5, 0:3]  # [3]

    return float(np.mean(skip_attention))


def compute_all_metrics(model: str = 'llama') -> None:
    """
    Compute all metrics for all heads and create rankings.

    Args:
        model: Model name ('llama' or 'gpt2')
    """
    print("=" * 80)
    print("HEAD METRICS COMPUTER - Stories 2.1-2.4")
    print("=" * 80)

    # Load aggregated attention
    results_dir = Path(__file__).parent.parent / 'results' / model
    avg_path = results_dir / 'attention_patterns_avg.npy'

    print(f"\nLoading attention patterns from {avg_path}...")
    attention_avg = np.load(avg_path).astype(np.float32)
    print(f"✓ Loaded: {attention_avg.shape}")

    n_layers, n_heads, _, _ = attention_avg.shape

    # Compute metrics for all heads
    print(f"\nComputing metrics for {n_layers * n_heads} heads...")

    head_metrics = []

    for layer in range(n_layers):
        for head in range(n_heads):
            attn = attention_avg[layer, head]  # [6, 6]

            # Compute all three metrics
            flow_score = compute_flow_score(attn)
            hub_score, hub_position = compute_hub_score(attn)
            skip_score = compute_skip_score(attn)

            # Determine layer type
            if layer < n_layers // 3:
                layer_type = 'early'
            elif layer < 2 * n_layers // 3:
                layer_type = 'middle'
            else:
                layer_type = 'late'

            head_metrics.append({
                'layer': layer,
                'head': head,
                'layer_type': layer_type,
                'flow_score': flow_score,
                'hub_score': hub_score,
                'hub_position': hub_position,
                'skip_score': skip_score,
                'max_attention': float(attn.max()),
                'mean_attention': float(attn.mean())
            })

    print(f"✓ Computed metrics for {len(head_metrics)} heads")

    # Create dataframe for easy manipulation
    df = pd.DataFrame(head_metrics)

    # Story 2.1: Rank by flow score
    print("\n" + "=" * 80)
    print("STORY 2.1: INFORMATION FLOW SCORES")
    print("=" * 80)

    df_flow = df.sort_values('flow_score', ascending=False).copy()

    print(f"\nTop 10 heads by forward flow:")
    for i, row in df_flow.head(10).iterrows():
        print(f"  {i+1:2d}. L{row['layer']:2d}H{row['head']:2d}: "
              f"flow={row['flow_score']:.3f} ({row['layer_type']})")

    flow_path = results_dir / 'heads_ranked_by_flow.csv'
    df_flow[['layer', 'head', 'flow_score', 'layer_type']].to_csv(flow_path, index=False)
    print(f"\n✓ Saved: {flow_path}")

    # Story 2.2: Rank by hub score
    print("\n" + "=" * 80)
    print("STORY 2.2: HUB CONNECTIVITY SCORES")
    print("=" * 80)

    df_hub = df.sort_values('hub_score', ascending=False).copy()

    print(f"\nTop 10 heads by hub connectivity:")
    for i, row in df_hub.head(10).iterrows():
        print(f"  {i+1:2d}. L{row['layer']:2d}H{row['head']:2d}: "
              f"hub={row['hub_score']:.3f} (pos {row['hub_position']})")

    hub_path = results_dir / 'heads_ranked_by_hub.csv'
    df_hub[['layer', 'head', 'hub_score', 'hub_position', 'layer_type']].to_csv(hub_path, index=False)
    print(f"\n✓ Saved: {hub_path}")

    # Story 2.3: Rank by skip score
    print("\n" + "=" * 80)
    print("STORY 2.3: SKIP CONNECTION SCORES")
    print("=" * 80)

    df_skip = df.sort_values('skip_score', ascending=False).copy()

    print(f"\nTop 10 heads by skip connections:")
    for i, row in df_skip.head(10).iterrows():
        print(f"  {i+1:2d}. L{row['layer']:2d}H{row['head']:2d}: "
              f"skip={row['skip_score']:.3f}")

    skip_path = results_dir / 'heads_ranked_by_skip.csv'
    df_skip[['layer', 'head', 'skip_score', 'layer_type']].to_csv(skip_path, index=False)
    print(f"\n✓ Saved: {skip_path}")

    # Story 2.4: Composite ranking
    print("\n" + "=" * 80)
    print("STORY 2.4: COMPOSITE RANKING")
    print("=" * 80)

    # Normalize scores to 0-1 range
    df['flow_norm'] = (df['flow_score'] - df['flow_score'].min()) / (df['flow_score'].max() - df['flow_score'].min() + 1e-10)
    df['hub_norm'] = (df['hub_score'] - df['hub_score'].min()) / (df['hub_score'].max() - df['hub_score'].min() + 1e-10)
    df['skip_norm'] = (df['skip_score'] - df['skip_score'].min()) / (df['skip_score'].max() - df['skip_score'].min() + 1e-10)

    # Composite score: weighted average
    df['composite_score'] = (
        0.4 * df['flow_norm'] +
        0.4 * df['hub_norm'] +
        0.2 * df['skip_norm']
    )

    # Assign functional types
    def assign_functional_type(row):
        scores = [
            ('Forward Flow', row['flow_norm']),
            ('Hub Aggregator', row['hub_norm']),
            ('Skip Connection', row['skip_norm'])
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Multi-purpose if top 2 scores are close
        if scores[0][1] > 0.7 and scores[1][1] > 0.7:
            return 'Multi-Purpose'
        return scores[0][0]

    df['functional_type'] = df.apply(assign_functional_type, axis=1)

    # Sort by composite score
    df = df.sort_values('composite_score', ascending=False)

    print(f"\nTop 10 critical heads by composite score:")
    for i, row in df.head(10).iterrows():
        print(f"  {i+1:2d}. L{row['layer']:2d}H{row['head']:2d}: "
              f"composite={row['composite_score']:.3f} ({row['functional_type']})")
        print(f"      flow={row['flow_score']:.3f}, "
              f"hub={row['hub_score']:.3f}, "
              f"skip={row['skip_score']:.3f}")

    # Save master ranking
    ranked_path = results_dir / 'ranked_heads.csv'
    df[['layer', 'head', 'layer_type', 'flow_score', 'hub_score', 'skip_score',
        'composite_score', 'functional_type', 'max_attention']].to_csv(ranked_path, index=False)
    print(f"\n✓ Saved master ranking: {ranked_path}")

    # Generate findings text
    findings = []
    findings.append("=" * 80)
    findings.append(f"{model.upper()} - CRITICAL HEADS FINDINGS")
    findings.append("=" * 80)
    findings.append("")

    # Top 3 critical heads
    findings.append("TOP 3 CRITICAL HEADS:")
    findings.append("")
    for i, row in df.head(3).iterrows():
        findings.append(f"{i+1}. L{row['layer']}H{row['head']} - {row['functional_type']}")
        findings.append(f"   Composite score: {row['composite_score']:.3f}")
        findings.append(f"   - Flow: {row['flow_score']:.3f} (forward information flow)")
        findings.append(f"   - Hub: {row['hub_score']:.3f} (creates hub at position {row['hub_position']})")
        findings.append(f"   - Skip: {row['skip_score']:.3f} (long-range connections)")
        findings.append(f"   - Max attention: {row['max_attention']:.3f}")
        findings.append("")

    # Distribution analysis
    findings.append("FUNCTIONAL TYPE DISTRIBUTION (Top 20 heads):")
    type_counts = df.head(20)['functional_type'].value_counts()
    for ftype, count in type_counts.items():
        findings.append(f"  {ftype}: {count}")
    findings.append("")

    # Layer analysis
    findings.append("LAYER DISTRIBUTION (Top 20 heads):")
    layer_counts = df.head(20)['layer_type'].value_counts()
    for ltype, count in layer_counts.items():
        findings.append(f"  {ltype.capitalize()}: {count}")
    findings.append("")

    # Score statistics
    findings.append("SCORE STATISTICS (All heads):")
    findings.append(f"  Flow: mean={df['flow_score'].mean():.3f}, max={df['flow_score'].max():.3f}")
    findings.append(f"  Hub: mean={df['hub_score'].mean():.3f}, max={df['hub_score'].max():.3f}")
    findings.append(f"  Skip: mean={df['skip_score'].mean():.3f}, max={df['skip_score'].max():.3f}")
    findings.append("")

    # Critical threshold
    critical_heads = df[df['composite_score'] > 0.7]
    findings.append(f"CRITICAL HEADS (composite > 0.7): {len(critical_heads)}")
    if len(critical_heads) > 0:
        findings.append("  Layers: " + ", ".join([f"L{row['layer']}H{row['head']}" for _, row in critical_heads.iterrows()]))
    findings.append("")

    findings_text = "\n".join(findings)

    findings_path = results_dir / 'critical_heads_findings.txt'
    with open(findings_path, 'w') as f:
        f.write(findings_text)
    print(f"✓ Saved findings: {findings_path}")

    # Print findings
    print("\n" + findings_text)

    print("\n" + "=" * 80)
    print("STORIES 2.1-2.4 COMPLETE ✓")
    print("=" * 80)
    print(f"\nIdentified {len(critical_heads)} critical heads (composite > 0.7)")
    print(f"Top head: L{df.iloc[0]['layer']}H{df.iloc[0]['head']} "
          f"({df.iloc[0]['functional_type']}, score={df.iloc[0]['composite_score']:.3f})")
    print("\nNext step: Run Story 2.5 to visualize critical heads")
    print("  python 7_visualize_critical_heads.py")


def main():
    parser = argparse.ArgumentParser(description='Compute head metrics')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to analyze (llama or gpt2)')
    args = parser.parse_args()

    compute_all_metrics(model=args.model)


if __name__ == '__main__':
    main()
