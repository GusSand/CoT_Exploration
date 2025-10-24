"""
Story 2.1: Multi-Position Probe Analysis

Train probes across multiple layers and token positions to understand
where correctness information is strongest.

Trains 18 probes: Layers 8, 14, 15 × Tokens 0-5
Generates heatmap visualization and statistical analysis.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Import probe trainer
import importlib.util
spec = importlib.util.spec_from_file_location("probe_trainer", Path(__file__).parent / "2_probe_trainer.py")
probe_trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe_trainer)
ProbeTrainer = probe_trainer.ProbeTrainer


def create_heatmap(results, output_path: str):
    """
    Create heatmap of probe accuracies.

    Args:
        results: List of ProbeResults
        output_path: Path to save figure
    """
    # Extract layers and tokens
    layers = sorted(set(r.layer for r in results))
    tokens = sorted(set(r.token for r in results))

    # Create accuracy matrix
    accuracy_matrix = np.zeros((len(layers), len(tokens)))

    for r in results:
        layer_idx = layers.index(r.layer)
        token_idx = tokens.index(r.token)
        accuracy_matrix[layer_idx, token_idx] = r.accuracy

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 4))

    sns.heatmap(
        accuracy_matrix,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        xticklabels=[f'Token {t}' for t in tokens],
        yticklabels=[f'Layer {l}' for l in layers],
        cbar_kws={'label': 'Probe Accuracy'},
        ax=ax
    )

    ax.set_title('Linear Probe Accuracy by Layer and Token Position', fontsize=14, fontweight='bold')
    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to {output_path}")

    return accuracy_matrix, layers, tokens


def analyze_distribution(results, layers, tokens):
    """
    Analyze information distribution across positions.

    Tests:
    1. Is information evenly distributed across tokens?
    2. Which layer has strongest signal?
    3. Which token position is most informative?
    """
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    # Group by layer
    for layer in layers:
        layer_results = [r for r in results if r.layer == layer]
        accuracies = [r.accuracy for r in layer_results]

        print(f"\nLayer {layer}:")
        print(f"  Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Min: {np.min(accuracies):.4f} (Token {layer_results[np.argmin(accuracies)].token})")
        print(f"  Max: {np.max(accuracies):.4f} (Token {layer_results[np.argmax(accuracies)].token})")
        print(f"  Range: {np.max(accuracies) - np.min(accuracies):.4f}")

        # Test uniformity (chi-squared test for equal performance)
        # H0: All tokens have equal accuracy
        variance = np.var(accuracies)
        print(f"  Variance: {variance:.6f}")

        if variance < 0.001:
            print(f"  → Information is EVENLY distributed (low variance)")
        else:
            print(f"  → Information is CONCENTRATED (high variance)")

    # Group by token
    print("\n" + "-"*60)
    print("BY TOKEN POSITION:")
    print("-"*60)

    for token in tokens:
        token_results = [r for r in results if r.token == token]
        accuracies = [r.accuracy for r in token_results]

        print(f"\nToken {token}:")
        print(f"  Mean accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"  Across layers: {[f'{acc:.3f}' for acc in accuracies]}")

    # Overall statistics
    print("\n" + "-"*60)
    print("OVERALL:")
    print("-"*60)

    all_accuracies = [r.accuracy for r in results]
    print(f"  Overall mean: {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
    print(f"  Overall range: [{np.min(all_accuracies):.4f}, {np.max(all_accuracies):.4f}]")

    # Best probe
    best = max(results, key=lambda r: r.accuracy)
    print(f"\n  Best probe: Layer {best.layer}, Token {best.token}")
    print(f"    Accuracy: {best.accuracy:.4f} [{best.accuracy_ci_lower:.4f}, {best.accuracy_ci_upper:.4f}]")

    # Worst probe
    worst = min(results, key=lambda r: r.accuracy)
    print(f"\n  Worst probe: Layer {worst.layer}, Token {worst.token}")
    print(f"    Accuracy: {worst.accuracy:.4f} [{worst.accuracy_ci_lower:.4f}, {worst.accuracy_ci_upper:.4f}]")

    print("="*60)


def main():
    """Run multi-position probe analysis."""
    # Paths
    project_root = Path(__file__).parent.parent.parent.parent.parent
    dataset_path = project_root / "src/experiments/linear_probes/data/probe_dataset_100.json"
    results_dir = project_root / "src/experiments/linear_probes/results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = ProbeTrainer(str(dataset_path), use_wandb=True, project_name="linear-probes-gsm8k")

    # Train probes for all layer-token combinations
    layers = [8, 14, 15]
    tokens = list(range(6))  # 0-5

    print("\n" + "="*60)
    print("MULTI-POSITION PROBE ANALYSIS")
    print("="*60)
    print(f"Layers: {layers}")
    print(f"Tokens: {tokens}")
    print(f"Total probes: {len(layers) * len(tokens)}")
    print("="*60)

    results = trainer.train_sweep(
        layers=layers,
        tokens=tokens,
        sweep_name="multi_position"
    )

    # Save results
    results_path = results_dir / "multi_position_results.json"
    trainer.save_results(results, str(results_path))

    # Create heatmap
    heatmap_path = results_dir / "multi_position_heatmap.png"
    accuracy_matrix, layer_order, token_order = create_heatmap(results, str(heatmap_path))

    # Statistical analysis
    analyze_distribution(results, layer_order, token_order)

    # Save summary statistics
    summary = {
        'overall_mean': float(np.mean([r.accuracy for r in results])),
        'overall_std': float(np.std([r.accuracy for r in results])),
        'by_layer': {},
        'by_token': {},
        'best_probe': {
            'layer': max(results, key=lambda r: r.accuracy).layer,
            'token': max(results, key=lambda r: r.accuracy).token,
            'accuracy': float(max(results, key=lambda r: r.accuracy).accuracy)
        },
        'worst_probe': {
            'layer': min(results, key=lambda r: r.accuracy).layer,
            'token': min(results, key=lambda r: r.accuracy).token,
            'accuracy': float(min(results, key=lambda r: r.accuracy).accuracy)
        }
    }

    for layer in layer_order:
        layer_accs = [r.accuracy for r in results if r.layer == layer]
        summary['by_layer'][f'layer_{layer}'] = {
            'mean': float(np.mean(layer_accs)),
            'std': float(np.std(layer_accs)),
            'min': float(np.min(layer_accs)),
            'max': float(np.max(layer_accs))
        }

    for token in token_order:
        token_accs = [r.accuracy for r in results if r.token == token]
        summary['by_token'][f'token_{token}'] = {
            'mean': float(np.mean(token_accs)),
            'std': float(np.std(token_accs)),
            'min': float(np.min(token_accs)),
            'max': float(np.max(token_accs))
        }

    summary_path = results_dir / "multi_position_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_path}")

    print("\n" + "="*60)
    print("MULTI-POSITION ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results: {results_path}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Summary: {summary_path}")
    print("="*60)


if __name__ == "__main__":
    main()
