"""
Compare Similarity Metrics: CODI vs Vanilla (4 Models)
======================================================

Compares geometric similarity metrics across 4 models:
1. CODI-GPT2 continuous thoughts
2. CODI-Llama continuous thoughts
3. Vanilla GPT-2 hidden states (control)
4. Vanilla Llama hidden states (control)

Generates comparison plots and statistical analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_codi_results(results_dir: Path) -> Dict:
    """Load CODI extended analysis results"""
    summary_path = results_dir / "summary_statistics.json"
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    return summary


def load_vanilla_results(results_dir: Path) -> Dict:
    """Load vanilla control results"""
    summary_path = results_dir / "summary.json"
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    return summary


def extract_codi_metrics(codi_summary: Dict) -> Tuple[float, float, float, float, float]:
    """Extract key metrics from CODI summary"""
    accuracy = codi_summary['overall_results']['accuracy']
    similarity_analysis = codi_summary['similarity_analysis']
    avg_cosine = similarity_analysis['avg_cosine_sim_top1']
    avg_l2 = similarity_analysis['avg_norm_l2_dist_top1']
    std_cosine = similarity_analysis.get('std_cosine_sim_top1', similarity_analysis.get('std_cosine_sim_all', 0.0))
    std_l2 = similarity_analysis.get('std_norm_l2_dist_top1', similarity_analysis.get('std_norm_l2_dist_all', 0.0))
    return accuracy, avg_cosine, std_cosine, avg_l2, std_l2


def extract_vanilla_metrics(vanilla_summary: Dict) -> Tuple[float, float, float, float, float]:
    """Extract metrics from vanilla control"""
    accuracy = vanilla_summary['accuracy']
    metrics = vanilla_summary['similarity_metrics']['all_tokens']
    avg_cosine = metrics['avg_cosine_similarity']
    std_cosine = metrics['std_cosine_similarity']
    avg_l2 = metrics['avg_norm_l2_distance']
    std_l2 = metrics['std_norm_l2_distance']
    return accuracy, avg_cosine, std_cosine, avg_l2, std_l2


def create_4model_comparison_plots(
    codi_gpt2_metrics: Tuple,
    codi_llama_metrics: Tuple,
    vanilla_gpt2_metrics: Tuple,
    vanilla_llama_metrics: Tuple,
    output_dir: Path
):
    """Create comprehensive 4-model comparison plots"""

    # Unpack metrics
    gpt2_codi_acc, gpt2_codi_cos, gpt2_codi_cos_std, gpt2_codi_l2, gpt2_codi_l2_std = codi_gpt2_metrics
    llama_codi_acc, llama_codi_cos, llama_codi_cos_std, llama_codi_l2, llama_codi_l2_std = codi_llama_metrics
    gpt2_van_acc, gpt2_van_cos, gpt2_van_cos_std, gpt2_van_l2, gpt2_van_l2_std = vanilla_gpt2_metrics
    llama_van_acc, llama_van_cos, llama_van_cos_std, llama_van_l2, llama_van_l2_std = vanilla_llama_metrics

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    models = ['CODI\nGPT-2', 'CODI\nLlama', 'Vanilla\nGPT-2', 'Vanilla\nLlama']
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#f39c12']

    # Plot 1: Accuracy
    accuracies = [gpt2_codi_acc * 100, llama_codi_acc * 100, gpt2_van_acc * 100, llama_van_acc * 100]
    bars1 = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
    axes[0].set_ylim(0, max(accuracies) * 1.2)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Plot 2: Cosine Similarity
    cosine_sims = [gpt2_codi_cos, llama_codi_cos, gpt2_van_cos, llama_van_cos]
    cosine_stds = [gpt2_codi_cos_std, llama_codi_cos_std, gpt2_van_cos_std, llama_van_cos_std]
    bars2 = axes[1].bar(models, cosine_sims, yerr=cosine_stds,
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                        capsize=5)
    axes[1].set_ylabel('Cosine Similarity', fontsize=14, fontweight='bold')
    axes[1].set_title('Avg Cosine Similarity\n(Hidden State vs Token Embedding)', fontsize=16, fontweight='bold')
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars2, cosine_sims, cosine_stds)):
        height = bar.get_height()
        y_pos = height + std if height >= 0 else height - std
        axes[1].text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{val:.3f}\n±{std:.3f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    # Plot 3: Normalized L2 Distance
    l2_dists = [gpt2_codi_l2, llama_codi_l2, gpt2_van_l2, llama_van_l2]
    l2_stds = [gpt2_codi_l2_std, llama_codi_l2_std, gpt2_van_l2_std, llama_van_l2_std]
    bars3 = axes[2].bar(models, l2_dists, yerr=l2_stds,
                        color=colors, alpha=0.7, edgecolor='black', linewidth=1.5,
                        capsize=5)
    axes[2].set_ylabel('Normalized L2 Distance', fontsize=14, fontweight='bold')
    axes[2].set_title('Avg Norm. L2 Distance\n(on Unit Sphere)', fontsize=16, fontweight='bold')
    axes[2].set_ylim(0, max(l2_dists) * 1.3)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val, std) in enumerate(zip(bars3, l2_dists, l2_stds)):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{val:.3f}\n±{std:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_4models.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_4models.pdf', bbox_inches='tight')
    print(f"Saved 4-model comparison plot to {output_dir / 'comparison_4models.png'}")
    plt.close()


def create_scatter_plot_4models(
    codi_gpt2_metrics: Tuple,
    codi_llama_metrics: Tuple,
    vanilla_gpt2_metrics: Tuple,
    vanilla_llama_metrics: Tuple,
    output_dir: Path
):
    """Create scatter plot: Cosine Similarity vs L2 Distance for 4 models"""

    gpt2_codi_acc, gpt2_codi_cos, gpt2_codi_cos_std, gpt2_codi_l2, gpt2_codi_l2_std = codi_gpt2_metrics
    llama_codi_acc, llama_codi_cos, llama_codi_cos_std, llama_codi_l2, llama_codi_l2_std = codi_llama_metrics
    gpt2_van_acc, gpt2_van_cos, gpt2_van_cos_std, gpt2_van_l2, gpt2_van_l2_std = vanilla_gpt2_metrics
    llama_van_acc, llama_van_cos, llama_van_cos_std, llama_van_l2, llama_van_l2_std = vanilla_llama_metrics

    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot points with different markers
    ax.scatter(gpt2_codi_cos, gpt2_codi_l2, s=350, c='#3498db', alpha=0.8,
              edgecolors='black', linewidth=2, label='CODI-GPT2', marker='o')
    ax.scatter(llama_codi_cos, llama_codi_l2, s=350, c='#e74c3c', alpha=0.8,
              edgecolors='black', linewidth=2, label='CODI-Llama', marker='s')
    ax.scatter(gpt2_van_cos, gpt2_van_l2, s=350, c='#95a5a6', alpha=0.8,
              edgecolors='black', linewidth=2, label='Vanilla GPT-2', marker='^')
    ax.scatter(llama_van_cos, llama_van_l2, s=350, c='#f39c12', alpha=0.8,
              edgecolors='black', linewidth=2, label='Vanilla Llama', marker='D')

    # Add error bars
    ax.errorbar(gpt2_codi_cos, gpt2_codi_l2, xerr=gpt2_codi_cos_std, yerr=gpt2_codi_l2_std,
               fmt='none', ecolor='#3498db', capsize=5, alpha=0.5)
    ax.errorbar(llama_codi_cos, llama_codi_l2, xerr=llama_codi_cos_std, yerr=llama_codi_l2_std,
               fmt='none', ecolor='#e74c3c', capsize=5, alpha=0.5)
    ax.errorbar(gpt2_van_cos, gpt2_van_l2, xerr=gpt2_van_cos_std, yerr=gpt2_van_l2_std,
               fmt='none', ecolor='#95a5a6', capsize=5, alpha=0.5)
    ax.errorbar(llama_van_cos, llama_van_l2, xerr=llama_van_cos_std, yerr=llama_van_l2_std,
               fmt='none', ecolor='#f39c12', capsize=5, alpha=0.5)

    # Mathematical relationship curve: norm_l2 ≈ sqrt(2 - 2*cos_sim)
    cos_range = np.linspace(-1, 1, 100)
    l2_theoretical = np.sqrt(2 - 2 * cos_range)
    ax.plot(cos_range, l2_theoretical, 'k--', alpha=0.3, linewidth=2,
           label='Theoretical: L2 = √(2-2·cos)')

    ax.set_xlabel('Cosine Similarity', fontsize=16, fontweight='bold')
    ax.set_ylabel('Normalized L2 Distance', fontsize=16, fontweight='bold')
    ax.set_title('Geometric Similarity: Cosine vs L2 Distance (4 Models)', fontsize=18, fontweight='bold')
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_4models.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'scatter_4models.pdf', bbox_inches='tight')
    print(f"Saved scatter plot to {output_dir / 'scatter_4models.png'}")
    plt.close()


def generate_text_report_4models(
    codi_gpt2_metrics: Tuple,
    codi_llama_metrics: Tuple,
    vanilla_gpt2_metrics: Tuple,
    vanilla_llama_metrics: Tuple,
    output_path: Path
):
    """Generate detailed text comparison report for 4 models"""

    gpt2_codi_acc, gpt2_codi_cos, gpt2_codi_cos_std, gpt2_codi_l2, gpt2_codi_l2_std = codi_gpt2_metrics
    llama_codi_acc, llama_codi_cos, llama_codi_cos_std, llama_codi_l2, llama_codi_l2_std = codi_llama_metrics
    gpt2_van_acc, gpt2_van_cos, gpt2_van_cos_std, gpt2_van_l2, gpt2_van_l2_std = vanilla_gpt2_metrics
    llama_van_acc, llama_van_cos, llama_van_cos_std, llama_van_l2, llama_van_l2_std = vanilla_llama_metrics

    report = []
    report.append("=" * 80)
    report.append("CODI vs Vanilla (4 Models): Geometric Similarity Analysis")
    report.append("=" * 80)
    report.append("")

    report.append("## Overview")
    report.append("")
    report.append("Comparison of geometric similarity metrics across 4 models:")
    report.append("1. CODI-GPT2: Continuous thoughts (fine-tuned)")
    report.append("2. CODI-Llama: Continuous thoughts (fine-tuned)")
    report.append("3. Vanilla GPT-2: Standard hidden states (no CODI)")
    report.append("4. Vanilla Llama: Standard hidden states (no CODI)")
    report.append("")

    report.append("=" * 80)
    report.append("RESULTS SUMMARY")
    report.append("=" * 80)
    report.append("")

    # Accuracy
    report.append("### Task Accuracy")
    report.append("")
    report.append(f"  CODI-GPT2:        {gpt2_codi_acc*100:6.1f}%")
    report.append(f"  CODI-Llama:       {llama_codi_acc*100:6.1f}%")
    report.append(f"  Vanilla GPT-2:    {gpt2_van_acc*100:6.1f}%")
    report.append(f"  Vanilla Llama:    {llama_van_acc*100:6.1f}%")
    report.append("")

    # Cosine Similarity
    report.append("### Cosine Similarity (Top-1 Token / Generated Token)")
    report.append("")
    report.append(f"  CODI-GPT2:        {gpt2_codi_cos:7.4f} (±{gpt2_codi_cos_std:.4f})")
    report.append(f"  CODI-Llama:       {llama_codi_cos:7.4f} (±{llama_codi_cos_std:.4f})")
    report.append(f"  Vanilla GPT-2:    {gpt2_van_cos:7.4f} (±{gpt2_van_cos_std:.4f})")
    report.append(f"  Vanilla Llama:    {llama_van_cos:7.4f} (±{llama_van_cos_std:.4f})")
    report.append("")

    # Normalized L2 Distance
    report.append("### Normalized L2 Distance (Top-1 Token / Generated Token)")
    report.append("")
    report.append(f"  CODI-GPT2:        {gpt2_codi_l2:7.4f} (±{gpt2_codi_l2_std:.4f})")
    report.append(f"  CODI-Llama:       {llama_codi_l2:7.4f} (±{llama_codi_l2_std:.4f})")
    report.append(f"  Vanilla GPT-2:    {gpt2_van_l2:7.4f} (±{gpt2_van_l2_std:.4f})")
    report.append(f"  Vanilla Llama:    {llama_van_l2:7.4f} (±{llama_van_l2_std:.4f})")
    report.append("")

    report.append("=" * 80)
    report.append("KEY FINDINGS")
    report.append("=" * 80)
    report.append("")

    # Find best model for each metric
    cosine_values = [gpt2_codi_cos, llama_codi_cos, gpt2_van_cos, llama_van_cos]
    model_names = ['CODI-GPT2', 'CODI-Llama', 'Vanilla GPT-2', 'Vanilla Llama']
    best_cosine_idx = np.argmax(cosine_values)

    report.append(f"1. BEST GEOMETRIC ALIGNMENT: {model_names[best_cosine_idx]}")
    report.append("")
    report.append(f"   {model_names[best_cosine_idx]} achieves the highest cosine similarity: {cosine_values[best_cosine_idx]:.4f}")
    report.append("")

    # CODI vs Vanilla comparison
    report.append("2. CODI FINE-TUNING IMPACT")
    report.append("")
    report.append(f"   GPT-2: CODI ({gpt2_codi_cos:.4f}) vs Vanilla ({gpt2_van_cos:.4f})")
    gpt2_improvement = ((gpt2_codi_cos - gpt2_van_cos) / abs(gpt2_van_cos)) * 100 if gpt2_van_cos != 0 else 0
    report.append(f"          Improvement: {gpt2_improvement:+.1f}%")
    report.append("")
    report.append(f"   Llama: CODI ({llama_codi_cos:.4f}) vs Vanilla ({llama_van_cos:.4f})")
    llama_improvement = ((llama_codi_cos - llama_van_cos) / abs(llama_van_cos)) * 100 if llama_van_cos != 0 else 0
    report.append(f"          Improvement: {llama_improvement:+.1f}%")
    report.append("")

    # Model capacity
    report.append("3. MODEL CAPACITY")
    report.append("")
    report.append(f"   Llama models (1B params) vs GPT-2 (124M params):")
    report.append(f"   - CODI: Llama {llama_codi_cos:.4f} vs GPT-2 {gpt2_codi_cos:.4f}")
    report.append(f"   - Vanilla: Llama {llama_van_cos:.4f} vs GPT-2 {gpt2_van_cos:.4f}")
    report.append("")

    # Vanilla comparison
    vanilla_gpt2_neg = gpt2_van_cos < 0
    vanilla_llama_neg = llama_van_cos < 0
    report.append("4. VANILLA MODEL BEHAVIOR")
    report.append("")
    if vanilla_gpt2_neg:
        report.append(f"   Vanilla GPT-2 shows NEGATIVE similarity ({gpt2_van_cos:.4f})")
        report.append("   → Hidden states point away from token embeddings")
    if vanilla_llama_neg:
        report.append(f"   Vanilla Llama shows NEGATIVE similarity ({llama_van_cos:.4f})")
        report.append("   → Hidden states point away from token embeddings")
    report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"Saved text report to {output_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description="Compare 4 models similarity metrics")
    parser.add_argument("--codi_gpt2_results", type=str, required=True)
    parser.add_argument("--codi_llama_results", type=str, required=True)
    parser.add_argument("--vanilla_gpt2_results", type=str, required=True)
    parser.add_argument("--vanilla_llama_results", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/comparison_4models")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    print(f"  CODI-GPT2: {args.codi_gpt2_results}")
    print(f"  CODI-Llama: {args.codi_llama_results}")
    print(f"  Vanilla GPT-2: {args.vanilla_gpt2_results}")
    print(f"  Vanilla Llama: {args.vanilla_llama_results}")
    print()

    # Load all results
    codi_gpt2_summary = load_codi_results(Path(args.codi_gpt2_results))
    codi_llama_summary = load_codi_results(Path(args.codi_llama_results))
    vanilla_gpt2_summary = load_vanilla_results(Path(args.vanilla_gpt2_results))
    vanilla_llama_summary = load_vanilla_results(Path(args.vanilla_llama_results))

    # Extract metrics
    codi_gpt2_metrics = extract_codi_metrics(codi_gpt2_summary)
    codi_llama_metrics = extract_codi_metrics(codi_llama_summary)
    vanilla_gpt2_metrics = extract_vanilla_metrics(vanilla_gpt2_summary)
    vanilla_llama_metrics = extract_vanilla_metrics(vanilla_llama_summary)

    print("Creating visualizations...")

    # Create plots
    create_4model_comparison_plots(
        codi_gpt2_metrics, codi_llama_metrics,
        vanilla_gpt2_metrics, vanilla_llama_metrics,
        output_dir
    )

    create_scatter_plot_4models(
        codi_gpt2_metrics, codi_llama_metrics,
        vanilla_gpt2_metrics, vanilla_llama_metrics,
        output_dir
    )

    # Generate text report
    report_text = generate_text_report_4models(
        codi_gpt2_metrics, codi_llama_metrics,
        vanilla_gpt2_metrics, vanilla_llama_metrics,
        output_dir / "comparison_report_4models.txt"
    )

    print("\n" + "=" * 80)
    print("4-MODEL COMPARISON COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - comparison_4models.png")
    print(f"  - scatter_4models.png")
    print(f"  - comparison_report_4models.txt")
    print()

    # Print key findings
    print(report_text)


if __name__ == "__main__":
    main()
