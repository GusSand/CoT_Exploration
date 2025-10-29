#!/usr/bin/env python3
"""
CT Hidden State Divergence Analysis

Research Question: When CT0 attention is blocked, at which step does the reasoning diverge?

Key Question from User:
"Does only the final answer change, or do the preliminary decodings of the continuous
chain of thought positions change?"

Method:
1. Load baseline and CT0-blocked hidden states for CT0-CT5
2. Compute cosine similarity between baseline and blocked states at each step
3. Track divergence patterns:
   - Does divergence start immediately (CT1)?
   - Does divergence accumulate (CT1→CT2→...→CT5)?
   - Is divergence constant or increasing?

Expected Findings:
- Scenario A (Late Divergence): High similarity CT1-CT5, divergence only at answer
  → CT0 only matters for final answer generation
- Scenario B (Early Divergence): Low similarity starting at CT1, increasing divergence
  → CT0 influences entire reasoning chain
- Scenario C (Gradual Divergence): Similarity decreases step-by-step
  → Cascading effect through reasoning chain

Time: 30 minutes
"""

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy.spatial.distance import cosine

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / 'results' / 'attention_data'
OUTPUT_DIR = SCRIPT_DIR.parent / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata():
    """Load metadata with problem info and correctness."""
    metadata_file = DATA_DIR / 'llama_metadata_final.json'
    with open(metadata_file) as f:
        return json.load(f)


def load_ct_hidden_states(problem_id: int, condition: str) -> Dict[int, np.ndarray]:
    """
    Load hidden states for all CT tokens.

    Returns:
        Dict mapping step -> hidden_states [n_layers, hidden_dim]
    """
    h5_file = DATA_DIR / f'llama_problem_{problem_id:04d}_{condition}_hidden.h5'

    if not h5_file.exists():
        return None

    hidden_by_step = {}

    with h5py.File(h5_file, 'r') as f:
        # Iterate through steps (0-5 for 6 CT tokens)
        for step in range(6):
            step_key = f'step_{step}'
            if step_key not in f:
                continue

            # Collect hidden states from all layers
            layer_hiddens = []
            layer_idx = 0
            while f'{step_key}/layer_{layer_idx}/hidden_state' in f:
                hidden = f[f'{step_key}/layer_{layer_idx}/hidden_state'][:]
                # Shape: [batch=1, seq_len=1, hidden_dim=2048]
                # Squeeze to get [hidden_dim]
                hidden = hidden.squeeze()
                layer_hiddens.append(hidden)
                layer_idx += 1

            if layer_hiddens:
                # Stack layers: [n_layers, hidden_dim]
                hidden_by_step[step] = np.stack(layer_hiddens, axis=0)

    return hidden_by_step


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns:
        Similarity in [0, 1] where 1 = identical, 0 = orthogonal
    """
    # Flatten if needed
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    # Cosine similarity = 1 - cosine distance
    return 1.0 - cosine(vec1, vec2)


def compute_l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute L2 (Euclidean) distance between two vectors."""
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    return np.linalg.norm(vec1 - vec2)


def analyze_ct_divergence(
    baseline_hidden: Dict[int, np.ndarray],
    ct0blocked_hidden: Dict[int, np.ndarray]
) -> Dict:
    """
    Analyze divergence between baseline and CT0-blocked hidden states.

    Returns:
        Dict with divergence metrics per step and layer
    """
    if not baseline_hidden or not ct0blocked_hidden:
        return None

    divergence = {}

    for step in range(6):
        if step not in baseline_hidden or step not in ct0blocked_hidden:
            continue

        baseline_h = baseline_hidden[step]  # [n_layers, hidden_dim]
        blocked_h = ct0blocked_hidden[step]  # [n_layers, hidden_dim]

        n_layers = baseline_h.shape[0]

        # Per-layer metrics
        layer_similarities = []
        layer_l2_distances = []

        for layer in range(n_layers):
            sim = compute_cosine_similarity(baseline_h[layer], blocked_h[layer])
            l2_dist = compute_l2_distance(baseline_h[layer], blocked_h[layer])

            layer_similarities.append(sim)
            layer_l2_distances.append(l2_dist)

        # Aggregate metrics
        divergence[step] = {
            'cosine_similarity_mean': float(np.mean(layer_similarities)),
            'cosine_similarity_std': float(np.std(layer_similarities)),
            'cosine_similarity_min': float(np.min(layer_similarities)),
            'cosine_similarity_max': float(np.max(layer_similarities)),
            'l2_distance_mean': float(np.mean(layer_l2_distances)),
            'l2_distance_std': float(np.std(layer_l2_distances)),
            'layer_similarities': layer_similarities,
            'layer_l2_distances': layer_l2_distances
        }

    return divergence


def run_ct_divergence_analysis(n_problems: int = 100):
    """
    Main analysis: Track CT hidden state divergence when CT0 is blocked.
    """
    print(f"\n{'='*60}")
    print(f"CT Hidden State Divergence Analysis")
    print(f"{'='*60}\n")

    print("Research Question:")
    print("  When CT0 attention is blocked, at which step does reasoning diverge?")
    print("  - Do intermediate CT tokens (CT1-CT5) change?")
    print("  - Or only the final answer?\n")

    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata()
    problems_to_analyze = metadata[:n_problems]
    print(f"Analyzing {len(problems_to_analyze)} problems")

    # Storage for aggregated results
    all_results = []
    aggregate_by_step = defaultdict(lambda: {
        'similarities': [],
        'l2_distances': []
    })

    # Stratify by impact
    by_impact = defaultdict(lambda: defaultdict(lambda: {
        'similarities': [],
        'l2_distances': []
    }))

    # Analyze each problem
    print("\nAnalyzing CT hidden state divergence...\n")

    n_success = 0
    n_failed = 0

    for problem in tqdm(problems_to_analyze, desc="Processing problems"):
        problem_id = problem['problem_id']

        # Load baseline and CT0-blocked hidden states
        baseline_hidden = load_ct_hidden_states(problem_id, 'baseline')
        ct0blocked_hidden = load_ct_hidden_states(problem_id, 'ct0blocked')

        if baseline_hidden is None or ct0blocked_hidden is None:
            n_failed += 1
            continue

        # Compute divergence
        divergence = analyze_ct_divergence(baseline_hidden, ct0blocked_hidden)

        if not divergence:
            n_failed += 1
            continue

        # Store results
        problem_result = {
            'problem_id': problem_id,
            'baseline_correct': problem.get('baseline', {}).get('correct', False),
            'ct0_blocked_correct': problem.get('ct0_blocked', {}).get('correct', False),
            'impact': problem.get('impact', 'unknown'),
            'divergence': divergence
        }

        all_results.append(problem_result)

        # Aggregate
        for step, metrics in divergence.items():
            aggregate_by_step[step]['similarities'].append(metrics['cosine_similarity_mean'])
            aggregate_by_step[step]['l2_distances'].append(metrics['l2_distance_mean'])

            # By impact
            impact = problem.get('impact', 'unknown')
            by_impact[impact][step]['similarities'].append(metrics['cosine_similarity_mean'])
            by_impact[impact][step]['l2_distances'].append(metrics['l2_distance_mean'])

        n_success += 1

    print(f"\n✓ Successfully analyzed {n_success} problems")
    print(f"✗ Failed to load {n_failed} problems")

    # Compute aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE DIVERGENCE PATTERNS")
    print("="*60 + "\n")

    aggregate_summary = {}

    for step in sorted(aggregate_by_step.keys()):
        data = aggregate_by_step[step]

        summary = {
            'cosine_similarity_mean': float(np.mean(data['similarities'])),
            'cosine_similarity_std': float(np.std(data['similarities'])),
            'cosine_similarity_min': float(np.min(data['similarities'])),
            'cosine_similarity_max': float(np.max(data['similarities'])),
            'l2_distance_mean': float(np.mean(data['l2_distances'])),
            'l2_distance_std': float(np.std(data['l2_distances'])),
            'n_samples': len(data['similarities'])
        }

        aggregate_summary[step] = summary

        # Interpret similarity
        sim = summary['cosine_similarity_mean']
        if sim > 0.95:
            interpretation = "Nearly Identical"
        elif sim > 0.85:
            interpretation = "Very Similar"
        elif sim > 0.70:
            interpretation = "Moderately Similar"
        elif sim > 0.50:
            interpretation = "Somewhat Similar"
        else:
            interpretation = "Diverged"

        print(f"CT{step} (Step {step}):")
        print(f"  Cosine Similarity: {sim:.4f} ± {summary['cosine_similarity_std']:.4f}  [{interpretation}]")
        print(f"  L2 Distance:       {summary['l2_distance_mean']:.2f} ± {summary['l2_distance_std']:.2f}")
        print()

    # Key insights
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60 + "\n")

    # Check divergence pattern
    similarities = [aggregate_summary[s]['cosine_similarity_mean'] for s in sorted(aggregate_summary.keys())]

    # Is CT1 already diverged?
    if len(similarities) > 1 and similarities[1] < 0.85:
        print("✓ EARLY DIVERGENCE DETECTED")
        print(f"  CT1 similarity: {similarities[1]:.4f} (< 0.85 threshold)")
        print("  → CT0 blocking affects reasoning from the FIRST step")
        print("  → Intermediate CT tokens DO change, not just final answer\n")
    elif len(similarities) > 1 and similarities[1] > 0.95:
        print("✓ LATE DIVERGENCE DETECTED")
        print(f"  CT1 similarity: {similarities[1]:.4f} (> 0.95 threshold)")
        print("  → CT0 blocking doesn't affect early reasoning")
        print("  → Divergence happens later in the chain\n")

    # Is divergence increasing?
    if len(similarities) > 2:
        divergence_trend = np.polyfit(range(len(similarities)), similarities, 1)[0]

        if divergence_trend < -0.02:
            print("✓ ACCUMULATING DIVERGENCE")
            print(f"  Similarity decreases by ~{abs(divergence_trend):.4f} per step")
            print("  → Divergence cascades through CT1→CT2→...→CT5")
            print("  → Each step amplifies the difference\n")
        elif divergence_trend > 0.02:
            print("✓ RECOVERING SIMILARITY")
            print(f"  Similarity increases by ~{divergence_trend:.4f} per step")
            print("  → Model compensates for CT0 blocking over time\n")
        else:
            print("✓ CONSTANT DIVERGENCE")
            print("  Similarity remains stable across steps")
            print("  → CT0 blocking causes fixed offset, no cascading\n")

    # Stratify by impact
    print("="*60)
    print("DIVERGENCE BY PROBLEM IMPACT")
    print("="*60 + "\n")

    for impact_type in ['degradation', 'no_change']:
        if impact_type not in by_impact:
            continue

        print(f"{impact_type.upper()}:")

        impact_similarities = []
        for step in sorted(by_impact[impact_type].keys()):
            sims = by_impact[impact_type][step]['similarities']
            if sims:
                mean_sim = np.mean(sims)
                impact_similarities.append(mean_sim)
                print(f"  CT{step}: {mean_sim:.4f} (n={len(sims)})")

        if len(impact_similarities) > 1:
            print(f"  Average similarity: {np.mean(impact_similarities):.4f}")
        print()

    # Helper to convert numpy types
    def convert_to_python(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(item) for item in obj]
        else:
            return obj

    # Save results
    output_file = OUTPUT_DIR / 'ct_hidden_state_divergence.json'
    results_dict = {
        'summary': {
            'n_problems_analyzed': n_success,
            'n_problems_failed': n_failed,
            'aggregate_by_step': aggregate_summary,
            'divergence_pattern': {
                'similarities_by_step': [aggregate_summary[s]['cosine_similarity_mean']
                                        for s in sorted(aggregate_summary.keys())],
                'trend_slope': float(divergence_trend) if len(similarities) > 2 else None
            }
        },
        'by_impact': {
            impact_type: {
                str(step): {
                    'cosine_similarity_mean': float(np.mean(data['similarities'])),
                    'l2_distance_mean': float(np.mean(data['l2_distances'])),
                    'n_samples': len(data['similarities'])
                }
                for step, data in impact_data.items()
            }
            for impact_type, impact_data in by_impact.items()
        },
        'problem_level_results': all_results[:20]  # Save first 20
    }

    # Convert all numpy types to Python native types
    results_dict = convert_to_python(results_dict)

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return aggregate_summary, by_impact, all_results


def create_visualizations(aggregate_summary: Dict, by_impact: Dict):
    """Create visualizations of CT divergence patterns."""
    print("\nCreating visualizations...")

    steps = sorted(aggregate_summary.keys())

    # Extract data
    similarities = [aggregate_summary[s]['cosine_similarity_mean'] for s in steps]
    similarities_std = [aggregate_summary[s]['cosine_similarity_std'] for s in steps]
    l2_distances = [aggregate_summary[s]['l2_distance_mean'] for s in steps]
    l2_distances_std = [aggregate_summary[s]['l2_distance_std'] for s in steps]

    # Figure 1: Cosine Similarity Across Steps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CT Hidden State Divergence: Baseline vs CT0-Blocked',
                 fontsize=14, fontweight='bold')

    # Cosine similarity
    ax = axes[0]
    x = np.arange(len(steps))

    ax.plot(x, similarities, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(x,
                     np.array(similarities) - np.array(similarities_std),
                     np.array(similarities) + np.array(similarities_std),
                     alpha=0.3, color='steelblue')

    # Add threshold lines
    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Nearly Identical (0.95)')
    ax.axhline(y=0.85, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Very Similar (0.85)')
    ax.axhline(y=0.70, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Diverged (<0.70)')

    ax.set_xlabel('CT Token', fontweight='bold', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontweight='bold', fontsize=12)
    ax.set_title('Hidden State Similarity Across CT Tokens', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'CT{s}' for s in steps])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # L2 distance
    ax = axes[1]

    ax.plot(x, l2_distances, marker='s', linewidth=2, markersize=8, color='coral')
    ax.fill_between(x,
                     np.array(l2_distances) - np.array(l2_distances_std),
                     np.array(l2_distances) + np.array(l2_distances_std),
                     alpha=0.3, color='coral')

    ax.set_xlabel('CT Token', fontweight='bold', fontsize=12)
    ax.set_ylabel('L2 Distance', fontweight='bold', fontsize=12)
    ax.set_title('Hidden State L2 Distance Across CT Tokens', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'CT{s}' for s in steps])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_file = FIGURES_DIR / 'ct_hidden_state_divergence.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

    # Figure 2: Divergence by Impact
    if 'degradation' in by_impact and 'no_change' in by_impact:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('CT Divergence by Problem Impact Type',
                     fontsize=14, fontweight='bold')

        for impact_type, color in [('degradation', 'red'), ('no_change', 'gray')]:
            if impact_type not in by_impact:
                continue

            impact_steps = sorted(by_impact[impact_type].keys())
            impact_sims = [np.mean(by_impact[impact_type][s]['similarities'])
                          for s in impact_steps]

            ax.plot(impact_steps, impact_sims, marker='o', linewidth=2,
                   markersize=8, label=impact_type.replace('_', ' ').title(),
                   color=color)

        ax.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.3)
        ax.axhline(y=0.85, color='orange', linestyle='--', linewidth=1, alpha=0.3)

        ax.set_xlabel('CT Token', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontweight='bold', fontsize=12)
        ax.set_title('Do Problems with Degradation Show More Divergence?', fontweight='bold')
        ax.set_xticks(impact_steps)
        ax.set_xticklabels([f'CT{s}' for s in impact_steps])
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig_file = FIGURES_DIR / 'ct_divergence_by_impact.png'
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {fig_file}")
        plt.close()

    print("✓ Visualizations complete\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze CT hidden state divergence')
    parser.add_argument('--n_problems', type=int, default=100,
                       help='Number of problems to analyze')
    args = parser.parse_args()

    # Run analysis
    aggregate_summary, by_impact, all_results = run_ct_divergence_analysis(n_problems=args.n_problems)

    # Create visualizations
    create_visualizations(aggregate_summary, by_impact)


if __name__ == '__main__':
    main()
