#!/usr/bin/env python3
"""
Generate Divergence Narratives: CT Hidden State Case Studies

Goal: Create detailed narratives showing HOW reasoning diverges step-by-step
      when CT0 attention is blocked.

For each case study:
1. Load hidden states for baseline and CT0-blocked
2. Compute per-layer divergence at each step
3. Create visualizations showing divergence trajectory
4. Generate narrative explaining what changes and when

Output:
- Individual case study markdown files with visualizations
- Combined summary document
- Saved to results/case_study_narratives/
"""

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from scipy.spatial.distance import cosine

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / 'results'
NARRATIVES_DIR = RESULTS_DIR / 'case_study_narratives'
NARRATIVES_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path('/home/paperspace/dev/CoT_Exploration/src/experiments/codi_attention_flow/ablation/results/attention_data')
SELECTED_CASES_FILE = RESULTS_DIR / 'selected_divergence_cases.json'


def load_selected_cases():
    """Load the selected case studies."""
    with open(SELECTED_CASES_FILE) as f:
        data = json.load(f)
    return data['cases']


def load_ct_hidden_states(problem_id: int, condition: str) -> Dict[int, np.ndarray]:
    """Load hidden states for all CT tokens."""
    h5_file = DATA_DIR / f'llama_problem_{problem_id:04d}_{condition}_hidden.h5'

    if not h5_file.exists():
        return None

    hidden_by_step = {}

    with h5py.File(h5_file, 'r') as f:
        for step in range(6):
            step_key = f'step_{step}'
            if step_key not in f:
                continue

            layer_hiddens = []
            layer_idx = 0
            while f'{step_key}/layer_{layer_idx}/hidden_state' in f:
                hidden = f[f'{step_key}/layer_{layer_idx}/hidden_state'][:]
                hidden = hidden.squeeze()
                layer_hiddens.append(hidden)
                layer_idx += 1

            if layer_hiddens:
                hidden_by_step[step] = np.stack(layer_hiddens, axis=0)

    return hidden_by_step


def compute_divergence_detailed(baseline_hidden: Dict, ct0blocked_hidden: Dict) -> Dict:
    """Compute detailed divergence metrics."""
    divergence = {}

    for step in range(6):
        if step not in baseline_hidden or step not in ct0blocked_hidden:
            continue

        baseline_h = baseline_hidden[step]  # [n_layers, hidden_dim]
        blocked_h = ct0blocked_hidden[step]

        n_layers = baseline_h.shape[0]

        # Per-layer metrics
        layer_similarities = []
        layer_l2_distances = []

        for layer in range(n_layers):
            vec1 = baseline_h[layer].flatten()
            vec2 = blocked_h[layer].flatten()

            sim = 1.0 - cosine(vec1, vec2)
            l2_dist = np.linalg.norm(vec1 - vec2)

            layer_similarities.append(sim)
            layer_l2_distances.append(l2_dist)

        divergence[step] = {
            'layer_similarities': layer_similarities,
            'layer_l2_distances': layer_l2_distances,
            'mean_similarity': float(np.mean(layer_similarities)),
            'std_similarity': float(np.std(layer_similarities)),
            'min_similarity': float(np.min(layer_similarities)),
            'max_similarity': float(np.max(layer_similarities)),
            'mean_l2': float(np.mean(layer_l2_distances)),
            'most_diverged_layer': int(np.argmin(layer_similarities)),
            'least_diverged_layer': int(np.argmax(layer_similarities))
        }

    return divergence


def create_divergence_visualization(case: Dict, divergence: Dict, output_file: Path):
    """Create visualization showing divergence trajectory."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Case Study: Problem {case["problem_id"]} - Hidden State Divergence Trajectory\n'
                 f'{case["selection_reason"]}',
                 fontsize=14, fontweight='bold')

    steps = sorted(divergence.keys())

    # Plot 1: Mean similarity across steps
    ax = axes[0, 0]
    similarities = [divergence[s]['mean_similarity'] for s in steps]
    stds = [divergence[s]['std_similarity'] for s in steps]

    ax.plot(steps, similarities, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.fill_between(steps,
                     np.array(similarities) - np.array(stds),
                     np.array(similarities) + np.array(stds),
                     alpha=0.3, color='steelblue')

    ax.axhline(y=0.95, color='green', linestyle='--', linewidth=1, alpha=0.3, label='Nearly Identical')
    ax.axhline(y=0.85, color='orange', linestyle='--', linewidth=1, alpha=0.3, label='Very Similar')
    ax.axhline(y=0.70, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Diverged')

    ax.set_xlabel('CT Step', fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontweight='bold')
    ax.set_title('Average Divergence Across Layers', fontweight='bold')
    ax.set_xticks(steps)
    ax.set_xticklabels([f'CT{s}' for s in steps])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Per-layer heatmap
    ax = axes[0, 1]

    # Create similarity matrix [steps x layers]
    n_layers = len(divergence[steps[0]]['layer_similarities'])
    sim_matrix = np.zeros((len(steps), n_layers))

    for i, step in enumerate(steps):
        sim_matrix[i, :] = divergence[step]['layer_similarities']

    sns.heatmap(sim_matrix.T, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=[f'CT{s}' for s in steps],
                yticklabels=[f'L{i}' for i in range(n_layers)],
                cbar_kws={'label': 'Cosine Similarity'})

    ax.set_xlabel('CT Step', fontweight='bold')
    ax.set_ylabel('Layer', fontweight='bold')
    ax.set_title('Per-Layer Divergence Heatmap', fontweight='bold')

    # Plot 3: L2 distance
    ax = axes[1, 0]

    l2_distances = [divergence[s]['mean_l2'] for s in steps]

    ax.plot(steps, l2_distances, marker='s', linewidth=2, markersize=8, color='coral')

    ax.set_xlabel('CT Step', fontweight='bold')
    ax.set_ylabel('L2 Distance', fontweight='bold')
    ax.set_title('L2 Distance Across Steps', fontweight='bold')
    ax.set_xticks(steps)
    ax.set_xticklabels([f'CT{s}' for s in steps])
    ax.grid(axis='y', alpha=0.3)

    # Plot 4: Divergence range (min/max across layers)
    ax = axes[1, 1]

    min_sims = [divergence[s]['min_similarity'] for s in steps]
    max_sims = [divergence[s]['max_similarity'] for s in steps]
    mean_sims = [divergence[s]['mean_similarity'] for s in steps]

    ax.plot(steps, mean_sims, marker='o', linewidth=2, markersize=6, color='steelblue', label='Mean')
    ax.fill_between(steps, min_sims, max_sims, alpha=0.3, color='steelblue', label='Min-Max Range')

    ax.axhline(y=0.85, color='orange', linestyle='--', linewidth=1, alpha=0.3)
    ax.axhline(y=0.70, color='red', linestyle='--', linewidth=1, alpha=0.3)

    ax.set_xlabel('CT Step', fontweight='bold')
    ax.set_ylabel('Cosine Similarity', fontweight='bold')
    ax.set_title('Layer-wise Divergence Range', fontweight='bold')
    ax.set_xticks(steps)
    ax.set_xticklabels([f'CT{s}' for s in steps])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def generate_narrative(case: Dict, divergence: Dict) -> str:
    """Generate detailed narrative for a case study."""
    narrative = f"""# Case Study: Problem {case['problem_id']}

**Selection Reason**: {case['selection_reason']}

**Impact Type**: {case['impact']} (Baseline: {'✓' if case['baseline_correct'] else '✗'}, CT0-blocked: {'✓' if case['ct0_blocked_correct'] else '✗'})

---

## Problem

**Question** (truncated):
```
{case['question'][:200]}...
```

**Gold Answer**: {case['gold_answer']}
**Baseline Prediction**: {case['baseline_answer']}
**CT0-Blocked Prediction**: {case['ct0_blocked_answer']}

---

## Divergence Profile

**Overall Metrics**:
- Total divergence: {case['total_divergence']:.3f}
- CT1 similarity: {case['ct1_similarity']:.3f} ({100 * (1 - case['ct1_similarity']):.1f}% diverged)
- CT4 similarity: {case['ct4_similarity']:.3f} ({100 * (1 - case['ct4_similarity']):.1f}% diverged)
- Divergence slope: {case['divergence_slope']:.3f} per step
- Pattern: {case['pattern']}

---

## Step-by-Step Divergence Analysis

"""

    steps = sorted(divergence.keys())

    for step in steps:
        metrics = divergence[step]
        sim = metrics['mean_similarity']
        divergence_pct = 100 * (1 - sim)

        # Interpretation
        if sim > 0.95:
            interp = "**Nearly identical** - no significant divergence"
        elif sim > 0.85:
            interp = "**Very similar** - minor divergence"
        elif sim > 0.70:
            interp = "**Moderately diverged** - noticeable differences"
        elif sim > 0.50:
            interp = "**Significantly diverged** - major differences"
        else:
            interp = "**Heavily diverged** - reasoning has fundamentally changed"

        narrative += f"""### CT{step} - Step {step}

**Similarity**: {sim:.3f} ({divergence_pct:.1f}% diverged)
**L2 Distance**: {metrics['mean_l2']:.2f}
**Interpretation**: {interp}

**Layer Analysis**:
- Most diverged layer: Layer {metrics['most_diverged_layer']} (similarity: {metrics['min_similarity']:.3f})
- Least diverged layer: Layer {metrics['least_diverged_layer']} (similarity: {metrics['max_similarity']:.3f})
- Layer variance: {metrics['std_similarity']:.3f}

"""

        if step == 0:
            narrative += "**Note**: CT0 is identical in both conditions (as expected - same generation process)\n\n"
        elif step == 1:
            if sim < 0.70:
                narrative += "**⚠️ IMMEDIATE DIVERGENCE**: CT1 shows significant divergence from the first step!\n\n"
            else:
                narrative += "**Stable start**: CT1 remains relatively similar despite CT0 blocking.\n\n"
        elif metrics['mean_similarity'] < divergence[step-1]['mean_similarity'] - 0.05:
            narrative += "**📉 CASCADING**: Divergence is accumulating from previous steps.\n\n"

    # Summary interpretation
    narrative += """---

## Interpretation

"""

    if case['pattern'] == 'early_divergence':
        narrative += """**Early Divergence Pattern**: This problem shows divergence immediately at CT1. The reasoning path
changes from the very first continuous thought token, suggesting CT0's encoded information is critical
for the initial reasoning step.

"""
    elif case['pattern'] == 'late_divergence':
        narrative += """**Late Divergence Pattern**: CT1 remains relatively stable, but later steps (CT3-CT4) show significant
divergence. This suggests the model can partially compensate initially, but the lack of CT0 information
causes problems as reasoning progresses.

"""
    elif case['pattern'] == 'cascading':
        narrative += """**Cascading Pattern**: Divergence increases steadily from step to step, with each CT token amplifying
the differences. This demonstrates how errors compound through the sequential reasoning chain.

"""
    else:
        narrative += """**Stable Pattern**: Despite CT0 blocking, the hidden states remain relatively stable. This may indicate
the model has alternative pathways to access needed information, or this particular problem is less
dependent on CT0's encoded information.

"""

    if case['impact'] == 'degradation':
        narrative += f"""**Impact on Answer**: Blocking CT0 caused the model to produce an **incorrect answer**
(baseline: {case['baseline_answer']}, blocked: {case['ct0_blocked_answer']}, gold: {case['gold_answer']}).
The hidden state divergence directly translated to reasoning failure.

"""
    else:
        narrative += f"""**Robustness**: Despite significant hidden state divergence, the model **still produced the correct answer**
in both conditions (answer: {case['baseline_answer']}). This suggests redundancy in the reasoning process
or that the specific diverged representations didn't affect the critical computation for this problem.

"""

    narrative += """---

## Key Takeaways

"""

    # Generate takeaways based on the specific case
    takeaways = []

    if case['ct1_similarity'] < 0.65:
        takeaways.append("- CT1 diverges **immediately** (< 65% similarity), confirming CT0's critical role from the first reasoning step")

    if case['divergence_slope'] < -0.04:
        takeaways.append(f"- Strong **cascading effect** (slope: {case['divergence_slope']:.3f}), showing how early divergence amplifies")

    if case['impact'] == 'no_change' and case['total_divergence'] > 0.4:
        takeaways.append("- **Resilient reasoning**: High divergence but correct answer demonstrates model robustness")

    if divergence[4]['std_similarity'] > 0.15:
        takeaways.append(f"- **Layer heterogeneity**: Different layers show varied divergence (std: {divergence[4]['std_similarity']:.3f}), suggesting specialized roles")

    most_diverged_layer = divergence[4]['most_diverged_layer']
    if most_diverged_layer in [0, 1, 2]:
        takeaways.append(f"- **Early layers most affected**: Layer {most_diverged_layer} shows maximum divergence, possibly related to input encoding")
    elif most_diverged_layer in [13, 14, 15]:
        takeaways.append(f"- **Late layers most affected**: Layer {most_diverged_layer} shows maximum divergence, possibly related to output preparation")

    for takeaway in takeaways:
        narrative += takeaway + "\n"

    return narrative


def main():
    print("="*60)
    print("Generating CT0 Divergence Case Study Narratives")
    print("="*60)
    print()

    # Load selected cases
    print("Loading selected cases...")
    cases = load_selected_cases()
    print(f"Loaded {len(cases)} case studies\n")

    # Generate narratives for each case
    all_narratives = []

    for i, case in enumerate(cases, 1):
        problem_id = case['problem_id']
        print(f"Processing Case {i}/{len(cases)}: Problem {problem_id}...")

        # Load hidden states
        baseline_hidden = load_ct_hidden_states(problem_id, 'baseline')
        ct0blocked_hidden = load_ct_hidden_states(problem_id, 'ct0blocked')

        if not baseline_hidden or not ct0blocked_hidden:
            print(f"  ✗ Failed to load hidden states, skipping")
            continue

        # Compute detailed divergence
        divergence = compute_divergence_detailed(baseline_hidden, ct0blocked_hidden)

        # Create visualization
        viz_file = NARRATIVES_DIR / f'case_{i:02d}_problem_{problem_id}_divergence.png'
        create_divergence_visualization(case, divergence, viz_file)
        print(f"  ✓ Created visualization: {viz_file.name}")

        # Generate narrative
        narrative = generate_narrative(case, divergence)

        # Save individual case study
        case_file = NARRATIVES_DIR / f'case_{i:02d}_problem_{problem_id}.md'
        with open(case_file, 'w') as f:
            f.write(narrative)
            f.write(f"\n## Visualization\n\n")
            f.write(f"![Divergence Trajectory](case_{i:02d}_problem_{problem_id}_divergence.png)\n")

        print(f"  ✓ Saved narrative: {case_file.name}\n")

        all_narratives.append({
            'case_number': i,
            'problem_id': problem_id,
            'narrative_file': case_file.name,
            'visualization_file': viz_file.name
        })

    # Create index document
    print("Creating index document...")
    index_file = NARRATIVES_DIR / 'README.md'
    with open(index_file, 'w') as f:
        f.write("# CT0 Hidden State Divergence - Case Studies\n\n")
        f.write("This directory contains detailed case studies showing how reasoning diverges ")
        f.write("step-by-step when CT0 attention is blocked.\n\n")
        f.write("## Case Studies\n\n")

        for item in all_narratives:
            f.write(f"### Case {item['case_number']}: Problem {item['problem_id']}\n")
            f.write(f"- Narrative: [{item['narrative_file']}]({item['narrative_file']})\n")
            f.write(f"- Visualization: [{item['visualization_file']}]({item['visualization_file']})\n\n")

    print(f"✓ Created index: {index_file}")
    print()
    print("="*60)
    print(f"✓ Generated {len(all_narratives)} case study narratives")
    print(f"  Location: {NARRATIVES_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
