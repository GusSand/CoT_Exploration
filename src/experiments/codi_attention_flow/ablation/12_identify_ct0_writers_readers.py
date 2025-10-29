#!/usr/bin/env python3
"""
CT0 Writers and Readers Analysis

Research Question: Who writes TO CT0 and who reads FROM CT0?

Following the bidirectional blocking results showing CT0 is a PASSIVE HUB:
- CT0 doesn't need to see the question (input blocking = 0% drop)
- Other positions need to see CT0 (output blocking = 16% drop)
- This suggests: Question → Writers → CT0 → Readers → Answer

This analysis identifies:
1. WRITERS: Positions that attend TO CT0 (write numerical info)
2. READERS: Positions that attend FROM CT0 (read stored info)

Method:
1. Load baseline attention patterns (no blocking)
2. For each problem, compute:
   - Attention TO CT0: Who writes to CT0?
   - Attention FROM CT0: Who reads from CT0?
3. Aggregate patterns across problems and layers
4. Stratify by:
   - Generation phase (CT token generation vs answer generation)
   - Problem correctness
   - Head specialization

Expected Findings:
- Writers: Early question tokens and possibly CT1-CT5
- Readers: Later CT tokens and answer tokens
- Clear temporal separation: writes happen early, reads happen later

Time: 1-2 hours
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

# Paths
DATA_DIR = Path('results/attention_data')
OUTPUT_DIR = Path(__file__).parent.parent / 'results'
FIGURES_DIR = OUTPUT_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_metadata():
    """Load metadata with problem info and correctness."""
    metadata_file = DATA_DIR / 'llama_metadata_final.json'
    with open(metadata_file) as f:
        return json.load(f)


def load_attention_data(problem_id: int) -> Dict[int, np.ndarray]:
    """
    Load baseline attention weights for a problem.

    Returns:
        Dict mapping step -> attention tensor [n_layers, n_heads, seq_len, seq_len]
    """
    h5_file = DATA_DIR / f'llama_problem_{problem_id:04d}_baseline_attention.h5'

    if not h5_file.exists():
        return None

    attention_by_step = {}

    with h5py.File(h5_file, 'r') as f:
        # Iterate through steps (0-5 for 6 CT tokens)
        for step in range(6):
            step_key = f'step_{step}'
            if step_key not in f:
                continue

            # Collect attention from all layers for this step
            layer_attentions = []
            layer_idx = 0
            while f'{step_key}/layer_{layer_idx}/attention' in f:
                attn = f[f'{step_key}/layer_{layer_idx}/attention'][:]
                layer_attentions.append(attn)
                layer_idx += 1

            if layer_attentions:
                # Stack layers: [n_layers, batch=1, n_heads, seq_len, seq_len]
                attention_by_step[step] = np.stack(layer_attentions, axis=0)

    return attention_by_step


def analyze_ct0_writers_readers(
    attention: np.ndarray,
    ct_start: int,
    step: int,
    n_layers: int = 16,
    n_heads: int = 32
) -> Dict:
    """
    Identify who writes TO CT0 and who reads FROM CT0.

    Args:
        attention: [n_layers, batch=1, n_heads, query_len, key_len]
                   During generation: query_len=1 (current token), key_len=full_seq
        ct_start: Position index where CT0 starts
        step: Current CT generation step (0-5)

    Returns:
        Dict with writer/reader statistics
    """
    # Expected shape: [n_layers, batch=1, n_heads, query_len, key_len]
    # query_len=1 during generation (current token being generated)
    # key_len=full sequence length up to current position

    ct0_pos = ct_start
    current_pos = ct_start + step  # Position of token being generated

    # Get key_len (full sequence length)
    key_len = attention.shape[-1]

    # For this step, we're generating CT{step}
    # The attention pattern tells us what CT{step} attends to

    # If we're generating CT0 (step=0), we can see what CT0 attends to
    # If we're generating later tokens, we can see if they attend to CT0

    # Remove batch dimension: [n_layers, n_heads, query_len=1, key_len]
    if attention.shape[1] == 1:
        attention = attention.squeeze(1)

    # Further squeeze query dimension: [n_layers, n_heads, key_len]
    if attention.shape[2] == 1:
        attention = attention.squeeze(2)

    # Now attention is [n_layers, n_heads, key_len]
    # This shows what the CURRENT token (being generated) attends to

    # READERS FROM CT0: If we're generating CT0 (step=0), this shows what CT0 reads
    if step == 0:
        # This is CT0's attention pattern
        avg_attention_from_ct0 = attention.mean(axis=(0, 1))  # [key_len]
        per_layer_from_ct0 = attention.mean(axis=1)  # [n_layers, key_len]
        per_head_from_ct0 = attention.mean(axis=0)  # [n_heads, key_len]
    else:
        # For other steps, we don't have CT0's attention in this data
        avg_attention_from_ct0 = np.zeros(key_len)
        per_layer_from_ct0 = np.zeros((n_layers, key_len))
        per_head_from_ct0 = np.zeros((n_heads, key_len))

    # WRITERS TO CT0: If CT0 exists (step > 0), we can see if current token attends to it
    # Note: current_pos is BEING generated, so it's not in the key_len yet
    # The attention tensor shows what the token BEING generated attends to
    if ct0_pos < key_len and step > 0:
        # Current token (CT{step})'s attention TO CT0
        attention_to_ct0_value = attention[:, :, ct0_pos]  # [n_layers, n_heads]

        # Track how much CT{step} attends to CT0
        # We'll store this as a single aggregate value for this step
        ct_to_ct0_attention = float(attention_to_ct0_value.mean())

        # For visualization, create a simple array
        avg_attention_to_ct0 = np.zeros(key_len)
        if ct0_pos < key_len:
            avg_attention_to_ct0[ct0_pos] = ct_to_ct0_attention

        per_layer_to_ct0 = np.zeros((n_layers, key_len))
        per_layer_to_ct0[:, ct0_pos] = attention_to_ct0_value.mean(axis=1)

        per_head_to_ct0 = np.zeros((n_heads, key_len))
        per_head_to_ct0[:, ct0_pos] = attention_to_ct0_value.mean(axis=0)
    else:
        ct_to_ct0_attention = 0.0
        avg_attention_to_ct0 = np.zeros(key_len)
        per_layer_to_ct0 = np.zeros((n_layers, key_len))
        per_head_to_ct0 = np.zeros((n_heads, key_len))

    # Identify top writers and readers
    ct_positions = list(range(ct_start, min(ct_start + 6, key_len)))
    question_positions = list(range(ct_start))  # Everything before CT tokens

    # Top writers (excluding CT0 itself)
    writer_scores = {}
    for pos in range(key_len):
        if pos != ct0_pos:
            writer_scores[pos] = float(avg_attention_to_ct0[pos])

    top_writers = sorted(writer_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # Top readers (what CT0 reads from)
    reader_scores = {}
    for pos in range(key_len):
        reader_scores[pos] = float(avg_attention_from_ct0[pos])

    top_readers = sorted(reader_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    # Aggregate by region
    # For step 0, CT{step} doesn't attend to CT0 (it IS CT0)
    # For steps 1-5, CT{step}'s attention to CT0 is tracked
    ct0_reads_from_question = sum(avg_attention_from_ct0[pos] for pos in question_positions)
    ct0_reads_from_ct = sum(avg_attention_from_ct0[pos] for pos in ct_positions if pos != ct0_pos)

    return {
        'step': step,
        'ct0_position': ct0_pos,
        'seq_len': key_len,
        'ct_start': ct_start,

        # Average patterns
        'avg_attention_to_ct0': avg_attention_to_ct0.tolist(),
        'avg_attention_from_ct0': avg_attention_from_ct0.tolist(),

        # Per-layer patterns
        'per_layer_to_ct0': per_layer_to_ct0.tolist(),
        'per_layer_from_ct0': per_layer_from_ct0.tolist(),

        # Top writers/readers
        'top_writers': top_writers,
        'top_readers': top_readers,

        # Aggregate statistics
        'ct_to_ct0_attention': ct_to_ct0_attention if step > 0 else 0.0,  # CT{step} → CT0
        'ct0_reads_from_question': float(ct0_reads_from_question),
        'ct0_reads_from_ct': float(ct0_reads_from_ct)
    }


def run_ct0_writers_readers_analysis(n_problems: int = 100):
    """
    Main analysis: Identify CT0 writers and readers across problems.
    """
    print(f"\n{'='*60}")
    print(f"CT0 Writers and Readers Analysis")
    print(f"{'='*60}\n")

    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata()
    problems_to_analyze = metadata[:n_problems]
    print(f"Analyzing {len(problems_to_analyze)} problems")

    # Storage for aggregated results
    all_results = []
    aggregate_writers = defaultdict(list)
    aggregate_readers = defaultdict(list)
    aggregate_by_step = defaultdict(lambda: {
        'writers_question': [],
        'writers_ct': [],
        'readers_question': [],
        'readers_ct': []
    })

    # Analyze each problem
    print("\nAnalyzing CT0 attention patterns...\n")

    n_success = 0
    n_failed = 0

    for problem in tqdm(problems_to_analyze, desc="Processing problems"):
        problem_id = problem['problem_id']

        # Load baseline attention data
        baseline_attn = load_attention_data(problem_id)

        if baseline_attn is None or not baseline_attn:
            n_failed += 1
            continue

        # Estimate CT start position from first step
        if 0 in baseline_attn:
            first_step_shape = baseline_attn[0].shape
            # Shape: [n_layers, batch=1, n_heads, seq_len, seq_len]
            # seq_len includes: question + BOT + CT0
            question_len = first_step_shape[-1] - 1  # seq_len - 1 (for BOT + CT0 being generated)
            ct_start = question_len + 1
        else:
            ct_start = 31  # Fallback

        problem_result = {
            'problem_id': problem_id,
            'baseline_correct': problem.get('baseline', {}).get('correct', False),
            'ct_start': ct_start,
            'by_step': {}
        }

        # Analyze each CT generation step
        for step in range(6):
            if step not in baseline_attn:
                continue

            analysis = analyze_ct0_writers_readers(
                baseline_attn[step], ct_start, step
            )

            problem_result['by_step'][step] = analysis

            # Aggregate statistics
            aggregate_writers[step].append(analysis['ct_to_ct0_attention'])
            aggregate_readers[step].append(analysis['ct0_reads_from_question'])

            aggregate_by_step[step]['writers_ct'].append(analysis['ct_to_ct0_attention'])
            aggregate_by_step[step]['readers_question'].append(analysis['ct0_reads_from_question'])
            aggregate_by_step[step]['readers_ct'].append(analysis['ct0_reads_from_ct'])

        all_results.append(problem_result)
        n_success += 1

    print(f"\n✓ Successfully analyzed {n_success} problems")
    print(f"✗ Failed to load {n_failed} problems")

    # Compute aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE PATTERNS: CT0 WRITERS AND READERS")
    print("="*60 + "\n")

    aggregate_summary = {}

    for step in sorted(aggregate_by_step.keys()):
        data = aggregate_by_step[step]

        summary = {
            'ct_to_ct0_attention_mean': float(np.mean(data['writers_ct'])),
            'ct_to_ct0_attention_std': float(np.std(data['writers_ct'])),
            'ct0_reads_from_question_mean': float(np.mean(data['readers_question'])),
            'ct0_reads_from_question_std': float(np.std(data['readers_question'])),
            'ct0_reads_from_ct_mean': float(np.mean(data['readers_ct'])),
            'ct0_reads_from_ct_std': float(np.std(data['readers_ct'])),
            'n_samples': len(data['writers_ct'])
        }

        aggregate_summary[step] = summary

        print(f"Step {step} (Generating CT{step}):")
        print(f"  WRITERS TO CT0:")
        print(f"    CT{step} → CT0: {summary['ct_to_ct0_attention_mean']:.4f} ± {summary['ct_to_ct0_attention_std']:.4f}")
        print(f"  READERS FROM CT0 (only for step 0):")
        print(f"    CT0 → Question tokens: {summary['ct0_reads_from_question_mean']:.4f} ± {summary['ct0_reads_from_question_std']:.4f}")
        print(f"    CT0 → Other CT tokens: {summary['ct0_reads_from_ct_mean']:.4f} ± {summary['ct0_reads_from_ct_std']:.4f}")
        print()

    # Key insight: Writer/Reader ratio
    print("="*60)
    print("KEY INSIGHTS")
    print("="*60 + "\n")

    # Calculate overall patterns
    # Writers: steps 1-5 (CT1-CT5 attending to CT0)
    total_ct_writes = np.mean([s['ct_to_ct0_attention_mean'] for k, s in aggregate_summary.items() if k > 0])

    # Readers: step 0 only (CT0's attention pattern)
    if 0 in aggregate_summary:
        ct0_reads_question = aggregate_summary[0]['ct0_reads_from_question_mean']
        ct0_reads_ct = aggregate_summary[0]['ct0_reads_from_ct_mean']
    else:
        ct0_reads_question = 0.0
        ct0_reads_ct = 0.0

    print(f"Overall Writer Pattern (CT1-CT5 → CT0):")
    print(f"  Average attention from CT1-CT5 to CT0: {total_ct_writes:.4f}")
    print()

    print(f"Overall Reader Pattern (CT0 → others, step 0 only):")
    print(f"  CT0 reads FROM question: {ct0_reads_question:.4f}")
    print(f"  CT0 reads FROM other CTs: {ct0_reads_ct:.4f}")
    print(f"  Reader ratio (Question/CT): {ct0_reads_question / (ct0_reads_ct + 1e-10):.2f}x")
    print()

    # Interpretation
    if total_ct_writes > 0.05:
        print("✓ FINDING: Other CT tokens (CT1-CT5) significantly attend TO CT0")
        print("  → CT0 acts as a hub that other CT positions write to")

    print()

    if ct0_reads_question > ct0_reads_ct:
        print("✓ FINDING: CT0 primarily READS FROM question tokens")
        print("  → CT0 processes question context during generation")
    else:
        print("✓ FINDING: CT0 primarily READS FROM other CT tokens")
        print("  → CT0 coordinates with other CT positions")

    # Save results
    output_file = OUTPUT_DIR / 'ct0_writers_readers_analysis.json'
    results_dict = {
        'summary': {
            'n_problems_analyzed': n_success,
            'n_problems_failed': n_failed,
            'aggregate_by_step': aggregate_summary,
            'overall_patterns': {
                'total_ct_writes': float(total_ct_writes),
                'ct0_reads_question': float(ct0_reads_question),
                'ct0_reads_ct': float(ct0_reads_ct),
                'reader_ratio_question_to_ct': float(ct0_reads_question / (ct0_reads_ct + 1e-10))
            }
        },
        'problem_level_results': all_results[:20]  # Save first 20 for inspection
    }

    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Analysis complete!")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*60}\n")

    return aggregate_summary, all_results


def create_visualizations(aggregate_summary: Dict):
    """Create visualizations of CT0 writer/reader patterns."""
    print("\nCreating visualizations...")

    steps = sorted(aggregate_summary.keys())

    # Extract data
    ct_to_ct0 = [aggregate_summary[s]['ct_to_ct0_attention_mean'] for s in steps]
    ct_to_ct0_std = [aggregate_summary[s]['ct_to_ct0_attention_std'] for s in steps]

    question_reads = [aggregate_summary[s]['ct0_reads_from_question_mean'] for s in steps]
    ct_reads = [aggregate_summary[s]['ct0_reads_from_ct_mean'] for s in steps]
    question_reads_std = [aggregate_summary[s]['ct0_reads_from_question_std'] for s in steps]
    ct_reads_std = [aggregate_summary[s]['ct0_reads_from_ct_std'] for s in steps]

    # Figure 1: Writers TO CT0 (CT1-CT5 attending to CT0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('CT Tokens Writing TO CT0: How much each CT{i} attends to CT0',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(steps))
    ax.bar(x, ct_to_ct0, yerr=ct_to_ct0_std, color='coral', alpha=0.8, capsize=5)

    ax.set_xlabel('CT Generation Step', fontweight='bold')
    ax.set_ylabel('Attention Weight (CT{i} → CT0)', fontweight='bold')
    ax.set_title('Writers TO CT0 Across Steps', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'CT{s}' for s in steps])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, val in enumerate(ct_to_ct0):
        if val > 0:
            ax.text(i, val + ct_to_ct0_std[i] + 0.005, f'{val:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_file = FIGURES_DIR / 'ct0_writers_by_step.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

    # Figure 2: CT0 Reads (step 0 only)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle('CT0 Attention Pattern (Step 0): What does CT0 attend to?',
                 fontsize=14, fontweight='bold')

    categories = ['Question\nTokens', 'Other CT\nTokens']
    if 0 in aggregate_summary:
        reads = [aggregate_summary[0]['ct0_reads_from_question_mean'],
                aggregate_summary[0]['ct0_reads_from_ct_mean']]
        reads_std = [aggregate_summary[0]['ct0_reads_from_question_std'],
                    aggregate_summary[0]['ct0_reads_from_ct_std']]
    else:
        reads = [0, 0]
        reads_std = [0, 0]

    x = np.arange(len(categories))
    ax.bar(x, reads, yerr=reads_std, color='forestgreen', alpha=0.8, capsize=5)

    ax.set_ylabel('Attention Weight (FROM CT0)', fontweight='bold')
    ax.set_title('What CT0 Reads (Attends To) During Generation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, val in enumerate(reads):
        ax.text(i, val + reads_std[i] + 0.01, f'{val:.3f}',
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    fig_file = FIGURES_DIR / 'ct0_readers_step0.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_file}")
    plt.close()

    print("✓ Visualizations complete\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze CT0 writers and readers')
    parser.add_argument('--n_problems', type=int, default=100,
                       help='Number of problems to analyze')
    args = parser.parse_args()

    # Run analysis
    aggregate_summary, all_results = run_ct0_writers_readers_analysis(n_problems=args.n_problems)

    # Create visualizations
    create_visualizations(aggregate_summary)


if __name__ == '__main__':
    main()
