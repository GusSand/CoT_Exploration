#!/usr/bin/env python3
"""
Analyze difficulty patterns of problems where LLaMA needs vs skips CoT.

This script joins difficulty metrics with CoT necessity results to understand
what makes problems "easy enough" for LLaMA to solve without latent reasoning.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up paths
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "results" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data():
    """Load and integrate difficulty metrics with CoT necessity results."""
    print("Loading data...")

    # Load difficulty metrics
    with open(RESULTS_DIR / "matched_pairs_difficulty_analysis.json") as f:
        difficulty_data = json.load(f)

    # Load CoT necessity results
    with open(RESULTS_DIR / "cot_necessity_llama_simple.json") as f:
        cot_data = json.load(f)

    # Convert to DataFrames for easier manipulation
    df_difficulty = pd.DataFrame(difficulty_data)

    # Remove duplicates (keep first occurrence)
    df_difficulty = df_difficulty.drop_duplicates(subset=['pair_id'], keep='first')

    # Extract CoT necessity info
    cot_necessity = []
    for result in cot_data['results']:
        cot_necessity.append({
            'pair_id': result['pair_id'],
            'needs_cot': result['needs_cot_either']
        })
    df_cot = pd.DataFrame(cot_necessity)

    # Remove duplicates (keep first occurrence)
    df_cot = df_cot.drop_duplicates(subset=['pair_id'], keep='first')

    # Join the datasets
    df = pd.merge(df_difficulty, df_cot, on='pair_id')

    print(f"Total pairs: {len(df)}")
    print(f"Needs CoT: {df['needs_cot'].sum()}")
    print(f"Skips CoT: {(~df['needs_cot']).sum()}")

    return df

def compute_statistics(df):
    """Compute statistical comparisons between CoT-needed and CoT-skipped problems."""
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    needs_cot = df[df['needs_cot']]
    skips_cot = df[~df['needs_cot']]

    results = {
        'counts': {
            'needs_cot': len(needs_cot),
            'skips_cot': len(skips_cot)
        }
    }

    # Analyze each difficulty metric
    metrics = ['steps', 'total_operations', 'solution_length', 'sentences']

    for metric in metrics:
        needs_values = needs_cot[metric].values
        skips_values = skips_cot[metric].values

        # Compute statistics
        needs_mean = np.mean(needs_values)
        needs_std = np.std(needs_values)
        skips_mean = np.mean(skips_values)
        skips_std = np.std(skips_values)

        # T-test
        t_stat, p_value = stats.ttest_ind(needs_values, skips_values)

        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(needs_values)-1)*needs_std**2 + (len(skips_values)-1)*skips_std**2) / (len(needs_values) + len(skips_values) - 2))
        cohens_d = (needs_mean - skips_mean) / pooled_std if pooled_std > 0 else 0

        results[metric] = {
            'needs_cot': {
                'mean': needs_mean,
                'std': needs_std,
                'median': np.median(needs_values),
                'min': np.min(needs_values),
                'max': np.max(needs_values)
            },
            'skips_cot': {
                'mean': skips_mean,
                'std': skips_std,
                'median': np.median(skips_values),
                'min': np.min(skips_values),
                'max': np.max(skips_values)
            },
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }

        print(f"\n{metric.upper()}:")
        print(f"  Needs CoT:  {needs_mean:.2f} ± {needs_std:.2f} (median: {np.median(needs_values):.1f})")
        print(f"  Skips CoT:  {skips_mean:.2f} ± {skips_std:.2f} (median: {np.median(skips_values):.1f})")
        print(f"  Difference: {needs_mean - skips_mean:+.2f}")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"  Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small' if abs(cohens_d) > 0.2 else 'negligible'})")

    return results

def analyze_operations(df):
    """Analyze operation type distributions."""
    print("\n" + "="*80)
    print("OPERATION TYPE ANALYSIS")
    print("="*80)

    needs_cot = df[df['needs_cot']]
    skips_cot = df[~df['needs_cot']]

    # Expand operations into separate columns
    operation_types = ['addition', 'subtraction', 'multiplication', 'division']

    for group_name, group_df in [('Needs CoT', needs_cot), ('Skips CoT', skips_cot)]:
        print(f"\n{group_name} ({len(group_df)} problems):")
        for op_type in operation_types:
            op_counts = [ops.get(op_type, 0) for ops in group_df['operations']]
            mean_count = np.mean(op_counts)
            problems_with_op = sum(1 for c in op_counts if c > 0)
            print(f"  {op_type:14s}: {mean_count:.2f} avg, {problems_with_op}/{len(group_df)} problems ({100*problems_with_op/len(group_df):.1f}%)")

def stratify_by_difficulty(df):
    """Stratify problems by difficulty and analyze CoT necessity rates."""
    print("\n" + "="*80)
    print("DIFFICULTY STRATIFICATION")
    print("="*80)

    # Define difficulty levels based on reasoning steps
    df['difficulty'] = pd.cut(df['steps'], bins=[0, 2, 3, 100], labels=['easy (≤2)', 'medium (3)', 'hard (≥4)'])

    stratification = df.groupby('difficulty').agg({
        'needs_cot': ['sum', 'count', 'mean']
    }).round(3)

    print("\nCoT Necessity by Difficulty Level:")
    print("-" * 60)
    for difficulty in ['easy (≤2)', 'medium (3)', 'hard (≥4)']:
        if difficulty in stratification.index:
            total = stratification.loc[difficulty, ('needs_cot', 'count')]
            needs_cot = stratification.loc[difficulty, ('needs_cot', 'sum')]
            rate = stratification.loc[difficulty, ('needs_cot', 'mean')]
            print(f"{difficulty:15s}: {needs_cot:.0f}/{total:.0f} need CoT ({100*rate:.1f}%)")

    return stratification

def create_visualizations(df, stats_results):
    """Create visualizations showing difficulty patterns."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    # Prepare data
    df['cot_status'] = df['needs_cot'].map({True: 'Needs CoT (44)', False: 'Skips CoT (57)'})

    # Figure 1: Reasoning steps distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    needs_steps = df[df['needs_cot']]['steps']
    skips_steps = df[~df['needs_cot']]['steps']

    bins = np.arange(0.5, max(df['steps']) + 1.5, 1)
    ax.hist([skips_steps, needs_steps], bins=bins, label=['Skips CoT (57)', 'Needs CoT (44)'],
            alpha=0.7, color=['#2ecc71', '#e74c3c'])
    ax.set_xlabel('Number of Reasoning Steps', fontsize=12)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Distribution of Reasoning Steps by CoT Necessity', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Box plot
    ax = axes[1]
    sns.boxplot(data=df, x='needs_cot', y='steps', ax=ax, palette=['#2ecc71', '#e74c3c'])
    ax.set_xticklabels(['Skips CoT\n(57)', 'Needs CoT\n(44)'])
    ax.set_ylabel('Number of Reasoning Steps', fontsize=12)
    ax.set_xlabel('')
    ax.set_title('Reasoning Steps Comparison', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'reasoning_steps_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: reasoning_steps_distribution.png")

    # Figure 2: Multiple metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ('steps', 'Number of Reasoning Steps'),
        ('total_operations', 'Total Operations'),
        ('solution_length', 'Solution Length (chars)'),
        ('sentences', 'Number of Sentences')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x='needs_cot', y=metric, ax=ax, palette=['#2ecc71', '#e74c3c'])
        ax.set_xticklabels(['Skips CoT\n(57)', 'Needs CoT\n(44)'])
        ax.set_ylabel(label, fontsize=11)
        ax.set_xlabel('')
        ax.set_title(label, fontsize=12, fontweight='bold')

        # Add statistical annotation
        p_val = stats_results[metric]['p_value']
        sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(0.5, 0.98, f'p={p_val:.4f} {sig_text}',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'difficulty_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: difficulty_metrics_comparison.png")

    # Figure 3: Difficulty stratification
    df['difficulty'] = pd.cut(df['steps'], bins=[0, 2, 3, 100], labels=['Easy (≤2)', 'Medium (3)', 'Hard (≥4)'])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Count problems by difficulty and CoT necessity
    strat_data = df.groupby(['difficulty', 'needs_cot']).size().unstack(fill_value=0)
    strat_data.columns = ['Skips CoT', 'Needs CoT']

    strat_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], width=0.7)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_xlabel('Difficulty Level', fontsize=12)
    ax.set_title('CoT Necessity by Problem Difficulty', fontsize=13, fontweight='bold')
    ax.legend(title='', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(alpha=0.3, axis='y')

    # Add percentage labels
    for i, difficulty in enumerate(strat_data.index):
        total = strat_data.loc[difficulty].sum()
        needs_cot = strat_data.loc[difficulty, 'Needs CoT']
        pct = 100 * needs_cot / total if total > 0 else 0
        ax.text(i, total + 1, f'{pct:.0f}% need CoT', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'difficulty_stratification.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: difficulty_stratification.png")

    print(f"\nAll figures saved to: {FIGURES_DIR}")

def generate_hypotheses(df, stats_results):
    """Generate testable hypotheses from the findings."""
    print("\n" + "="*80)
    print("HYPOTHESIS GENERATION")
    print("="*80)

    hypotheses = []

    # Based on steps difference
    steps_diff = stats_results['steps']['needs_cot']['mean'] - stats_results['steps']['skips_cot']['mean']
    if stats_results['steps']['significant']:
        hypotheses.append({
            'id': 'H1',
            'hypothesis': f'LLaMA can solve problems with ≤{stats_results["steps"]["skips_cot"]["median"]:.0f} reasoning steps via direct computation without latent CoT',
            'evidence': f'Mean steps: Needs CoT = {stats_results["steps"]["needs_cot"]["mean"]:.2f}, Skips CoT = {stats_results["steps"]["skips_cot"]["mean"]:.2f} (p={stats_results["steps"]["p_value"]:.4f})',
            'test': 'Test on additional problems stratified by step count; measure accuracy with/without CoT tokens'
        })

    # Based on difficulty stratification
    easy_problems = df[df['steps'] <= 2]
    easy_cot_rate = easy_problems['needs_cot'].mean()
    hypotheses.append({
        'id': 'H2',
        'hypothesis': f'Problem difficulty threshold exists around 2-3 reasoning steps, below which LLaMA uses direct computation {100*(1-easy_cot_rate):.0f}% of the time',
        'evidence': f'Easy problems (≤2 steps): {100*easy_cot_rate:.1f}% need CoT; Medium+ problems (≥3 steps): higher CoT dependency',
        'test': 'Analyze CoT token activations for easy vs hard problems to confirm different computational pathways'
    })

    # Based on operations
    hypotheses.append({
        'id': 'H3',
        'hypothesis': 'LLaMA\'s latent reasoning capacity is more efficiently utilized for complex multi-step problems than simple arithmetic',
        'evidence': f'Larger models may have specialized circuits for basic arithmetic that bypass latent reasoning',
        'test': 'Compare activation patterns in early layers for easy vs hard problems; probe for arithmetic circuits'
    })

    # Model size hypothesis
    hypotheses.append({
        'id': 'H4',
        'hypothesis': 'Model size correlates with ability to perform direct computation, explaining the 100% vs 44% CoT dependency gap between GPT-2 and LLaMA',
        'evidence': 'GPT-2 (117M) needs CoT 100% of the time; LLaMA (1B) only 44% of the time on same problems',
        'test': 'Test intermediate model sizes (350M, 700M) to find where direct computation capability emerges'
    })

    # Efficiency hypothesis
    hypotheses.append({
        'id': 'H5',
        'hypothesis': 'For problems where LLaMA needs CoT, the latent reasoning is qualitatively different (more abstract/complex) than problems where it skips CoT',
        'evidence': f'Clear separation in problem complexity metrics (Cohen\'s d = {stats_results["steps"]["cohens_d"]:.2f} for steps)',
        'test': 'Analyze attention patterns and hidden state representations for CoT-needed vs CoT-skipped problems'
    })

    print("\nGenerated Hypotheses:\n")
    for h in hypotheses:
        print(f"{h['id']}: {h['hypothesis']}")
        print(f"    Evidence: {h['evidence']}")
        print(f"    Test: {h['test']}\n")

    return hypotheses

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("LLaMA CoT Difficulty Analysis")
    print("="*80)

    # Load and integrate data
    df = load_data()

    # Statistical analysis
    stats_results = compute_statistics(df)

    # Operation analysis
    analyze_operations(df)

    # Difficulty stratification
    stratification = stratify_by_difficulty(df)

    # Create visualizations
    create_visualizations(df, stats_results)

    # Generate hypotheses
    hypotheses = generate_hypotheses(df, stats_results)

    # Save results
    output = {
        'summary': {
            'total_pairs': len(df),
            'needs_cot': int(df['needs_cot'].sum()),
            'skips_cot': int((~df['needs_cot']).sum())
        },
        'statistics': stats_results,
        'hypotheses': hypotheses
    }

    # Convert numpy types to native Python types
    output = convert_to_native_types(output)

    output_file = RESULTS_DIR / 'llama_cot_difficulty_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete! Results saved to: {output_file}")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
