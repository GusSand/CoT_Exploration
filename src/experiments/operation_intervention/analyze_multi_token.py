"""
Analyze Multi-Token Intervention Results

Compares single-token vs multi-token intervention effects to test if disrupting
both planning (Token 1) and execution (Token 5) produces larger causal effects.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_results(results_path):
    """Load experiment results."""
    return json.load(open(results_path))


def compute_accuracy_by_condition(results):
    """Compute accuracy for each condition."""
    accuracies = defaultdict(list)

    for prob in results:
        for cond_name, cond_data in prob['conditions'].items():
            accuracies[cond_name].append(1 if cond_data['is_correct'] else 0)

    return {cond: np.mean(acc) for cond, acc in accuracies.items()}


def compute_answer_changes(results):
    """Count how many problems changed answers vs baseline."""
    changes = defaultdict(int)
    total = len(results)

    # Initialize all conditions
    all_conditions = set()
    for prob in results:
        all_conditions.update(prob['conditions'].keys())
    all_conditions.discard('baseline')

    for cond in all_conditions:
        changes[cond] = 0

    for prob in results:
        baseline_ans = prob['conditions']['baseline']['predicted_answer']
        for cond_name in prob['conditions'].keys():
            if cond_name == 'baseline':
                continue
            intervention_ans = prob['conditions'][cond_name]['predicted_answer']
            if str(baseline_ans) != str(intervention_ans):
                changes[cond_name] += 1

    return {cond: count / total for cond, count in changes.items()}


def compute_accuracy_by_operation(results):
    """Compute accuracy by operation type for each condition."""
    by_op = defaultdict(lambda: defaultdict(list))

    for prob in results:
        op_type = prob['operation_type']
        for cond_name, cond_data in prob['conditions'].items():
            by_op[op_type][cond_name].append(1 if cond_data['is_correct'] else 0)

    # Convert to means
    return {op: {cond: np.mean(acc) for cond, acc in conds.items()}
            for op, conds in by_op.items()}


def compute_statistics(results):
    """Compute statistical tests comparing conditions."""
    stats_results = {}

    # Extract accuracies
    baseline = np.array([1 if prob['conditions']['baseline']['is_correct'] else 0 for prob in results])
    token1_only = np.array([1 if prob['conditions']['token1_only']['is_correct'] else 0 for prob in results])
    token5_only = np.array([1 if prob['conditions']['token5_only']['is_correct'] else 0 for prob in results])
    multi_token = np.array([1 if prob['conditions']['multi_token']['is_correct'] else 0 for prob in results])
    token1_random = np.array([1 if prob['conditions']['token1_random']['is_correct'] else 0 for prob in results])
    multi_random = np.array([1 if prob['conditions']['multi_random']['is_correct'] else 0 for prob in results])

    # Paired t-tests
    from scipy.stats import ttest_rel

    stats_results['paired_tests'] = {
        'baseline_vs_token1': {
            't_statistic': float(ttest_rel(baseline, token1_only)[0]),
            'p_value': float(ttest_rel(baseline, token1_only)[1])
        },
        'baseline_vs_token5': {
            't_statistic': float(ttest_rel(baseline, token5_only)[0]),
            'p_value': float(ttest_rel(baseline, token5_only)[1])
        },
        'baseline_vs_multi': {
            't_statistic': float(ttest_rel(baseline, multi_token)[0]),
            'p_value': float(ttest_rel(baseline, multi_token)[1])
        },
        'token1_vs_multi': {
            't_statistic': float(ttest_rel(token1_only, multi_token)[0]),
            'p_value': float(ttest_rel(token1_only, multi_token)[1])
        },
        'baseline_vs_token1_random': {
            't_statistic': float(ttest_rel(baseline, token1_random)[0]),
            'p_value': float(ttest_rel(baseline, token1_random)[1])
        },
        'baseline_vs_multi_random': {
            't_statistic': float(ttest_rel(baseline, multi_random)[0]),
            'p_value': float(ttest_rel(baseline, multi_random)[1])
        }
    }

    # Effect sizes (Cohen's h for proportions)
    def cohens_h(p1, p2):
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    stats_results['effect_sizes'] = {
        'baseline_vs_token1': float(cohens_h(np.mean(baseline), np.mean(token1_only))),
        'baseline_vs_token5': float(cohens_h(np.mean(baseline), np.mean(token5_only))),
        'baseline_vs_multi': float(cohens_h(np.mean(baseline), np.mean(multi_token))),
        'baseline_vs_token1_random': float(cohens_h(np.mean(baseline), np.mean(token1_random))),
        'baseline_vs_multi_random': float(cohens_h(np.mean(baseline), np.mean(multi_random)))
    }

    return stats_results


def plot_accuracy_comparison(accuracies, output_dir):
    """Plot accuracy by condition with focus on single vs multi-token."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group conditions
    conditions = ['baseline', 'token1_only', 'token5_only', 'multi_token', 'token1_random', 'token5_random', 'multi_random']
    labels = ['Baseline', 'Token 1\nOnly', 'Token 5\nOnly', 'Multi\nToken', 'Token 1\nRandom', 'Token 5\nRandom', 'Multi\nRandom']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#95a5a6', '#bdc3c7', '#7f8c8d']

    values = [accuracies[c] * 100 for c in conditions]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Token vs Single-Token Intervention Effects', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=accuracies['baseline']*100, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Baseline')
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_token_accuracy.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'multi_token_accuracy.pdf', bbox_inches='tight')
    plt.close()


def plot_answer_changes(change_rates, output_dir):
    """Plot answer change rates."""
    fig, ax = plt.subplots(figsize=(12, 7))

    conditions = ['token1_only', 'token5_only', 'multi_token', 'token1_random', 'token5_random', 'multi_random']
    labels = ['Token 1\nOnly', 'Token 5\nOnly', 'Multi\nToken', 'Token 1\nRandom', 'Token 5\nRandom', 'Multi\nRandom']
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#95a5a6', '#bdc3c7', '#7f8c8d']

    values = [change_rates[c] * 100 for c in conditions]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_ylabel('Answer Change Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('How Often Interventions Changed Answers from Baseline', fontsize=15, fontweight='bold')
    ax.set_ylim(0, max(values) + 10)

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_token_changes.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'multi_token_changes.pdf', bbox_inches='tight')
    plt.close()


def plot_by_operation_comparison(by_operation, output_dir):
    """Compare single vs multi-token effects by operation type."""
    fig, ax = plt.subplots(figsize=(13, 7))

    operations = ['pure_addition', 'pure_multiplication', 'mixed']
    op_labels = ['Addition', 'Multiplication', 'Mixed']
    conditions = ['baseline', 'token1_only', 'token5_only', 'multi_token']
    cond_labels = ['Baseline', 'Token 1', 'Token 5', 'Multi']
    colors_cond = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']

    x = np.arange(len(operations))
    width = 0.2

    for i, cond in enumerate(conditions):
        values = [by_operation[op][cond] * 100 for op in operations]
        ax.bar(x + i*width, values, width, label=cond_labels[i], color=colors_cond[i], alpha=0.8)

    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Intervention Effects by Operation Type', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(op_labels, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_token_by_operation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'multi_token_by_operation.pdf', bbox_inches='tight')
    plt.close()


def main():
    # Paths
    base_dir = Path(__file__).parent
    results_path = base_dir / 'multi_token_results.json'
    output_dir = base_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    print("Loading multi-token results...")
    results = load_results(results_path)

    print(f"Analyzing {len(results)} problems...")

    # Compute metrics
    accuracies = compute_accuracy_by_condition(results)
    change_rates = compute_answer_changes(results)
    by_operation = compute_accuracy_by_operation(results)
    statistics = compute_statistics(results)

    # Print summary
    print("\n=== MULTI-TOKEN INTERVENTION RESULTS ===")
    print("\n--- Accuracy by Condition ---")
    for cond in ['baseline', 'token1_only', 'token5_only', 'multi_token', 'token1_random', 'token5_random', 'multi_random']:
        acc = accuracies[cond]
        print(f"{cond:20s}: {acc*100:5.1f}%")

    print("\n--- Answer Change Rates (vs Baseline) ---")
    for cond in ['token1_only', 'token5_only', 'multi_token', 'token1_random', 'token5_random', 'multi_random']:
        rate = change_rates[cond]
        print(f"{cond:20s}: {rate*100:5.1f}%")

    print("\n--- Statistical Tests (Paired t-tests) ---")
    for comp, test in statistics['paired_tests'].items():
        print(f"{comp:30s}: t={test['t_statistic']:6.3f}, p={test['p_value']:.4f}")

    print("\n--- Effect Sizes (Cohen's h) ---")
    for comp, h in statistics['effect_sizes'].items():
        print(f"{comp:30s}: {h:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(accuracies, output_dir)
    plot_answer_changes(change_rates, output_dir)
    plot_by_operation_comparison(by_operation, output_dir)

    # Save detailed results
    analysis_results = {
        'accuracies': {k: float(v) for k, v in accuracies.items()},
        'change_rates': {k: float(v) for k, v in change_rates.items()},
        'by_operation': {op: {cond: float(acc) for cond, acc in conds.items()}
                         for op, conds in by_operation.items()},
        'statistics': statistics
    }

    json.dump(analysis_results, open(output_dir / 'multi_token_analysis.json', 'w'), indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"   - 3 visualizations (PNG + PDF)")
    print(f"   - multi_token_analysis.json")


if __name__ == '__main__':
    main()
