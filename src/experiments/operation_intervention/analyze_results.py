"""
Analyze Operation Intervention Results

Generates visualizations and statistical analysis of the causal intervention experiment.
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
plt.rcParams['figure.figsize'] = (12, 8)

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

def compute_answer_changes(results):
    """Count how many problems changed answers vs baseline."""
    changes = defaultdict(int)
    total = len(results)

    for prob in results:
        baseline_ans = prob['conditions']['baseline']['predicted_answer']
        for cond_name in ['to_addition', 'to_multiplication', 'to_mixed', 'random_control', 'wrong_token_control']:
            intervention_ans = prob['conditions'][cond_name]['predicted_answer']
            if str(baseline_ans) != str(intervention_ans):
                changes[cond_name] += 1

    # Ensure all conditions are in the dict
    for cond in ['to_addition', 'to_multiplication', 'to_mixed', 'random_control', 'wrong_token_control']:
        if cond not in changes:
            changes[cond] = 0

    return {cond: count / total for cond, count in changes.items()}

def plot_accuracy_comparison(accuracies, output_dir):
    """Plot accuracy by condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ['baseline', 'to_addition', 'to_multiplication', 'to_mixed', 'random_control', 'wrong_token_control']
    labels = ['Baseline', 'To Addition', 'To Multiplication', 'To Mixed', 'Random Control', 'Wrong Token']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#95a5a6', '#9b59b6']

    values = [accuracies[c] * 100 for c in conditions]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Intervention Effects on Problem-Solving Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=accuracies['baseline']*100, color='green', linestyle='--', alpha=0.5, label='Baseline')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_condition.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_by_condition.pdf', bbox_inches='tight')
    plt.close()

def plot_accuracy_by_operation(by_operation, output_dir):
    """Plot accuracy by operation type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    operations = ['pure_addition', 'pure_multiplication', 'mixed']
    op_labels = ['Pure Addition', 'Pure Multiplication', 'Mixed']
    conditions = ['baseline', 'to_addition', 'to_multiplication', 'to_mixed']
    cond_labels = ['Baseline', 'To Add', 'To Mult', 'To Mixed']

    x = np.arange(len(operations))
    width = 0.2

    for i, cond in enumerate(conditions):
        values = [by_operation[op][cond] * 100 for op in operations]
        ax.bar(x + i*width, values, width, label=cond_labels[i], alpha=0.8)

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy by Problem Operation Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(op_labels)
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_by_operation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_by_operation.pdf', bbox_inches='tight')
    plt.close()

def plot_answer_changes(change_rates, output_dir):
    """Plot answer change rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ['to_addition', 'to_multiplication', 'to_mixed', 'random_control', 'wrong_token_control']
    labels = ['To Addition', 'To Multiplication', 'To Mixed', 'Random Control', 'Wrong Token']
    colors = ['#3498db', '#e74c3c', '#f39c12', '#95a5a6', '#9b59b6']

    values = [change_rates[c] * 100 for c in conditions]

    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Answer Change Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('How Often Interventions Changed Answers', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 60)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'answer_change_rates.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'answer_change_rates.pdf', bbox_inches='tight')
    plt.close()

def compute_statistics(results):
    """Compute statistical tests."""
    stats_results = {}

    # Extract accuracies
    baseline = np.array([1 if prob['conditions']['baseline']['is_correct'] else 0 for prob in results])
    to_add = np.array([1 if prob['conditions']['to_addition']['is_correct'] else 0 for prob in results])
    to_mult = np.array([1 if prob['conditions']['to_multiplication']['is_correct'] else 0 for prob in results])
    to_mixed = np.array([1 if prob['conditions']['to_mixed']['is_correct'] else 0 for prob in results])
    random_ctrl = np.array([1 if prob['conditions']['random_control']['is_correct'] else 0 for prob in results])
    wrong_token = np.array([1 if prob['conditions']['wrong_token_control']['is_correct'] else 0 for prob in results])

    # Paired t-tests (treating as continuous even though binary)
    from scipy.stats import ttest_rel

    stats_results['paired_tests'] = {
        'baseline_vs_addition': {
            't_statistic': float(ttest_rel(baseline, to_add)[0]),
            'p_value': float(ttest_rel(baseline, to_add)[1])
        },
        'baseline_vs_random': {
            't_statistic': float(ttest_rel(baseline, random_ctrl)[0]),
            'p_value': float(ttest_rel(baseline, random_ctrl)[1])
        },
        'baseline_vs_wrong_token': {
            't_statistic': float(ttest_rel(baseline, wrong_token)[0]),
            'p_value': float(ttest_rel(baseline, wrong_token)[1])
        }
    }

    # Effect sizes (Cohen's h for proportions)
    def cohens_h(p1, p2):
        return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    stats_results['effect_sizes'] = {
        'baseline_vs_addition': float(cohens_h(np.mean(baseline), np.mean(to_add))),
        'baseline_vs_multiplication': float(cohens_h(np.mean(baseline), np.mean(to_mult))),
        'baseline_vs_mixed': float(cohens_h(np.mean(baseline), np.mean(to_mixed))),
        'baseline_vs_random': float(cohens_h(np.mean(baseline), np.mean(random_ctrl))),
        'baseline_vs_wrong_token': float(cohens_h(np.mean(baseline), np.mean(wrong_token)))
    }

    # Contingency tables for reference
    stats_results['contingency_tables'] = {
        'baseline_vs_addition': {
            'both_correct': int(np.sum(baseline * to_add)),
            'baseline_only': int(np.sum(baseline * (1 - to_add))),
            'intervention_only': int(np.sum((1 - baseline) * to_add)),
            'both_wrong': int(np.sum((1 - baseline) * (1 - to_add)))
        },
        'baseline_vs_random': {
            'both_correct': int(np.sum(baseline * random_ctrl)),
            'baseline_only': int(np.sum(baseline * (1 - random_ctrl))),
            'intervention_only': int(np.sum((1 - baseline) * random_ctrl)),
            'both_wrong': int(np.sum((1 - baseline) * (1 - random_ctrl)))
        }
    }

    return stats_results

def main():
    # Paths
    base_dir = Path(__file__).parent
    results_path = base_dir / 'intervention_results.json'
    output_dir = base_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    print("Loading results...")
    results = load_results(results_path)

    print(f"Analyzing {len(results)} problems...")

    # Compute metrics
    accuracies = compute_accuracy_by_condition(results)
    by_operation = compute_accuracy_by_operation(results)
    change_rates = compute_answer_changes(results)
    statistics = compute_statistics(results)

    # Print summary
    print("\n=== Accuracy by Condition ===")
    for cond, acc in accuracies.items():
        print(f"{cond}: {acc*100:.1f}%")

    print("\n=== Answer Change Rates ===")
    for cond, rate in change_rates.items():
        print(f"{cond}: {rate*100:.1f}%")

    print("\n=== Effect Sizes (Cohen's h) ===")
    for comp, h in statistics['effect_sizes'].items():
        print(f"{comp}: {h:.3f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_accuracy_comparison(accuracies, output_dir)
    plot_accuracy_by_operation(by_operation, output_dir)
    plot_answer_changes(change_rates, output_dir)

    # Save detailed results
    analysis_results = {
        'accuracies': {k: float(v) for k, v in accuracies.items()},
        'by_operation': {op: {cond: float(acc) for cond, acc in conds.items()}
                         for op, conds in by_operation.items()},
        'change_rates': {k: float(v) for k, v in change_rates.items()},
        'statistics': statistics
    }

    json.dump(analysis_results, open(output_dir / 'analysis_summary.json', 'w'), indent=2)

    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    print(f"   - 3 visualizations (PNG + PDF)")
    print(f"   - analysis_summary.json")

if __name__ == '__main__':
    main()
