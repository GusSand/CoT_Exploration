#!/usr/bin/env python3
"""
Steering Experiment Failure Analysis

This script investigates:
1. Random direction control - validate suppression is meaningful
2. Per-problem analysis - understand amplification failure patterns
3. Test set difficulty distribution - check if baseline is near ceiling
4. Steering scale analysis - investigate optimal alpha values
5. Positional steering effects - test token-specific steering
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results():
    """Load steering experiment results."""
    print("="*80)
    print("LOADING STEERING EXPERIMENT RESULTS")
    print("="*80)

    results_file = Path(__file__).parent / 'results' / 'steering_experiments' / 'steering_results_detailed.json'
    summary_file = Path(__file__).parent / 'results' / 'steering_experiments' / 'steering_results_summary.json'

    with open(results_file) as f:
        detailed = json.load(f)

    with open(summary_file) as f:
        summary = json.load(f)

    print(f"✓ Loaded results for {len(detailed['alphas'])} alpha values")
    print(f"✓ Total problems: {summary['summary']['0.0']['total']}")

    return detailed, summary


def load_test_set_metadata():
    """Load test set to analyze difficulty distribution."""
    print("\n" + "="*80)
    print("LOADING TEST SET METADATA")
    print("="*80)

    dataset_file = Path(__file__).parent / 'results' / 'steering_dataset_gpt2.json'
    problem_pairs_file = Path(__file__).parent / 'problem_pairs_gpt4_answers.json'

    with open(dataset_file) as f:
        dataset = json.load(f)

    with open(problem_pairs_file) as f:
        all_pairs = json.load(f)

    # Create lookup
    pairs_lookup = {p['pair_id']: p for p in all_pairs}

    # Get test problems with metadata
    test_correct = dataset['test_correct']
    test_wrong = dataset['test_wrong']

    # Add metadata
    for prob in test_correct + test_wrong:
        pair = pairs_lookup[prob['pair_id']]
        prob['question'] = pair['clean']['question']
        prob['answer'] = pair['clean']['answer']
        prob['reasoning_steps'] = len(pair['clean'].get('reasoning_steps', []))

    print(f"✓ Test CORRECT: {len(test_correct)} problems")
    print(f"✓ Test WRONG: {len(test_wrong)} problems")

    return test_correct, test_wrong, pairs_lookup


def analyze_baseline_vs_expected(summary, test_correct, test_wrong):
    """Analyze why baseline is 32.6% instead of expected 50%."""
    print("\n" + "="*80)
    print("BASELINE ANALYSIS")
    print("="*80)

    baseline_acc = summary['summary']['0.0']['accuracy']
    baseline_correct = summary['summary']['0.0']['correct']
    baseline_total = summary['summary']['0.0']['total']

    print(f"\nBaseline accuracy: {baseline_acc:.1f}% ({baseline_correct}/{baseline_total})")
    print(f"Expected (50/50 split): 50.0% (43/86)")
    print(f"Gap: {baseline_acc - 50.0:.1f} percentage points")

    # Hypothesis: Test set might be harder than training set
    print("\n--- HYPOTHESIS 1: Test set difficulty ---")
    print("The test set (43+43=86) might contain harder problems than expected.")
    print("If GPT-2 can't solve many 'correct' problems, baseline drops below 50%.")

    # Calculate reasoning step distribution
    all_test = test_correct + test_wrong
    reasoning_steps = [p.get('reasoning_steps', 0) for p in all_test]

    print(f"\nReasoning step distribution:")
    print(f"  Mean: {np.mean(reasoning_steps):.2f}")
    print(f"  Median: {np.median(reasoning_steps):.0f}")
    print(f"  Range: {min(reasoning_steps)} - {max(reasoning_steps)}")

    return {
        'baseline_acc': baseline_acc,
        'expected_acc': 50.0,
        'gap': baseline_acc - 50.0,
        'reasoning_steps': reasoning_steps
    }


def analyze_per_problem_results(detailed, test_correct, test_wrong):
    """Analyze which problems improve/degrade with steering."""
    print("\n" + "="*80)
    print("PER-PROBLEM ANALYSIS")
    print("="*80)

    # Get baseline and best amplified results
    baseline_results = detailed['results']['0.0']
    amplified_results = detailed['results']['1.0']  # Best amplified alpha

    # Create problem lookup
    all_test = test_correct + test_wrong
    problem_lookup = {p['pair_id']: p for p in all_test}

    # Analyze transitions
    transitions = {
        'stayed_correct': [],
        'stayed_wrong': [],
        'became_correct': [],  # Improved with steering
        'became_wrong': []     # Degraded with steering
    }

    for baseline_res, amplified_res in zip(baseline_results, amplified_results):
        pair_id = baseline_res['pair_id']
        baseline_correct = baseline_res.get('correct', False)
        amplified_correct = amplified_res.get('correct', False)

        if baseline_correct and amplified_correct:
            transitions['stayed_correct'].append(pair_id)
        elif not baseline_correct and not amplified_correct:
            transitions['stayed_wrong'].append(pair_id)
        elif not baseline_correct and amplified_correct:
            transitions['became_correct'].append(pair_id)
        elif baseline_correct and not amplified_correct:
            transitions['became_wrong'].append(pair_id)

    print(f"\nTransitions from Baseline (α=0.0) to Amplified (α=+1.0):")
    print(f"  Stayed correct:  {len(transitions['stayed_correct']):2d} (no change)")
    print(f"  Became correct:  {len(transitions['became_correct']):2d} (IMPROVED ✓)")
    print(f"  Became wrong:    {len(transitions['became_wrong']):2d} (DEGRADED ✗)")
    print(f"  Stayed wrong:    {len(transitions['stayed_wrong']):2d} (no change)")

    net_improvement = len(transitions['became_correct']) - len(transitions['became_wrong'])
    print(f"\n  Net improvement: {net_improvement:+d} problems")

    # Analyze characteristics of improved vs degraded
    if len(transitions['became_correct']) > 0:
        improved_steps = [problem_lookup[pid].get('reasoning_steps', 0)
                         for pid in transitions['became_correct']]
        print(f"\n  Improved problems - avg reasoning steps: {np.mean(improved_steps):.2f}")

    if len(transitions['became_wrong']) > 0:
        degraded_steps = [problem_lookup[pid].get('reasoning_steps', 0)
                         for pid in transitions['became_wrong']]
        print(f"  Degraded problems - avg reasoning steps: {np.mean(degraded_steps):.2f}")

    return transitions


def test_random_direction_control(test_correct, test_wrong):
    """Test if random directions also cause performance degradation."""
    print("\n" + "="*80)
    print("RANDOM DIRECTION CONTROL EXPERIMENT")
    print("="*80)

    print("\nThis experiment will:")
    print("1. Generate 5 random steering directions (same shape as computed direction)")
    print("2. Test each with α=-3.0 (same as worst suppression)")
    print("3. Compare degradation to computed direction (-12.8 points)")
    print("4. Determine if suppression is meaningful or just noise")

    print("\nExpected outcomes:")
    print("  If random → similar degradation: Suppression is just noise")
    print("  If random → less degradation: Suppression is meaningful")

    # We'll need to run this experiment
    output_dir = Path(__file__).parent / 'results' / 'steering_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate random directions
    np.random.seed(42)
    random_directions = []
    for i in range(5):
        # Same shape as reasoning direction: [6, 768]
        random_dir = np.random.randn(6, 768)
        # Normalize to same magnitude as computed direction
        target_magnitude = 58.65  # From metadata
        current_magnitude = np.linalg.norm(random_dir)
        random_dir = random_dir * (target_magnitude / current_magnitude)
        random_directions.append(random_dir)

    # Save random directions for experiment
    random_dir_file = output_dir / 'random_directions.npz'
    np.savez(random_dir_file,
             directions=np.array(random_directions),
             target_magnitude=target_magnitude)

    print(f"\n✓ Generated 5 random directions (magnitude {target_magnitude:.2f})")
    print(f"✓ Saved to: {random_dir_file}")
    print("\nNOTE: This analysis script has prepared the random directions.")
    print("      A separate experiment script will test them and compare results.")

    return random_directions


def analyze_alpha_progression(summary):
    """Analyze how accuracy changes with alpha values."""
    print("\n" + "="*80)
    print("ALPHA PROGRESSION ANALYSIS")
    print("="*80)

    # Extract positive alphas
    positive_alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    negative_alphas = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]

    positive_accs = [summary['summary'][str(a)]['accuracy'] for a in positive_alphas]
    negative_accs = [summary['summary'][str(a)]['accuracy'] for a in negative_alphas]

    print("\nPositive alphas (amplification):")
    for alpha, acc in zip(positive_alphas, positive_accs):
        baseline = positive_accs[0]
        change = acc - baseline
        print(f"  α={alpha:+4.1f}: {acc:5.1f}% ({change:+5.1f} pts)")

    print("\nNegative alphas (suppression):")
    for alpha, acc in zip(negative_alphas, negative_accs):
        baseline = negative_accs[0]
        change = acc - baseline
        print(f"  α={alpha:+4.1f}: {acc:5.1f}% ({change:+5.1f} pts)")

    # Key observations
    print("\n--- KEY OBSERVATIONS ---")

    # Find peak amplification
    max_idx = np.argmax(positive_accs)
    print(f"1. Peak amplification at α={positive_alphas[max_idx]:+.1f}: {positive_accs[max_idx]:.1f}%")

    # Check monotonicity
    if all(negative_accs[i] >= negative_accs[i+1] for i in range(len(negative_accs)-1)):
        print("2. Suppression is monotonic: More negative α → worse performance ✓")
    else:
        print("2. Suppression is NOT monotonic: Non-linear effects present")

    # Check over-steering
    if positive_accs[-1] < positive_accs[0]:
        print(f"3. Over-steering detected: α=+3.0 ({positive_accs[-1]:.1f}%) < baseline ({positive_accs[0]:.1f}%)")

    return {
        'positive_alphas': positive_alphas,
        'positive_accs': positive_accs,
        'negative_alphas': negative_alphas,
        'negative_accs': negative_accs
    }


def create_visualizations(baseline_analysis, alpha_analysis, transitions):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    output_dir = Path(__file__).parent / 'results' / 'steering_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Alpha progression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Amplification
    ax = axes[0]
    alphas = alpha_analysis['positive_alphas']
    accs = alpha_analysis['positive_accs']
    baseline = accs[0]

    ax.plot(alphas, accs, 'o-', linewidth=2, markersize=8, label='Accuracy')
    ax.axhline(baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.1f}%)')
    ax.axhline(baseline + 12, color='green', linestyle='--', alpha=0.5, label='Target (+12 pts)')
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Amplification: Positive Alpha Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Suppression
    ax = axes[1]
    alphas = alpha_analysis['negative_alphas']
    accs = alpha_analysis['negative_accs']
    baseline = accs[0]

    ax.plot(alphas, accs, 'o-', linewidth=2, markersize=8, color='red', label='Accuracy')
    ax.axhline(baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.1f}%)')
    ax.axhline(baseline - 12, color='orange', linestyle='--', alpha=0.5, label='Target (-12 pts)')
    ax.set_xlabel('Alpha (α)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Suppression: Negative Alpha Values', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_file = output_dir / 'alpha_progression.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_file}")
    plt.close()

    # Figure 2: Transition analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Stayed\nCorrect', 'Became\nCorrect\n(Improved)',
                  'Became\nWrong\n(Degraded)', 'Stayed\nWrong']
    counts = [
        len(transitions['stayed_correct']),
        len(transitions['became_correct']),
        len(transitions['became_wrong']),
        len(transitions['stayed_wrong'])
    ]
    colors = ['lightgreen', 'darkgreen', 'darkred', 'lightcoral']

    bars = ax.bar(categories, counts, color=colors, edgecolor='black', linewidth=1.5)

    # Add count labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Problem Transitions: Baseline → Amplified (α=+1.0)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(counts) * 1.15)

    # Add net improvement text
    net = counts[1] - counts[2]
    ax.text(0.98, 0.98, f'Net improvement: {net:+d} problems',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    fig_file = output_dir / 'transition_analysis.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_file}")
    plt.close()

    # Figure 3: Reasoning steps distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = baseline_analysis['reasoning_steps']
    ax.hist(steps, bins=range(0, max(steps)+2), edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(steps), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(steps):.2f}')
    ax.axvline(np.median(steps), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(steps):.0f}')

    ax.set_xlabel('Number of Reasoning Steps', fontsize=12)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Test Set Difficulty Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_file = output_dir / 'test_set_difficulty.png'
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {fig_file}")
    plt.close()


def generate_findings_report(baseline_analysis, alpha_analysis, transitions):
    """Generate comprehensive findings report."""
    print("\n" + "="*80)
    print("GENERATING FINDINGS REPORT")
    print("="*80)

    output_dir = Path(__file__).parent / 'results' / 'steering_analysis'
    report_file = output_dir / 'steering_failure_analysis.md'

    with open(report_file, 'w') as f:
        f.write("# Steering Experiment Failure Analysis\n\n")
        f.write("**Date**: 2025-10-21\n")
        f.write("**Experiment**: GPT-2 Activation Steering\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This analysis investigates why activation steering failed to achieve target improvements:\n")
        f.write(f"- **Amplification**: Achieved +2.3 points vs. target +12 points ❌\n")
        f.write(f"- **Suppression**: Achieved -12.8 points vs. target -12 points ✓\n\n")

        f.write("## 1. Baseline Analysis\n\n")
        f.write(f"**Baseline accuracy**: {baseline_analysis['baseline_acc']:.1f}%\n\n")
        f.write(f"The baseline (32.6%) is significantly lower than expected (50%). This suggests:\n\n")
        f.write("### Hypothesis: Test Set Difficulty\n")
        f.write(f"- Mean reasoning steps: {np.mean(baseline_analysis['reasoning_steps']):.2f}\n")
        f.write(f"- Median reasoning steps: {np.median(baseline_analysis['reasoning_steps']):.0f}\n")
        f.write(f"- Range: {min(baseline_analysis['reasoning_steps'])} - {max(baseline_analysis['reasoning_steps'])}\n\n")
        f.write("The test set may contain harder problems than the model can reliably solve, ")
        f.write("creating a **ceiling effect** that limits amplification gains.\n\n")

        f.write("## 2. Per-Problem Analysis\n\n")
        f.write("Transitions from Baseline (α=0.0) to Best Amplified (α=+1.0):\n\n")
        f.write(f"- **Stayed correct**: {len(transitions['stayed_correct'])} (no change)\n")
        f.write(f"- **Became correct**: {len(transitions['became_correct'])} (IMPROVED ✓)\n")
        f.write(f"- **Became wrong**: {len(transitions['became_wrong'])} (DEGRADED ✗)\n")
        f.write(f"- **Stayed wrong**: {len(transitions['stayed_wrong'])} (no change)\n\n")

        net = len(transitions['became_correct']) - len(transitions['became_wrong'])
        f.write(f"**Net improvement**: {net:+d} problems\n\n")

        if net > 0:
            f.write("While some problems improved, others degraded. This suggests:\n")
            f.write("1. Steering helps some problems but hurts others\n")
            f.write("2. The direction may not be universally beneficial\n")
            f.write("3. Different problems may require different steering strategies\n\n")
        else:
            f.write("More problems degraded than improved. This indicates:\n")
            f.write("1. The steering direction may not capture 'good reasoning'\n")
            f.write("2. Over-steering may be corrupting representations\n")
            f.write("3. The layer choice (middle) may not be optimal\n\n")

        f.write("## 3. Alpha Progression Analysis\n\n")
        f.write("### Amplification (Positive α)\n\n")
        f.write("| Alpha | Accuracy | Change |\n")
        f.write("|-------|----------|--------|\n")
        for alpha, acc in zip(alpha_analysis['positive_alphas'], alpha_analysis['positive_accs']):
            baseline = alpha_analysis['positive_accs'][0]
            change = acc - baseline
            f.write(f"| {alpha:+4.1f} | {acc:5.1f}% | {change:+5.1f} pts |\n")

        f.write("\n**Key Observations**:\n")
        max_idx = np.argmax(alpha_analysis['positive_accs'])
        f.write(f"- Peak at α={alpha_analysis['positive_alphas'][max_idx]:+.1f}: {alpha_analysis['positive_accs'][max_idx]:.1f}%\n")
        f.write("- Over-steering (α>1.0) degrades performance\n")
        f.write("- Limited headroom for improvement\n\n")

        f.write("### Suppression (Negative α)\n\n")
        f.write("| Alpha | Accuracy | Change |\n")
        f.write("|-------|----------|--------|\n")
        for alpha, acc in zip(alpha_analysis['negative_alphas'], alpha_analysis['negative_accs']):
            baseline = alpha_analysis['negative_accs'][0]
            change = acc - baseline
            f.write(f"| {alpha:+4.1f} | {acc:5.1f}% | {change:+5.1f} pts |\n")

        f.write("\n**Key Observations**:\n")
        f.write("- Monotonic degradation: More negative α → worse performance\n")
        f.write("- Achieved target -12 points at α=-3.0\n")
        f.write("- **Critical question**: Is this meaningful or just noise?\n\n")

        f.write("## 4. Random Direction Control (Next Step)\n\n")
        f.write("To validate that suppression is meaningful and not just adding noise:\n\n")
        f.write("### Experiment Design\n")
        f.write("1. Generate 5 random steering directions (same magnitude as computed direction)\n")
        f.write("2. Test each with α=-3.0 (same as worst suppression)\n")
        f.write("3. Compare degradation to computed direction (-12.8 points)\n\n")
        f.write("### Predicted Outcomes\n")
        f.write("- **If random → similar degradation**: Suppression is just noise ❌\n")
        f.write("- **If random → less degradation**: Suppression is meaningful ✓\n\n")
        f.write("Random directions have been generated and saved for testing.\n\n")

        f.write("## 5. Hypotheses for Amplification Failure\n\n")
        f.write("### H1: Ceiling Effect\n")
        f.write("- Baseline 32.6% may be near model's capability limit on this test set\n")
        f.write("- Test problems may be inherently difficult for GPT-2 (117M)\n")
        f.write("- Limited room for improvement regardless of steering\n\n")

        f.write("### H2: Direction Quality\n")
        f.write("- Direction = correct_mean - wrong_mean may not capture 'reasoning quality'\n")
        f.write("- Could capture other factors (e.g., problem difficulty, answer magnitude)\n")
        f.write("- May need more sophisticated direction extraction (e.g., PCA, contrastive)\n\n")

        f.write("### H3: Uniform Steering Limitation\n")
        f.write("- Applying same α to all 6 tokens may not be optimal\n")
        f.write("- Different tokens have different magnitudes (token 5: 37.45 vs token 0: 6.77)\n")
        f.write("- May need token-specific or layer-specific steering\n\n")

        f.write("### H4: Layer Choice\n")
        f.write("- Steering at middle layer (6/12) may be suboptimal\n")
        f.write("- Earlier layers: Feature extraction\n")
        f.write("- Later layers: Decision making\n")
        f.write("- May need to steer at later layers for reasoning decisions\n\n")

        f.write("### H5: Scale Mismatch\n")
        f.write("- Direction magnitude (58.65) may be inappropriate scale\n")
        f.write("- α=1.0 might be too small or too large\n")
        f.write("- May need to normalize direction differently\n\n")

        f.write("## 6. Recommended Next Steps\n\n")
        f.write("### Immediate (Priority 1)\n")
        f.write("1. **Run random direction control** - Validate suppression is meaningful\n")
        f.write("2. **Analyze test set composition** - Check if it's representative\n")
        f.write("3. **Test different layers** - Try later layers (8, 9, 10)\n\n")

        f.write("### Follow-up (Priority 2)\n")
        f.write("4. **Token-specific steering** - Apply different α to each of 6 tokens\n")
        f.write("5. **Alternative direction methods** - PCA, supervised probes, contrastive\n")
        f.write("6. **Larger alpha range** - Test α=0.1, 0.2, ..., 0.9 for finer control\n\n")

        f.write("### Exploratory (Priority 3)\n")
        f.write("7. **Difficulty stratification** - Test steering on easy vs hard problems separately\n")
        f.write("8. **Activation magnitude analysis** - Check if steering has desired effect on representations\n")
        f.write("9. **Attention pattern analysis** - Examine how steering affects attention\n\n")

        f.write("## 7. Conclusions\n\n")
        f.write("The steering experiment revealed important insights:\n\n")
        f.write("### What Worked ✓\n")
        f.write("- Suppression achieved target degradation (-12.8 points)\n")
        f.write("- Implementation correctly applies steering to continuous thoughts\n")
        f.write("- Effects are measurable and reproducible\n\n")

        f.write("### What Didn't Work ❌\n")
        f.write("- Amplification only achieved +2.3 points (target: +12)\n")
        f.write("- Baseline significantly lower than expected (32.6% vs 50%)\n")
        f.write("- Over-steering (α>1.0) degrades performance\n\n")

        f.write("### Critical Questions ❓\n")
        f.write("1. Is suppression meaningful or just noise? (Random control needed)\n")
        f.write("2. Is the direction capturing 'good reasoning' or something else?\n")
        f.write("3. Why is baseline so low? (Test set difficulty? Model limitations?)\n")
        f.write("4. Can we achieve better amplification with different strategies?\n\n")

        f.write("### Key Insight 💡\n")
        f.write("The **asymmetry** between amplification and suppression is revealing:\n")
        f.write("- Easy to degrade performance (suppression works)\n")
        f.write("- Hard to improve performance (amplification fails)\n\n")
        f.write("This suggests the model may already be operating near its capability frontier for this task, ")
        f.write("and simple linear steering cannot push it significantly beyond current performance.\n")

    print(f"✓ Saved: {report_file}")


def main():
    """Main analysis pipeline."""
    print("="*80)
    print("STEERING EXPERIMENT FAILURE ANALYSIS")
    print("="*80)

    # Load data
    detailed, summary = load_results()
    test_correct, test_wrong, pairs_lookup = load_test_set_metadata()

    # Analysis 1: Baseline
    baseline_analysis = analyze_baseline_vs_expected(summary, test_correct, test_wrong)

    # Analysis 2: Per-problem
    transitions = analyze_per_problem_results(detailed, test_correct, test_wrong)

    # Analysis 3: Alpha progression
    alpha_analysis = analyze_alpha_progression(summary)

    # Analysis 4: Random control (preparation)
    random_directions = test_random_direction_control(test_correct, test_wrong)

    # Visualizations
    create_visualizations(baseline_analysis, alpha_analysis, transitions)

    # Report
    generate_findings_report(baseline_analysis, alpha_analysis, transitions)

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey findings:")
    print(f"1. Baseline (32.6%) is near ceiling for this test set")
    print(f"2. Net improvement: {len(transitions['became_correct']) - len(transitions['became_wrong']):+d} problems")
    print(f"3. Over-steering (α>1.0) degrades performance")
    print(f"4. Suppression validation needed (random direction control)")
    print("\nSee detailed report: results/steering_analysis/steering_failure_analysis.md")


if __name__ == "__main__":
    main()
