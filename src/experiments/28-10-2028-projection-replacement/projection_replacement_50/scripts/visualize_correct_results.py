import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)

def load_results(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data['results'], data['summary']

def plot_comprehensive_analysis(results, summary, output_dir):
    """Create comprehensive visualization"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    categories = ['No Intervention', 'With Intervention']
    accuracies = [summary['accuracy_no_intervention'], summary['accuracy_with_intervention']]
    colors = ['#3498db', '#e74c3c']
    bars = ax1.bar(categories, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy: With vs Without Intervention', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)

    # 2. Answer Changes
    ax2 = fig.add_subplot(gs[0, 2])
    changed = summary['answers_changed']
    unchanged = summary['total'] - changed
    ax2.pie([unchanged, changed], labels=['Unchanged', 'Changed'],
            autopct='%1.0f%%', colors=['#2ecc71', '#f39c12'],
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title(f'Answers Changed\n({changed}/{summary["total"]})',
                  fontsize=12, fontweight='bold')

    # 3. Position-wise number distribution
    ax3 = fig.add_subplot(gs[1, :])
    position_counts = {}
    for r in results:
        for pos in r['number_positions']:
            position_counts[pos] = position_counts.get(pos, 0) + 1

    if position_counts:
        positions = sorted(position_counts.keys())
        counts = [position_counts[p] for p in positions]
        pos_labels = ['BoT'] + [f'T{i}' for i in range(1, 7)]
        pos_labels_filtered = [pos_labels[p] for p in positions if p < len(pos_labels)]

        bars = ax3.bar(pos_labels_filtered, counts, color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=2)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax3.set_xlabel('Chain-of-Thought Position', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Examples with Numbers', fontsize=12, fontweight='bold')
        ax3.set_title('Where Do Numbers Appear in CoT?', fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

    # 4. Detailed breakdown
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    # Analyze what happened
    changed_correct_to_wrong = 0
    changed_wrong_to_correct = 0
    changed_wrong_to_wrong = 0

    for r in results:
        if r['predicted_with_intervention'] is not None:
            if r['predicted_no_intervention'] != r['predicted_with_intervention']:
                if r['correct_no_intervention'] and not r['correct_with_intervention']:
                    changed_correct_to_wrong += 1
                elif not r['correct_no_intervention'] and r['correct_with_intervention']:
                    changed_wrong_to_correct += 1
                else:
                    changed_wrong_to_wrong += 1

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  INTERVENTION EFFECT SUMMARY                                                  ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                ║
    ║  Total Examples: {summary['total']}                                                              ║
    ║                                                                                ║
    ║  Accuracy WITHOUT Intervention: {summary['accuracy_no_intervention']:.1f}% ({summary['correct_no_intervention']}/{summary['total']})                              ║
    ║  Accuracy WITH Intervention:    {summary['accuracy_with_intervention']:.1f}% ({summary['correct_with_intervention']}/{summary['total']})                              ║
    ║  Difference:                    +{summary['accuracy_with_intervention']-summary['accuracy_no_intervention']:.1f}%                                                  ║
    ║                                                                                ║
    ║  Answers Changed: {changed}/{summary['total']} ({changed/summary['total']*100:.0f}%)                                                    ║
    ║    • Correct → Wrong:  {changed_correct_to_wrong}                                                          ║
    ║    • Wrong → Correct:  {changed_wrong_to_correct}                                                          ║
    ║    • Wrong → Wrong:    {changed_wrong_to_wrong}                                                          ║
    ║                                                                                ║
    ║  KEY FINDING:                                                                  ║
    ║  Numbers in continuous CoT CAN affect final answers, but only in {changed/summary['total']*100:.0f}% of cases.   ║
    ║  The intervention slightly improves accuracy by +{summary['accuracy_with_intervention']-summary['accuracy_no_intervention']:.1f}%.                         ║
    ║                                                                                ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """

    ax4.text(0.5, 0.5, summary_text, fontsize=11, family='monospace',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('CODI-LLaMA: Number Embedding Intervention Analysis on GSM8K',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_dir / 'intervention_analysis_complete.png', dpi=300, bbox_inches='tight')
    print(f"[OK] Saved complete analysis to {output_dir / 'intervention_analysis_complete.png'}")
    plt.close()

def print_example_cases(results, output_file):
    """Print detailed example cases"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("EXAMPLES WHERE INTERVENTION CHANGED THE ANSWER\n")
        f.write("="*80 + "\n\n")

        changed_examples = [r for r in results if r['predicted_with_intervention'] is not None
                           and r['predicted_no_intervention'] != r['predicted_with_intervention']]

        for i, r in enumerate(changed_examples[:5], 1):
            f.write(f"Example {i}:\n")
            f.write(f"Question: {r['question'][:150]}...\n")
            f.write(f"Ground Truth: {r['ground_truth']}\n\n")

            f.write(f"WITHOUT Intervention:\n")
            f.write(f"  Answer: {r['answer_no_intervention']}\n")
            f.write(f"  Predicted: {r['predicted_no_intervention']}\n")
            f.write(f"  Correct: {'✓' if r['correct_no_intervention'] else '✗'}\n\n")

            f.write(f"WITH Intervention:\n")
            f.write(f"  Answer: {r['answer_with_intervention']}\n")
            f.write(f"  Predicted: {r['predicted_with_intervention']}\n")
            f.write(f"  Correct: {'✓' if r['correct_with_intervention'] else '✗'}\n\n")

            f.write(f"Number positions: {r['number_positions']}\n")
            f.write(f"Intervened positions: {r['intervened_positions']}\n")
            f.write("\n" + "-"*80 + "\n\n")

        f.write("\n" + "="*80 + "\n")
        f.write("EXAMPLES WHERE INTERVENTION DID NOT CHANGE THE ANSWER\n")
        f.write("="*80 + "\n\n")

        unchanged_examples = [r for r in results if r['predicted_with_intervention'] is not None
                             and r['predicted_no_intervention'] == r['predicted_with_intervention']]

        for i, r in enumerate(unchanged_examples[:5], 1):
            f.write(f"Example {i}:\n")
            f.write(f"Question: {r['question'][:150]}...\n")
            f.write(f"Ground Truth: {r['ground_truth']}\n")
            f.write(f"Predicted (both): {r['predicted_no_intervention']}\n")
            f.write(f"Correct: {'✓' if r['correct_no_intervention'] else '✗'}\n")
            f.write(f"Number positions: {r['number_positions']}\n")
            f.write("\n" + "-"*80 + "\n\n")

    print(f"[OK] Saved example cases to {output_file}")

def main():
    results_file = Path("C:/Users/Paper001/Documents/claude/results_correct/llama_intervention_results_50_correct.json")
    output_dir = Path("C:/Users/Paper001/Documents/claude/results_correct")

    print("Loading results...")
    results, summary = load_results(results_file)
    print(f"[OK] Loaded {len(results)} results")

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Accuracy WITHOUT intervention: {summary['accuracy_no_intervention']:.1f}%")
    print(f"Accuracy WITH intervention:    {summary['accuracy_with_intervention']:.1f}%")
    print(f"Difference:                    +{summary['accuracy_with_intervention']-summary['accuracy_no_intervention']:.1f}%")
    print(f"\nAnswers changed: {summary['answers_changed']}/{summary['total']} ({summary['answers_changed']/summary['total']*100:.0f}%)")

    print("\nGenerating visualizations...")
    plot_comprehensive_analysis(results, summary, output_dir)

    example_file = output_dir / "example_cases_correct.txt"
    print_example_cases(results, example_file)

    print("\n[OK] All analysis complete!")

if __name__ == "__main__":
    main()
