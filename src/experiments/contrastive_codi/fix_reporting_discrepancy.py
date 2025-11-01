#!/usr/bin/env python3
"""Fix the reporting discrepancy by regenerating comparison results from individual probe files."""

import json
import numpy as np
from pathlib import Path

def fix_reporting_discrepancy():
    """Regenerate probe_comparison_results.json from individual probe result files."""

    results_dir = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi/results")

    print("🔧 Fixing reporting discrepancy...")
    print(f"📁 Results directory: {results_dir}")

    # Load individual probe results
    print("📂 Loading individual probe result files...")

    with open(results_dir / "ct_token_probe_results.json", 'r') as f:
        ct_results = json.load(f)

    with open(results_dir / "regular_hidden_probe_results.json", 'r') as f:
        regular_results = json.load(f)

    print(f"✅ Loaded {len(ct_results)} CT probe results")
    print(f"✅ Loaded {len(regular_results)} regular probe results")

    # Check the individual results
    print("\\n🔍 Individual CT probe results:")
    for result in ct_results[:3]:  # Show first 3
        layer = result['layer']
        cv_acc = result['cv_accuracy_mean']
        holdout_acc = result['holdout_accuracy']
        print(f"  {layer}: CV accuracy = {cv_acc:.3f}, Holdout accuracy = {holdout_acc:.3f}")

    print("\\n🔍 Individual regular probe results:")
    for result in regular_results[:3]:  # Show first 3
        layer = result['layer']
        cv_acc = result['cv_accuracy_mean']
        holdout_acc = result['holdout_accuracy']
        print(f"  {layer}: CV accuracy = {cv_acc:.3f}, Holdout accuracy = {holdout_acc:.3f}")

    # Regenerate comparison results
    print("\\n🔄 Regenerating comparison results...")

    comparison = {
        'ct_token_results': [],
        'regular_hidden_results': [],
        'summary': {}
    }

    ct_accuracies = []
    reg_accuracies = []

    for ct_res, reg_res in zip(ct_results, regular_results):
        layer = ct_res['layer']
        ct_acc = ct_res['cv_accuracy_mean']
        reg_acc = reg_res['cv_accuracy_mean']

        ct_accuracies.append(ct_acc)
        reg_accuracies.append(reg_acc)

        comparison['ct_token_results'].append({
            'layer': layer,
            'accuracy': ct_acc,
            'accuracy_std': ct_res['cv_accuracy_std'],
            'auc': ct_res['cv_auc_mean'],
            'auc_std': ct_res['cv_auc_std']
        })

        comparison['regular_hidden_results'].append({
            'layer': layer,
            'accuracy': reg_acc,
            'accuracy_std': reg_res['cv_accuracy_std'],
            'auc': reg_res['cv_auc_mean'],
            'auc_std': reg_res['cv_auc_std']
        })

    # Summary statistics
    comparison['summary'] = {
        'ct_tokens_best_accuracy': max(ct_accuracies),
        'ct_tokens_mean_accuracy': np.mean(ct_accuracies),
        'regular_hidden_best_accuracy': max(reg_accuracies),
        'regular_hidden_mean_accuracy': np.mean(reg_accuracies),
        'ct_tokens_advantage': np.mean(ct_accuracies) - np.mean(reg_accuracies),
        'random_baseline': 0.5
    }

    # Save corrected comparison results
    output_file = results_dir / "probe_comparison_results_CORRECTED.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"✅ Corrected comparison saved to: {output_file}")

    # Show summary
    print("\\n📊 Corrected Summary:")
    print(f"  CT Tokens Mean Accuracy: {comparison['summary']['ct_tokens_mean_accuracy']:.3f}")
    print(f"  Regular Hidden Mean Accuracy: {comparison['summary']['regular_hidden_mean_accuracy']:.3f}")
    print(f"  CT Tokens Advantage: {comparison['summary']['ct_tokens_advantage']:+.3f}")
    print(f"  Random Baseline: {comparison['summary']['random_baseline']:.3f}")

    # Compare with existing (incorrect) file
    print("\\n🔍 Comparing with existing (incorrect) file:")

    try:
        with open(results_dir / "probe_comparison_results.json", 'r') as f:
            old_comparison = json.load(f)

        old_ct_mean = old_comparison['summary']['ct_tokens_mean_accuracy']
        old_reg_mean = old_comparison['summary']['regular_hidden_mean_accuracy']

        print(f"  OLD CT Tokens Mean: {old_ct_mean:.3f}")
        print(f"  OLD Regular Hidden Mean: {old_reg_mean:.3f}")
        print(f"  NEW CT Tokens Mean: {comparison['summary']['ct_tokens_mean_accuracy']:.3f}")
        print(f"  NEW Regular Hidden Mean: {comparison['summary']['regular_hidden_mean_accuracy']:.3f}")

        if abs(old_ct_mean - 1.0) < 0.01:
            print("  ❌ Confirmed: Old file has incorrect 1.000 values")
        else:
            print("  ✅ Old file looks correct")

    except FileNotFoundError:
        print("  ⚠️  Original comparison file not found")

    # Replace the incorrect file
    print("\\n🔄 Replacing incorrect comparison file...")
    old_file = results_dir / "probe_comparison_results.json"
    backup_file = results_dir / "probe_comparison_results_BACKUP.json"

    # Backup old file
    if old_file.exists():
        old_file.rename(backup_file)
        print(f"  📁 Backed up old file to: {backup_file}")

    # Replace with corrected file
    output_file.rename(old_file)
    print(f"  ✅ Replaced with corrected file: {old_file}")

    print("\\n🎉 Reporting discrepancy fixed!")
    print("  Now regenerate the final report to see correct results.")

if __name__ == "__main__":
    fix_reporting_discrepancy()