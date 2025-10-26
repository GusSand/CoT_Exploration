"""
Extract final metrics from baseline training results to create summary.
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
baseline_results_path = BASE_DIR / "results" / "sae_training_results.json"
summary_output_path = BASE_DIR / "analysis" / "sae_training_summary.json"

# Load baseline results
with open(baseline_results_path, 'r') as f:
    baseline = json.load(f)

# Extract final metrics for each position
summary = {}
for pos_key, history in baseline.items():
    summary[pos_key] = {
        'final_metrics': {
            'explained_variance': history['explained_variance'][-1],
            'test_loss': history['test_loss'][-1],
            'feature_death_rate': history['feature_death_rate'][-1],
            'l0_norm': history['l0_norm'][-1]
        }
    }

# Save summary
with open(summary_output_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Created baseline summary: {summary_output_path}")
print("\nFinal Metrics (Baseline - 800 problems):")
for pos in range(6):
    pos_key = str(pos)
    if pos_key in summary:
        metrics = summary[pos_key]['final_metrics']
        print(f"  Position {pos}: EV={metrics['explained_variance']:.4f}, "
              f"Val Loss={metrics['test_loss']:.4f}, "
              f"Death Rate={metrics['feature_death_rate']:.4f}")
