"""
Re-analyze key features (1893 from pos 3, 148 from pos 1) with full dataset models.
Compare with baseline to see if interpretability holds.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

from sae_model import SparseAutoencoder

# Paths
BASE_DIR = Path(__file__).parent.parent
VIZ_DIR = BASE_DIR / "analysis" / "visualizations"
MODELS_DIR_BASELINE = BASE_DIR / "models"
MODELS_DIR_FULL = BASE_DIR / "models_full_dataset"

print("="*80)
print("KEY FEATURE ANALYSIS: Baseline vs Full Dataset Models")
print("="*80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Load test data
print("Loading test data...")
test_data_path = BASE_DIR / "results" / "enriched_test_data_with_cot.pt"
test_data = torch.load(test_data_path, weights_only=False)

activations = test_data['hidden_states'].to(device)
metadata = test_data['metadata']

print(f"✓ Loaded {len(activations)} test samples")

# ============================================================================
# Analyze Position 3, Feature 1893
# ============================================================================
print("\n" + "="*80)
print("POSITION 3, FEATURE 1893")
print("="*80)

position = 3
feature_id = 1893

# Load baseline model
baseline_model_path = MODELS_DIR_BASELINE / "pos_3_final.pt"
sae_baseline = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
sae_baseline.load_state_dict(torch.load(baseline_model_path, map_location=device))
sae_baseline.eval()

# Load full dataset model
full_model_path = MODELS_DIR_FULL / "pos_3_final.pt"
sae_full = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
sae_full.load_state_dict(torch.load(full_model_path, map_location=device))
sae_full.eval()

# Filter test data for position 3
pos3_indices = [i for i, p in enumerate(metadata['positions']) if p == position]
pos3_activations = activations[pos3_indices]
pos3_cot_sequences = [metadata['cot_token_ids'][i] for i in pos3_indices]

print(f"\nAnalyzing {len(pos3_activations)} samples from position {position}")

# Extract features from both models
with torch.no_grad():
    _, features_baseline = sae_baseline(pos3_activations)
    _, features_full = sae_full(pos3_activations)

# Get activations for feature 1893
feature_acts_baseline = features_baseline[:, feature_id].cpu().numpy()
feature_acts_full = features_full[:, feature_id].cpu().numpy()

# Find top activating samples for each model
threshold = np.percentile(feature_acts_baseline, 90)
top_baseline = np.where(feature_acts_baseline > threshold)[0]

threshold_full = np.percentile(feature_acts_full, 90)
top_full = np.where(feature_acts_full > threshold_full)[0]

print(f"\n📊 Feature 1893 Statistics:")
print(f"  Baseline: {len(top_baseline)} top activations (threshold: {threshold:.4f})")
print(f"  Full Dataset: {len(top_full)} top activations (threshold: {threshold_full:.4f})")

# Analyze token patterns for baseline
print(f"\n🔍 Baseline Model - Top Token Patterns:")
token_counts_baseline = Counter()
for idx in top_baseline[:50]:  # Top 50 samples
    tokens = pos3_cot_sequences[idx]
    if isinstance(tokens, list):
        for token in tokens:
            token_counts_baseline[token] += 1

for token, count in token_counts_baseline.most_common(10):
    token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
    pct = (count / len(top_baseline[:50])) * 100
    print(f"  '{token_str}': {count} ({pct:.1f}%)")

# Analyze token patterns for full dataset
print(f"\n🔍 Full Dataset Model - Top Token Patterns:")
token_counts_full = Counter()
for idx in top_full[:50]:  # Top 50 samples
    tokens = pos3_cot_sequences[idx]
    if isinstance(tokens, list):
        for token in tokens:
            token_counts_full[token] += 1

for token, count in token_counts_full.most_common(10):
    token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
    pct = (count / len(top_full[:50])) * 100
    print(f"  '{token_str}': {count} ({pct:.1f}%)")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline distribution
ax1 = axes[0]
ax1.hist(feature_acts_baseline, bins=50, alpha=0.7, color='salmon', edgecolor='black')
ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'90th percentile: {threshold:.3f}')
ax1.set_xlabel('Feature Activation', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Baseline Model (800 problems)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Full dataset distribution
ax2 = axes[1]
ax2.hist(feature_acts_full, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
ax2.axvline(threshold_full, color='green', linestyle='--', linewidth=2, label=f'90th percentile: {threshold_full:.3f}')
ax2.set_xlabel('Feature Activation', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Full Dataset Model (7,473 problems)', fontweight='bold', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Position 3, Feature 1893: Activation Distribution Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = VIZ_DIR / "feature_1893_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_path}")
plt.close()

# ============================================================================
# Analyze Position 1, Feature 148
# ============================================================================
print("\n" + "="*80)
print("POSITION 1, FEATURE 148")
print("="*80)

position = 1
feature_id = 148

# Load baseline model
baseline_model_path = MODELS_DIR_BASELINE / "pos_1_final.pt"
sae_baseline = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
sae_baseline.load_state_dict(torch.load(baseline_model_path, map_location=device))
sae_baseline.eval()

# Load full dataset model
full_model_path = MODELS_DIR_FULL / "pos_1_final.pt"
sae_full = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
sae_full.load_state_dict(torch.load(full_model_path, map_location=device))
sae_full.eval()

# Filter test data for position 1
pos1_indices = [i for i, p in enumerate(metadata['positions']) if p == position]
pos1_activations = activations[pos1_indices]
pos1_cot_sequences = [metadata['cot_token_ids'][i] for i in pos1_indices]

print(f"\nAnalyzing {len(pos1_activations)} samples from position {position}")

# Extract features from both models
with torch.no_grad():
    _, features_baseline = sae_baseline(pos1_activations)
    _, features_full = sae_full(pos1_activations)

# Get activations for feature 148
feature_acts_baseline = features_baseline[:, feature_id].cpu().numpy()
feature_acts_full = features_full[:, feature_id].cpu().numpy()

# Find top activating samples for each model
threshold = np.percentile(feature_acts_baseline, 90)
top_baseline = np.where(feature_acts_baseline > threshold)[0]

threshold_full = np.percentile(feature_acts_full, 90)
top_full = np.where(feature_acts_full > threshold_full)[0]

print(f"\n📊 Feature 148 Statistics:")
print(f"  Baseline: {len(top_baseline)} top activations (threshold: {threshold:.4f})")
print(f"  Full Dataset: {len(top_full)} top activations (threshold: {threshold_full:.4f})")

# Analyze token patterns for baseline
print(f"\n🔍 Baseline Model - Top Token Patterns:")
token_counts_baseline = Counter()
for idx in top_baseline[:50]:
    tokens = pos1_cot_sequences[idx]
    if isinstance(tokens, list):
        for token in tokens:
            token_counts_baseline[token] += 1

for token, count in token_counts_baseline.most_common(10):
    token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
    pct = (count / len(top_baseline[:50])) * 100
    print(f"  '{token_str}': {count} ({pct:.1f}%)")

# Analyze token patterns for full dataset
print(f"\n🔍 Full Dataset Model - Top Token Patterns:")
token_counts_full = Counter()
for idx in top_full[:50]:
    tokens = pos1_cot_sequences[idx]
    if isinstance(tokens, list):
        for token in tokens:
            token_counts_full[token] += 1

for token, count in token_counts_full.most_common(10):
    token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
    pct = (count / len(top_full[:50])) * 100
    print(f"  '{token_str}': {count} ({pct:.1f}%)")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline distribution
ax1 = axes[0]
ax1.hist(feature_acts_baseline, bins=50, alpha=0.7, color='salmon', edgecolor='black')
ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'90th percentile: {threshold:.3f}')
ax1.set_xlabel('Feature Activation', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Baseline Model (800 problems)', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Full dataset distribution
ax2 = axes[1]
ax2.hist(feature_acts_full, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
ax2.axvline(threshold_full, color='green', linestyle='--', linewidth=2, label=f'90th percentile: {threshold_full:.3f}')
ax2.set_xlabel('Feature Activation', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Full Dataset Model (7,473 problems)', fontweight='bold', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.suptitle('Position 1, Feature 148: Activation Distribution Comparison',
             fontsize=14, fontweight='bold')
plt.tight_layout()
output_path = VIZ_DIR / "feature_148_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization: {output_path}")
plt.close()

print("\n" + "="*80)
print("KEY FEATURE ANALYSIS COMPLETE!")
print("="*80)
print("\n📋 Summary:")
print("  - Feature 1893 (Position 3) and Feature 148 (Position 1) analyzed")
print("  - Compared activation patterns between baseline and full dataset models")
print("  - Token enrichment patterns examined for both models")
print(f"  - Visualizations saved to: {VIZ_DIR}")
