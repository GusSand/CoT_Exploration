"""
Run feature analysis on FULL DATASET SAE models to generate feature catalog.
This is needed to create the updated cross_position_token_0_comparison visualization.
"""
import torch
import json
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from tqdm import tqdm
from scipy import stats

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "operation_circuits"))
from sae_model import SparseAutoencoder

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models_full_dataset"  # Full dataset models
ANALYSIS_DIR = BASE_DIR / "analysis"
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = ANALYSIS_DIR / "visualizations"

print("="*80)
print("SAE FEATURE ANALYSIS - FULL DATASET MODELS")
print("="*80)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Load test data
print("Loading test data...")
test_data_path = BASE_DIR / "results" / "enriched_test_data_with_cot.pt"
test_data = torch.load(test_data_path, weights_only=False)

activations = test_data['hidden_states'].to(device)
metadata = test_data['metadata']

print(f"✓ Loaded {len(activations)} test samples")

# Load full dataset SAE models
print("\nLoading full dataset SAE models...")
saes = {}
for position in range(6):
    model_path = MODELS_DIR / f"pos_{position}_final.pt"
    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        continue

    sae = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
    sae.load_state_dict(torch.load(model_path, map_location=device))
    sae.eval()
    saes[position] = sae
    print(f"  ✓ Loaded Position {position}")

print(f"✓ Loaded {len(saes)} SAE models")

# Extract features for all positions
print("\nExtracting features from full dataset models...")
extracted_features = {}

for position in range(6):
    if position not in saes:
        continue

    # Filter samples for this position
    pos_indices = [i for i, p in enumerate(metadata['positions']) if p == position]
    pos_activations = activations[pos_indices]
    pos_cot_token_ids = [metadata['cot_token_ids'][i] for i in pos_indices]

    print(f"\nPosition {position}: {len(pos_activations)} samples")

    # Extract features
    with torch.no_grad():
        _, features = saes[position](pos_activations)

    extracted_features[f"position_{position}"] = {
        'features': features.cpu(),
        'cot_token_ids': pos_cot_token_ids
    }

print(f"\n✓ Extracted features for all positions")

# Analyze feature-token correlations
print("\n" + "="*80)
print("FEATURE-COT TOKEN CORRELATION ANALYSIS")
print("="*80)

catalog = {"positions": {}}

for position in range(6):
    if position not in saes:
        continue

    print(f"\nAnalyzing position {position}...")

    pos_key = f"position_{position}"
    features = extracted_features[pos_key]['features']
    cot_token_ids = extracted_features[pos_key]['cot_token_ids']

    n_samples, n_features = features.shape

    # Find top 100 features by activation
    feature_activations = features.mean(dim=0)
    top_100_indices = torch.argsort(feature_activations, descending=True)[:100]

    position_catalog = []

    for feature_idx in tqdm(top_100_indices, desc=f"Position {position}"):
        feature_idx = feature_idx.item()
        feature_acts = features[:, feature_idx].numpy()

        # Find top activating samples (90th percentile)
        threshold = np.percentile(feature_acts, 90)
        top_samples = np.where(feature_acts > threshold)[0]

        if len(top_samples) < 10:
            continue

        # Count tokens in top activating samples
        token_counts_top = Counter()
        for sample_idx in top_samples:
            tokens = cot_token_ids[sample_idx]
            if isinstance(tokens, list):
                for token in tokens:
                    token_counts_top[token] += 1

        # Count tokens in all samples (baseline)
        token_counts_all = Counter()
        for tokens in cot_token_ids:
            if isinstance(tokens, list):
                for token in tokens:
                    token_counts_all[token] += 1

        # Calculate enrichment for top 10 tokens
        enriched_tokens = []
        for token, count_top in token_counts_top.most_common(10):
            count_all = token_counts_all[token]

            # Enrichment = (count_top / len(top_samples)) / (count_all / n_samples)
            freq_top = count_top / len(top_samples)
            freq_all = count_all / n_samples
            enrichment = freq_top / freq_all if freq_all > 0 else 0

            # Statistical significance (Fisher's exact test)
            # Contingency table: [[top_with_token, top_without], [other_with, other_without]]
            a = count_top
            b = len(top_samples) - a
            c = count_all - a
            d = n_samples - len(top_samples) - c

            # Ensure all values are non-negative
            if a >= 0 and b >= 0 and c >= 0 and d >= 0 and a + c > 0 and b + d > 0:
                try:
                    _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
                except Exception as e:
                    p_value = 1.0
            else:
                p_value = 1.0

            token_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)

            enriched_tokens.append({
                "token_str": token_str,
                "token_id": int(token) if isinstance(token, int) else token,
                "enrichment": float(enrichment),
                "p_value": float(p_value),
                "count_top": int(count_top),
                "count_all": int(count_all)
            })

        if enriched_tokens:
            position_catalog.append({
                "feature_id": int(feature_idx),
                "mean_activation": float(feature_activations[feature_idx]),
                "n_top_samples": int(len(top_samples)),
                "enriched_tokens": enriched_tokens
            })

    catalog["positions"][str(position)] = {
        "n_features": int(n_features),
        "n_interpretable": len(position_catalog),
        "top_100_features": position_catalog
    }

    print(f"  Found {len(position_catalog)} interpretable features")

# Save catalog
catalog_path = ANALYSIS_DIR / "feature_catalog_full_dataset.json"
with open(catalog_path, 'w') as f:
    json.dump(catalog, f, indent=2)

print(f"\n✓ Saved feature catalog: {catalog_path}")

print("\n" + "="*80)
print("FEATURE ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults:")
for position in range(6):
    if str(position) in catalog["positions"]:
        n_interp = catalog["positions"][str(position)]["n_interpretable"]
        print(f"  Position {position}: {n_interp} interpretable features")
