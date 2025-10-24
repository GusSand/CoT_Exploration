"""
Ablation study: Which layers besides L14 contribute to error prediction?

Test different layer combinations to find which layers are helpful:
- L14 only: 66.67% (baseline)
- L12+L14, L13+L14, L14+L15, L14+L16 (pairs)
- L12+L13+L14, L14+L15+L16 (triplets)
- All L12-L16: 69.40% (known)

Goal: Find minimal layer set that achieves near-optimal accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
import json
from tqdm import tqdm
from itertools import combinations


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 2048, n_features: int = 2048):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def encode(self, x):
        return torch.relu(self.encoder(x))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


def load_sae(weights_path: str, device='cuda'):
    checkpoint = torch.load(weights_path, map_location=device)
    sae = SparseAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        n_features=checkpoint['config']['n_features']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    return sae, checkpoint


def encode_dataset(dataset, sae, device='cuda'):
    """Encode full dataset once, return dict of features by layer."""
    print("Encoding dataset...")

    all_solutions = []

    # Process incorrect
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        thoughts = sol['continuous_thoughts']
        sae_features = {}
        for layer_name, thought_list in thoughts.items():
            thought_tensor = torch.tensor(thought_list, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, features = sae(thought_tensor)
            sae_features[layer_name] = features.cpu().numpy()

        all_solutions.append({
            'sae_features': sae_features,
            'is_correct': False
        })

    # Process correct
    for sol in tqdm(dataset['correct_solutions'], desc="Correct"):
        thoughts = sol['continuous_thoughts']
        sae_features = {}
        for layer_name, thought_list in thoughts.items():
            thought_tensor = torch.tensor(thought_list, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, features = sae(thought_tensor)
            sae_features[layer_name] = features.cpu().numpy()

        all_solutions.append({
            'sae_features': sae_features,
            'is_correct': True
        })

    return all_solutions


def create_feature_matrix(encoded_solutions, layer_subset, n_features=2048):
    """Create feature matrix using only specified layers."""
    n_samples = len(encoded_solutions)
    n_vectors = len(layer_subset) * 6  # 6 tokens per layer
    X = np.zeros((n_samples, n_features * n_vectors))
    y = np.zeros(n_samples, dtype=int)

    for i, sol in enumerate(encoded_solutions):
        y[i] = 1 if sol['is_correct'] else 0

        concat_features = []
        for layer_name in layer_subset:
            layer_features = sol['sae_features'][layer_name]  # [6, n_features]
            for token_idx in range(6):
                concat_features.append(layer_features[token_idx])

        X[i] = np.concatenate(concat_features)

    return X, y


def test_layer_combination(encoded_solutions, layer_subset, n_features=2048, test_size=0.2, random_state=42):
    """Train classifier on given layer subset and return test accuracy."""
    X, y = create_feature_matrix(encoded_solutions, layer_subset, n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=random_state, verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, X.shape[1]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("LAYER ABLATION STUDY - L12-L16 ERROR PREDICTION")
    print("="*80)

    # Load dataset
    dataset_path = 'src/experiments/sae_error_analysis/data/error_analysis_dataset_l12_l16.json'
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Dataset: {dataset['metadata']['total']} solutions")

    # Load SAE
    sae_path = 'src/experiments/sae_error_analysis/sae_l12_l16/sae_weights.pt'
    sae, checkpoint = load_sae(sae_path, device)
    n_features = checkpoint['config']['n_features']
    print(f"SAE: {n_features} features\n")

    # Encode dataset once
    encoded_solutions = encode_dataset(dataset, sae, device)

    all_layers = ['L12', 'L13', 'L14', 'L15', 'L16']

    print("\n" + "="*80)
    print("TESTING LAYER COMBINATIONS")
    print("="*80)

    results = []

    # Test single layers
    print("\n1. Single Layers:")
    for layer in all_layers:
        acc, feat_dim = test_layer_combination(encoded_solutions, [layer], n_features)
        results.append({
            'layers': [layer],
            'accuracy': acc,
            'feature_dim': feat_dim,
            'n_layers': 1
        })
        print(f"  {layer:20s}: {acc*100:.2f}% ({feat_dim:,} features)")

    # Test L14 + one other layer (pairs with L14)
    print("\n2. L14 + One Other Layer:")
    for layer in ['L12', 'L13', 'L15', 'L16']:
        layers = sorted([layer, 'L14'])
        acc, feat_dim = test_layer_combination(encoded_solutions, layers, n_features)
        results.append({
            'layers': layers,
            'accuracy': acc,
            'feature_dim': feat_dim,
            'n_layers': 2
        })
        print(f"  {'+'.join(layers):20s}: {acc*100:.2f}% ({feat_dim:,} features)")

    # Test interesting triplets
    print("\n3. Triplets:")
    triplets = [
        ['L12', 'L13', 'L14'],  # Earlier side
        ['L14', 'L15', 'L16'],  # Later side
        ['L12', 'L14', 'L16'],  # Spread out
        ['L13', 'L14', 'L15'],  # Centered on L14
    ]
    for layers in triplets:
        acc, feat_dim = test_layer_combination(encoded_solutions, layers, n_features)
        results.append({
            'layers': layers,
            'accuracy': acc,
            'feature_dim': feat_dim,
            'n_layers': 3
        })
        print(f"  {'+'.join(layers):20s}: {acc*100:.2f}% ({feat_dim:,} features)")

    # Test quads
    print("\n4. Quads:")
    quads = [
        ['L12', 'L13', 'L14', 'L15'],
        ['L13', 'L14', 'L15', 'L16'],
        ['L12', 'L13', 'L14', 'L16'],
        ['L12', 'L14', 'L15', 'L16'],
    ]
    for layers in quads:
        acc, feat_dim = test_layer_combination(encoded_solutions, layers, n_features)
        results.append({
            'layers': layers,
            'accuracy': acc,
            'feature_dim': feat_dim,
            'n_layers': 4
        })
        print(f"  {'+'.join(layers):20s}: {acc*100:.2f}% ({feat_dim:,} features)")

    # Test all 5 layers (known result)
    print("\n5. All Layers:")
    acc, feat_dim = test_layer_combination(encoded_solutions, all_layers, n_features)
    results.append({
        'layers': all_layers,
        'accuracy': acc,
        'feature_dim': feat_dim,
        'n_layers': 5
    })
    print(f"  {'+'.join(all_layers):20s}: {acc*100:.2f}% ({feat_dim:,} features)")

    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Sort by accuracy
    results_sorted = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    print("\nTop 10 Combinations by Accuracy:")
    for i, res in enumerate(results_sorted[:10], 1):
        layers_str = '+'.join(res['layers'])
        acc = res['accuracy'] * 100
        improvement = (res['accuracy'] - 0.6667) * 100  # vs L14-only baseline
        print(f"{i:2d}. {layers_str:25s}: {acc:.2f}% (+{improvement:+.2f} pts, {res['feature_dim']:,} features)")

    # Find best single layer
    single_layer_results = [r for r in results if r['n_layers'] == 1]
    best_single = max(single_layer_results, key=lambda x: x['accuracy'])
    print(f"\nBest single layer: {best_single['layers'][0]} ({best_single['accuracy']*100:.2f}%)")

    # Find best pair
    pair_results = [r for r in results if r['n_layers'] == 2]
    best_pair = max(pair_results, key=lambda x: x['accuracy'])
    print(f"Best pair: {'+'.join(best_pair['layers'])} ({best_pair['accuracy']*100:.2f}%)")

    # Find best triplet
    triplet_results = [r for r in results if r['n_layers'] == 3]
    best_triplet = max(triplet_results, key=lambda x: x['accuracy'])
    print(f"Best triplet: {'+'.join(best_triplet['layers'])} ({best_triplet['accuracy']*100:.2f}%)")

    # Marginal contribution analysis
    print("\n" + "="*80)
    print("MARGINAL CONTRIBUTION OF EACH LAYER")
    print("="*80)
    print("(How much does adding each layer to L14 improve accuracy?)")

    l14_only = next(r for r in results if r['layers'] == ['L14'])
    l14_acc = l14_only['accuracy']

    for layer in ['L12', 'L13', 'L15', 'L16']:
        pair = next(r for r in results if sorted(r['layers']) == sorted([layer, 'L14']))
        improvement = (pair['accuracy'] - l14_acc) * 100
        print(f"  L14 + {layer}: {pair['accuracy']*100:.2f}% ({improvement:+.2f} pts)")

    # Cost-benefit
    print("\n" + "="*80)
    print("COST-BENEFIT ANALYSIS")
    print("="*80)
    print("(Accuracy gain per additional feature dimension)")

    for res in results_sorted[:5]:
        layers_str = '+'.join(res['layers'])
        acc_gain = (res['accuracy'] - l14_acc) * 100
        feat_increase = res['feature_dim'] - l14_only['feature_dim']
        if feat_increase > 0:
            cost_per_pct = feat_increase / acc_gain if acc_gain > 0 else float('inf')
            print(f"  {layers_str:25s}: +{acc_gain:.2f}% for +{feat_increase:,} features (cost: {cost_per_pct:,.0f} feat/1%)")

    # Save results
    output_dir = Path('src/experiments/sae_error_analysis/results')
    output_path = output_dir / 'layer_ablation_results.json'

    with open(output_path, 'w') as f:
        json.dump({
            'results': results,
            'best_single': best_single,
            'best_pair': best_pair,
            'best_triplet': best_triplet,
            'l14_baseline': l14_acc
        }, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
