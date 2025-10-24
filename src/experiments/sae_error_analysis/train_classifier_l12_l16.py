"""
Train error classifier on L12-L16 SAE features.

Goal: Test if L12-L16 features achieve 70-75% accuracy (vs 66.67% L14-only baseline).

Pipeline:
1. Load L12-L16 dataset
2. Load trained L12-L16 SAE
3. Encode continuous thoughts → SAE features
4. Concatenate 5 layers × 6 tokens = 30 vectors per problem
5. Train logistic regression: SAE features → correct/incorrect
6. Compare to L14-only baseline (66.67%)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import wandb
from datetime import datetime


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder."""

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
    """Load trained SAE from checkpoint."""
    checkpoint = torch.load(weights_path, map_location=device)
    sae = SparseAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        n_features=checkpoint['config']['n_features']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    return sae, checkpoint


def encode_continuous_thoughts_to_sae_features(dataset, sae, device='cuda'):
    """Encode continuous thoughts to SAE features."""
    print("\nEncoding continuous thoughts to SAE features...")

    encoded_data = []

    # Process incorrect solutions
    print("  Encoding incorrect solutions...")
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        thoughts = sol['continuous_thoughts']

        # Encode each layer's thoughts
        sae_features = {}
        for layer_name, thought_list in thoughts.items():
            thought_tensor = torch.tensor(thought_list, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, features = sae(thought_tensor)  # [6, n_features]

            sae_features[layer_name] = features.cpu().numpy()

        encoded_data.append({
            'pair_id': sol['pair_id'],
            'variant': sol['variant'],
            'is_correct': False,
            'sae_features': sae_features  # Dict: layer_name -> [6, n_features]
        })

    # Process correct solutions
    print("  Encoding correct solutions...")
    for sol in tqdm(dataset['correct_solutions'], desc="Correct"):
        thoughts = sol['continuous_thoughts']

        sae_features = {}
        for layer_name, thought_list in thoughts.items():
            thought_tensor = torch.tensor(thought_list, dtype=torch.float32).to(device)

            with torch.no_grad():
                _, features = sae(thought_tensor)

            sae_features[layer_name] = features.cpu().numpy()

        encoded_data.append({
            'pair_id': sol['pair_id'],
            'variant': sol['variant'],
            'is_correct': True,
            'sae_features': sae_features
        })

    print(f"\n  Encoded {len(encoded_data)} solutions")
    print(f"    Incorrect: {sum(1 for d in encoded_data if not d['is_correct'])}")
    print(f"    Correct: {sum(1 for d in encoded_data if d['is_correct'])}")

    return encoded_data


def concatenate_features(encoded_data, n_features, layer_order):
    """
    Concatenate all L12-L16 SAE feature vectors per problem.

    5 layers × 6 tokens = 30 vectors (vs 6 for L14-only)

    Returns:
        X: [n_samples, n_features * 30] feature matrix
        y: [n_samples] binary labels (0=incorrect, 1=correct)
    """
    print("\nConcatenating SAE features (L12-L16)...")

    n_samples = len(encoded_data)
    X = np.zeros((n_samples, n_features * 30))  # 5 layers × 6 tokens
    y = np.zeros(n_samples, dtype=int)

    for i, sample in enumerate(encoded_data):
        sae_features = sample['sae_features']
        y[i] = 1 if sample['is_correct'] else 0

        # Concatenate: L12_t0, L12_t1, ..., L16_t5
        concat_features = []
        for layer_name in layer_order:
            layer_features = sae_features[layer_name]  # [6, n_features]
            for token_idx in range(6):
                concat_features.append(layer_features[token_idx])

        X[i] = np.concatenate(concat_features)

    print(f"  Feature matrix shape: {X.shape} (L12-L16)")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution: {np.sum(y==0)} incorrect, {np.sum(y==1)} correct")

    return X, y


def train_error_classifier(X, y, test_size=0.2, random_state=42, baseline_acc=0.6667):
    """
    Train binary logistic regression.

    Args:
        X: [n_samples, n_features * 30] feature matrix (L12-L16)
        y: [n_samples] binary labels
        baseline_acc: L14-only baseline (66.67%)

    Returns:
        Results dict
    """
    print("\n" + "="*80)
    print("TRAINING ERROR CLASSIFIER (L12-L16)")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Feature dim: {X_train.shape[1]:,} (L12-L16, vs 12,288 for L14-only)")

    # Train classifier
    print(f"\nTraining logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=random_state, verbose=1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # Calculate performance vs baseline
    acc_vs_baseline = test_acc - baseline_acc
    target_met = test_acc >= 0.70

    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nBaseline (L14-only): {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    print(f"L12-L16:             {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Improvement:         {acc_vs_baseline:+.4f} ({acc_vs_baseline*100:+.2f} percentage points)")

    print(f"\nTarget (70-75%): {'✅ MET' if target_met else '❌ NOT MET'}")
    print(f"Better than L14-only: {'✅ YES' if acc_vs_baseline > 0 else '❌ NO'}")

    # Classification report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (TEST SET)")
    print("="*80)
    print(classification_report(y_test, y_pred_test,
                                target_names=['Incorrect', 'Correct']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Incorrect  Correct")
    print(f"Actual Incorrect    {cm[0,0]:4d}      {cm[0,1]:4d}")
    print(f"Actual Correct      {cm[1,0]:4d}      {cm[1,1]:4d}")

    return {
        'classifier': clf,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'baseline_acc': baseline_acc,
        'acc_vs_baseline': acc_vs_baseline,
        'target_met': target_met,
        'confusion_matrix': cm
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset_l12_l16.json')
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_error_analysis/sae_l12_l16/sae_weights.pt')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/results')
    parser.add_argument('--baseline_acc', type=float, default=0.6667,
                        help='L14-only baseline accuracy')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SAE ERROR CLASSIFICATION - L12-L16")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"SAE weights: {args.sae_weights}")
    print(f"Baseline (L14-only): {args.baseline_acc:.4f}")

    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    print(f"\nDataset: {dataset['metadata']['total']} solutions")

    # Load SAE
    sae, checkpoint = load_sae(args.sae_weights, device)
    print(f"\nSAE: {checkpoint['config']['n_features']} features")

    # Encode dataset
    encoded_data = encode_continuous_thoughts_to_sae_features(dataset, sae, device)

    # Concatenate features
    layer_order = dataset['metadata']['layers']
    X, y = concatenate_features(encoded_data, checkpoint['config']['n_features'], layer_order)

    # Train classifier
    results = train_error_classifier(X, y, baseline_acc=args.baseline_acc)

    # Save results
    results_path = output_dir / 'error_classification_l12_l16_results.json'
    results_summary = {
        'experiment': 'SAE Error Classification - L12-L16',
        'date': datetime.now().isoformat(),
        'layers_used': layer_order,
        'baseline_layers': ['L14'],
        'baseline_accuracy': float(results['baseline_acc']),
        'test_accuracy': float(results['test_acc']),
        'improvement_vs_baseline': float(results['acc_vs_baseline']),
        'target_met': bool(results['target_met']),
        'target_range': '70-75%',
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': classification_report(
            results['y_test'], results['y_pred_test'],
            target_names=['Incorrect', 'Correct'],
            output_dict=True
        )
    }

    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Baseline (L14-only): {results['baseline_acc']*100:.2f}%")
    print(f"L12-L16:             {results['test_acc']*100:.2f}%")
    print(f"Improvement:         {results['acc_vs_baseline']*100:+.2f} percentage points")
    print(f"\n{'✅ TARGET MET (70-75%)' if results['target_met'] else '❌ TARGET NOT MET'}")


if __name__ == "__main__":
    main()
