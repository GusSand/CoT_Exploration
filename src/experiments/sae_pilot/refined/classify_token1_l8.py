"""
Classify operations using refined SAE features from Token 1 L8 only.

This tests the hypothesis that Token 1 L8 (most discriminative from operation
circuits: 77.5% solo accuracy) preserves more operation-specific information
than mean pooling across all tokens/layers.
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


def extract_token1_l8_features(features_np, metadata, n_problems=600):
    """
    Extract features from Token 1 L8 only (most discriminative position).

    Args:
        features_np: [n_vectors, n_features] array
        metadata: Dict with 'problems', 'tokens', 'layers', 'operation_types'
        n_problems: Number of unique problems

    Returns:
        problem_features: [n_problems, n_features] - one vector per problem
        problem_labels: [n_problems] operation types
    """
    print("\nExtracting Token 1 L8 features only...")

    n_features = features_np.shape[1]
    problem_features = np.zeros((n_problems, n_features))
    problem_labels = []
    problem_op_types = {}
    problem_found = [False] * n_problems

    # Extract only Token 1, Layer 8 vectors
    n_extracted = 0
    for vector_idx in range(len(features_np)):
        token = metadata['tokens'][vector_idx]
        layer = metadata['layers'][vector_idx]
        problem_idx = metadata['problems'][vector_idx]
        op_type = metadata['operation_types'][vector_idx]

        # Filter for Token 1 and Layer 8
        if token == 1 and layer == 8:
            problem_features[problem_idx] = features_np[vector_idx]
            problem_op_types[problem_idx] = op_type
            problem_found[problem_idx] = True
            n_extracted += 1

    # Create label array
    problem_labels = [problem_op_types[i] for i in range(n_problems)]

    print(f"  Extracted: {n_extracted} vectors (Token 1, L8 only)")
    print(f"  Problems: {n_problems}")
    print(f"  Features per problem: {n_features}")
    print(f"  All problems found: {all(problem_found)}")

    return problem_features, problem_labels


def train_classifier(features, labels, test_size=0.2, random_state=42):
    """Train logistic regression classifier on features."""
    print("\nTraining classifier...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    print(f"  Train size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")

    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"  Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    return clf, X_test, y_test, y_pred_test, test_acc


def visualize_results(y_test, y_pred, accuracy, output_dir: Path, pilot_acc=0.70, baseline_acc=0.833):
    """Visualize classification results."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Confusion matrix
    ax = axes[0]
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(set(y_test))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Token 1 L8)\nAccuracy: {accuracy*100:.2f}%')

    # 2. Accuracy comparison
    ax = axes[1]
    models = ['Baseline\n(Raw L8)', 'Pilot\n(Mean Pool)', 'Refined\n(Token 1 L8)']
    accuracies = [baseline_acc * 100, pilot_acc * 100, accuracy * 100]
    colors = ['gray', 'orange', 'green' if accuracy >= baseline_acc else 'blue']

    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Performance Comparison')
    ax.set_ylim([0, 105])
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Target (80%)')
    ax.axhline(y=baseline_acc*100, color='blue', linestyle='--', alpha=0.5,
               label=f'Baseline ({baseline_acc*100:.1f}%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'token1_l8_classification.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_pilot/refined/sae_weights.pt')
    parser.add_argument('--activations', type=str,
                        default='src/experiments/sae_pilot/data/sae_training_activations.pt')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_pilot/refined')
    parser.add_argument('--pilot_acc', type=float, default=0.70,
                        help='Pilot SAE accuracy (mean pooling)')
    parser.add_argument('--baseline_acc', type=float, default=0.833,
                        help='Baseline accuracy from operation circuits')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)

    # Load SAE
    print("Loading refined SAE...")
    sae, checkpoint = load_sae(args.sae_weights, device)
    print(f"  Config: {checkpoint['config']['n_features']} features, "
          f"L1={checkpoint['config']['l1_coefficient']}")

    # Load activations
    print("\nLoading activations...")
    activation_data = torch.load(args.activations)
    activations = activation_data['activations']
    metadata = activation_data['metadata']
    print(f"  Activations shape: {activations.shape}")

    # Extract SAE features
    print("\nExtracting SAE features...")
    with torch.no_grad():
        x = activations.to(device)
        _, features = sae(x)
        features_np = features.cpu().numpy()

    print(f"  Features shape: {features_np.shape}")

    # Extract Token 1 L8 only
    problem_features, problem_labels = extract_token1_l8_features(
        features_np, metadata, n_problems=600
    )

    # Train classifier
    clf, X_test, y_test, y_pred, test_acc = train_classifier(
        problem_features, problem_labels
    )

    # Print detailed report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (TOKEN 1 L8 ONLY)")
    print("="*80)
    print(classification_report(y_test, y_pred))

    # Visualize
    visualize_results(y_test, y_pred, test_acc, output_dir, args.pilot_acc, args.baseline_acc)

    # Save results
    results = {
        'test_accuracy': test_acc,
        'pilot_accuracy': args.pilot_acc,
        'baseline_accuracy': args.baseline_acc,
        'improvement_vs_pilot': test_acc - args.pilot_acc,
        'improvement_vs_baseline': test_acc - args.baseline_acc,
        'target_met': test_acc >= 0.80,
        'beats_pilot': test_acc >= args.pilot_acc,
        'beats_baseline': test_acc >= args.baseline_acc,
        'method': 'Token 1 L8 only (most discriminative position)',
        'sae_config': checkpoint['config'],
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    results_path = output_dir / 'token1_l8_classification_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Classification complete! Results saved to {results_path}")

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    print(f"Refined SAE (Token 1 L8): {test_acc*100:.2f}%")
    print(f"Pilot SAE (Mean Pool): {args.pilot_acc*100:.2f}%")
    print(f"Baseline (Raw L8): {args.baseline_acc*100:.2f}%")
    print(f"\nImprovement vs Pilot: {(test_acc - args.pilot_acc)*100:+.2f} pts")
    print(f"Improvement vs Baseline: {(test_acc - args.baseline_acc)*100:+.2f} pts")
    print(f"\nTarget (>80%): {'✅ MET' if test_acc >= 0.80 else '❌ NOT MET'}")
    print(f"Beats Pilot: {'✅ YES' if test_acc >= args.pilot_acc else '❌ NO'}")
    print(f"Beats Baseline: {'✅ YES' if test_acc >= args.baseline_acc else '❌ NO'}")
    print("="*80)


if __name__ == "__main__":
    main()
