"""
Test concatenation vs mean pooling aggregation.

Concatenates all 18 SAE feature vectors (3 layers × 6 tokens) per problem
instead of averaging them.

Hypothesis: Mean pooling loses position-specific information.
Concatenation preserves layer/token signals.
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


def concatenate_all_features(features_np, metadata, n_problems=600):
    """
    Concatenate all 18 feature vectors per problem (3 layers × 6 tokens).

    Args:
        features_np: [n_vectors, n_features] array
        metadata: Dict with 'problems', 'tokens', 'layers', 'operation_types'
        n_problems: Number of unique problems

    Returns:
        problem_features: [n_problems, n_features * 18] - concatenated vectors
        problem_labels: [n_problems] operation types
    """
    print("\nConcatenating all features (3 layers × 6 tokens = 18 vectors)...")

    n_features = features_np.shape[1]
    # Each problem gets 18 feature vectors concatenated
    problem_features = np.zeros((n_problems, n_features * 18))
    problem_labels = []
    problem_op_types = {}

    # Track which positions we've filled for each problem
    problem_vectors = {i: [] for i in range(n_problems)}

    # Collect all vectors for each problem
    for vector_idx in range(len(features_np)):
        token = metadata['tokens'][vector_idx]
        layer = metadata['layers'][vector_idx]
        problem_idx = metadata['problems'][vector_idx]
        op_type = metadata['operation_types'][vector_idx]

        # Store vector with position info
        problem_vectors[problem_idx].append({
            'features': features_np[vector_idx],
            'layer': layer,
            'token': token
        })
        problem_op_types[problem_idx] = op_type

    # Concatenate in consistent order: L4T0, L4T1, ..., L4T5, L8T0, ..., L14T5
    layer_order = [4, 8, 14]
    token_order = [0, 1, 2, 3, 4, 5]

    for problem_idx in range(n_problems):
        vectors = problem_vectors[problem_idx]

        # Create lookup dict
        lookup = {(v['layer'], v['token']): v['features'] for v in vectors}

        # Concatenate in order
        concat_features = []
        for layer in layer_order:
            for token in token_order:
                if (layer, token) in lookup:
                    concat_features.append(lookup[(layer, token)])
                else:
                    # Should never happen, but fill with zeros if missing
                    concat_features.append(np.zeros(n_features))

        problem_features[problem_idx] = np.concatenate(concat_features)

    # Create label array
    problem_labels = [problem_op_types[i] for i in range(n_problems)]

    print(f"  Problems: {n_problems}")
    print(f"  Features per problem: {problem_features.shape[1]:,} ({n_features} × 18)")
    print(f"  Shape: {problem_features.shape}")

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
    print(f"  Feature dim: {X_train.shape[1]:,}")

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


def visualize_results(y_test, y_pred, accuracy, output_dir: Path,
                      mean_pool_acc=0.70, token1_l8_acc=0.633, baseline_acc=0.833):
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
    ax.set_title(f'Confusion Matrix (Concatenation)\nAccuracy: {accuracy*100:.2f}%')

    # 2. Accuracy comparison
    ax = axes[1]
    models = ['Baseline\n(Raw)', 'Mean Pool\n(Pilot)', 'Token 1 L8\n(Refined)', 'Concatenate\n(All 18)']
    accuracies = [baseline_acc * 100, mean_pool_acc * 100, token1_l8_acc * 100, accuracy * 100]
    colors = ['gray', 'orange', 'blue', 'green' if accuracy >= baseline_acc else 'purple']

    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Performance: Aggregation Strategy Comparison')
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
                ha='center', va='bottom', fontweight='bold', fontsize=9)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'concatenate_classification.{ext}'
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

    # Concatenate all features
    problem_features, problem_labels = concatenate_all_features(
        features_np, metadata, n_problems=600
    )

    # Train classifier
    clf, X_test, y_test, y_pred, test_acc = train_classifier(
        problem_features, problem_labels
    )

    # Print detailed report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT (CONCATENATE ALL 18 VECTORS)")
    print("="*80)
    print(classification_report(y_test, y_pred))

    # Visualize
    visualize_results(y_test, y_pred, test_acc, output_dir)

    # Save results
    results = {
        'test_accuracy': test_acc,
        'mean_pool_accuracy': 0.70,
        'token1_l8_accuracy': 0.633,
        'baseline_accuracy': 0.833,
        'method': 'Concatenate all 18 vectors (3 layers × 6 tokens)',
        'feature_dim': problem_features.shape[1],
        'improvement_vs_mean_pool': test_acc - 0.70,
        'improvement_vs_token1_l8': test_acc - 0.633,
        'improvement_vs_baseline': test_acc - 0.833,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    results_path = output_dir / 'concatenate_classification_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Classification complete! Results saved to {results_path}")

    # Final verdict
    print("\n" + "="*80)
    print("AGGREGATION STRATEGY COMPARISON")
    print("="*80)
    print(f"Baseline (Raw L8):          83.3%")
    print(f"Mean Pool (Pilot):          70.0%")
    print(f"Token 1 L8 (Refined):       63.3%")
    print(f"Concatenate (All 18):       {test_acc*100:.1f}%")
    print(f"\nImprovement vs Mean Pool:   {(test_acc - 0.70)*100:+.1f} pts")
    print(f"Improvement vs Token 1 L8:  {(test_acc - 0.633)*100:+.1f} pts")
    print(f"Improvement vs Baseline:    {(test_acc - 0.833)*100:+.1f} pts")
    print(f"\nTarget (>80%): {'✅ MET' if test_acc >= 0.80 else '❌ NOT MET'}")
    print(f"Beats Mean Pool: {'✅ YES' if test_acc >= 0.70 else '❌ NO'}")
    print(f"Beats Baseline: {'✅ YES' if test_acc >= 0.833 else '❌ NO'}")
    print("="*80)


if __name__ == "__main__":
    main()
