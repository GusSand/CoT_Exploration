"""
Train error classifier on SAE features from continuous thoughts.

Pipeline:
1. Load error_analysis_dataset.json (462 incorrect + 452 correct solutions)
2. Load refined SAE (2048 features, L1=0.0005)
3. Encode continuous thoughts → SAE features
4. Concatenate all 18 vectors per problem (3 layers × 6 tokens)
5. Train logistic regression: SAE features → correct/incorrect
6. Evaluate: target >60% accuracy (better than coin flip)
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
    """
    Encode continuous thoughts to SAE features.

    Args:
        dataset: Dict with 'correct_solutions' and 'incorrect_solutions'
        sae: Trained sparse autoencoder
        device: 'cuda' or 'cpu'

    Returns:
        encoded_data: List of dicts with SAE features + labels
    """
    print("\nEncoding continuous thoughts to SAE features...")

    encoded_data = []

    # Process incorrect solutions
    print("  Encoding incorrect solutions...")
    for sol in tqdm(dataset['incorrect_solutions'], desc="Incorrect"):
        thoughts = sol['continuous_thoughts']

        # Encode each layer's thoughts
        sae_features = {}
        for layer_name, thought_list in thoughts.items():
            # thought_list is list of 6 vectors [2048]
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


def concatenate_features(encoded_data, n_features):
    """
    Concatenate all 18 SAE feature vectors per problem.

    3 layers (early, middle, late) × 6 tokens = 18 vectors

    Args:
        encoded_data: List of dicts with 'sae_features' and 'is_correct'
        n_features: Number of SAE features (2048)

    Returns:
        X: [n_samples, n_features * 18] feature matrix
        y: [n_samples] binary labels (0=incorrect, 1=correct)
    """
    print("\nConcatenating SAE features...")

    n_samples = len(encoded_data)
    X = np.zeros((n_samples, n_features * 18))
    y = np.zeros(n_samples, dtype=int)

    layer_order = ['early', 'middle', 'late']  # L4, L8, L14

    for i, sample in enumerate(encoded_data):
        sae_features = sample['sae_features']
        y[i] = 1 if sample['is_correct'] else 0

        # Concatenate in order: early_t0, early_t1, ..., late_t5
        concat_features = []
        for layer_name in layer_order:
            layer_features = sae_features[layer_name]  # [6, n_features]
            for token_idx in range(6):
                concat_features.append(layer_features[token_idx])

        X[i] = np.concatenate(concat_features)

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution: {np.sum(y==0)} incorrect, {np.sum(y==1)} correct")

    return X, y


def train_error_classifier(X, y, test_size=0.2, random_state=42):
    """
    Train binary logistic regression: SAE features → correct/incorrect.

    Args:
        X: [n_samples, n_features * 18] feature matrix
        y: [n_samples] binary labels (0=incorrect, 1=correct)

    Returns:
        Results dict with classifier, predictions, and metrics
    """
    print("\n" + "="*80)
    print("TRAINING ERROR CLASSIFIER")
    print("="*80)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    print(f"  Feature dim: {X_train.shape[1]:,}")

    # Train classifier
    print(f"\nTraining logistic regression...")
    clf = LogisticRegression(max_iter=1000, random_state=random_state, verbose=1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Probabilities
    y_prob_train = clf.predict_proba(X_train)[:, 1]
    y_prob_test = clf.predict_proba(X_test)[:, 1]

    # Accuracy
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"\nTarget (>60%): {'✅ MET' if test_acc >= 0.60 else '❌ NOT MET'}")
    print(f"Better than coin flip: {'✅ YES' if test_acc >= 0.55 else '❌ NO'}")

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
        'y_prob_train': y_prob_train, 'y_prob_test': y_prob_test,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'confusion_matrix': cm
    }


def visualize_results(results, output_dir: Path):
    """Generate visualizations for error classification results."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Confusion Matrix
    ax = axes[0, 0]
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Incorrect', 'Correct'],
                yticklabels=['Incorrect', 'Correct'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix\nTest Accuracy: {results["test_acc"]*100:.2f}%')

    # 2. Prediction confidence distribution
    ax = axes[0, 1]
    y_test = results['y_test']
    y_prob = results['y_prob_test']

    # Plot distributions for correct and incorrect
    incorrect_probs = y_prob[y_test == 0]
    correct_probs = y_prob[y_test == 1]

    ax.hist(incorrect_probs, bins=20, alpha=0.6, label='Incorrect', color='red')
    ax.hist(correct_probs, bins=20, alpha=0.6, label='Correct', color='green')
    ax.axvline(x=0.5, color='black', linestyle='--', label='Decision boundary')
    ax.set_xlabel('Predicted P(Correct)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. ROC-style curve (sorted predictions)
    ax = axes[1, 0]
    sorted_idx = np.argsort(y_prob)[::-1]  # Sort descending
    sorted_y = y_test[sorted_idx]
    cumulative_correct = np.cumsum(sorted_y) / np.sum(sorted_y)
    cumulative_incorrect = np.cumsum(1 - sorted_y) / np.sum(1 - sorted_y)

    x = np.arange(len(sorted_y)) / len(sorted_y) * 100
    ax.plot(x, cumulative_correct * 100, label='Correct (True Positive Rate)',
            color='green', linewidth=2)
    ax.plot(x, cumulative_incorrect * 100, label='Incorrect (False Positive Rate)',
            color='red', linewidth=2)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('% of samples (sorted by confidence)')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Cumulative Prediction Curve')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])

    # 4. Accuracy comparison
    ax = axes[1, 1]
    methods = ['Random\nBaseline', 'Target\n(>60%)', 'SAE Error\nClassifier']
    accuracies = [50, 60, results['test_acc'] * 100]
    colors = ['gray', 'orange', 'green' if results['test_acc'] >= 0.60 else 'red']

    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Error Classification Performance')
    ax.set_ylim([0, 105])
    ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5,
               label='Target (60%)')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.3,
               label='Random (50%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'error_classification_results.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset.json',
                        help='Path to error analysis dataset')
    parser.add_argument('--sae_weights', type=str,
                        default='src/experiments/sae_pilot/refined/sae_weights.pt',
                        help='Path to trained SAE weights')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/results',
                        help='Output directory for results')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SAE ERROR CLASSIFICATION EXPERIMENT")
    print("="*80)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"SAE weights: {args.sae_weights}")
    print(f"Output: {output_dir}")

    # Load dataset
    print("\n" + "="*80)
    print("LOADING ERROR ANALYSIS DATASET")
    print("="*80)
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    print(f"Metadata:")
    print(f"  Total solutions: {dataset['metadata']['total']}")
    print(f"  Correct: {dataset['metadata']['n_correct']}")
    print(f"  Incorrect: {dataset['metadata']['n_incorrect']}")
    print(f"  Layers: {dataset['metadata']['layers']}")
    print(f"  Latent tokens: {dataset['metadata']['n_latent_tokens']}")

    # Load SAE
    print("\n" + "="*80)
    print("LOADING REFINED SAE")
    print("="*80)
    sae, checkpoint = load_sae(args.sae_weights, device)
    print(f"SAE config:")
    print(f"  Input dim: {checkpoint['config']['input_dim']}")
    print(f"  Features: {checkpoint['config']['n_features']}")
    print(f"  L1 coefficient: {checkpoint['config']['l1_coefficient']}")

    # Encode continuous thoughts to SAE features
    encoded_data = encode_continuous_thoughts_to_sae_features(dataset, sae, device)

    # Concatenate features (3 layers × 6 tokens = 18 vectors)
    X, y = concatenate_features(encoded_data, checkpoint['config']['n_features'])

    # Train error classifier
    results = train_error_classifier(X, y,
                                     test_size=args.test_size,
                                     random_state=args.random_seed)

    # Visualize
    visualize_results(results, output_dir)

    # Save results
    results_summary = {
        'experiment': 'SAE Error Classification',
        'dataset': {
            'n_samples': len(encoded_data),
            'n_correct': dataset['metadata']['n_correct'],
            'n_incorrect': dataset['metadata']['n_incorrect']
        },
        'sae_config': {
            'input_dim': checkpoint['config']['input_dim'],
            'n_features': checkpoint['config']['n_features'],
            'l1_coefficient': checkpoint['config']['l1_coefficient']
        },
        'features': {
            'aggregation': 'concatenate',
            'n_vectors': 18,
            'feature_dim': X.shape[1]
        },
        'performance': {
            'train_accuracy': float(results['train_acc']),
            'test_accuracy': float(results['test_acc']),
            'target_met': results['test_acc'] >= 0.60,
            'better_than_random': results['test_acc'] >= 0.55
        },
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': classification_report(
            results['y_test'], results['y_pred_test'],
            target_names=['Incorrect', 'Correct'],
            output_dict=True
        )
    }

    results_path = output_dir / 'error_classification_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")

    # Save encoded dataset for later analysis
    encoded_dataset_path = output_dir / 'encoded_error_dataset.pt'
    torch.save({
        'X': X,
        'y': y,
        'encoded_data': encoded_data,
        'sae_config': checkpoint['config']
    }, encoded_dataset_path)
    print(f"✅ Encoded dataset saved to {encoded_dataset_path}")

    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Test Accuracy: {results['test_acc']*100:.2f}%")
    print(f"Target (>60%): {'✅ MET' if results['test_acc'] >= 0.60 else '❌ NOT MET'}")
    print(f"Better than random: {'✅ YES' if results['test_acc'] >= 0.55 else '❌ NO'}")
    print("="*80)


if __name__ == "__main__":
    main()
