"""
Comprehensive comparison of all SAE variants:
1. ReLU SAE (baseline from sae_pilot)
2. TopK SAE (K=100, d=512 - baseline from user)
3. Vanilla Matryoshka SAE (hierarchical, 3 levels)
4. Matryoshka-TopK Hybrid (hierarchical + TopK)

Compares:
- Reconstruction metrics (EV, L0, feature death, utilization)
- Classification performance
- Feature efficiency
"""

import torch
import numpy as np
from pathlib import Path
import json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from matryoshka_sae import MatryoshkaSAE
from matryoshka_topk_sae import MatryoshkaTopKSAE


def load_relu_sae(checkpoint_path, device='cuda'):
    """Load ReLU SAE from sae_pilot."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'sae_pilot'))
    from sae import SparseAutoencoder

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = SparseAutoencoder(
        input_dim=config['input_dim'],
        n_features=config['n_features'],
        l1_coefficient=config['l1_coefficient']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_matryoshka_sae(checkpoint_path, device='cuda'):
    """Load vanilla Matryoshka SAE."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MatryoshkaSAE(
        input_dim=config['input_dim'],
        levels=config['levels'],
        l1_coefficient=config['l1_coefficient'],
        level_weights=config['level_weights']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_matryoshka_topk_sae(checkpoint_path, device='cuda'):
    """Load Matryoshka-TopK hybrid SAE."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = MatryoshkaTopKSAE(
        input_dim=config['input_dim'],
        levels=config['levels'],
        k_values=config['k_values'],
        level_weights=config['level_weights']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def compute_relu_metrics(model, data, device='cuda'):
    """Compute metrics for ReLU SAE."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        reconstruction, features = model(data)

        # Explained variance
        ss_res = torch.sum((data - reconstruction) ** 2)
        ss_tot = torch.sum((data - data.mean(0, keepdim=True)) ** 2)
        explained_var = 1 - (ss_res / ss_tot)

        # L0 norm
        l0_norm = (features != 0).float().sum(1).mean()

        # Feature utilization
        total_features = features.shape[1]
        active_features = (features.abs().sum(0) > 0).sum().item()
        dead_features = total_features - active_features
        feature_death_rate = dead_features / total_features

        return {
            'architecture': 'ReLU SAE',
            'total_features': total_features,
            'explained_variance': explained_var.item(),
            'l0_norm': l0_norm.item(),
            'active_features': active_features,
            'dead_features': dead_features,
            'feature_death_rate': feature_death_rate,
            'utilization_pct': (active_features / total_features) * 100,
            'features_for_classification': features.cpu().numpy()
        }


def compute_matryoshka_metrics(model, data, device='cuda'):
    """Compute metrics for vanilla Matryoshka SAE."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        reconstructions, features = model(data)
        metrics = model.compute_metrics(data, reconstructions, features)

        # Extract Level 3 (finest) metrics
        level_3 = metrics['level_3']

        return {
            'architecture': 'Vanilla Matryoshka',
            'total_features': level_3['features'],
            'explained_variance': level_3['explained_variance'],
            'l0_norm': level_3['l0_norm'],
            'active_features': level_3['active_features'],
            'dead_features': level_3['dead_features'],
            'feature_death_rate': level_3['feature_death_rate'],
            'utilization_pct': (level_3['active_features'] / level_3['features']) * 100,
            'features_for_classification': features['concatenated'].cpu().numpy(),
            'hierarchical_metrics': metrics
        }


def compute_matryoshka_topk_metrics(model, data, device='cuda'):
    """Compute metrics for Matryoshka-TopK hybrid SAE."""
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        reconstructions, features = model(data)
        metrics = model.compute_metrics(data, reconstructions, features)

        # Extract Level 3 (finest) metrics
        level_3 = metrics['level_3']

        return {
            'architecture': 'Matryoshka-TopK Hybrid',
            'total_features': level_3['features'],
            'expected_k': level_3['expected_k'],
            'explained_variance': level_3['explained_variance'],
            'l0_norm': level_3['l0_norm'],
            'active_features': level_3['active_features'],
            'dead_features': level_3['dead_features'],
            'feature_death_rate': level_3['feature_death_rate'],
            'utilization_pct': level_3['utilization_pct'],
            'features_for_classification': features['concatenated'].cpu().numpy(),
            'hierarchical_metrics': metrics
        }


def get_topk_baseline():
    """Return TopK baseline metrics from user-provided data."""
    return {
        'architecture': 'TopK SAE',
        'total_features': 512,
        'expected_k': 100,
        'explained_variance': 0.878,  # 87.8%
        'l0_norm': 100.0,
        'active_features': 512,
        'dead_features': 0,
        'feature_death_rate': 0.0,
        'utilization_pct': 100.0,
        'features_for_classification': None  # Model not available
    }


def classify_features(features, labels, test_size=0.2, random_state=42):
    """Train classifier on features."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def print_comparison_table(results):
    """Print comprehensive comparison table."""
    print("\n" + "=" * 90)
    print("COMPREHENSIVE SAE COMPARISON")
    print("=" * 90)

    print(f"\n{'Architecture':<25} {'Features':<12} {'EV':<10} {'L0':<10} {'Death':<10} {'Util':<10}")
    print("-" * 90)

    for result in results:
        arch = result['architecture']
        n_feat = result['total_features']
        ev = result['explained_variance']
        l0 = result['l0_norm']
        death = result['feature_death_rate']
        util = result['utilization_pct']

        print(f"{arch:<25} {n_feat:<12,} {ev:<9.1%} {l0:<9.1f} {death:<9.1%} {util:<9.1f}%")

    print("\n" + "=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)

    # Find best EV
    best_ev = max(results, key=lambda x: x['explained_variance'])
    print(f"\n1. BEST Explained Variance: {best_ev['architecture']}")
    print(f"   {best_ev['explained_variance']:.1%} EV")

    # Find best utilization
    best_util = max(results, key=lambda x: x['utilization_pct'])
    print(f"\n2. BEST Feature Utilization: {best_util['architecture']}")
    print(f"   {best_util['utilization_pct']:.1f}% utilization ({best_util['active_features']:,}/{best_util['total_features']:,} active)")

    # Find lowest feature death
    lowest_death = min(results, key=lambda x: x['feature_death_rate'])
    print(f"\n3. LOWEST Feature Death: {lowest_death['architecture']}")
    print(f"   {lowest_death['feature_death_rate']:.1%} feature death")

    # Find most efficient (active features vs EV)
    print(f"\n4. EFFICIENCY (Active Features vs Performance):")
    for result in sorted(results, key=lambda x: x['active_features']):
        efficiency_score = result['explained_variance'] / (result['active_features'] / 100)
        print(f"   {result['architecture']:<25} {result['active_features']:>5,} active → {result['explained_variance']:>5.1%} EV (score: {efficiency_score:.3f})")


def print_classification_results(classification_results):
    """Print classification comparison."""
    print("\n" + "=" * 90)
    print("CLASSIFICATION PERFORMANCE (Operation Detection)")
    print("=" * 90)

    print(f"\n{'Architecture':<25} {'Features':<12} {'Accuracy':<12} {'vs Best':<15}")
    print("-" * 70)

    # Find best accuracy
    best_acc = max([r['accuracy'] for r in classification_results if r['accuracy'] is not None])

    for result in classification_results:
        arch = result['architecture']
        n_feat = result['total_features']

        if result['accuracy'] is None:
            print(f"{arch:<25} {n_feat:<12,} {'N/A':<12} {'(no model)':<15}")
        else:
            acc = result['accuracy']
            diff = (acc - best_acc) * 100
            print(f"{arch:<25} {n_feat:<12,} {acc:<11.1%} {diff:+.1f} pts")


if __name__ == "__main__":
    base_dir = Path(__file__).parent

    # Model paths
    relu_checkpoint = base_dir.parent / "sae_pilot/results/sae_weights.pt"
    matryoshka_checkpoint = base_dir / "models/pos3_hierarchical.pt"
    matryoshka_topk_checkpoint = base_dir / "models/pos3_hierarchical_topk.pt"
    data_path = base_dir / "data/position_3_activations.pt"

    print("=" * 90)
    print("LOADING MODELS")
    print("=" * 90)

    # Load data
    print("\nLoading validation data...")
    data_full = torch.load(data_path)
    activations = data_full['activations']

    # Use last 20% as validation
    n_samples = activations.shape[0]
    n_val = int(n_samples * 0.2)
    val_data = activations[-n_val:]

    print(f"  Validation samples: {len(val_data):,}")

    # Load operation labels for classification
    print("\nLoading operation labels...")
    from datasets import load_dataset

    dataset = load_dataset('gsm8k', 'main', split='train')
    problem_ids = data_full['metadata']['problem_ids']

    def extract_operation(cot_text):
        mult_count = sum(1 for c in ['*', '×'] if c in cot_text)
        add_count = sum(1 for c in ['+'] if c in cot_text)
        div_count = sum(1 for c in ['/', '÷'] if c in cot_text)
        counts = {'multiplication': mult_count, 'addition': add_count, 'division': div_count}
        return max(counts, key=counts.get)

    labels = []
    valid_indices = []

    for idx, pid in enumerate(problem_ids):
        try:
            problem_idx = int(pid.split('_')[1])
            if problem_idx < len(dataset):
                cot = dataset[problem_idx]['answer']
                op = extract_operation(cot)
                labels.append(op)
                valid_indices.append(idx)
        except:
            continue

    labels = np.array(labels)
    valid_activations = activations[valid_indices]

    print(f"  Labeled samples: {len(labels):,}")

    # Compute metrics for each model
    results = []
    classification_results = []

    # 1. ReLU SAE
    print("\n1. ReLU SAE...")
    relu_model, relu_config = load_relu_sae(str(relu_checkpoint))
    relu_metrics = compute_relu_metrics(relu_model, val_data)
    results.append(relu_metrics)

    relu_features_full = compute_relu_metrics(relu_model, valid_activations)['features_for_classification']
    relu_class = classify_features(relu_features_full, labels)
    classification_results.append({
        'architecture': 'ReLU SAE',
        'total_features': relu_metrics['total_features'],
        'accuracy': relu_class['accuracy']
    })

    # 2. TopK SAE (baseline only - no model)
    print("2. TopK SAE (baseline from user data)...")
    topk_metrics = get_topk_baseline()
    results.append(topk_metrics)
    classification_results.append({
        'architecture': 'TopK SAE',
        'total_features': topk_metrics['total_features'],
        'accuracy': None  # No model available
    })

    # 3. Vanilla Matryoshka SAE
    print("3. Vanilla Matryoshka SAE...")
    matryoshka_model, matryoshka_config = load_matryoshka_sae(str(matryoshka_checkpoint))
    matryoshka_metrics = compute_matryoshka_metrics(matryoshka_model, val_data)
    results.append(matryoshka_metrics)

    matryoshka_features_full = compute_matryoshka_metrics(matryoshka_model, valid_activations)['features_for_classification']
    matryoshka_class = classify_features(matryoshka_features_full, labels)
    classification_results.append({
        'architecture': 'Vanilla Matryoshka',
        'total_features': matryoshka_metrics['total_features'],
        'accuracy': matryoshka_class['accuracy']
    })

    # 4. Matryoshka-TopK Hybrid
    print("4. Matryoshka-TopK Hybrid...")
    matryoshka_topk_model, matryoshka_topk_config = load_matryoshka_topk_sae(str(matryoshka_topk_checkpoint))
    matryoshka_topk_metrics = compute_matryoshka_topk_metrics(matryoshka_topk_model, val_data)
    results.append(matryoshka_topk_metrics)

    matryoshka_topk_features_full = compute_matryoshka_topk_metrics(matryoshka_topk_model, valid_activations)['features_for_classification']
    matryoshka_topk_class = classify_features(matryoshka_topk_features_full, labels)
    classification_results.append({
        'architecture': 'Matryoshka-TopK Hybrid',
        'total_features': matryoshka_topk_metrics['total_features'],
        'accuracy': matryoshka_topk_class['accuracy']
    })

    # Print results
    print_comparison_table(results)
    print_classification_results(classification_results)

    # Save results
    output_path = base_dir / "results/comprehensive_comparison.json"

    comparison_data = {
        'reconstruction_metrics': results,
        'classification_results': classification_results
    }

    with open(output_path, 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")
    print("\n✓ Comprehensive comparison complete!")
