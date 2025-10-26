"""
Train non-linear probes (MLPs) on continuous thought activations with PROPER train/test split.

Tests whether the 59.73% linear probe performance is limited by the linear
assumption, or if deception is inherently weakly encoded in continuous thoughts.

This version:
- Uses 80/20 train/test split
- Hyperparameter search on train set only
- Final evaluation on held-out test set
- Reports both train and test metrics to detect overfitting
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class MLPProbe(nn.Module):
    """MLP classifier for deception detection."""

    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 2))  # Binary classification

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def train_mlp_probe(X_train, y_train, X_val, y_val, hidden_dims, lr, dropout, device, max_epochs=100):
    """Train single MLP probe with early stopping."""

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model
    model = MLPProbe(X_train.shape[1], hidden_dims, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Early stopping
    best_val_acc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation
        y_true, y_pred, y_prob = evaluate(model, val_loader, device)
        val_acc = accuracy_score(y_true, y_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Load best model
    model.load_state_dict(best_model_state)

    return model, best_val_acc


def load_continuous_activations(data_path: str):
    """Load continuous thought activations."""
    print(f"Loading continuous thought activations from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"  Loaded {len(data['samples'])} samples")
    print(f"  Honest: {data['n_honest']}, Deceptive: {data['n_deceptive']}")

    return data


def main():
    print("=" * 80)
    print("Training Non-Linear Probes with PROPER Train/Test Split")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2.json"

    data = load_continuous_activations(str(data_path))
    samples = data['samples']

    print(f"\n[1/6] Preparing data...")

    # Extract features and labels
    X = []
    y = []

    for sample in samples:
        # Concatenate activations from layers 4, 8, 11 (6 tokens each)
        # Structure: thoughts[layer] = [token0, token1, ..., token5]
        activations = []
        for layer_key in ['layer_4', 'layer_8', 'layer_11']:
            for token_idx in range(6):
                activation = sample['thoughts'][layer_key][token_idx]
                activations.append(activation)

        X.append(np.concatenate(activations))
        y.append(1 if sample['is_honest'] else 0)  # 1=honest, 0=deceptive

    X = np.array(X)
    y = np.array(y)

    print(f"  Total samples: {len(X)}")
    print(f"  Features shape: {X.shape}")
    print(f"  Labels: Honest={np.sum(y == 1)}, Deceptive={np.sum(y == 0)}")
    print(f"  Balance: {np.mean(y)*100:.1f}% honest, {(1-np.mean(y))*100:.1f}% deceptive")
    print(f"\n  ⚠️  WARNING: Data is imbalanced due to extraction bug")
    print(f"      Expected: 1000 honest + 1000 deceptive = 2000 total")
    print(f"      Actual:   {np.sum(y == 1)} honest + {np.sum(y == 0)} deceptive = {len(X)} total")

    # PROPER train/test split (80/20)
    print(f"\n[2/6] Creating train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Train: {len(X_train)} samples (Honest={np.sum(y_train == 1)}, Deceptive={np.sum(y_train == 0)})")
    print(f"  Test:  {len(X_test)} samples (Honest={np.sum(y_test == 1)}, Deceptive={np.sum(y_test == 0)})")

    # Standardize (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Architecture configurations
    architectures = {
        'shallow': [256],
        'medium': [512, 256],
        'deep': [768, 512, 256]
    }

    # Hyperparameters to search
    learning_rates = [1e-4, 1e-3, 1e-2]
    dropouts = [0.1, 0.3, 0.5]

    print(f"\n[3/6] Training configurations...")
    print(f"  Architectures: {list(architectures.keys())}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Dropouts: {dropouts}")
    print(f"  5-fold cross-validation on TRAIN set only")

    # Store results for each architecture
    results = {}

    for arch_name, hidden_dims in architectures.items():
        print(f"\n{'=' * 80}")
        print(f"Architecture: {arch_name.upper()} - {hidden_dims}")
        print(f"{'=' * 80}")

        best_config = None
        best_score = 0

        # Hyperparameter search ON TRAIN SET ONLY
        print(f"\n  Hyperparameter search on train set...")

        for lr in learning_rates:
            for dropout in dropouts:
                # 5-fold cross-validation ON TRAIN SET
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
                    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]

                    _, val_acc = train_mlp_probe(
                        X_tr, y_tr, X_val, y_val,
                        hidden_dims, lr, dropout, device
                    )
                    fold_scores.append(val_acc)

                avg_score = np.mean(fold_scores)
                print(f"    lr={lr:.0e}, dropout={dropout:.1f}: {avg_score*100:.2f}%")

                if avg_score > best_score:
                    best_score = avg_score
                    best_config = {'lr': lr, 'dropout': dropout}

        print(f"\n  ✅ Best config: lr={best_config['lr']:.0e}, dropout={best_config['dropout']:.1f}")
        print(f"  ✅ CV accuracy (train): {best_score*100:.2f}%")

        # Train final model on FULL TRAIN SET with best config
        print(f"\n[4/6] Training final model on full train set...")

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = MLPProbe(X_train_scaled.shape[1], hidden_dims, best_config['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])

        # Train for fixed epochs
        for epoch in range(50):
            train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on TRAIN set
        train_eval_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        y_train_true, y_train_pred, y_train_prob = evaluate(model, train_eval_loader, device)

        train_acc = accuracy_score(y_train_true, y_train_pred)
        train_f1 = f1_score(y_train_true, y_train_pred)
        train_auroc = roc_auc_score(y_train_true, y_train_prob)
        train_conf = confusion_matrix(y_train_true, y_train_pred)

        # Evaluate on TEST set
        print(f"\n[5/6] Evaluating on held-out test set...")

        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_scaled),
            torch.LongTensor(y_test)
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        y_test_true, y_test_pred, y_test_prob = evaluate(model, test_loader, device)

        test_acc = accuracy_score(y_test_true, y_test_pred)
        test_f1 = f1_score(y_test_true, y_test_pred)
        test_auroc = roc_auc_score(y_test_true, y_test_prob)
        test_conf = confusion_matrix(y_test_true, y_test_pred)

        print(f"\n[6/6] Final Results - {arch_name.upper()}:")
        print(f"\n  TRAIN Performance:")
        print(f"    Accuracy: {train_acc*100:.2f}%")
        print(f"    F1-Score: {train_f1:.3f}")
        print(f"    AUROC:    {train_auroc:.3f}")

        print(f"\n  TEST Performance (held-out):")
        print(f"    Accuracy: {test_acc*100:.2f}%")
        print(f"    F1-Score: {test_f1:.3f}")
        print(f"    AUROC:    {test_auroc:.3f}")

        print(f"\n  Overfitting Gap: {(train_acc - test_acc)*100:+.2f} percentage points")

        print(f"\n  Test Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Deceptive  Honest")
        print(f"  Actual:")
        print(f"    Deceptive      {test_conf[0][0]:3d}       {test_conf[0][1]:3d}")
        print(f"    Honest         {test_conf[1][0]:3d}       {test_conf[1][1]:3d}")

        # Calculate recall
        deceptive_recall = test_conf[0][0] / (test_conf[0][0] + test_conf[0][1])
        honest_recall = test_conf[1][1] / (test_conf[1][0] + test_conf[1][1])
        print(f"\n  Deceptive Recall: {deceptive_recall*100:.1f}%")
        print(f"  Honest Recall:    {honest_recall*100:.1f}%")

        # Store results
        results[arch_name] = {
            'architecture': hidden_dims,
            'best_lr': float(best_config['lr']),
            'best_dropout': float(best_config['dropout']),
            'cv_accuracy': float(best_score),
            'train_accuracy': float(train_acc),
            'train_f1': float(train_f1),
            'train_auroc': float(train_auroc),
            'test_accuracy': float(test_acc),
            'test_f1': float(test_f1),
            'test_auroc': float(test_auroc),
            'overfitting_gap': float(train_acc - test_acc),
            'test_confusion_matrix': test_conf.tolist(),
            'test_deceptive_recall': float(deceptive_recall),
            'test_honest_recall': float(honest_recall),
            'test_report': classification_report(y_test_true, y_test_pred, output_dict=True)
        }

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY: Non-Linear vs Linear Probes (TEST SET PERFORMANCE)")
    print(f"{'=' * 80}")
    print(f"\n{'Architecture':<15} {'CV Acc':<12} {'Train Acc':<12} {'Test Acc':<12} {'Test AUROC':<12} {'Overfit':<10}")
    print(f"{'-'*73}")

    for arch_name, result in results.items():
        print(f"{arch_name.capitalize():<15} {result['cv_accuracy']*100:>6.2f}%      "
              f"{result['train_accuracy']*100:>6.2f}%      {result['test_accuracy']*100:>6.2f}%      "
              f"{result['test_auroc']:>6.3f}        {result['overfitting_gap']*100:>+5.2f}pp")

    # Linear baseline (from previous experiment)
    linear_acc = 0.5973
    linear_auroc = 0.643
    linear_f1 = 0.599

    # Response token baseline
    response_acc = 0.7050
    response_auroc = 0.777

    print(f"{'-'*73}")
    print(f"{'Linear (baseline)':<15} {'N/A':<12} {'N/A':<12} {linear_acc*100:>6.2f}%      {linear_auroc:>6.3f}        {'N/A':<10}")
    print(f"{'Response tokens':<15} {'N/A':<12} {'N/A':<12} {response_acc*100:>6.2f}%      {response_auroc:>6.3f}        {'N/A':<10}")

    # Find best architecture by TEST accuracy
    best_arch = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    improvement_vs_linear = (best_arch[1]['test_accuracy'] - linear_acc) * 100
    improvement_vs_response = (best_arch[1]['test_accuracy'] - response_acc) * 100

    print(f"\n{'=' * 80}")
    print(f"BEST: {best_arch[0].upper()} - Test Accuracy: {best_arch[1]['test_accuracy']*100:.2f}%")
    print(f"{'=' * 80}")
    print(f"  vs Linear baseline:      {improvement_vs_linear:+.2f} points")
    print(f"  vs Response tokens:      {improvement_vs_response:+.2f} points")

    if best_arch[1]['test_accuracy'] > response_acc:
        print(f"\n  ✅ Non-linear continuous thoughts BEAT response tokens!")
    elif best_arch[1]['test_accuracy'] > linear_acc:
        print(f"\n  ✅ Non-linearity helps, but still below response tokens")
    else:
        print(f"\n  ❌ No improvement over linear baseline")

    print(f"{'=' * 80}")

    # Save results
    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_nonlinear_proper_split_gpt2.json"

    final_results = {
        'model': 'gpt2',
        'probe_type': 'nonlinear_mlp_proper_split',
        'data_stats': {
            'total_samples': int(len(X)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'honest_samples': int(np.sum(y == 1)),
            'deceptive_samples': int(np.sum(y == 0)),
            'balance_ratio': float(np.mean(y))
        },
        'architectures': results,
        'baselines': {
            'linear_continuous': {
                'accuracy': linear_acc,
                'f1_score': linear_f1,
                'auroc': linear_auroc
            },
            'linear_response': {
                'accuracy': response_acc,
                'auroc': response_auroc
            }
        },
        'best_architecture': {
            'name': best_arch[0],
            'test_accuracy': best_arch[1]['test_accuracy'],
            'test_auroc': best_arch[1]['test_auroc'],
            'improvement_over_linear': float(improvement_vs_linear),
            'improvement_over_response': float(improvement_vs_response)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Saved: {output_file}")

    return results


if __name__ == "__main__":
    main()
