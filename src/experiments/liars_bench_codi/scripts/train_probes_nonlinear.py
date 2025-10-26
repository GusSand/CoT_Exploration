"""
Train non-linear probes (MLPs) on continuous thought activations.

Tests whether the 59.73% linear probe performance is limited by the linear
assumption, or if deception is inherently weakly encoded in continuous thoughts.

Architectures tested:
1. Shallow MLP: 1 hidden layer (256 units)
2. Medium MLP: 2 hidden layers (512, 256 units)
3. Deep MLP: 3 hidden layers (768, 512, 256 units)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
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
    print("Training Non-Linear Probes on Continuous Thought Activations")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "probe_dataset_gpt2.json"

    data = load_continuous_activations(str(data_path))
    samples = data['samples']

    print(f"\n[1/5] Preparing data...")

    # Extract features and labels
    X = []
    y = []
    seen_questions = set()

    for idx, sample in enumerate(samples):
        # Structure: thoughts[layer] is a list (batch_size) of lists (num_tokens) of lists (hidden_size)
        # Due to a bug in extraction, each sample contains the ENTIRE batch's activations
        # We need to match each sample to its correct batch element based on index

        # Skip duplicates (same question means same batch was stored multiple times)
        if sample['question'] in seen_questions:
            continue
        seen_questions.add(sample['question'])

        # Get the correct batch element index
        batch_size = len(sample['thoughts']['layer_4'])
        batch_elem_idx = idx % batch_size

        # Concatenate activations from layers 4, 8, 11 (6 tokens each)
        activations = []
        for layer_key in ['layer_4', 'layer_8', 'layer_11']:
            layer_data = sample['thoughts'][layer_key]
            # Extract all 6 tokens for this batch element
            for token_idx in range(6):
                activation = layer_data[batch_elem_idx][token_idx]
                activations.append(activation)

        X.append(np.concatenate(activations))
        y.append(1 if sample['is_honest'] else 0)  # 1=honest, 0=deceptive

    X = np.array(X)
    y = np.array(y)

    print(f"  Features shape: {X.shape}")
    print(f"  Labels: Honest={np.sum(y == 1)}, Deceptive={np.sum(y == 0)}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Architecture configurations
    architectures = {
        'shallow': [256],
        'medium': [512, 256],
        'deep': [768, 512, 256]
    }

    # Hyperparameters to search
    learning_rates = [1e-4, 1e-3, 1e-2]
    dropouts = [0.1, 0.3, 0.5]

    print(f"\n[2/5] Training configurations...")
    print(f"  Architectures: {list(architectures.keys())}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Dropouts: {dropouts}")
    print(f"  5-fold cross-validation")

    # Store results for each architecture
    results = {}

    for arch_name, hidden_dims in architectures.items():
        print(f"\n{'=' * 80}")
        print(f"Architecture: {arch_name.upper()} - {hidden_dims}")
        print(f"{'=' * 80}")

        best_config = None
        best_score = 0

        # Hyperparameter search
        print(f"\n[3/5] Hyperparameter search...")

        for lr in learning_rates:
            for dropout in dropouts:
                # 5-fold cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                fold_scores = []

                for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y)):
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    _, val_acc = train_mlp_probe(
                        X_train, y_train, X_val, y_val,
                        hidden_dims, lr, dropout, device
                    )
                    fold_scores.append(val_acc)

                avg_score = np.mean(fold_scores)
                print(f"  lr={lr:.0e}, dropout={dropout:.1f}: {avg_score*100:.2f}%")

                if avg_score > best_score:
                    best_score = avg_score
                    best_config = {'lr': lr, 'dropout': dropout}

        print(f"\n  ✅ Best config: lr={best_config['lr']:.0e}, dropout={best_config['dropout']:.1f}")
        print(f"  ✅ CV accuracy: {best_score*100:.2f}%")

        # Train final model on full dataset with best config
        print(f"\n[4/5] Training final model on full dataset...")

        # Use full dataset
        train_dataset = TensorDataset(
            torch.FloatTensor(X_scaled),
            torch.LongTensor(y)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        model = MLPProbe(X_scaled.shape[1], hidden_dims, best_config['dropout']).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=best_config['lr'])

        # Train for fixed epochs
        for epoch in range(50):
            train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate on full dataset
        full_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        y_true, y_pred, y_prob = evaluate(model, full_loader, device)

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auroc = roc_auc_score(y_true, y_prob)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"\n[5/5] Final Results - {arch_name.upper()}:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  AUROC:    {auroc:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Deceptive  Honest")
        print(f"  Actual:")
        print(f"    Deceptive      {conf_matrix[0][0]:3d}      {conf_matrix[0][1]:3d}")
        print(f"    Honest         {conf_matrix[1][0]:3d}      {conf_matrix[1][1]:3d}")

        # Store results
        results[arch_name] = {
            'architecture': hidden_dims,
            'best_lr': float(best_config['lr']),
            'best_dropout': float(best_config['dropout']),
            'cv_accuracy': float(best_score),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'auroc': float(auroc),
            'confusion_matrix': conf_matrix.tolist(),
            'report': classification_report(y_true, y_pred, output_dict=True)
        }

    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY: Non-Linear vs Linear Probes")
    print(f"{'=' * 80}")
    print(f"\n{'Architecture':<15} {'Accuracy':<12} {'AUROC':<12} {'F1-Score':<12}")
    print(f"{'-'*51}")

    for arch_name, result in results.items():
        print(f"{arch_name.capitalize():<15} {result['accuracy']*100:>6.2f}%      {result['auroc']:>6.3f}      {result['f1_score']:>6.3f}")

    # Linear baseline
    linear_acc = 0.5973
    linear_auroc = 0.643
    linear_f1 = 0.599
    print(f"{'-'*51}")
    print(f"{'Linear (baseline)':<15} {linear_acc*100:>6.2f}%      {linear_auroc:>6.3f}      {linear_f1:>6.3f}")

    # Find best architecture
    best_arch = max(results.items(), key=lambda x: x[1]['accuracy'])
    improvement = (best_arch[1]['accuracy'] - linear_acc) * 100

    print(f"\n{'=' * 80}")
    if improvement > 0:
        print(f"✅ IMPROVEMENT: {best_arch[0].upper()} beats linear by {improvement:+.2f} points")
    else:
        print(f"❌ NO IMPROVEMENT: Non-linear probes do not beat linear baseline")
        print(f"   Conclusion: Deception is weakly encoded, not a linear modeling issue")
    print(f"{'=' * 80}")

    # Save results
    output_dir = script_dir.parent / "results"
    output_file = output_dir / "probe_results_nonlinear_gpt2.json"

    final_results = {
        'model': 'gpt2',
        'probe_type': 'nonlinear_mlp',
        'architectures': results,
        'linear_baseline': {
            'accuracy': linear_acc,
            'f1_score': linear_f1,
            'auroc': linear_auroc
        },
        'best_architecture': {
            'name': best_arch[0],
            'accuracy': best_arch[1]['accuracy'],
            'improvement_over_linear': float(improvement)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✅ Saved: {output_file}")

    return results


if __name__ == "__main__":
    main()
