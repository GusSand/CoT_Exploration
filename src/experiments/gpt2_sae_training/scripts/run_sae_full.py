"""
GPT-2 Experiment 3: SAE Training (FULL IMPLEMENTATION)

Trains Sparse Autoencoder on GPT-2 continuous thoughts:
- Input: 768 dims (GPT-2 hidden size)
- Features: 4096 sparse features
- Layers: 4, 8, 11
- Tokens: 6

Then uses SAE features for error prediction.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for GPT-2 activations."""

    def __init__(self, input_dim=768, n_features=4096):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

    def forward(self, x):
        # Encode
        hidden = F.relu(self.encoder(x))

        # Decode
        recon = self.decoder(hidden)

        return recon, hidden


def load_activations(data_path: str):
    """Load all activations for SAE training."""
    print(f"Loading activations from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']

    # Collect activations from layers 4, 8, 11
    layers = [4, 8, 11]
    activations = []

    for sample in samples:
        for layer in layers:
            layer_key = f'layer_{layer}'
            for token_idx in range(6):
                act = sample['thoughts'][layer_key][token_idx]
                activations.append(act)

    activations = torch.tensor(activations, dtype=torch.float32)

    print(f"  Collected {len(activations)} activation vectors")
    print(f"  Shape: {activations.shape}")

    return activations


def train_sae(activations, n_features=4096, n_epochs=25, batch_size=256, lr=0.001, l1_coef=0.001, device='cuda'):
    """Train Sparse Autoencoder."""

    print(f"\nTraining SAE...")
    print(f"  Features: {n_features}")
    print(f"  Epochs: {n_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  L1 coefficient: {l1_coef}")

    # Model
    model = SparseAutoencoder(input_dim=768, n_features=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    activations = activations.to(device)
    n_samples = len(activations)

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_sparsity = 0

        # Shuffle
        indices = torch.randperm(n_samples)

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = activations[batch_indices]

            # Forward
            recon, hidden = model(batch)

            # Loss
            recon_loss = F.mse_loss(recon, batch)
            sparsity_loss = hidden.abs().mean()
            loss = recon_loss + l1_coef * sparsity_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparsity += sparsity_loss.item()

        # Epoch summary
        n_batches = (n_samples + batch_size - 1) // batch_size
        print(f"  Epoch {epoch+1}/{n_epochs}: Loss={total_loss/n_batches:.4f}, "
              f"Recon={total_recon/n_batches:.4f}, Sparsity={total_sparsity/n_batches:.4f}")

    print("  SAE training complete!")

    return model


def extract_sae_features(model, data_path: str, device='cuda'):
    """Extract SAE features for all samples."""

    print(f"\nExtracting SAE features...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    model.eval()

    sae_samples = []

    with torch.no_grad():
        for sample in samples:
            # Concatenate activations from layers 4, 8, 11 and all 6 tokens
            sample_features = []
            for layer in [4, 8, 11]:
                layer_key = f'layer_{layer}'
                for token_idx in range(6):
                    act = torch.tensor(sample['thoughts'][layer_key][token_idx], dtype=torch.float32).to(device)
                    _, hidden = model(act.unsqueeze(0))
                    sample_features.extend(hidden.cpu().squeeze(0).tolist())

            sae_samples.append({
                'id': sample['id'],
                'is_correct': sample['is_correct'],
                'sae_features': sample_features
            })

    print(f"  Extracted features for {len(sae_samples)} samples")
    print(f"  Feature dimension per sample: {len(sae_samples[0]['sae_features'])}")

    return sae_samples


def train_error_classifier(sae_samples):
    """Train error prediction classifier on SAE features."""

    print(f"\nTraining error classifier...")

    # Prepare data
    X = np.array([s['sae_features'] for s in sae_samples])
    y = np.array([1 if s['is_correct'] else 0 for s in sae_samples])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\n  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Incorrect', 'Correct']))

    return {
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc)
    }


def main():
    """Run GPT-2 SAE experiment."""

    print("="*60)
    print("GPT-2 EXPERIMENT 3: SAE TRAINING")
    print("="*60)

    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "src/experiments/gpt2_shared_data/gpt2_predictions_1000.json"
    output_dir = project_root / "src/experiments/gpt2_sae_training/results"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load activations
    activations = load_activations(str(data_path))

    # Train SAE
    model = train_sae(activations, n_features=4096, n_epochs=25, device=device)

    # Save SAE
    model_path = output_dir / "sae_model_gpt2.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n  SAE model saved: {model_path}")

    # Extract SAE features
    sae_samples = extract_sae_features(model, str(data_path), device=device)

    # Save SAE features
    features_path = output_dir / "sae_features_gpt2.json"
    with open(features_path, 'w') as f:
        json.dump({
            'metadata': {
                'model': 'GPT-2 CODI',
                'n_samples': len(sae_samples),
                'sae_features': 4096,
                'layers': [4, 8, 11],
                'tokens': 6,
                'date': datetime.now().isoformat()
            },
            'samples': sae_samples
        }, f)
    print(f"  SAE features saved: {features_path}")

    # Train error classifier
    results = train_error_classifier(sae_samples)

    # Save results
    results_path = output_dir / "classification_results_gpt2.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved: {results_path}")

    print("\n" + "="*60)
    print("✅ GPT-2 SAE TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
