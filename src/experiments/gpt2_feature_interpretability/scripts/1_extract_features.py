"""
Extract sparse features from all 72 GPT-2 SAEs.

For each of 1,000 problems × 12 layers × 6 positions:
- Load corresponding SAE checkpoint
- Extract sparse features (512 dims, ~150 active)
- Save all features to single file for correlation analysis

Output: gpt2_extracted_features.pt (~300-400 MB)
"""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add TopK SAE to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "topk_grid_pilot"))
from topk_sae import TopKAutoencoder


def load_sae(position, layer, base_dir='src/experiments/gpt2_sae_training/results/sweet_spot_all'):
    """Load a single SAE checkpoint."""
    checkpoint_path = Path(base_dir) / f'gpt2_sweet_spot_pos{position}_layer{layer}.pt'
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    sae = TopKAutoencoder(
        input_dim=checkpoint['config']['input_dim'],
        latent_dim=checkpoint['config']['latent_dim'],
        k=checkpoint['config']['k']
    )
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.eval()

    return sae


def load_activations(position, layer, data_path='src/experiments/gpt2_sae_training/data'):
    """Load activations for a specific position-layer combination."""
    # Load full dataset
    train_data = torch.load(f'{data_path}/gpt2_full_train_activations.pt', weights_only=False)
    val_data = torch.load(f'{data_path}/gpt2_full_val_activations.pt', weights_only=False)

    # Combine train and val
    all_activations = torch.cat([train_data['activations'], val_data['activations']], dim=0)
    all_positions = np.concatenate([train_data['metadata']['positions'], val_data['metadata']['positions']])
    all_layers = np.concatenate([train_data['metadata']['layers'], val_data['metadata']['layers']])
    all_problem_ids = np.concatenate([train_data['metadata']['problem_ids'], val_data['metadata']['problem_ids']])

    # Filter to this position and layer
    mask = (all_positions == position) & (all_layers == layer)
    activations = all_activations[mask]
    problem_ids = all_problem_ids[mask]

    return activations, problem_ids


def extract_all_features():
    """Extract features from all 72 SAEs."""
    print("="*80)
    print("EXTRACTING FEATURES FROM ALL 72 GPT-2 SAEs")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Storage for all features
    all_features = {}
    all_metadata = {
        'problem_ids': [],
        'layers': [],
        'positions': [],
    }

    total_saes = 12 * 6
    count = 0

    for layer in range(12):
        for position in range(6):
            count += 1
            print(f"[{count}/{total_saes}] Processing Layer {layer}, Position {position}...")

            # Load SAE
            sae = load_sae(position, layer).to(device)

            # Load activations
            activations, problem_ids = load_activations(position, layer)
            activations = activations.to(device)

            print(f"  Activations: {activations.shape}")

            # Extract features
            with torch.no_grad():
                reconstruction, features, metrics = sae(activations)

            # Move to CPU and store
            features_cpu = features.cpu()

            # Store features by (layer, position)
            key = (layer, position)
            all_features[key] = features_cpu

            # Store metadata
            for pid in problem_ids:
                all_metadata['problem_ids'].append(int(pid))
                all_metadata['layers'].append(layer)
                all_metadata['positions'].append(position)

            # Print stats
            active_pct = (features_cpu != 0).float().mean().item() * 100
            print(f"  Features: {features_cpu.shape}, Active: {active_pct:.1f}%, L0: {metrics['l0_mean']:.1f}")

            # Clear GPU memory
            del sae, activations, features
            if count % 10 == 0:
                torch.cuda.empty_cache()

    print(f"\n✓ Extracted features from all {total_saes} SAEs")

    # Verify metadata
    total_samples = len(all_metadata['problem_ids'])
    expected_samples = 1000 * 12 * 6
    print(f"\nMetadata verification:")
    print(f"  Total samples: {total_samples}")
    print(f"  Expected: {expected_samples}")
    print(f"  Match: {total_samples == expected_samples}")

    # Save features
    output_dir = Path('src/experiments/gpt2_feature_interpretability/data')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'gpt2_extracted_features.pt'
    print(f"\nSaving features to: {output_path}")

    torch.save({
        'features': all_features,  # Dict[(layer, pos)] -> tensor (N, 512)
        'metadata': all_metadata,
        'config': {
            'model': 'gpt2',
            'num_layers': 12,
            'num_positions': 6,
            'latent_dim': 512,
            'k': 150,
            'num_problems': 1000,
        }
    }, output_path)

    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"✓ Saved ({size_mb:.1f} MB)")

    # Summary
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"  Total SAEs: {total_saes}")
    print(f"  Total samples: {total_samples}")
    print(f"  Output: {output_path}")
    print("="*80)


if __name__ == '__main__':
    extract_all_features()
