"""
Matryoshka-TopK Hybrid SAE: Hierarchical structure with TopK activation.

Combines:
1. Matryoshka's hierarchical representation (coarse → medium → fine)
2. TopK's efficient activation (exactly K features per sample)

Architecture:
- Level 1: 128 features, TopK(K=25)  → Coarse patterns
- Level 2: 256 features, TopK(K=35)  → Medium patterns
- Level 3: 512 features, TopK(K=40)  → Fine patterns
- Total active per sample: 100 (matches TopK baseline)

Key differences from vanilla Matryoshka:
- TopK activation instead of ReLU + L1
- Smaller feature budgets per level
- No L1 penalty (TopK enforces sparsity)
- 100% feature utilization expected
"""

import torch
import torch.nn as nn
import numpy as np


class MatryoshkaTopKSAE(nn.Module):
    """Hierarchical SAE with TopK activation at each level."""

    def __init__(
        self,
        input_dim: int = 2048,
        levels: list = [128, 256, 512],
        k_values: list = [25, 35, 40],
        level_weights: list = [0.3, 0.3, 0.4]
    ):
        """
        Args:
            input_dim: Input activation dimension
            levels: Number of features at each hierarchy level
            k_values: TopK values for each level
            level_weights: Reconstruction loss weights per level
        """
        super().__init__()

        self.input_dim = input_dim
        self.levels = levels
        self.k_values = k_values
        self.level_weights = level_weights

        assert len(levels) == 3, "Expected 3 levels"
        assert len(k_values) == 3, "Expected 3 K values"
        assert len(level_weights) == 3, "Expected 3 level weights"

        # Encoders for each level
        self.encoder_1 = nn.Linear(input_dim, levels[0], bias=True)
        self.encoder_2 = nn.Linear(input_dim, levels[1], bias=True)
        self.encoder_3 = nn.Linear(input_dim, levels[2], bias=True)

        # Shared decoder (from concatenated features)
        total_features = sum(levels)
        self.decoder = nn.Linear(total_features, input_dim, bias=True)

        # Initialize decoder weights (columns normalized)
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def topk_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Apply TopK activation: keep top K values, zero rest.

        Args:
            x: Pre-activation values [batch, features]
            k: Number of features to keep

        Returns:
            TopK activated features [batch, features]
        """
        # Get TopK indices
        topk_vals, topk_indices = torch.topk(x, k, dim=1)

        # Create sparse tensor
        result = torch.zeros_like(x)
        result.scatter_(1, topk_indices, topk_vals)

        return result

    def encode_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Encode at specific hierarchy level with TopK.

        Args:
            x: Input activations [batch, input_dim]
            level: Level index (0, 1, or 2)

        Returns:
            TopK features for that level
        """
        if level == 0:
            pre_act = self.encoder_1(x)
            return self.topk_activation(pre_act, self.k_values[0])
        elif level == 1:
            pre_act = self.encoder_2(x)
            return self.topk_activation(pre_act, self.k_values[1])
        elif level == 2:
            pre_act = self.encoder_3(x)
            return self.topk_activation(pre_act, self.k_values[2])
        else:
            raise ValueError(f"Invalid level: {level}")

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass through all levels.

        Args:
            x: Input activations [batch, input_dim]

        Returns:
            (reconstructions_dict, features_dict)
            - reconstructions_dict: {level_i: reconstruction from levels 0..i}
            - features_dict: {level_i: TopK features at level i}
        """
        # Encode at all levels with TopK
        z1 = self.encode_level(x, 0)  # [batch, 128], K=25
        z2 = self.encode_level(x, 1)  # [batch, 256], K=35
        z3 = self.encode_level(x, 2)  # [batch, 512], K=40

        # Nested reconstruction (hierarchical)
        # Level 1: use only z1
        z1_padded = torch.cat([
            z1,
            torch.zeros(x.shape[0], self.levels[1], device=x.device),
            torch.zeros(x.shape[0], self.levels[2], device=x.device)
        ], dim=1)
        recon_1 = self.decoder(z1_padded)

        # Level 2: use z1 + z2
        z2_padded = torch.cat([
            z1,
            z2,
            torch.zeros(x.shape[0], self.levels[2], device=x.device)
        ], dim=1)
        recon_2 = self.decoder(z2_padded)

        # Level 3: use z1 + z2 + z3
        z3_full = torch.cat([z1, z2, z3], dim=1)
        recon_3 = self.decoder(z3_full)

        reconstructions = {
            'level_1': recon_1,
            'level_2': recon_2,
            'level_3': recon_3
        }

        features = {
            'level_1': z1,
            'level_2': z2,
            'level_3': z3,
            'concatenated': z3_full  # All features combined
        }

        return reconstructions, features

    def loss(
        self,
        x: torch.Tensor,
        reconstructions: dict,
        features: dict
    ) -> tuple:
        """Compute hierarchical reconstruction loss.

        No L1 penalty - TopK enforces sparsity inherently.

        Args:
            x: Original input
            reconstructions: Dict of reconstructions per level
            features: Dict of features per level (unused, for compatibility)

        Returns:
            (total_loss, loss_components_dict)
        """
        # MSE reconstruction loss at each level
        mse_1 = torch.mean((x - reconstructions['level_1']) ** 2)
        mse_2 = torch.mean((x - reconstructions['level_2']) ** 2)
        mse_3 = torch.mean((x - reconstructions['level_3']) ** 2)

        # Weighted sum (prioritize fine level)
        w1, w2, w3 = self.level_weights
        total_loss = w1 * mse_1 + w2 * mse_2 + w3 * mse_3

        components = {
            'total': total_loss.item(),
            'mse_level_1': mse_1.item(),
            'mse_level_2': mse_2.item(),
            'mse_level_3': mse_3.item(),
            'l1': 0.0  # No L1 penalty
        }

        return total_loss, components

    def compute_metrics(
        self,
        x: torch.Tensor,
        reconstructions: dict,
        features: dict
    ) -> dict:
        """Compute validation metrics for each level.

        Args:
            x: Original input
            reconstructions: Dict of reconstructions per level
            features: Dict of features per level

        Returns:
            Dictionary of metrics per level
        """
        metrics = {}

        # Compute for each level
        for level_idx, level_name in enumerate(['level_1', 'level_2', 'level_3']):
            recon = reconstructions[level_name]
            feats = features[level_name]

            # Explained variance (R²)
            ss_res = torch.sum((x - recon) ** 2)
            ss_tot = torch.sum((x - x.mean(0, keepdim=True)) ** 2)
            explained_var = 1 - (ss_res / ss_tot)

            # L0 norm (average active features per sample)
            # With TopK, this should be exactly K
            l0_norm = (feats != 0).float().sum(1).mean()

            # Feature utilization
            # For TopK, expect 100% (all features used at least once)
            total_features = feats.shape[1]
            active_features = (feats.abs().sum(0) > 0).sum().item()
            dead_features = total_features - active_features
            feature_death_rate = dead_features / total_features

            # Mean activation magnitude (for active features)
            mean_activation = feats[feats != 0].abs().mean().item() if (feats != 0).any() else 0.0
            max_activation = feats.abs().max().item()

            metrics[level_name] = {
                'features': total_features,
                'expected_k': self.k_values[level_idx],
                'explained_variance': explained_var.item(),
                'l0_norm': l0_norm.item(),
                'active_features': active_features,
                'dead_features': dead_features,
                'feature_death_rate': feature_death_rate,
                'utilization_pct': (active_features / total_features) * 100,
                'mean_activation': mean_activation,
                'max_activation': max_activation
            }

        return metrics

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'architecture': 'MatryoshkaTopKSAE',
            'input_dim': self.input_dim,
            'levels': self.levels,
            'k_values': self.k_values,
            'level_weights': self.level_weights,
            'total_features': sum(self.levels),
            'total_active_per_sample': sum(self.k_values)
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test instantiation
    print("=" * 60)
    print("Matryoshka-TopK SAE Test")
    print("=" * 60)

    model = MatryoshkaTopKSAE(
        input_dim=2048,
        levels=[128, 256, 512],
        k_values=[25, 35, 40],
        level_weights=[0.3, 0.3, 0.4]
    )

    config = model.get_config()
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print(f"\nParameters: {count_parameters(model):,}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(32, 2048)

    reconstructions, features = model(x)

    print("\nReconstructions:")
    for level_name, recon in reconstructions.items():
        print(f"  {level_name}: {recon.shape}")

    print("\nFeatures:")
    for level_name, feat in features.items():
        if level_name == 'concatenated':
            active = (feat != 0).float().sum(1).mean()
            print(f"  {level_name}: {feat.shape} (avg active: {active:.1f})")
        else:
            active = (feat != 0).float().sum(1).mean()
            expected_k = model.k_values[['level_1', 'level_2', 'level_3'].index(level_name)]
            print(f"  {level_name}: {feat.shape} (active: {active:.1f}, expected K={expected_k})")

    # Test loss computation
    loss, components = model.loss(x, reconstructions, features)
    print("\nLoss components:")
    for key, value in components.items():
        print(f"  {key}: {value:.6f}")

    print("\n✓ Model test complete!")
