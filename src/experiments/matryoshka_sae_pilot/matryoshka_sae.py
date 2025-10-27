"""
Matryoshka Sparse Autoencoder for Continuous Thought Decoding.

Implements hierarchical SAE with nested feature levels [512, 1024, 2048].
Based on SAEBench paper Section 4.1 - Matryoshka representation learning.

Key Innovation:
    - Level 1 (512): Coarse features (high-level reasoning patterns)
    - Level 2 (1024): Medium features (intermediate abstractions)
    - Level 3 (2048): Fine features (detailed computations)
    - Nested training: Each level builds on previous levels

Architecture:
    Input (2048,) → 3 Encoders → Features [512, 1024, 2048] → Shared Decoder → Output (2048,)

Loss:
    L = w1*MSE(x, recon1) + w2*MSE(x, recon2) + w3*MSE(x, recon3) + λ*L1(all features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List


class MatryoshkaSAE(nn.Module):
    """Matryoshka Sparse Autoencoder with hierarchical feature levels.

    Args:
        input_dim: Dimension of input vectors (2048 for LLaMA)
        levels: List of feature dimensions [512, 1024, 2048]
        l1_coefficient: Sparsity penalty (0.0005 default)
        level_weights: Reconstruction loss weights [0.3, 0.3, 0.4] (prioritize fine)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        levels: List[int] = [512, 1024, 2048],
        l1_coefficient: float = 0.0005,
        level_weights: List[float] = [0.3, 0.3, 0.4]
    ):
        super().__init__()

        assert len(levels) == 3, "Matryoshka SAE requires exactly 3 levels"
        assert len(level_weights) == 3, "Must provide 3 level weights"
        assert abs(sum(level_weights) - 1.0) < 1e-6, "Level weights must sum to 1.0"

        self.input_dim = input_dim
        self.levels = levels
        self.l1_coefficient = l1_coefficient
        self.level_weights = level_weights

        # Three separate encoders (one per level)
        self.encoder_1 = nn.Linear(input_dim, levels[0], bias=True)  # 2048 → 512
        self.encoder_2 = nn.Linear(input_dim, levels[1], bias=True)  # 2048 → 1024
        self.encoder_3 = nn.Linear(input_dim, levels[2], bias=True)  # 2048 → 2048

        # Shared decoder for all levels (key for nested learning)
        # Takes max feature dimension and outputs to input space
        self.decoder = nn.Linear(levels[2], input_dim, bias=True)  # 2048 → 2048

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for encoder in [self.encoder_1, self.encoder_2, self.encoder_3]:
            nn.init.xavier_uniform_(encoder.weight)
            nn.init.zeros_(encoder.bias)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Encode input at specific level.

        Args:
            x: Input tensor (batch_size, input_dim)
            level: Level index (0, 1, or 2)

        Returns:
            Sparse features for that level
        """
        if level == 0:
            return F.relu(self.encoder_1(x))  # 512 features
        elif level == 1:
            return F.relu(self.encoder_2(x))  # 1024 features
        elif level == 2:
            return F.relu(self.encoder_3(x))  # 2048 features
        else:
            raise ValueError(f"Invalid level {level}. Must be 0, 1, or 2.")

    def decode_level(self, features: torch.Tensor, level: int) -> torch.Tensor:
        """Decode features from specific level.

        For nested reconstruction:
        - Level 0 (512): Pad to 2048 with zeros
        - Level 1 (1024): Pad to 2048 with zeros
        - Level 2 (2048): Use directly

        Args:
            features: Feature tensor (batch_size, level_dim)
            level: Level index (0, 1, or 2)

        Returns:
            Reconstruction (batch_size, input_dim)
        """
        batch_size = features.shape[0]

        # Pad features to max dimension (2048) if needed
        if level == 0:
            # 512 → 2048: pad with zeros
            padded = torch.zeros(batch_size, self.levels[2], device=features.device)
            padded[:, :self.levels[0]] = features
        elif level == 1:
            # 1024 → 2048: pad with zeros
            padded = torch.zeros(batch_size, self.levels[2], device=features.device)
            padded[:, :self.levels[1]] = features
        else:
            # 2048 → 2048: no padding needed
            padded = features

        return self.decoder(padded)

    def forward(self, x: torch.Tensor) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """Forward pass through all levels of Matryoshka SAE.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Tuple of (reconstructions, features) where each is a tuple of 3 tensors
        """
        # Encode at all three levels
        z1 = self.encode_level(x, 0)  # 512 features (coarse)
        z2 = self.encode_level(x, 1)  # 1024 features (medium)
        z3 = self.encode_level(x, 2)  # 2048 features (fine)

        # Decode from all three levels
        recon1 = self.decode_level(z1, 0)  # Reconstruction from 512 features
        recon2 = self.decode_level(z2, 1)  # Reconstruction from 1024 features
        recon3 = self.decode_level(z3, 2)  # Reconstruction from 2048 features

        return (recon1, recon2, recon3), (z1, z2, z3)

    def loss(
        self,
        x: torch.Tensor,
        reconstructions: Tuple[torch.Tensor, ...],
        features: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """Compute Matryoshka SAE loss with weighted reconstruction per level.

        Args:
            x: Original input (batch_size, input_dim)
            reconstructions: Tuple of (recon1, recon2, recon3)
            features: Tuple of (z1, z2, z3)

        Returns:
            Dictionary with loss components
        """
        recon1, recon2, recon3 = reconstructions
        z1, z2, z3 = features

        # Reconstruction loss per level
        recon_loss_1 = F.mse_loss(recon1, x)
        recon_loss_2 = F.mse_loss(recon2, x)
        recon_loss_3 = F.mse_loss(recon3, x)

        # Weighted reconstruction loss (prioritize fine level)
        weighted_recon_loss = (
            self.level_weights[0] * recon_loss_1 +
            self.level_weights[1] * recon_loss_2 +
            self.level_weights[2] * recon_loss_3
        )

        # Sparsity loss (L1 penalty on all features)
        l1_loss_1 = z1.abs().sum(dim=-1).mean()
        l1_loss_2 = z2.abs().sum(dim=-1).mean()
        l1_loss_3 = z3.abs().sum(dim=-1).mean()
        total_l1_loss = l1_loss_1 + l1_loss_2 + l1_loss_3

        # Total loss
        total_loss = weighted_recon_loss + self.l1_coefficient * total_l1_loss

        return {
            'total_loss': total_loss,
            'weighted_recon_loss': weighted_recon_loss,
            'recon_loss_1': recon_loss_1,
            'recon_loss_2': recon_loss_2,
            'recon_loss_3': recon_loss_3,
            'l1_loss_1': l1_loss_1,
            'l1_loss_2': l1_loss_2,
            'l1_loss_3': l1_loss_3,
            'total_l1_loss': total_l1_loss,
            'l1_coefficient': self.l1_coefficient
        }

    @torch.no_grad()
    def compute_metrics(
        self,
        x: torch.Tensor,
        reconstructions: Tuple[torch.Tensor, ...],
        features: Tuple[torch.Tensor, ...]
    ) -> Dict[str, Dict[str, float]]:
        """Compute quality metrics for each level.

        Args:
            x: Original input
            reconstructions: Tuple of reconstructions
            features: Tuple of feature activations

        Returns:
            Dictionary mapping level → metrics
        """
        recon1, recon2, recon3 = reconstructions
        z1, z2, z3 = features

        metrics = {}

        for level_idx, (recon, z, level_dim) in enumerate([
            (recon1, z1, self.levels[0]),
            (recon2, z2, self.levels[1]),
            (recon3, z3, self.levels[2])
        ]):
            # Explained variance
            total_var = x.var(dim=0).sum().item()
            residual_var = (x - recon).var(dim=0).sum().item()
            ev = 1.0 - (residual_var / total_var)

            # L0 norm (number of active features)
            l0_norm = (z > 0).float().sum(dim=-1).mean().item()

            # Feature death rate
            active_features = (z > 0).any(dim=0).float().sum().item()
            dead_features = level_dim - active_features
            death_rate = dead_features / level_dim

            # Sparsity
            mean_activation = z.mean().item()
            max_activation = z.max().item()

            metrics[f'level_{level_idx + 1}'] = {
                'features': level_dim,
                'explained_variance': ev,
                'l0_norm': l0_norm,
                'active_features': int(active_features),
                'dead_features': int(dead_features),
                'feature_death_rate': death_rate,
                'mean_activation': mean_activation,
                'max_activation': max_activation
            }

        return metrics

    def num_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_features_at_level(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Extract features at specific level for downstream tasks.

        Args:
            x: Input tensor (batch_size, input_dim)
            level: Level index (0, 1, or 2)

        Returns:
            Features at that level
        """
        with torch.no_grad():
            return self.encode_level(x, level)


def create_matryoshka_sae(
    input_dim: int = 2048,
    levels: List[int] = [512, 1024, 2048],
    l1_coefficient: float = 0.0005,
    level_weights: List[float] = [0.3, 0.3, 0.4],
    device: str = 'cuda'
) -> MatryoshkaSAE:
    """Factory function to create Matryoshka SAE.

    Args:
        input_dim: Input dimension (2048 for LLaMA)
        levels: Feature dimensions [512, 1024, 2048]
        l1_coefficient: Sparsity penalty
        level_weights: Reconstruction loss weights
        device: Device to place model on

    Returns:
        Initialized MatryoshkaSAE model
    """
    model = MatryoshkaSAE(
        input_dim=input_dim,
        levels=levels,
        l1_coefficient=l1_coefficient,
        level_weights=level_weights
    ).to(device)

    num_params = model.num_parameters()
    print(f"Created Matryoshka SAE")
    print(f"  Levels: {levels}")
    print(f"  Level weights: {level_weights}")
    print(f"  L1 coefficient: {l1_coefficient}")
    print(f"  Total parameters: {num_params:,} ({num_params * 4 / 1e6:.1f} MB fp32)")

    return model


if __name__ == "__main__":
    # Test the architecture
    print("Testing Matryoshka SAE Architecture\n")

    # Create model
    model = create_matryoshka_sae(device='cpu')

    # Test forward pass
    batch_size = 16
    x = torch.randn(batch_size, 2048)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    reconstructions, features = model(x)

    print("\nReconstructions:")
    for i, recon in enumerate(reconstructions):
        print(f"  Level {i+1}: {recon.shape}")

    print("\nFeatures:")
    for i, feat in enumerate(features):
        print(f"  Level {i+1}: {feat.shape}")

    # Test loss computation
    loss_dict = model.loss(x, reconstructions, features)
    print("\nLoss components:")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.item():.6f}")
        else:
            print(f"  {key}: {value:.6f}")

    # Test metrics
    metrics = model.compute_metrics(x, reconstructions, features)
    print("\nMetrics per level:")
    for level, level_metrics in metrics.items():
        print(f"\n{level}:")
        for metric, value in level_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    print("\n✓ Architecture test passed!")
