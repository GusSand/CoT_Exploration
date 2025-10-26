"""
Sparse Autoencoder (SAE) Model for Continuous Thought Decoding.

Architecture:
    Input (2048,) → Encoder → Features (2048,) → Decoder → Output (2048,)

Loss:
    L = MSE(output, input) + λ * L1(features)

Based on proven architecture from refined SAE (4.2% feature death, 89% explained variance).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for continuous thought vectors.

    Args:
        input_dim: Dimension of input vectors (2048 for LLaMA)
        n_features: Number of sparse features (2048 proven optimal)
        l1_coefficient: Sparsity penalty (0.0005 proven optimal)
    """

    def __init__(
        self,
        input_dim: int = 2048,
        n_features: int = 2048,
        l1_coefficient: float = 0.0005
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features
        self.l1_coefficient = l1_coefficient

        # Encoder: input_dim → n_features
        self.encoder = nn.Linear(input_dim, n_features, bias=True)

        # Decoder: n_features → input_dim
        self.decoder = nn.Linear(n_features, input_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse features.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Sparse features (batch_size, n_features)
        """
        features = F.relu(self.encoder(x))
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to reconstruction.

        Args:
            features: Sparse features (batch_size, n_features)

        Returns:
            Reconstruction (batch_size, input_dim)
        """
        reconstruction = self.decoder(features)
        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SAE.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            Tuple of (reconstruction, features)
        """
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute SAE loss with reconstruction and sparsity terms.

        Args:
            x: Original input (batch_size, input_dim)
            reconstruction: Reconstructed output (batch_size, input_dim)
            features: Sparse features (batch_size, n_features)

        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # Sparsity loss (L1 penalty on features)
        l1_loss = features.abs().sum(dim=-1).mean()

        # Total loss
        total_loss = recon_loss + self.l1_coefficient * l1_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'l1_loss': l1_loss,
            'l1_coefficient': self.l1_coefficient
        }

    @torch.no_grad()
    def get_feature_statistics(self, features: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about feature activations.

        Args:
            features: Feature activations (batch_size, n_features)

        Returns:
            Dictionary with statistics
        """
        # L0 norm (number of active features per sample)
        l0_norm = (features > 0).float().sum(dim=-1).mean().item()

        # Feature death rate (% features that never activate)
        active_features = (features > 0).any(dim=0).float().sum().item()
        dead_features = self.n_features - active_features
        death_rate = dead_features / self.n_features

        # Sparsity metrics
        mean_activation = features.mean().item()
        max_activation = features.max().item()

        return {
            'l0_norm_mean': l0_norm,
            'feature_death_rate': death_rate,
            'active_features': int(active_features),
            'dead_features': int(dead_features),
            'mean_activation': mean_activation,
            'max_activation': max_activation
        }

    @torch.no_grad()
    def compute_explained_variance(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> float:
        """Compute explained variance (R²-like metric).

        Args:
            x: Original input
            reconstruction: Reconstructed output

        Returns:
            Explained variance (0-1, higher is better)
        """
        # Total variance in original data
        total_variance = x.var(dim=0).sum().item()

        # Residual variance (reconstruction error)
        residual_variance = (x - reconstruction).var(dim=0).sum().item()

        # Explained variance
        explained_variance = 1.0 - (residual_variance / total_variance)

        return explained_variance

    @torch.no_grad()
    def compute_cosine_similarity(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor
    ) -> float:
        """Compute average cosine similarity between input and reconstruction.

        Args:
            x: Original input
            reconstruction: Reconstructed output

        Returns:
            Mean cosine similarity
        """
        cosine_sim = F.cosine_similarity(x, reconstruction, dim=-1).mean().item()
        return cosine_sim


def create_position_specific_saes(
    n_positions: int = 6,
    input_dim: int = 2048,
    n_features: int = 2048,
    l1_coefficient: float = 0.0005,
    device: str = 'cuda'
) -> Dict[int, SparseAutoencoder]:
    """Create 6 position-specific SAEs.

    Args:
        n_positions: Number of continuous thought positions (6)
        input_dim: Input dimension (2048 for LLaMA)
        n_features: Number of features per SAE (2048)
        l1_coefficient: Sparsity penalty (0.0005)
        device: Device to place models on

    Returns:
        Dictionary mapping position → SAE model
    """
    saes = {}
    for position in range(n_positions):
        sae = SparseAutoencoder(
            input_dim=input_dim,
            n_features=n_features,
            l1_coefficient=l1_coefficient
        ).to(device)
        saes[position] = sae

    total_params = sum(sae.num_parameters() for sae in saes.values())
    print(f"Created {n_positions} SAEs")
    print(f"Total parameters: {total_params:,} ({total_params * 4 / 1e6:.1f} MB fp32)")

    return saes


# Add utility method to SparseAutoencoder
def num_parameters(self) -> int:
    """Count number of trainable parameters."""
    return sum(p.numel() for p in self.parameters() if p.requires_grad)


SparseAutoencoder.num_parameters = num_parameters
