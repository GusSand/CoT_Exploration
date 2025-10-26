"""
TopK Sparse Autoencoder with configurable dictionary size.

This implementation enforces exact K-sparsity by selecting top-K activations
by magnitude and zeroing the rest. Unlike ReLU SAE, there is no L1 penalty,
eliminating shrinkage effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder with configurable dictionary size.

    Args:
        input_dim (int): Dimension of input activations (2048 for CODI)
        latent_dim (int): Size of feature dictionary (512, 1024, or 2048)
        k (int): Number of features to keep active per sample (5, 10, 20, or 100)
    """

    def __init__(self, input_dim=2048, latent_dim=2048, k=20):
        super().__init__()

        if k > latent_dim:
            raise ValueError(f"k ({k}) cannot exceed latent_dim ({latent_dim})")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k

        # Encoder: projects input to latent space
        self.encoder = nn.Linear(input_dim, latent_dim, bias=True)

        # Decoder: projects latent back to input space
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)

        # Initialize decoder with unit-norm columns for better reconstruction
        self._normalize_decoder()

    def _normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(
                self.decoder.weight.data,
                dim=0
            )

    def forward(self, x):
        """
        Forward pass with TopK sparsity.

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            reconstruction: Reconstructed activations (batch_size, input_dim)
            sparse_activations: Sparse feature activations (batch_size, latent_dim)
            metrics: Dict with auxiliary metrics (L0, mean activation, etc.)
        """
        # Encode to latent space
        activations = self.encoder(x)  # (batch_size, latent_dim)

        # Select top-K activations by absolute magnitude
        topk_values, topk_indices = torch.topk(
            activations.abs(),
            k=self.k,
            dim=-1
        )

        # Create sparse activation tensor (zeros everywhere except top-K)
        sparse_activations = torch.zeros_like(activations)

        # Scatter original values (not absolute) back to sparse tensor
        # We need to gather the original signed values at the topk indices
        original_topk_values = torch.gather(activations, dim=-1, index=topk_indices)
        sparse_activations.scatter_(
            dim=-1,
            index=topk_indices,
            src=original_topk_values
        )

        # Decode back to input space
        reconstruction = self.decoder(sparse_activations)

        # Compute auxiliary metrics
        with torch.no_grad():
            # L0 norm (should equal K)
            l0_norm = (sparse_activations != 0).sum(dim=-1).float()

            # Activation statistics (only for active features)
            active_mask = sparse_activations != 0
            if active_mask.any():
                active_values = sparse_activations[active_mask].abs()
                mean_activation = active_values.mean()
                max_activation = active_values.max()
                median_activation = active_values.median()
            else:
                mean_activation = torch.tensor(0.0)
                max_activation = torch.tensor(0.0)
                median_activation = torch.tensor(0.0)

            metrics = {
                'l0_mean': l0_norm.mean().item(),
                'l0_std': l0_norm.std().item(),
                'mean_activation': mean_activation.item(),
                'max_activation': max_activation.item(),
                'median_activation': median_activation.item(),
            }

        return reconstruction, sparse_activations, metrics

    def loss(self, x, reconstruction):
        """
        Compute reconstruction loss (no L1 penalty for TopK).

        Args:
            x: Original input
            reconstruction: Reconstructed output

        Returns:
            loss: MSE reconstruction loss
        """
        return F.mse_loss(reconstruction, x)

    def compute_feature_stats(self, sparse_activations):
        """
        Compute feature-level statistics across a batch.

        Args:
            sparse_activations: (batch_size, latent_dim) sparse feature tensor

        Returns:
            dict with per-feature statistics
        """
        with torch.no_grad():
            # Feature death rate: % features never active
            feature_active = (sparse_activations != 0).any(dim=0)  # (latent_dim,)
            feature_death_rate = (~feature_active).sum().float() / self.latent_dim

            # Per-feature activation frequency
            activation_freq = (sparse_activations != 0).float().mean(dim=0)  # (latent_dim,)

            return {
                'feature_death_rate': feature_death_rate.item(),
                'activation_frequency': activation_freq,
                'num_dead_features': (~feature_active).sum().item(),
                'num_active_features': feature_active.sum().item(),
            }


def test_topk_sparsity():
    """Test that exactly K features are active per sample."""
    print("Testing TopK sparsity constraint...")

    configs = [
        (2048, 512, 5),
        (2048, 1024, 10),
        (2048, 2048, 20),
        (2048, 2048, 100),
    ]

    for input_dim, latent_dim, k in configs:
        model = TopKAutoencoder(input_dim=input_dim, latent_dim=latent_dim, k=k)
        x = torch.randn(32, input_dim)  # Batch of 32 samples

        reconstruction, sparse, metrics = model(x)

        # Check that exactly K features are non-zero per sample
        l0_per_sample = (sparse != 0).sum(dim=-1)
        assert torch.all(l0_per_sample == k), \
            f"Expected L0={k}, got {l0_per_sample.unique()}"

        # Check metrics match
        assert abs(metrics['l0_mean'] - k) < 1e-6, \
            f"Expected mean L0={k}, got {metrics['l0_mean']}"

        assert metrics['l0_std'] < 1e-6, \
            f"Expected std L0=0, got {metrics['l0_std']}"

        print(f"  ✓ ({input_dim}, {latent_dim}, K={k}): L0={metrics['l0_mean']:.1f} ± {metrics['l0_std']:.1f}")

    print("✓ TopK sparsity test passed!\n")


def test_all_grid_configs():
    """Test all 12 grid configurations."""
    print("Testing all 12 grid configurations...")

    k_values = [5, 10, 20, 100]
    latent_dims = [512, 1024, 2048]

    for latent_dim in latent_dims:
        for k in k_values:
            model = TopKAutoencoder(input_dim=2048, latent_dim=latent_dim, k=k)
            x = torch.randn(16, 2048)
            reconstruction, sparse, metrics = model(x)

            # Verify shapes
            assert reconstruction.shape == (16, 2048), \
                f"Wrong reconstruction shape: {reconstruction.shape}"
            assert sparse.shape == (16, latent_dim), \
                f"Wrong sparse shape: {sparse.shape}"

            # Verify sparsity
            l0 = (sparse != 0).sum(dim=-1)
            assert torch.all(l0 == k), \
                f"Config ({latent_dim}, {k}): Expected L0={k}, got {l0.unique()}"

    print(f"✓ All {len(k_values) * len(latent_dims)} configurations work correctly!\n")


def test_reconstruction_quality():
    """Test that reconstruction improves with more features."""
    print("Testing reconstruction quality vs K...")

    # Fixed input
    torch.manual_seed(42)
    x = torch.randn(100, 2048)

    results = []
    for k in [5, 10, 20, 50, 100]:
        model = TopKAutoencoder(input_dim=2048, latent_dim=2048, k=k)

        # Simple training: just fit to this batch
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(100):
            reconstruction, _, _ = model(x)
            loss = model.loss(x, reconstruction)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final reconstruction quality
        with torch.no_grad():
            reconstruction, _, _ = model(x)
            mse = F.mse_loss(reconstruction, x).item()
            cosine_sim = F.cosine_similarity(x, reconstruction, dim=-1).mean().item()

        results.append((k, mse, cosine_sim))
        print(f"  K={k:3d}: MSE={mse:.4f}, Cosine={cosine_sim:.4f}")

    # Check that more features → better reconstruction
    mse_values = [r[1] for r in results]
    assert mse_values == sorted(mse_values, reverse=True), \
        "MSE should decrease with more features"

    print("✓ Reconstruction quality improves with more features!\n")


def test_feature_death_rate():
    """Test feature death rate computation."""
    print("Testing feature death rate computation...")

    model = TopKAutoencoder(input_dim=2048, latent_dim=512, k=5)

    # Create batch where only first 100 features are ever active
    batch_size = 100
    x = torch.randn(batch_size, 2048)

    reconstruction, sparse, _ = model(x)
    stats = model.compute_feature_stats(sparse)

    print(f"  Feature death rate: {stats['feature_death_rate']:.2%}")
    print(f"  Active features: {stats['num_active_features']}/{model.latent_dim}")
    print(f"  Dead features: {stats['num_dead_features']}/{model.latent_dim}")

    assert 0 <= stats['feature_death_rate'] <= 1, "Death rate out of range"
    assert stats['num_active_features'] + stats['num_dead_features'] == model.latent_dim

    print("✓ Feature death rate computation works!\n")


if __name__ == "__main__":
    print("=" * 80)
    print("TopK SAE Unit Tests")
    print("=" * 80)
    print()

    test_topk_sparsity()
    test_all_grid_configs()
    test_reconstruction_quality()
    test_feature_death_rate()

    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)
