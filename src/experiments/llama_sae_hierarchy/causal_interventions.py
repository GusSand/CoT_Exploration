"""
Causal Intervention Engine for SAE Features.

Supports three types of interventions:
1. Ablate: Zero out specific features
2. Swap: Exchange activations between two features
3. Amplify: Scale specific feature activations

Usage:
    from causal_interventions import FeatureInterventionEngine

    engine = FeatureInterventionEngine(sae_model)
    modified_acts = engine.ablate_feature(activations, feature_idx=449)
"""

import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Add topk_grid_pilot to path
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder


class FeatureInterventionEngine:
    """
    Manages causal interventions on SAE features.

    This engine allows modifying feature activations to test causal hypotheses:
    - Ablation: Test if feature is necessary
    - Swap: Test if features encode specific content
    - Amplification: Test if feature is sufficient
    """

    def __init__(self, sae_model: TopKAutoencoder):
        """
        Initialize intervention engine.

        Args:
            sae_model: Trained TopK SAE model
        """
        self.sae = sae_model
        self.sae.eval()  # Always eval mode for interventions

    def encode(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode activations to sparse feature space.

        Args:
            activations: Input activations (batch_size, input_dim)

        Returns:
            reconstruction: Reconstructed activations
            sparse: Sparse feature activations (batch_size, latent_dim)
        """
        with torch.no_grad():
            reconstruction, sparse, _ = self.sae(activations)
        return reconstruction, sparse

    def decode(self, sparse: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activation space.

        Args:
            sparse: Sparse feature activations (batch_size, latent_dim)

        Returns:
            reconstruction: Reconstructed activations (batch_size, input_dim)
        """
        with torch.no_grad():
            reconstruction = self.sae.decoder(sparse)
        return reconstruction

    def ablate_feature(
        self,
        activations: torch.Tensor,
        feature_idx: int
    ) -> torch.Tensor:
        """
        Ablate (zero out) a specific feature.

        Args:
            activations: Input activations (batch_size, input_dim)
            feature_idx: Index of feature to ablate (0 to latent_dim-1)

        Returns:
            modified_activations: Activations with feature ablated
        """
        _, sparse = self.encode(activations)

        # Zero out target feature
        sparse_ablated = sparse.clone()
        sparse_ablated[:, feature_idx] = 0

        # Decode back
        modified_activations = self.decode(sparse_ablated)

        return modified_activations

    def ablate_features(
        self,
        activations: torch.Tensor,
        feature_indices: List[int]
    ) -> torch.Tensor:
        """
        Ablate multiple features simultaneously.

        Args:
            activations: Input activations (batch_size, input_dim)
            feature_indices: List of feature indices to ablate

        Returns:
            modified_activations: Activations with features ablated
        """
        _, sparse = self.encode(activations)

        # Zero out all target features
        sparse_ablated = sparse.clone()
        for idx in feature_indices:
            sparse_ablated[:, idx] = 0

        modified_activations = self.decode(sparse_ablated)

        return modified_activations

    def swap_features(
        self,
        activations: torch.Tensor,
        feature_a: int,
        feature_b: int
    ) -> torch.Tensor:
        """
        Swap activations between two features.

        Args:
            activations: Input activations (batch_size, input_dim)
            feature_a: First feature index
            feature_b: Second feature index

        Returns:
            modified_activations: Activations with features swapped
        """
        _, sparse = self.encode(activations)

        # Swap features
        sparse_swapped = sparse.clone()
        sparse_swapped[:, feature_a], sparse_swapped[:, feature_b] = \
            sparse[:, feature_b].clone(), sparse[:, feature_a].clone()

        modified_activations = self.decode(sparse_swapped)

        return modified_activations

    def amplify_feature(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        scale: float
    ) -> torch.Tensor:
        """
        Amplify (scale) a specific feature activation.

        Args:
            activations: Input activations (batch_size, input_dim)
            feature_idx: Index of feature to amplify
            scale: Scaling factor (e.g., 2.0 = double, 0.5 = halve)

        Returns:
            modified_activations: Activations with feature amplified
        """
        _, sparse = self.encode(activations)

        # Scale target feature
        sparse_amplified = sparse.clone()
        sparse_amplified[:, feature_idx] *= scale

        modified_activations = self.decode(sparse_amplified)

        return modified_activations

    def measure_feature_impact(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        metric: str = 'reconstruction_diff'
    ) -> Dict[str, float]:
        """
        Measure the impact of ablating a feature.

        Args:
            activations: Input activations (batch_size, input_dim)
            feature_idx: Feature to ablate
            metric: Metric to compute ('reconstruction_diff', 'l2_norm')

        Returns:
            impact_metrics: Dict with impact measurements
        """
        # Original reconstruction
        reconstruction_orig, sparse_orig = self.encode(activations)

        # Ablated reconstruction
        reconstruction_ablated = self.ablate_feature(activations, feature_idx)

        # Compute metrics
        metrics = {}

        if metric == 'reconstruction_diff' or metric == 'all':
            diff = (reconstruction_orig - reconstruction_ablated).abs()
            metrics['mean_abs_diff'] = float(diff.mean())
            metrics['max_abs_diff'] = float(diff.max())

        if metric == 'l2_norm' or metric == 'all':
            l2_diff = torch.norm(reconstruction_orig - reconstruction_ablated, dim=-1)
            metrics['mean_l2_diff'] = float(l2_diff.mean())
            metrics['max_l2_diff'] = float(l2_diff.max())

        # Feature statistics
        feature_active_mask = sparse_orig[:, feature_idx] != 0
        metrics['feature_activation_freq'] = float(feature_active_mask.float().mean())
        metrics['feature_mean_magnitude'] = float(sparse_orig[:, feature_idx].abs().mean())

        return metrics

    def identity_test(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Sanity check: Swapping feature with itself should not change output.

        Args:
            activations: Input activations
            feature_idx: Feature to test
            tolerance: Maximum allowed difference

        Returns:
            passed: True if test passed
        """
        # Original reconstruction
        reconstruction_orig, _ = self.encode(activations)

        # Swap feature with itself
        reconstruction_swapped = self.swap_features(activations, feature_idx, feature_idx)

        # Check if identical
        max_diff = (reconstruction_orig - reconstruction_swapped).abs().max()

        return float(max_diff) < tolerance

    def null_ablation_test(
        self,
        activations: torch.Tensor,
        feature_idx: int,
        tolerance: float = 1e-5
    ) -> bool:
        """
        Sanity check: Ablating inactive feature should not change output.

        Args:
            activations: Input activations
            feature_idx: Feature to test
            tolerance: Maximum allowed difference

        Returns:
            passed: True if test passed (feature was already inactive)
        """
        # Get sparse representation
        _, sparse = self.encode(activations)

        # Check if feature is active
        if (sparse[:, feature_idx] != 0).any():
            return False  # Test only valid for inactive features

        # Original reconstruction
        reconstruction_orig, _ = self.encode(activations)

        # Ablate inactive feature
        reconstruction_ablated = self.ablate_feature(activations, feature_idx)

        # Check if identical
        max_diff = (reconstruction_orig - reconstruction_ablated).abs().max()

        return float(max_diff) < tolerance

    def run_sanity_checks(
        self,
        activations: torch.Tensor,
        verbose: bool = True
    ) -> Dict[str, bool]:
        """
        Run all sanity checks on a batch of activations.

        Args:
            activations: Test activations (batch_size, input_dim)
            verbose: Print results

        Returns:
            results: Dict with test results
        """
        if verbose:
            print("\n" + "="*80)
            print("Running Sanity Checks")
            print("="*80 + "\n")

        results = {}

        # Test 1: Identity swap
        if verbose:
            print("Test 1: Identity Swap (swap feature with itself)")
        test_feature = 0
        identity_passed = self.identity_test(activations, test_feature)
        results['identity_swap'] = identity_passed
        if verbose:
            status = "✓ PASSED" if identity_passed else "✗ FAILED"
            print(f"  {status}: Feature {test_feature} swapped with itself\n")

        # Test 2: Null ablation
        if verbose:
            print("Test 2: Null Ablation (ablate inactive feature)")

        # Find an inactive feature
        _, sparse = self.encode(activations)
        inactive_features = torch.where((sparse != 0).sum(dim=0) == 0)[0]

        if len(inactive_features) > 0:
            test_feature = int(inactive_features[0])
            null_ablation_passed = self.null_ablation_test(activations, test_feature)
            results['null_ablation'] = null_ablation_passed
            if verbose:
                status = "✓ PASSED" if null_ablation_passed else "✗ FAILED"
                print(f"  {status}: Feature {test_feature} (inactive) ablated\n")
        else:
            results['null_ablation'] = None
            if verbose:
                print(f"  ⊘ SKIPPED: No inactive features found (all K={self.sae.k} features active)\n")

        # Test 3: Reconstruction fidelity
        if verbose:
            print("Test 3: Reconstruction Fidelity")
        reconstruction, _ = self.encode(activations)
        mse = torch.mean((activations - reconstruction) ** 2)
        reconstruction_good = mse < 10.0  # Reasonable threshold
        results['reconstruction_fidelity'] = reconstruction_good
        if verbose:
            status = "✓ PASSED" if reconstruction_good else "✗ FAILED"
            print(f"  {status}: MSE = {float(mse):.4f} (threshold: 10.0)\n")

        # Summary
        if verbose:
            print("="*80)
            passed = sum(1 for v in results.values() if v is True)
            total = sum(1 for v in results.values() if v is not None)
            print(f"Summary: {passed}/{total} tests passed")
            print("="*80 + "\n")

        return results


def demo():
    """Demonstration of intervention engine."""
    print("\n" + "="*80)
    print("Feature Intervention Engine - Demo")
    print("="*80 + "\n")

    # Create dummy SAE for demo
    print("Initializing TopK SAE...")
    sae = TopKAutoencoder(input_dim=2048, latent_dim=512, k=100)
    print(f"  SAE: {sae.input_dim} → {sae.latent_dim} (K={sae.k})\n")

    # Create engine
    engine = FeatureInterventionEngine(sae)

    # Generate random test data
    print("Generating test data...")
    batch_size = 10
    test_activations = torch.randn(batch_size, 2048)
    print(f"  Test activations: {test_activations.shape}\n")

    # Run sanity checks
    engine.run_sanity_checks(test_activations, verbose=True)

    # Demo interventions
    print("="*80)
    print("Demo: Feature Interventions")
    print("="*80 + "\n")

    print("1. Ablation:")
    ablated = engine.ablate_feature(test_activations, feature_idx=10)
    print(f"   Original shape: {test_activations.shape}")
    print(f"   Ablated shape: {ablated.shape}")
    diff = (test_activations - ablated).abs().mean()
    print(f"   Mean difference: {diff:.6f}\n")

    print("2. Swap:")
    swapped = engine.swap_features(test_activations, feature_a=10, feature_b=20)
    print(f"   Swapped features 10 ↔ 20")
    diff = (test_activations - swapped).abs().mean()
    print(f"   Mean difference: {diff:.6f}\n")

    print("3. Amplification:")
    amplified = engine.amplify_feature(test_activations, feature_idx=10, scale=2.0)
    print(f"   Amplified feature 10 by 2.0x")
    diff = (test_activations - amplified).abs().mean()
    print(f"   Mean difference: {diff:.6f}\n")

    print("4. Impact Measurement:")
    impact = engine.measure_feature_impact(test_activations, feature_idx=10, metric='all')
    print(f"   Feature 10 impact metrics:")
    for k, v in impact.items():
        print(f"     {k}: {v:.6f}")

    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    demo()
