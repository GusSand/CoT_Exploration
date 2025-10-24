#!/usr/bin/env python3
"""
Test top-k projection method for CODI discretization
Projects continuous vector onto subspace spanned by top-k vocabulary embeddings
"""
import torch
import numpy as np

def project_onto_topk_vocab(continuous_vector, vocab_embeddings_topk, normalize=False):
    """
    Project continuous vector onto subspace spanned by top-k vocab embeddings.

    Args:
        continuous_vector: Tensor of shape [hidden] - the continuous activation
        vocab_embeddings_topk: Tensor of shape [k, hidden] - top k vocab token embeddings
        normalize: If True, scale result to match continuous norm

    Returns:
        Projected vector of shape [hidden]
    """
    # vocab_embeddings_topk is a matrix V with k rows (each row is a vocab embedding)
    # We want to find coefficients α such that result = Vᵀα minimizes ||c - Vᵀα||²

    # This is a least-squares problem: α = (V Vᵀ)⁻¹ V c
    # Where V is [k, hidden] matrix

    V = vocab_embeddings_topk  # [k, hidden]
    c = continuous_vector  # [hidden]

    # Compute Gram matrix G = V Vᵀ (k x k)
    G = torch.mm(V, V.t())  # [k, k]

    # Compute V c (k-dimensional vector)
    Vc = torch.mv(V, c)  # [k]

    # Solve G α = Vc for α
    # Use torch.linalg.solve for numerical stability
    try:
        alpha = torch.linalg.solve(G, Vc)  # [k]
    except:
        # If singular, use pseudo-inverse
        alpha = torch.linalg.lstsq(G, Vc).solution  # [k]

    # Compute projection: result = Vᵀα = Σ αᵢ vᵢ
    projected = torch.mv(V.t(), alpha)  # [hidden]

    if normalize:
        # Scale to match original norm
        continuous_norm = torch.norm(continuous_vector)
        projected_norm = torch.norm(projected)
        projected = projected * (continuous_norm / (projected_norm + 1e-8))

    return projected, alpha

def test_topk_projection():
    """
    Test on simple examples
    """
    print("="*80)
    print("TESTING TOP-K PROJECTION")
    print("="*80)

    # Test 1: k=1 should match single-token projection
    print("\n" + "="*80)
    print("TEST 1: k=1 (should match single-token projection)")
    print("="*80)

    continuous = torch.tensor([10.0, 20.0, 30.0])
    vocab_1 = torch.tensor([1.0, 0.5, 0.2])

    # Method 1: Our top-k with k=1
    projected_k1, alpha = project_onto_topk_vocab(continuous, vocab_1.unsqueeze(0), normalize=True)

    # Method 2: Direct single-token projection
    continuous_norm = torch.norm(continuous)
    vocab_norm = torch.norm(vocab_1)
    direct = vocab_1 * (continuous_norm / vocab_norm)

    print(f"Continuous vector: {continuous.numpy()}")
    print(f"Vocab embedding: {vocab_1.numpy()}")
    print(f"Top-k (k=1) result: {projected_k1.numpy()}")
    print(f"Direct result: {direct.numpy()}")
    print(f"Difference: {torch.norm(projected_k1 - direct).item():.6e}")
    print(f"Alpha coefficient: {alpha.item():.4f}")

    # Test 2: k=3 projection
    print("\n" + "="*80)
    print("TEST 2: k=3 (project onto 3-dimensional subspace)")
    print("="*80)

    continuous = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    vocab_top3 = torch.tensor([
        [1.0, 0.5, 0.2, 0.1, 0.0],  # token 1
        [0.2, 1.0, 0.3, 0.2, 0.1],  # token 2
        [0.1, 0.2, 1.0, 0.4, 0.2],  # token 3
    ])

    # Unnormalized projection
    projected_unnorm, alpha_unnorm = project_onto_topk_vocab(continuous, vocab_top3, normalize=False)

    # Normalized projection
    projected_norm, alpha_norm = project_onto_topk_vocab(continuous, vocab_top3, normalize=True)

    print(f"Continuous vector: {continuous.numpy()}")
    print(f"Continuous norm: {torch.norm(continuous).item():.4f}")
    print(f"\nTop-3 vocab embeddings:")
    for i, v in enumerate(vocab_top3):
        print(f"  Token {i+1}: {v.numpy()} (norm: {torch.norm(v).item():.4f})")

    print(f"\nUNNORMALIZED projection:")
    print(f"  Result: {projected_unnorm.numpy()}")
    print(f"  Result norm: {torch.norm(projected_unnorm).item():.4f}")
    print(f"  Alpha coefficients: {alpha_unnorm.numpy()}")
    print(f"  Verification: Sum alpha_i*v_i = {torch.sum(alpha_unnorm.unsqueeze(1) * vocab_top3, dim=0).numpy()}")

    print(f"\nNORMALIZED projection:")
    print(f"  Result: {projected_norm.numpy()}")
    print(f"  Result norm: {torch.norm(projected_norm).item():.4f}")
    print(f"  Alpha coefficients: {alpha_norm.numpy()}")

    # Check orthogonality of residual
    residual = continuous - projected_unnorm
    print(f"\nResidual (c - projection): {residual.numpy()}")
    print(f"Residual norm: {torch.norm(residual).item():.4f}")
    print(f"Residual orthogonal to subspace?")
    for i, v in enumerate(vocab_top3):
        dot = torch.dot(residual, v).item()
        print(f"  residual · v{i+1} = {dot:.6e}")

    # Test 3: k=5, continuous vector closely aligned with one vocab vector
    print("\n" + "="*80)
    print("TEST 3: Continuous nearly aligned with one vocab vector")
    print("="*80)

    vocab_base = torch.randn(5, 10)  # 5 random vocab embeddings
    continuous = vocab_base[2] * 10 + torch.randn(10) * 0.1  # Mostly aligned with vocab 3

    projected, alpha = project_onto_topk_vocab(continuous, vocab_base, normalize=True)

    print(f"Continuous norm: {torch.norm(continuous).item():.4f}")
    print(f"Projected norm: {torch.norm(projected).item():.4f}")
    print(f"Alpha coefficients: {alpha.numpy()}")
    print(f"Dominant coefficient (should be alpha_3): alpha[2] = {alpha[2].item():.4f}")
    print(f"Projection error: {torch.norm(continuous - projected).item():.4f}")

def test_comparison_k1_vs_k5():
    """
    Compare k=1 vs k=5 projection on same vector
    """
    print("\n" + "="*80)
    print("COMPARISON: k=1 vs k=5")
    print("="*80)

    torch.manual_seed(42)
    continuous = torch.randn(10) * 20  # Random continuous vector
    vocab_embeddings = torch.randn(5, 10)  # 5 vocab embeddings

    # k=1 projection (onto top token only)
    proj_k1, alpha_k1 = project_onto_topk_vocab(continuous, vocab_embeddings[0:1], normalize=True)

    # k=5 projection (onto all 5 tokens)
    proj_k5, alpha_k5 = project_onto_topk_vocab(continuous, vocab_embeddings, normalize=True)

    print(f"Continuous: {continuous.numpy()}")
    print(f"Continuous norm: {torch.norm(continuous).item():.4f}")

    print(f"\nk=1 projection:")
    print(f"  Result: {proj_k1.numpy()}")
    print(f"  Result norm: {torch.norm(proj_k1).item():.4f}")
    print(f"  Alpha: {alpha_k1.numpy()}")
    print(f"  Error: {torch.norm(continuous - proj_k1).item():.4f}")

    print(f"\nk=5 projection:")
    print(f"  Result: {proj_k5.numpy()}")
    print(f"  Result norm: {torch.norm(proj_k5).item():.4f}")
    print(f"  Alpha: {alpha_k5.numpy()}")
    print(f"  Error (unnormalized): {torch.norm(continuous - proj_k5).item():.4f}")

    print(f"\nKey insight:")
    print(f"  k=5 has MORE degrees of freedom, so can better approximate continuous vector")
    print(f"  But after normalization, both preserve original norm")

if __name__ == "__main__":
    test_topk_projection()
    test_comparison_k1_vs_k5()
