#!/usr/bin/env python3
"""
Quick test to verify measurement capture is working correctly.
"""
import json
import torch
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'scripts' / 'experiments'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG
from run_ablation_N_tokens_llama import NTokenPatcher

# Import functions from FULL script
import sys
import importlib.util
spec = importlib.util.spec_from_file_location(
    "full_ccta",
    Path(__file__).parent / "1_run_token_ablation_FULL.py"
)
full_ccta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(full_ccta)

compute_kl_divergence = full_ccta.compute_kl_divergence
compute_attention_disruption = full_ccta.compute_attention_disruption
run_with_measurements = full_ccta.run_with_measurements


def test_single_problem():
    print("="*80)
    print("TESTING FIXED MEASUREMENT CAPTURE")
    print("="*80)

    # Load model
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
    print(f"\nLoading model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)
    model = cacher.model
    device = cacher.device

    # Load test problem
    dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    problem = dataset[0]
    print(f"\nTest problem: {problem['gsm8k_id']}")
    print(f"Question: {problem['question'][:100]}...")

    # Test baseline measurement
    print("\n" + "="*80)
    print("Testing BASELINE measurement capture...")
    print("="*80)

    patcher = NTokenPatcher(cacher, num_tokens=6)
    test_layer = 'middle'

    output, logits, attention = run_with_measurements(
        patcher, problem['question'], test_layer, model, device, patch_activations=None
    )

    print(f"✓ Output: {output[:100]}...")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"✓ Attention shape: {attention.shape}")
    print(f"✓ Attention range: [{attention.min():.4f}, {attention.max():.4f}]")

    # Test corrupted measurement
    print("\n" + "="*80)
    print("Testing CORRUPTED measurement capture...")
    print("="*80)

    # Cache activations
    baseline_acts = patcher.cache_N_token_activations(problem['question'], test_layer)

    # Zero out token 0
    corrupted_acts = [act.clone() for act in baseline_acts]
    corrupted_acts[0] = torch.zeros_like(baseline_acts[0])

    output_corr, logits_corr, attention_corr = run_with_measurements(
        patcher, problem['question'], test_layer, model, device, patch_activations=corrupted_acts
    )

    print(f"✓ Output: {output_corr[:100]}...")
    print(f"✓ Logits shape: {logits_corr.shape}")
    print(f"✓ Logits range: [{logits_corr.min():.2f}, {logits_corr.max():.2f}]")
    print(f"✓ Attention shape: {attention_corr.shape}")
    print(f"✓ Attention range: [{attention_corr.min():.4f}, {attention_corr.max():.4f}]")

    # Compute measurements
    print("\n" + "="*80)
    print("Testing MEASUREMENT FUNCTIONS...")
    print("="*80)

    kl_div = compute_kl_divergence(logits, logits_corr)
    attn_disruption = compute_attention_disruption(attention, attention_corr)

    print(f"✓ KL Divergence: {kl_div:.6f}")
    print(f"✓ Attention Disruption: {attn_disruption:.6f}")

    if kl_div > 0 or attn_disruption > 0:
        print("\n✅ SUCCESS! Measurements are now non-zero!")
        print("="*80)
        return True
    else:
        print("\n❌ FAILED: Measurements are still zero")
        print("="*80)
        return False


if __name__ == "__main__":
    success = test_single_problem()
    sys.exit(0 if success else 1)
