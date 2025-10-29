# Single Layer Probe Experiment (L10→L11)

**Date**: October 28, 2024  
**Status**: Successful baseline

## Overview

This experiment trained a single linear probe to transform hidden states from Layer 10 to Layer 11 of the CODI-GPT2 model during continuous thought iterations.

## Objective

Establish a baseline for tuned lens probing by training on a single layer pair before scaling to multiple layers.

## Training Strategy: Hybrid Balancing

The probe was trained using a hybrid balancing approach:

1. **Scan Phase**: Process 20,000 examples from GSM8k-Aug dataset
2. **Balance Phase**: Create balanced dataset of 70,000 samples by downsampling high-probability matches and upsampling low-probability matches to ensure diverse token representation

This addresses class imbalance where Layer 10 may already predict Layer 11 output with high confidence for common tokens.

## Results

**Hard Accuracy**: 32.67% (exact token match)

This represents meaningful progress - the probe successfully learns to bridge the Layer 10→11 transformation gap beyond random chance.

## Logs

- train_hybrid_70K.log - Full training output (100 epochs)
- test_hybrid_100ex.log - Hard accuracy evaluation
- soft_metrics_hybrid_100ex.log - Cosine similarity & KL divergence
- semantic_accuracy_hybrid.log - Token-level semantic matching

## Analysis

See PROBE_COMPARISON_SUMMARY.md for detailed comparison of training strategies and performance analysis.

## Model Weights

Trained probe weights: ../../../../models/tuned_lens/single_probe_L10_to_L11/

## Training Code

See ../../../../src/experiments/tuned_lens/single_probe/train_probe_HYBRID.py

## Next Steps

We attempted to scale this to all layer pairs (L0-L10 → L11), but discovered a LoRA loading issue that affected probe quality.
