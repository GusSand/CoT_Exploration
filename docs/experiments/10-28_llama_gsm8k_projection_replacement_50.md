# LLaMA Projection Replacement Intervention - 50 Sample Pilot

**Date**: 2025-10-28
**Model**: CODI-LLaMA (Llama-3.2-1B + LoRA)
**Dataset**: 50 GSM8K test samples (random selection)
**Experiment**: Number embedding projection intervention in continuous CoT

---

## Objective

Test whether replacing number token embeddings in continuous chain-of-thought (CoT) representations affects final answer generation in CODI-LLaMA, thereby validating whether semantic number information is preserved in the latent continuous space.

**Research Question**: If we intervene on number token predictions during CoT by replacing their embeddings with a different target number embedding, does this change the final answer?

---

## Background

### CODI Architecture

CODI (Continuous Thought via Distillation) represents chain-of-thought reasoning in continuous latent space rather than discrete text. The model:
1. Encodes question → generates 6 latent CoT tokens (BoT + T1-T6)
2. Each token can be decoded to text for inspection
3. Final answer is generated conditioned on the latent CoT

**Key Innovation**: CoT exists as continuous representations, not discrete tokens.

### Motivation

Previous proof-of-concept notebook showed one example where:
- CoT contained numerical reasoning (decoded tokens showed numbers)
- Intervening on number embeddings did NOT change the final answer
- **Hypothesis**: Numbers in decoded CoT are artifacts; actual computation uses distributed continuous representations

**This experiment scales to 50 samples to validate/falsify this hypothesis.**

---

## Methodology

### Intervention Design

**Projection Replacement Method**:


**Parameters**:
- Target token: '5' (single digit number)
- k=3: Strength of replacement projection
- Intervention position: All CoT positions (BoT, T1-T6)

### Causal Intervention Protocol

**Critical Implementation**: Two-pass causal intervention
1. **Pass 1 (Baseline)**:
   - Run complete CoT WITHOUT intervention
   - Decode tokens at each position
   - Generate final answer

2. **Pass 2 (Intervention)**:
   - Run complete CoT WITH causal intervention
   - At each position: if number detected → intervene → affects next position
   - Decode tokens at each position
   - Generate final answer

3. **Compare**: Baseline answer vs Intervention answer

**Why Causal**: Intervention must occur DURING forward pass, not post-hoc, because each CoT position conditions on previous positions.
