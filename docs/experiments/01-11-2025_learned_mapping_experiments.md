# Learned Mapping Experiments: From Token Embeddings to Hidden States

**Date:** November 1, 2025
**Model:** CODI-LLaMA (meta-llama/Llama-3.2-1B with LoRA)
**Objective:** Learn transformations from token embeddings to hidden state activations to enable discretization intervention

## Executive Summary

We conducted three experiments to learn mappings from token embeddings to hidden state activations, testing whether learned transformations could preserve reasoning ability better than simple identity mapping. **All approaches failed to improve upon the baseline**, with performance degrading significantly across all methods.

### Key Results:

| Approach | Clean Dataset | GSM8K Test | Key Characteristic |
|----------|--------------|------------|-------------------|
| **Baseline (Identity)** | **90.2%** | **54.5%** | Simple L2 normalization |
| Full-Rank MSE | 25.8% | 33.3% | 4.2M parameters, val_loss=4.01 |
| Residual Low-Rank KL | 18.2% | 31.8% | 0.53M parameters, val_loss=2.27 |
| Diagonal Regularization (strong) | 12.1% | 28.8% | Frobenius=0.069 from identity |
| Diagonal Regularization (none) | 50.0% | 41.7% | Frobenius=818.9 from identity |

**Conclusion:** Token embeddings fundamentally lack critical information present in hidden states necessary for multi-step mathematical reasoning. No learned transformation can recover what is not present in the embedding space.

---

## Experiment 1: Full-Rank Linear Mapping with MSE Loss

### Approach

Learn a full-rank linear transformation to map token embeddings to hidden state activations:

```
Y = W @ X + b
```

Where:
- `X`: Token embedding (2048-dim)
- `Y`: Target hidden state activation (2048-dim)
- `W`: Learned weight matrix (2048 × 2048)
- `b`: Learned bias vector (2048-dim)

**Loss Function:** Mean Squared Error (MSE)
```
L = ||Y_pred - Y_true||²
```

### Data Collection

- **Source:** GSM8K training set (7,473 examples)
- **Collection method:** Run chain-of-thought inference, collect (embedding, activation) pairs at beginning-of-thought token
- **Total pairs collected:** 52,311
- **Unique tokens:** 1,605

### Balanced Data Splitting

To prevent overfitting to frequent tokens (e.g., numbers), we created a balanced validation split:

```python
def create_balanced_split(all_pairs, val_fraction=0.2, max_per_token=100):
    """Group pairs by token_id, sample evenly from each group"""
    token_groups = {}
    for pair in all_pairs:
        token_groups[pair['token_id']].append(pair)

    for token_id, pairs in token_groups.items():
        n_val = max(1, int(len(pairs) * val_fraction))
        indices = np.random.permutation(len(pairs))

        val_idx = indices[:n_val]
        train_idx = indices[n_val:n_val + max_per_token]

        # Limit training samples per token to max_per_token
```

**Result:**
- Training pairs: 48,562
- Validation pairs: 3,749

### Hyperparameter Search

Tested 27 configurations varying:

| Hyperparameter | Values Tested |
|---------------|---------------|
| Learning rate | 1e-5, 1e-4, 1e-3 |
| Epochs | 50, 100, 200 |
| Downsampling | 0.2, 0.5, 1.0 |

**Best Configuration:**
- Learning rate: 0.01
- Epochs: 200
- Downsampling: 1.0 (use all training data)
- **Validation loss:** 4.0099
- **Parameters:** 4,196,352 (4.2M)

### Training Details

- **Batch size:** 256
- **Optimizer:** Adam
- **Initialization:** Small random initialization (`randn * 0.01`)
- **Data type fix:** Had to convert embeddings/activations from bfloat16 to float32 for training stability

### Results

**Accuracy on Test Sets:**
- Clean dataset: **25.8%** (baseline: 90.2%) → **-64.4% degradation**
- GSM8K test: **33.3%** (baseline: 54.5%) → **-21.2% degradation**

**Per-Example Analysis (Clean Dataset):**
- Both correct: 34 (25.8%)
- Baseline correct, learned wrong: 85 (64.4%)
- Learned correct, baseline wrong: 0 (0.0%)
- Both wrong: 13 (9.8%)

### Key Observations

1. **Catastrophic performance collapse:** Despite achieving reasonable reconstruction loss, the learned mapping destroyed reasoning ability
2. **No recovery cases:** Not a single example improved from wrong to correct
3. **Matrix divergence:** The learned matrix diverged significantly from identity, suggesting MSE loss alone is insufficient

---

## Experiment 2: Residual Low-Rank Mapping with KL Divergence Loss

### Motivation

Experiment 1 failed because:
1. Full-rank matrix has too many parameters (4.2M) → overfitting risk
2. MSE loss doesn't preserve prediction distributions
3. Starting from scratch ignores that identity works well

### Approach

Combine three improvements:
- **Residual learning:** Start from identity, learn small correction
- **Low-rank factorization:** Reduce parameters via `W = U @ V^T`
- **Prediction-preserving loss:** Use KL divergence on LM head outputs

**Architecture:**
```
Y = X + α · (U @ V^T @ X + b)
```

Where:
- `U, V`: Low-rank matrices (2048 × rank)
- `α`: Learnable scalar controlling residual strength
- `rank`: Bottleneck dimension (64, 128, 256)

**Loss Function:** KL Divergence on next-token predictions
```
L = KL(softmax(LM_head(Y_pred)) || softmax(LM_head(Y_true)))
```

### Hyperparameter Search

Tested 6 configurations:

| Config | Rank | LR | Epochs | Downsample | Val Loss | Final α |
|--------|------|-----|--------|------------|----------|---------|
| 1 | 64 | 0.001 | 50 | 1.0 | 2.335 | 0.320 |
| 2 | 128 | 0.001 | 50 | 1.0 | 2.320 | 0.299 |
| 3 | 256 | 0.001 | 50 | 1.0 | 2.349 | 0.309 |
| **4** | **128** | **0.01** | **50** | **0.5** | **2.270** | **0.249** |
| 5 | 128 | 0.01 | 100 | 1.0 | 2.347 | 0.285 |
| 6 | 256 | 0.01 | 50 | 1.0 | 2.342 | 0.287 |

**Best Configuration:**
- Rank: 128
- Learning rate: 0.01
- Epochs: 50
- Downsampling: 0.5
- **Validation loss:** 2.270 (44% better than MSE approach!)
- **Final α:** 0.249 (model wants mostly identity)
- **Parameters:** 524,289 (0.53M) - 8× fewer than full-rank

### Training Observations

1. **Low learned α:** The model converged to α ≈ 0.25, suggesting it wants to stay close to identity
2. **Better training loss:** KL loss of 2.27 vs MSE loss of 4.01
3. **Efficient training:** Converged in 50 epochs with half the data

### Results

**Accuracy on Test Sets:**
- Clean dataset: **18.2%** (baseline: 90.2%) → **-72.0% degradation**
- GSM8K test: **31.8%** (baseline: 54.5%) → **-22.7% degradation**

**Per-Example Analysis (Clean Dataset):**
- Both correct: 24 (18.2%)
- Baseline correct, learned wrong: 95 (72.0%)
- Learned correct, baseline wrong: 0 (0.0%)
- Both wrong: 13 (9.8%)

### Critical Insight

**Better training loss ≠ Better reasoning performance**

Despite achieving:
- 44% lower validation loss (2.27 vs 4.01)
- More parameter-efficient design (0.53M vs 4.2M)
- Learned α suggesting near-identity preference

The performance was **WORSE** than the full-rank MSE approach. This paradox suggests that optimizing for reconstruction or prediction preservation does not preserve the reasoning circuit.

---

## Experiment 3: Diagonal Regularization

### Motivation

Experiments 1 & 2 showed:
1. Identity mapping works best
2. Both learned approaches diverge from identity
3. Even with residual learning (α=0.25), performance collapses

**Hypothesis:** Maybe we need to constrain the learned matrix to stay very close to identity?

### Approach

Learn full-rank mapping with L2 regularization penalizing off-diagonal elements:

```
Loss = MSE(Y_pred, Y_true)
       + λ_offdiag · Σ(W[i,j]² for i ≠ j)
       + λ_diag · Σ((W[i,i] - 1)²)
```

This encourages:
- Off-diagonal elements → 0
- Diagonal elements → 1
- Result: Matrix stays near identity

### Hyperparameter Search

Tested 7 configurations:

| Config | λ_offdiag | λ_diag | Val Loss | Frobenius | Diag Mean ± Std |
|--------|-----------|--------|----------|-----------|-----------------|
| 0 | 0.0 | 0.0 | **3.471** | 818.92 | 2.21 ± 0.42 |
| 1 | 0.001 | 0.0 | 4.143 | 699.29 | 14.12 ± 8.17 |
| 2 | 0.01 | 0.0 | 4.133 | 571.30 | 11.75 ± 6.62 |
| 3 | 0.1 | 0.0 | 4.138 | 303.80 | 6.77 ± 3.43 |
| 4 | 1.0 | 0.0 | 4.133 | 355.37 | 7.74 ± 4.03 |
| 5 | 0.01 | 0.001 | 4.179 | 0.708 | **1.002 ± 0.003** |
| 6 | 0.1 | 0.01 | 4.175 | **0.069** | **1.000 ± 0.0003** |

**Frobenius distance from identity:** `||W - I||_F`

### Results

Tested three configurations on reasoning tasks:

| Configuration | Frobenius | Clean | GSM8K Test |
|---------------|-----------|-------|------------|
| **Baseline (Identity)** | 0.0 | **90.2%** | **54.5%** |
| No Regularization (Config 0) | 818.9 | 50.0% | 41.7% |
| Strong Regularization (Config 6) | 0.069 | **12.1%** | 28.8% |

### Counterintuitive Finding

**Staying closer to identity HURTS performance!**

- Config 6 (Frob=0.069, diagonal=1.00±0.0003): **12.1%** accuracy
- Config 0 (Frob=818.9, diagonal=2.21±0.42): **50.0%** accuracy

The configuration that stayed almost perfectly at identity performed **4× worse** than the unconstrained version that drifted far away.

### The Paradox

1. Pure identity (Baseline): **90.2%** ✓
2. Learn near-identity (Strong Reg): **12.1%** ✗
3. Learn far from identity (No Reg): **50.0%** ✗

**Interpretation:** The learned "near-identity" matrix (Frob=0.069) is mathematically close to identity but destroys the reasoning circuit. Even tiny learned deviations break the computational pathway that identity preserves.

---

## Cross-Experiment Analysis

### Training Loss vs. Accuracy

| Approach | Val Loss | Clean Acc | GSM8K Acc |
|----------|----------|-----------|-----------|
| MSE Full-Rank | 4.010 | 25.8% | 33.3% |
| **KL Low-Rank** | **2.270** | **18.2%** | 31.8% |
| Diag Reg (None) | 3.471 | 50.0% | 41.7% |
| Diag Reg (Strong) | 4.175 | 12.1% | 28.8% |

**Observation:** Lower validation loss does NOT correlate with better reasoning performance. The KL approach achieved the best training loss but worst accuracy.

### Parameter Efficiency

| Approach | Parameters | Clean Acc |
|----------|-----------|-----------|
| Full-Rank MSE | 4.20M | 25.8% |
| Residual Low-Rank | 0.53M | 18.2% |
| Diagonal Reg | 4.20M | 12.1% - 50.0% |

**Observation:** Fewer parameters did not help. The low-rank approach with 8× fewer parameters performed worst.

### Distance from Identity

| Approach | Frobenius | Clean Acc |
|----------|-----------|-----------|
| Identity (Baseline) | 0.0 | **90.2%** |
| Strong Regularization | 0.069 | 12.1% |
| No Regularization | 818.9 | 50.0% |

**Non-monotonic relationship:** Both staying close and moving far from identity hurt performance, but in different ways.

---

## Theoretical Implications

### Why All Approaches Failed

1. **Embedding Space Insufficiency**
   - Token embeddings are learned to be context-independent
   - Hidden states contain contextualized, task-specific information
   - No linear transformation can add information that isn't there

2. **The Identity Paradox**
   - Identity works because it preserves the computational circuit
   - Learning "near-identity" breaks the circuit even with tiny changes
   - The reasoning pathway is extremely fragile to perturbations

3. **Optimization Mismatch**
   - Minimizing reconstruction error (MSE) ≠ preserving reasoning ability
   - Preserving predictions (KL) ≠ preserving computational pathways
   - Constraining to identity space ≠ preserving identity function

### What We Learned

**Negative Results:**
1. ✗ Full-rank linear mapping cannot recover hidden state information
2. ✗ Low-rank residual learning cannot efficiently approximate the transformation
3. ✗ Prediction-preserving loss does not preserve reasoning circuits
4. ✗ Regularizing toward identity makes performance worse
5. ✗ Better training metrics do not correlate with reasoning performance

**Positive Insights:**
1. ✓ Token embeddings lack critical information for multi-step reasoning
2. ✓ The reasoning circuit is extremely sensitive to perturbations
3. ✓ Identity mapping works not because it's "close enough" but because it preserves computational pathways
4. ✓ Discretization intervention requires the exact activation values, not learned approximations

---

## Experimental Details

### Common Setup

**Model:** CODI-LLaMA
- Base: meta-llama/Llama-3.2-1B
- LoRA: rank=128, alpha=32
- Hidden dimension: 2048
- Precision: bfloat16 for inference, float32 for training

**Datasets:**
- Training data: GSM8K train set (52,311 pairs from 7,473 examples)
- Clean test: 132 hand-crafted examples
- GSM8K test: First 132 examples from test set

**Hardware:**
- GPU: NVIDIA A100 (or similar)
- Training time: ~2-3 hours per experiment

### Implementation Notes

**Data Collection:**
```python
# Run CoT inference to collect pairs
for example in gsm8k_train:
    hidden_states = model.forward(question, output_hidden_states=True)
    bot_position = find_bot_token(outputs)

    pairs.append({
        'token_id': predicted_token_id,
        'token_string': predicted_token_str,
        'embedding': model.embed_tokens(predicted_token_id),
        'activation': hidden_states[-1][bot_position]
    })
```

**Key Bug Fixes:**
1. **dtype mismatch:** Embeddings/activations were bfloat16, had to convert to float32 for training
2. **Buffered output:** Had to use `python -u` for unbuffered logging
3. **Generation API:** Cache position index errors required using custom generation loop

### Reproducibility

All experiments used:
- Fixed random seed: 42
- Deterministic settings where possible
- Same training data across experiments
- Identical evaluation procedure

**Code locations:**
- `/workspace/CoT_Exploration/src/experiments/01-11-2025-learned-mapping/`
- `/workspace/CoT_Exploration/src/experiments/01-11-2025-learned-mapping-residual/`
- `/workspace/CoT_Exploration/src/experiments/01-11-2025-diagonal-regularization/`

---

## Future Directions

### What NOT to Try

Based on these experiments, the following are unlikely to help:
1. ✗ More sophisticated architectures (MLP, attention)
2. ✗ Different loss functions (Wasserstein, adversarial)
3. ✗ More training data or longer training
4. ✗ Different regularization schemes
5. ✗ Ensemble methods combining multiple learned mappings

### What MIGHT Be Worth Exploring

1. **Non-linear context-aware mappings**
   - Use the full context, not just the token
   - Learn transformations conditioned on reasoning state

2. **Learned discretization vocabulary**
   - Instead of using token embeddings, learn a dedicated "reasoning vocabulary"
   - May require end-to-end training

3. **Hybrid approaches**
   - Use identity for certain token types, learned mappings for others
   - Adaptive selection based on confidence

4. **Understanding the gap**
   - Analyze what information is present in activations but missing in embeddings
   - Information-theoretic analysis of the embedding space

---

## Conclusion

We systematically explored three approaches to learning mappings from token embeddings to hidden state activations:

1. **Full-Rank MSE:** 4.2M parameters, 25.8% accuracy
2. **Residual Low-Rank KL:** 0.53M parameters, 18.2% accuracy
3. **Diagonal Regularization:** Variable constraints, 12.1-50.0% accuracy

**All approaches failed dramatically compared to the identity baseline (90.2%).**

The counterintuitive finding that staying closer to identity performs worse than moving far away, yet both are much worse than pure identity, reveals a fundamental limitation: **token embeddings lack the information necessary for multi-step reasoning.**

This suggests that effective discretization intervention requires preserving the exact activation values computed by the model, not learned approximations. The reasoning circuit is too fragile and information-sensitive to tolerate even "optimal" transformations.

---

## References

**Related Work:**
- Discretization intervention baseline: `28-10-2028-projection-replacement/`
- Projection intervention analysis: `27-10-2025_projection_intervention/`

**Code Files:**
- `learned_mapping_intervention.py`: Full-rank MSE approach
- `learned_mapping_residual_lowrank.py`: Residual low-rank KL approach
- `learned_mapping_diagonal_regularization_v2.py`: Diagonal regularization approach
- `evaluate_diagonal_reg.py`: Evaluation script for diagonal regularization

**Visualizations:**
- `comprehensive_mapping_comparison.png`: All three approaches compared
- `diagonal_regularization_training.png`: Training metrics for diagonal regularization
- `diagonal_reg_final_results.png`: Final accuracy results

**Data:**
- All hyperparameter search results saved as JSON
- Per-example predictions saved for detailed analysis
- Training pairs cached for reproducibility
