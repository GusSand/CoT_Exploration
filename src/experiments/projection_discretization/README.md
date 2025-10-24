# Projection-Based Discretization Experiments for CODI

This package contains all code, results, and analysis for the projection-based discretization experiments on CODI (Continuous Chain-of-Thought) models.

## Experiment Overview

**Research Question**: Can we discretize continuous thought tokens in CODI using projection onto vocabulary subspaces while maintaining reasoning performance?

**Key Findings**:
- Projection-based discretization severely degrades performance
- Top-k projection (k=5) provides 70.6% improvement over single-token (k=1) but still loses 56pp vs vanilla
- Floating point errors as small as 1e-7 can catastrophically derail chain-of-thought reasoning
- Chain-of-thought amplifies brittleness through sequential error accumulation

## Directory Structure

```
projection_experiment_package/
├── README.md                          # This file
├── FINDINGS_SUMMARY.md                # Comprehensive analysis document
├── code/
│   ├── run_gpt2_topk_projection.py   # GPT-2 experiment code
│   ├── run_llama_topk_projection.py  # Llama experiment code
│   ├── test_topk_projection.py       # Local testing/validation
│   └── visualize_topk_results.py     # Visualization script
├── results/
│   ├── vanilla_full/
│   │   └── final_results.json         # Baseline (42.53% accuracy)
│   ├── k1_thought_normalized_full/
│   │   └── final_results.json         # k=1 projection (10.84% accuracy)
│   ├── k5_thought_normalized_full/
│   │   └── final_results.json         # k=5 projection (18.50% accuracy)
│   └── accuracy_comparison.png        # Visualization
└── scripts/
    ├── run_gpt2_experiments.sh        # Batch script for GPT-2
    └── run_llama_experiments.sh       # Batch script for Llama
```

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install torch transformers datasets peft matplotlib

# Clone CODI repository (if not already)
git clone <codi-repo-url>
cd CoT_Exploration
```

### 2. Run Experiments

**GPT-2:**
```bash
cd src/experiments/projection_discretization

# Vanilla baseline
python code/run_gpt2_topk_projection.py \
    --batch_size 1 \
    --discretize_positions none \
    --output_dir ./results/vanilla_full \
    --device cuda

# k=1 projection
python code/run_gpt2_topk_projection.py \
    --batch_size 1 \
    --discretize_positions thought_only \
    --normalize \
    --k_nearest 1 \
    --output_dir ./results/k1_thought_normalized_full \
    --device cuda

# k=5 projection
python code/run_gpt2_topk_projection.py \
    --batch_size 1 \
    --discretize_positions thought_only \
    --normalize \
    --k_nearest 5 \
    --output_dir ./results/k5_thought_normalized_full \
    --device cuda
```

**Llama:**
```bash
# Same commands, but use run_llama_topk_projection.py
python code/run_llama_topk_projection.py [same args as above]
```

### 3. Visualize Results

```bash
python code/visualize_topk_results.py
# Generates: results/accuracy_comparison.png
```

## Key Code Components

### Top-k Projection Function

The core projection function implements two methods:

**k=1 (Direct Replacement):**
```python
vocab_embedding = top_1_vocab_token
result = vocab_embedding * (continuous_norm / vocab_norm)
```

**k>1 (Subspace Projection):**
```python
# Solve (V V^T)α = V c where V is k×d matrix of vocab embeddings
G = torch.mm(V, V.t())  # Gram matrix (k×k)
Vc = torch.mv(V, c)     # Right-hand side (k,)
α = torch.linalg.solve(G, Vc)
projected = torch.mv(V.t(), α)
result = projected * (continuous_norm / projected_norm)
```

### Discretization Positions

- `none`: No discretization (vanilla baseline)
- `thought_only`: Discretize thought tokens only (not BoT/EoT)
- `all`: Discretize all latent tokens (most aggressive)

## Results Summary

### GPT-2 on GSM8K (1319 examples)

| Method | Accuracy | Correct | Drop vs Vanilla | Relative to k=1 |
|--------|----------|---------|-----------------|-----------------|
| Vanilla | 42.53% | 561/1319 | — | — |
| k=1 | 10.84% | 143/1319 | -31.69pp | — |
| k=5 | 18.50% | 244/1319 | -24.03pp | +70.6% |

**Key Insights:**
- k=5 gets 101 additional correct answers vs k=1
- k=5 reduces approximation error by 38% vs k=1
- But discretization fundamentally breaks continuous CoT reasoning
- Computational cost is nearly identical (~2.3s per example)

## Reproducing Results

### Prerequisites

- CODI checkpoint: `/workspace/CoT_Exploration/models/CODI-gpt2`
- GSM8K dataset (auto-downloaded via HuggingFace datasets)
- GPU with 16GB+ VRAM (or use CPU with `--device cpu`)

### Expected Runtime

- **GPU (A100)**: ~1-1.5 hours per configuration
- **CPU**: ~6-8 hours per configuration

### Validation

To validate the projection implementation:

```bash
python code/test_topk_projection.py
# Expected output:
# k=1 approximation error: ~0.85
# k=5 approximation error: ~0.52 (38% reduction)
```

## Implementation Details

### Numerical Stability

**Critical**: Use direct replacement formula for k=1:
```python
# GOOD (4 operations, ~2 ULPs error)
result = vocab * (||continuous|| / ||vocab||)

# BAD (8 operations, ~4-5 ULPs error)
direction = vocab / ||vocab||
proj_scalar = continuous · direction
projected = proj_scalar * direction
result = projected * (||continuous|| / ||projected||)
```

The 8-operation version accumulates 2x the floating point error, which compounds through 6 thought iterations and flips token selections.

### Model Configuration

```python
model_args = ModelArguments(
    model_name_or_path="gpt2",  # or "meta-llama/Llama-2-7b-hf"
    lora_init=True,
    lora_r=128,
    lora_alpha=32,
    ckpt_dir="<path-to-checkpoint>",
    full_precision=True
)

training_args = TrainingArguments(
    inf_latent_iterations=6,   # 6 thought iterations
    use_prj=True,
    prj_dim=768,               # Hidden dimension
    greedy=True,               # Greedy decoding
    remove_eos=True
)
```

## Citation

If you use this code or findings, please cite:

```bibtex
@article{codi_projection_2024,
  title={On the Brittleness of Projection-Based Discretization for Continuous Chain-of-Thought},
  author={...},
  journal={...},
  year={2024}
}
```

## Additional Documentation

See `FINDINGS_SUMMARY.md` for comprehensive analysis including:
- Detailed floating point error analysis
- Mathematical derivations
- Chain-of-thought brittleness discussion
- Recommendations for future work

## Contact

For questions or issues, please open a GitHub issue.
