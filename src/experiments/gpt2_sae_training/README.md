# GPT-2 TopK SAE Training

**Experiment**: Parameter sweep to identify optimal TopK Sparse Autoencoder configuration for GPT-2 continuous thought activations.

**Date**: 2025-10-27
**Status**: ✅ Complete
**Sweet Spot**: d=512, K=150 (94.8% EV, 4.1% death rate)

---

## Quick Links

- **📖 Usage Guide**: [`docs/code/gpt2_sae_usage_guide.md`](../../../docs/code/gpt2_sae_usage_guide.md)
- **📊 Experiment Report**: [`docs/experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md`](../../../docs/experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md)
- **📦 Data Inventory**: [`docs/DATA_INVENTORY.md`](../../../docs/DATA_INVENTORY.md#18-gpt-2-topk-sae-parameter-sweep-datasets) (Section 18)

---

## Directory Structure

```
gpt2_sae_training/
├── README.md                    # This file
│
├── data/                        # Activation datasets
│   ├── gpt2_full_train_activations.pt    # 57,600 train samples (177 MB)
│   └── gpt2_full_val_activations.pt      # 14,400 val samples (44 MB)
│
├── results/                     # Training results
│   ├── sweet_spot_all/
│   │   ├── gpt2_sweet_spot_pos{0-5}_layer{0-11}.pt  # 72 SAE checkpoints
│   │   └── sweet_spot_metrics_all.json               # All metrics
│   │
│   ├── analysis_summary.json                         # Parameter sweep analysis
│   ├── gpt2_grid_metrics_pos3_layer8_config{0-7}.json  # Individual configs
│   ├── gpt2_sweet_spot_reconstruction_loss.png       # Layer×Position heatmap
│   └── gpt2_sweet_spot_feature_death_rate.png        # Layer×Position heatmap
│
└── scripts/                     # Training scripts
    ├── convert_gpt2_data.py                          # JSON → PT conversion
    ├── train_gpt2_grid.py                            # Parameter sweep
    ├── run_parallel_training.sh                      # Parallel execution
    ├── analyze_results.py                            # Results analysis
    ├── train_sweet_spot_all_layers_positions.py     # Train all 72 SAEs
    └── visualize_sweet_spot.py                       # Create heatmaps
```

---

## Quick Start

### Load a Sweet Spot SAE

```python
import torch
import sys
from pathlib import Path

# Add TopK SAE to path
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder

# Load SAE for Position 3, Layer 8
checkpoint = torch.load(
    'src/experiments/gpt2_sae_training/results/sweet_spot_all/gpt2_sweet_spot_pos3_layer8.pt',
    weights_only=False
)

sae = TopKAutoencoder(
    input_dim=768,   # GPT-2 hidden dim
    latent_dim=512,  # Sweet spot
    k=150            # Sweet spot
)
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()

print(f"✓ Loaded SAE: EV={checkpoint['metrics']['explained_variance']:.3f}")
```

### Extract Features

```python
# Input: GPT-2 activations (batch_size, 768)
activations = torch.randn(10, 768)

with torch.no_grad():
    reconstruction, features, metrics = sae(activations)

print(f"Features shape: {features.shape}")  # (10, 512)
print(f"Active features: {metrics['l0_mean']:.1f}")  # ~150
```

**Full usage examples**: See [`docs/code/gpt2_sae_usage_guide.md`](../../../docs/code/gpt2_sae_usage_guide.md)

---

## Sweet Spot Configuration

**Selected**: d=512, K=150

**Performance**:
- ✅ 94.8% explained variance (highest)
- ✅ 4.1% feature death rate (lowest)
- ✅ 29.3% sparsity (150/512 active)
- ✅ 95.9% feature utilization

**Why this config?**
1. Best balanced trade-off between reconstruction quality and feature utilization
2. All 8 tested configs exceeded 70% EV, but d=512, K=150 was consistently best
3. Near-zero feature death in late layers (0-2%)
4. Handles both easy (early layers) and hard (late layers) activations

---

## Parameter Sweep Results

Tested 8 configurations:

| Config | d | K | Sparsity | EV | Death Rate |
|--------|---|---|----------|-----|------------|
| **Sweet Spot** | **512** | **150** | **29.3%** | **94.8%** | **4.1%** |
| Runner-up | 512 | 100 | 19.5% | 94.1% | 26.0% |
| 3rd place | 384 | 75 | 19.5% | 93.3% | 36.5% |
| 4th place | 256 | 75 | 29.3% | 92.7% | 26.6% |
| 5th place | 256 | 50 | 19.5% | 91.4% | 43.8% |
| 6th place | 192 | 40 | 20.8% | 90.6% | 39.6% |
| 7th place | 256 | 30 | 11.7% | 88.5% | 59.0% |
| 8th place | 192 | 20 | 10.4% | 83.3% | 76.6% |

**Key Findings**:
- All 8 configs exceeded 70% EV threshold
- Larger dictionaries (d=512) significantly outperform smaller ones
- Higher K reduces feature death (K=150 > K=100 > K=75)

---

## Layer × Position Analysis

Trained sweet spot (d=512, K=150) on all 72 combinations (12 layers × 6 positions).

**Overall Statistics**:
- Mean EV: 93.2% ± 6.1%
- Mean death rate: 4.5% ± 7.4%
- Range: 75.6% - 99.7% EV

**Performance Patterns**:
- **Early layers (L0-L3)**: High EV (>96%), low death (<20%)
- **Middle layers (L4-L7)**: Medium EV (~93-97%), very low death (<10%)
- **Late layers (L8-L11)**: Lower EV (~75-95%), near-zero death (<2%)

**Position Specialization**:
- **Odd positions (1,3,5)**: Consistently higher EV, easier to reconstruct
- **Even positions (0,2,4)**: Lower EV in late layers, complex/abstract encoding

**Visualizations**: See `results/gpt2_sweet_spot_*.png`

---

## Comparison to LLaMA

| Metric | GPT-2 (124M) | LLaMA (1B) | Insight |
|--------|--------------|------------|---------|
| Input dim | 768 | 2048 | GPT-2 is 3× smaller |
| Sweet spot d | 512 | 512 | Same dictionary size |
| Sweet spot K | 150 | 100 | GPT-2 needs 50% more active features |
| Sparsity | 29.3% | 19.5% | GPT-2 requires denser representations |
| Expansion ratio | 0.67× | 0.25× | GPT-2 uses larger relative dictionary |

**Key Insight**: Smaller models (GPT-2) require denser, less specialized representations to capture reasoning, while larger models (LLaMA) can use sparser, more specialized features.

---

## Reproducing the Experiment

### Step 1: Convert Data (if needed)

```bash
python src/experiments/gpt2_sae_training/scripts/convert_gpt2_data.py
```

**Input**: `src/experiments/gpt2_shared_data/gpt2_predictions_1000_checkpoint_1000.json` (1.7 GB)
**Output**: Train (177 MB) + Val (44 MB) PT files
**Time**: <1 minute

### Step 2: Run Parameter Sweep

```bash
# Sequential (slower)
python src/experiments/gpt2_sae_training/scripts/train_gpt2_grid.py

# Parallel (faster)
bash src/experiments/gpt2_sae_training/scripts/run_parallel_training.sh
```

**Time**: ~20 seconds (parallel on A100)
**Output**: 8 checkpoints + metrics

### Step 3: Analyze Results

```bash
python src/experiments/gpt2_sae_training/scripts/analyze_results.py
```

**Output**: `results/analysis_summary.json` with sweet spot selection

### Step 4: Train Sweet Spot on All Layers×Positions

```bash
python src/experiments/gpt2_sae_training/scripts/train_sweet_spot_all_layers_positions.py
```

**Time**: ~3 minutes
**Output**: 72 SAE checkpoints

### Step 5: Create Visualizations

```bash
python src/experiments/gpt2_sae_training/scripts/visualize_sweet_spot.py
```

**Output**: 2 heatmap PNG files

---

## Files Size Reference

| File Type | Count | Total Size | Notes |
|-----------|-------|------------|-------|
| Activation data (.pt) | 2 | 221 MB | Train + val splits |
| SAE checkpoints (.pt) | 72 | ~216 MB | All layers×positions (excluded from git) |
| Parameter sweep checkpoints | 8 | ~16 MB | Config metrics (excluded from git) |
| Result metrics (.json) | 11 | ~20 KB | Analysis summaries |
| Visualizations (.png) | 2 | ~640 KB | Heatmaps |
| Scripts (.py) | 7 | ~30 KB | Training & analysis |

**Total repository size** (excluding .pt files): ~700 KB
**Total experiment artifacts** (including .pt): ~437 MB

---

## Dependencies

```python
# Core
torch >= 2.0.0
numpy >= 1.20.0
scikit-learn >= 1.0.0

# Visualization
matplotlib >= 3.5.0
seaborn >= 0.11.0

# CODI (for live inference)
transformers
peft
```

**TopK SAE**: `src/experiments/topk_grid_pilot/topk_sae.py`

---

## Common Use Cases

### 1. Error Prediction
Use SAE features to predict if GPT-2 will answer incorrectly.

```python
# Extract features for correct/incorrect samples
# Train classifier on features
# Achieve ~70-80% accuracy (see LLaMA SAE experiments)
```

### 2. Feature Interpretability
Analyze which features correlate with specific operations or concepts.

```python
# Extract top-K active features
# Correlate with CoT tokens (numbers, operators)
# Build feature catalog (monosemantic features)
```

### 3. Cross-Layer Analysis
Compare how features evolve across layers.

```python
# Track feature activations from L0 → L11
# Identify layer-specific vs shared features
# Understand reasoning progression
```

### 4. Position Specialization
Study how different positions encode different information.

```python
# Compare features across positions 0-5
# Identify position-specific patterns
# Validate odd/even position hypothesis
```

---

## Citation

```bibtex
@misc{gpt2_topk_sae_2025,
  title={GPT-2 TopK SAE Parameter Sweep},
  author={CoT Exploration Project},
  year={2025},
  note={Sweet Spot: d=512, K=150, 94.8\% EV}
}
```

---

## Related Work

- **LLaMA TopK SAE**: `src/experiments/topk_grid_pilot/`
- **SAE Error Analysis**: `src/experiments/sae_error_analysis/`
- **Matryoshka SAE**: `src/experiments/matryoshka_sae_pilot/`

---

## Contact & Issues

For questions or issues:
1. Check the **Usage Guide**: `docs/code/gpt2_sae_usage_guide.md`
2. Review the **Experiment Report**: `docs/experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md`
3. Consult the **Research Journal**: `docs/research_journal.md`

---

**Last Updated**: 2025-10-27
**Experiment Time**: ~30 minutes (5 min training + 25 min documentation)
**Status**: ✅ Complete & Production-Ready
