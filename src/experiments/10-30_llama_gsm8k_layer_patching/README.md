# Layer-wise CoT Activation Patching Experiment

**Date**: 2025-10-30
**Model**: LLaMA-3.2-1B-Instruct + CODI
**Dataset**: GSM8K Clean/Corrupted Pairs (66 pairs)
**Goal**: Identify which layers are most critical for reasoning by patching continuous thought token activations

## Overview

This experiment investigates the layer-wise contribution of continuous thought (CoT) tokens to reasoning in CODI-trained LLMs. By patching activations from clean examples into corrupted examples at different layers, we can identify which layers are most critical for reasoning.

### Experimental Design

For each clean/corrupted pair:
1. Extract clean CoT token activations at all 22 layers
2. Run baseline corrupted model (no patching)
3. For each layer L:
   - Patch corrupted CoT activations with clean CoT activations at layer L
   - Compute KL divergence between patched and baseline output distributions
4. Higher KL divergence → layer is more critical for reasoning

### Dataset

- **Source**: `corrected_llama_cot_clean_corrupted_pairs/llama_clean_corrupted_pairs.json`
- **Size**: 66 pairs (132 total examples)
- **Type**: GSM8K math problems with small numerical perturbations
- **Clean vs Corrupted**: Typically one number changed by ±1
- **All answers verified correct**

## Project Structure

```
10-30_llama_gsm8k_layer_patching/
├── config.py                      # Configuration parameters
├── run_all.sh                     # Master script to run all steps
├── README.md                      # This file
├── core/
│   ├── model_loader.py            # CODI model loading and activation extraction
│   ├── activation_patcher.py      # Patching infrastructure with PyTorch hooks
│   └── metrics.py                 # KL divergence and other metrics
├── scripts/
│   ├── 1_load_data.py             # Data validation
│   ├── 2_run_patching.py          # Main patching experiment + W&B logging
│   ├── 3_visualize_individual.py  # Per-example heatmaps
│   └── 4_visualize_aggregate.py   # Aggregate analysis
└── results/
    ├── prepared_pairs.json        # Validated dataset
    ├── patching_results.json      # Raw results (pair × layer)
    ├── aggregate_statistics.json  # Aggregate metrics
    ├── individual_heatmaps/       # 66 per-example visualizations
    └── aggregate_analysis/        # 4 aggregate visualizations
```

## Installation

Dependencies are already installed in the project environment. Key packages:
- `torch` - Model inference
- `transformers` - LLaMA loading
- `peft` - LoRA adapters
- `wandb` - Experiment tracking
- `matplotlib`, `seaborn` - Visualization
- `numpy` - Numerical operations

## Usage

### Quick Start

Run the entire experiment with one command:

```bash
cd src/experiments/10-30_llama_gsm8k_layer_patching
bash run_all.sh
```

**Estimated runtime**: 2-3 hours for 66 pairs × 22 layers

### Step-by-Step Execution

If you want to run steps individually:

```bash
# Step 1: Validate dataset
python scripts/1_load_data.py

# Step 2: Run patching experiment (longest step ~2 hours)
python scripts/2_run_patching.py

# Step 3: Generate individual heatmaps
python scripts/3_visualize_individual.py

# Step 4: Generate aggregate visualizations
python scripts/4_visualize_aggregate.py
```

## Configuration

Edit `config.py` to modify parameters:

- `NUM_LAYERS`: Number of layers to test (default: 22 for LLaMA-1B)
- `NUM_LATENT`: Number of CoT tokens (default: 5)
- `DEVICE`: GPU device (default: "cuda")
- `BATCH_SIZE`: Batch size (default: 1 to avoid OOM)
- `SEED`: Random seed for reproducibility (default: 42)

## Output

### Quantitative Results

1. **patching_results.json**: Raw results for all pairs
   - Per-pair, per-layer KL divergence
   - L2 distance between logits
   - Prediction change rates

2. **aggregate_statistics.json**: Summary statistics
   - Mean/std KL divergence per layer
   - Top 5 critical layers identified

### Visualizations

1. **Individual heatmaps** (66 files): `results/individual_heatmaps/`
   - X-axis: Layer number (0-21)
   - Y-axis: KL divergence
   - Shows per-example critical layers

2. **Aggregate visualizations** (4 files): `results/aggregate_analysis/`
   - `aggregate_kl_by_layer.png`: Mean KL with error bars across all pairs
   - `heatmap_all_pairs_layers.png`: Full matrix showing all pairs × layers
   - `critical_layers_bar_chart.png`: Top 10 critical layers
   - `layer_similarity_analysis.png`: Which layers show minimal patching effect

## Interpretation

### KL Divergence Metric

- **High KL**: Patching at this layer significantly changes the output → layer is critical
- **Low KL**: Patching has minimal effect → representations already similar

### Expected Patterns

Based on prior mechanistic interpretability research:
- **Early layers (0-5)**: Likely low KL (feature extraction, not reasoning)
- **Middle layers (6-16)**: Likely high KL (reasoning computation)
- **Late layers (17-21)**: Possibly moderate KL (answer formatting)

## W&B Logging

All experiments are logged to Weights & Biases:
- Project: `cot-exploration`
- Experiment name: `10-30_llama_gsm8k_layer_patching_<timestamp>`

Logged metrics:
- Per-pair, per-layer KL divergence
- Aggregate statistics per layer
- Configuration parameters

## Validation

The experiment includes several validation checks:

1. **Data validation**: All 66 pairs have valid clean/corrupted variants
2. **Sanity checks**:
   - Patching layer 0 vs layer 21 should show different effects
   - Patching corrupted with itself should give KL ≈ 0
3. **Metric validation**: KL divergence is always non-negative

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:
- Reduce `BATCH_SIZE` in `config.py` (already set to 1)
- Use smaller model or fewer examples
- Enable gradient checkpointing (requires code modification)

### Model Loading Errors

Ensure checkpoint exists at:
```
/home/paperspace/codi_ckpt/llama_gsm8k/pytorch_model.bin
```

### Import Errors

Make sure CODI submodule is properly initialized:
```bash
cd /home/paperspace/dev/CoT_Exploration
git submodule update --init --recursive
```

## Citation

If you use this experiment in your research, please cite:

```bibtex
@article{codi2025,
  title={CODI: Continuous Chain-of-Thought via Self-Distillation},
  journal={arXiv preprint arXiv:2502.21074},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue in the repository.

## License

MIT License - See repository root for details.
