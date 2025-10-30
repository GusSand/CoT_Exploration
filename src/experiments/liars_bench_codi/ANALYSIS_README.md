# Pre-Compression Deception Signal Analysis

**Date**: 2025-10-30
**Model**: LLaMA-3.2-1B CODI
**Dataset**: LIARS-BENCH
**Research Question**: WHERE is deception information lost during continuous thought compression?

---

## Overview

This analysis investigates the mechanistic failure of CODI on deception detection by probing activations at multiple layers and token positions to identify where the deception signal degrades.

### Hypothesis
Deception signals exist in early/mid layers but disappear during compression to continuous thought tokens (CT0-CT5), while remaining detectable in final response tokens.

### Methodology
- **Layers probed**: [0, 3, 6, 9, 12, 15] (6 layers across LLaMA-1B's 16 layers)
- **Positions probed**: question_last, ct0-ct5, answer_first (8 positions)
- **Total probes**: 6 layers × 8 positions = 48
- **Probe type**: Logistic Regression (sklearn)
- **Data**: 288 train / 288 test (perfectly balanced, proper held-out)

---

## Quick Start

### 1. Training (Incremental Approach)

```bash
# Train 5 epochs first (~50 minutes)
cd /home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi
bash scripts/train_llama1b_incremental.sh

# Check results, then optionally continue to 10 epochs
# (Modify script to --num_train_epochs 10)
```

### 2. Run Complete Analysis Pipeline

```bash
# After 5-epoch training completes:
bash scripts/run_full_analysis.sh 5ep

# This runs:
# - Activation extraction (~30 min)
# - Probe training (~2 hours)
# - Visualization generation
# - Statistical analysis

# For 10 or 15 epochs:
bash scripts/run_full_analysis.sh 10ep
bash scripts/run_full_analysis.sh 15ep
```

### 3. View Results

```bash
# Visualizations
ls results/figures/multilayer_*_llama1b_5ep.png

# Probe results
cat results/multilayer_probe_results_llama1b_5ep.json

# Statistical analysis & key findings
cat results/multilayer_statistical_analysis_llama1b_5ep.json
```

---

## Scripts

### Training
- `train_llama1b_test.sh` - 1-epoch test run (validation)
- `train_llama1b_incremental.sh` - 5-epoch incremental training
- `train_llama1b.sh` - Full 15-epoch training (if needed)

### Analysis Pipeline
- `extract_activations_llama1b_multilayer.py` - Extract from 6 layers × 8 positions
- `train_multilayer_probes_llama1b.py` - Train 48 logistic regression probes
- `visualize_multilayer_results.py` - Generate heatmap + line plots
- `analyze_multilayer_patterns.py` - Statistical analysis + key findings
- `run_full_analysis.sh` - Master script (runs all above)

---

## Expected Results

### If Signal Exists & Degrades (Compression Hypothesis):
- Early layers (0, 3, 6): 55-65% accuracy (weak signal)
- Mid layers (9, 12): Signal degradation begins
- Late layers (15): CT ~50% (lost), response 60-70% (preserved)

### If No Signal (Representation Hypothesis):
- All layers/positions: ~50% accuracy (random chance)
- Indicates model doesn't encode deception, not just compression issue

---

## Data

**Training Data** (CODI):
- File: `data/processed/train_proper.json`
- Samples: 6,405 (honest answers only)
- Purpose: Train CODI to answer honestly

**Probe Data**:
- Train: `data/processed/probe_train_proper.json` (288 samples, 144 honest/144 deceptive)
- Test: `data/processed/probe_test_proper.json` (288 samples, 144 honest/144 deceptive)
- Purpose: Train/test deception detection probes
- Split: Question-level (proper held-out, zero overlap)

**Activations** (Generated):
- Format: `data/processed/multilayer_activations_llama1b_{epoch}_{split}.json`
- Structure: `activations[layer][position]` = list of 2048-dim vectors
- Size: ~500MB per split

---

## Checkpoints

**Location**: `~/codi_ckpt/llama1b_liars_bench_proper/`

**Structure**:
```
liars_bench_llama1b_codi/
└── Llama-3.2-1B-Instruct/
    ├── ep_5/lr_0.0008/seed_42/checkpoint-250/    # 5 epochs
    ├── ep_10/lr_0.0008/seed_42/checkpoint-500/   # 10 epochs (if continued)
    └── ep_15/lr_0.0008/seed_42/checkpoint-750/   # 15 epochs (if continued)
```

**Size**: ~3.5GB per checkpoint

---

## Troubleshooting

### Training doesn't start
```bash
# Check tmux session
tmux ls
tmux attach -t llama1b_5ep

# Check logs
tail -f ~/codi_ckpt/llama1b_liars_bench_proper/train_5ep.log
```

### Checkpoint not found
```bash
# Verify checkpoint path
ls ~/codi_ckpt/llama1b_liars_bench_proper/liars_bench_llama1b_codi/Llama-3.2-1B-Instruct/
```

### Activation extraction fails
- Check GPU memory: `nvidia-smi`
- Reduce batch size if OOM
- Ensure checkpoint path is correct

### Probe training slow
- Expected: ~2 hours for 48 probes
- Check progress: grep "Training probes" in output
- Uses CPU (sklearn), no GPU needed

---

## Timeline

| Task | Time | Cumulative |
|------|------|------------|
| 5-epoch training | 50 min | 0:50 |
| Activation extraction | 30 min | 1:20 |
| Probe training | 2 hours | 3:20 |
| Visualization + analysis | 5 min | 3:25 |
| **Total (5 epochs)** | **~3.5 hours** | |
| | | |
| Additional 5 epochs (to 10) | 50 min | +0:50 |
| Re-run analysis | 2.5 hours | +2:30 |
| **Total (10 epochs)** | **~6.5 hours** | |

---

## Citation

Based on experiment plan in:
`docs/experiments/10-30_llama1b_liars_bench_precompression_signal_analysis.md`

---

## Contact

For questions about this analysis, see the detailed experiment documentation or review the conversation logs in `docs/conversations/`.
