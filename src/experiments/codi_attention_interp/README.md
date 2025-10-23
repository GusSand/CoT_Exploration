# CODI Attention & Importance Analysis

Experiments to answer:
- **RQ1**: How can we causally attribute the importance of continuous thought tokens in CODI's compressed reasoning?
- **RQ2**: How does a continuous thought's importance relate to its attention patterns?

## Quick Start: Test Pipeline (10 problems)

The test pipeline has already been validated. Results are in `results/` and figures in `figures/`.

## Running Full Experiment (100 problems)

### Step 1: Create Full Dataset

```bash
# Sample 100 balanced problems from the 1,000-problem stratified dataset
python scripts/create_full_dataset.py
```

This creates `results/full_dataset_100.json` with 25 problems from each difficulty level.

### Step 2: Run Token Ablation (~10 minutes)

```bash
# Run ablation on 100 problems (700 experiments total)
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python scripts/1_run_token_ablation.py
```

Output: `results/token_ablation_results_100.json`

### Step 3: Extract Attention Weights (~2 minutes)

```bash
# Extract attention on same 100 problems
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python scripts/2_extract_attention.py
```

Output: `results/attention_weights_100.json`

### Step 4: Analyze & Visualize (~30 seconds)

```bash
# Generate correlation analysis and figures
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python scripts/3_analyze_and_visualize.py
```

Outputs:
- `figures/1_importance_by_position.{pdf,png}` - Bar chart of token importance
- `figures/2_importance_heatmap.{pdf,png}` - Problems × tokens matrix
- `figures/3_attention_vs_importance.{pdf,png}` - Correlation scatter plot
- `figures/4_correlation_by_position.{pdf,png}` - Per-token analysis
- `results/summary_statistics.json` - All stats

### One-Line Command (Run All)

```bash
# Run everything sequentially
cd /home/paperspace/dev/CoT_Exploration
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH bash -c "
    python src/experiments/codi_attention_interp/scripts/create_full_dataset.py && \
    python src/experiments/codi_attention_interp/scripts/1_run_token_ablation.py && \
    python src/experiments/codi_attention_interp/scripts/2_extract_attention.py && \
    python src/experiments/codi_attention_interp/scripts/3_analyze_and_visualize.py
"
```

**Total estimated time: ~13 minutes**

## Test Results Summary (10 problems)

### RQ1: Token Importance

| Token | Importance | Interpretation |
|-------|-----------|----------------|
| Token 5 | 40% | **Most critical** - final reasoning step |
| Token 1 | 20% | Moderate importance |
| Token 4 | 20% | Moderate importance |
| Tokens 0,2,3 | 10% each | Lower importance |

### RQ2: Attention-Importance Correlation

| Layer | Correlation | P-value | Significance |
|-------|------------|---------|--------------|
| **Layer 8 (middle)** | **r=0.367** | **p=0.004** | ✅ **Significant** |
| Layer 14 (late) | r=0.211 | p=0.105 | Trend (needs more data) |
| Layer 4 (early) | r=0.013 | p=0.919 | No correlation |

**Key Finding**: Middle layer attention (L8) significantly predicts token importance!

## File Structure

```
codi_attention_interp/
├── README.md                               # This file
├── scripts/
│   ├── 0_test_model_loading.py            # Quick validation script
│   ├── create_test_dataset.py             # Creates 10-problem test set
│   ├── create_full_dataset.py             # Creates 100-problem full set
│   ├── 1_run_token_ablation.py            # Individual token ablation
│   ├── 2_extract_attention.py             # Attention weight extraction
│   └── 3_analyze_and_visualize.py         # Analysis and visualization
├── results/
│   ├── test_dataset_10.json               # Test dataset
│   ├── full_dataset_100.json              # Full dataset (to be created)
│   ├── token_ablation_results_test.json   # Test ablation results
│   ├── attention_weights_test.json        # Test attention results
│   └── summary_statistics.json            # Statistical summary
└── figures/
    ├── 1_importance_by_position.{pdf,png}
    ├── 2_importance_heatmap.{pdf,png}
    ├── 3_attention_vs_importance.{pdf,png}
    └── 4_correlation_by_position.{pdf,png}
```

## Next Steps for Full Analysis

1. **Run on 100 problems** for stronger statistical power
2. **Compositional analysis**: Test token pairs/triplets (from your suggested experiments)
3. **Residual stream decomposition**: Understand how tokens build computation
4. **Discrete→Continuous attention**: Which question tokens route to which continuous thoughts

## Notes

- Uses LLaMA-3.2-1B CODI checkpoint at `~/codi_ckpt/llama_gsm8k/`
- Tests at middle layer (L8) for ablation
- Extracts attention at layers 4, 8, 14 (early, middle, late)
- All scripts support `--test_mode` flag for 10-problem testing
