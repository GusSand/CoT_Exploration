# CODI Discretization Analysis

This directory contains a comprehensive analysis of how discretizing continuous thought representations affects CODI's mathematical reasoning performance on the GSM8K dataset.

## Overview

We tested three discretization modes across two model architectures:
- **Vanilla**: All thoughts remain continuous (baseline)
- **Alternating**: Discretize only odd positions (T1, T3, T5)
- **Full**: Discretize all thought positions (T0-T5)

## Key Findings

### GPT-2 (117M parameters)
- **Vanilla**: 42.53% accuracy
- **Alternating**: 25.93% accuracy (-16.60pp, -39.0% relative)
- **Full**: 24.64% accuracy (-17.89pp, -42.1% relative)

### Llama-1B (1.0B parameters)
- **Vanilla**: 37.68% accuracy
- **Alternating**: 0.83% accuracy (-36.85pp, -97.8% relative)
- **Full**: 2.12% accuracy (-35.56pp, -94.4% relative)

**Critical Finding**: Llama shows near-complete performance collapse with discretization, suggesting that larger models may rely more heavily on fine-grained continuous representations for chain-of-thought reasoning.

## Directory Structure

```
discretization_analysis/
├── README.md                    # This file
├── scripts/                     # Analysis and visualization scripts
│   ├── run_gpt2_analysis.py    # Batched GPU analysis for GPT-2
│   ├── run_llama_analysis.py   # Batched GPU analysis for Llama
│   ├── visualize_results.py    # Generate plots from results
│   └── fix_json.py             # Fix Infinity values in JSON
├── reports/                     # Detailed findings and analysis
│   ├── GPT2_REPORT.md          # Complete GPT-2 analysis
│   ├── LLAMA_REPORT.md         # Complete Llama analysis
│   └── COMPARATIVE_ANALYSIS.md # Cross-model comparison
└── plots/                       # Visualization outputs
    ├── gpt2/                    # GPT-2 visualizations
    └── llama/                   # Llama visualizations
```

## Methodology

1. **Models**: CODI-GPT2 (117M) and CODI-Llama3.2-1B (1.0B)
2. **Dataset**: GSM8K test set (1,319 examples)
3. **Hardware**: NVIDIA A100 GPU with BFloat16 precision
4. **Batch Size**: 16 examples/batch for efficient GPU utilization
5. **Evaluation**: Exact match accuracy on numerical answers

## Running the Analysis

### GPT-2 Analysis
```bash
python scripts/run_gpt2_analysis.py --batch_size 16 --device cuda
```

### Llama Analysis  
```bash
python scripts/run_llama_analysis.py --batch_size 16 --device cuda
```

### Generate Visualizations
```bash
python scripts/visualize_results.py --results_file results.json --output_dir plots/
```

## Performance Metrics

- **GPU Speedup**: 85.5x faster than CPU (GPT-2)
- **Processing Speed**: 62.83 examples/minute (Llama on A100)
- **Total Time**: ~21 minutes per model for full 1,319 examples

## Implications

1. **Continuous representations are essential** for CODI's reasoning ability
2. **Larger models are more sensitive** to discretization
3. **Even partial discretization** (alternating mode) causes severe degradation
4. **Design recommendation**: Maintain continuous latent thoughts rather than forcing discrete commitments

## Citation

If you use this analysis in your research, please cite:
```bibtex
@misc{codi_discretization_2025,
  title={Impact of Discretization on Continuous Chain-of-Thought Reasoning},
  author={CODI Discretization Analysis},
  year={2025},
  note={Analysis of CODI models on GSM8K}
}
```

## Contact

For questions or issues, please open an issue in the repository.
