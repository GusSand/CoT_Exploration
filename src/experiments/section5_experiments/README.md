# CODI Section 5 Interpretability Analysis

This directory contains enhanced scripts for reproducing and extending Section 5 (Further Analysis) from the CODI paper.

## Overview

Section 5 of the CODI paper analyzes the interpretability of continuous thought tokens by:
1. Decoding continuous thoughts to vocabulary space
2. Examining attention patterns
3. Validating whether decoded outputs correspond to correct intermediate computation steps

Our implementation extends the original analysis with:
- **Segregated outputs**: Separate JSON files for correct vs. incorrect predictions
- **Intermediate computation validation**: Automated checking of whether decoded continuous thoughts match reference CoT steps
- **Enhanced visualizations**: HTML and text reports with detailed interpretability analysis
- **Structured exports**: JSON and CSV formats for further analysis

## Files

### Core Scripts

- `section5_analysis.py`: Enhanced analysis script with output segregation and computation validation
- `visualize_interpretability.py`: Generates HTML and text visualizations
- `scripts/run_section5_analysis.sh`: Convenient wrapper script to run complete analysis

### Original Files (from CODI repository)

- `probe_latent_token.py`: Original interpretability probing script
- `test.py`: Standard evaluation script
- `scripts/probe_latent_token.sh`: Original probe script

## Quick Start

###  1. Set Up Environment

```bash
# Create virtual environment
python3.12 -m venv env
source env/bin/activate

# Install dependencies
cd /workspace/CoT_Exploration/codi
pip install -r requirements.txt
```

### 2. Download Pretrained Model

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zen-E/CODI-gpt2', local_dir='models/CODI-gpt2')"
```

### 3. Run Section 5 Analysis

```bash
bash scripts/run_section5_analysis.sh
```

This will:
1. Run the model on the GSM8K test set
2. Decode all continuous thoughts to vocabulary space
3. Validate intermediate computations against reference CoT
4. Save outputs separately for correct/incorrect predictions
5. Generate interactive HTML visualizations
6. Create CSV exports for spreadsheet analysis

## Output Structure

After running the analysis, you'll find:

```
outputs/section5_analysis/
└── section5_run_YYYYMMDD_HHMMSS/
    ├── summary_statistics.json           # Overall metrics
    ├── interpretability_analysis.csv     # Spreadsheet-friendly data
    ├── interpretability_visualization.html # Interactive visualization
    ├── interpretability_visualization.txt  # Text report
    ├── correct_predictions/
    │   └── predictions.json              # All correct predictions
    └── incorrect_predictions/
        └── predictions.json              # All incorrect predictions
```

### Output Schema

Each prediction entry contains:

```json
{
  "question_id": 0,
  "question_text": "...",
  "reference_cot": "«10÷5=2»«2×2=4»«6×4=24»",
  "ground_truth_answer": 24.0,
  "predicted_answer": 24.0,
  "is_correct": true,

  "continuous_thoughts": [
    {
      "iteration": 0,
      "type": "initial",
      "topk_indices": [20, 767, 1821, ...],
      "topk_probs": [0.15, 0.12, 0.08, ...],
      "topk_decoded": [" 20", " 7", "7", ...]
    },
    ...
  ],

  "reference_steps": ["10÷5=2", "2×2=4", "6×4=24"],
  "decoded_steps": [" 20", " 7", " 27"],
  "step_correctness": [false, false, false],
  "overall_step_accuracy": 0.0
}
```

## Key Metrics

### Overall Results (from Summary Statistics)

- **Accuracy**: Overall prediction accuracy on test set
- **Correct/Incorrect Counts**: Number of predictions in each category

### Step Correctness Analysis

For correctly predicted answers, we analyze:
- **Problems with 1 step**: % of decoded thoughts matching reference
- **Problems with 2 steps**: % match rate
- **Problems with 3 steps**: % match rate
- etc.

This reproduces Table 3 from the paper:

| Total Steps | Accuracy |
|-------------|----------|
| 1           | 97.1%    |
| 2           | 83.9%    |
| 3           | 75.0%    |

## Advanced Usage

### Custom Analysis

You can run the analysis script directly with custom parameters:

```bash
python section5_analysis.py \
    --data_name "zen-E/GSM8k-Aug" \
    --model_name_or_path gpt2 \
    --ckpt_dir models/CODI-gpt2 \
    --batch_size 32 \
    --inf_latent_iterations 6 \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --use_prj True \
    --prj_dim 768 \
    --greedy True
```

### Visualization Only

If you already have results, generate visualizations separately:

```bash
python visualize_interpretability.py \
    --input_dir outputs/section5_analysis/section5_run_YYYYMMDD_HHMMSS \
    --max_examples 100 \
    --output_name my_visualization
```

### Different Datasets

To analyze on OOD datasets:

```bash
# SVAMP
python section5_analysis.py --data_name "svamp" ...

# GSM-Hard
python section5_analysis.py --data_name "gsm-hard" ...

# MultiArith
python section5_analysis.py --data_name "multi-arith" ...
```

## Interpreting Results

### Continuous Thoughts

Each continuous thought is decoded to vocabulary space. Look for:
- **Top-1 token**: Most likely token (highlighted in visualizations)
- **Top-K tokens**: Alternative interpretations
- **Numerical patterns**: Whether decoded tokens contain intermediate results

### Step Correctness

The validator checks if:
1. Decoded continuous thoughts contain numerical values
2. These values match intermediate results from reference CoT
3. The sequence of operations is preserved

**Note**: This is a heuristic analysis. The model may encode information differently than exact token-level CoT steps.

### Attention Patterns

(Future enhancement) Will show which question tokens each continuous thought attends to most strongly.

## Reproducing Paper Results

To reproduce Table 3 from Section 5.1:

1. Run analysis on GSM8K test set (done automatically)
2. Check `summary_statistics.json` → `step_correctness_analysis`
3. Compare with paper's reported values:
   - 1 step: 97.1% (paper) vs. your result
   - 2 steps: 83.9% (paper) vs. your result
   - 3 steps: 75.0% (paper) vs. your result

Expected variance: ±2-3% due to implementation differences

## Troubleshooting

### Model Not Found

```
ERROR: Checkpoint directory not found
```

**Solution**: Download the model first:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='zen-E/CODI-gpt2', local_dir='models/CODI-gpt2')"
```

### CUDA Out of Memory

**Solution**: Reduce batch size:
```bash
# Edit scripts/run_section5_analysis.sh
BATCH_SIZE=16  # or 8
```

### Python Version

Requires Python 3.10+. If you see NetworkX errors:
```bash
python3.12 -m venv env
```

## Citation

If you use these scripts, please cite both the original CODI paper and acknowledge this reproduction:

```bibtex
@article{shen2025codi,
  title={CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation},
  author={Zhenyi Shen and Hanqi Yan and Linhai Zhang and Zhanghao Hu and Yali Du and Yulan He},
  year={2025},
  journal={arXiv preprint arxiv:2502.21074}
}
```

## Contact

For issues or questions about this Section 5 reproduction:
- Open an issue in the CoT_Exploration repository
- Refer to the original CODI repository: https://github.com/zhenyi4/codi
