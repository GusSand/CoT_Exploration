# Operation-Specific Circuits Experiment

## Overview

This experiment investigates whether CODI's continuous thought representations encode operation-specific information. Specifically, we test if problems requiring different arithmetic operations (pure addition, pure multiplication, or mixed) have distinguishable continuous thought patterns.

**Hypothesis**: If CODI learns operation-specific circuits, then continuous thoughts should cluster by operation type, and we should be able to classify problems based on their thought representations.

## Experiment Design

### Operation Categories

Problems are classified into three categories based on their solution steps:

1. **Pure Addition**: Only addition/subtraction operations
2. **Pure Multiplication**: Only multiplication/division operations
3. **Mixed**: Both addition/multiplication operations

### Dataset Sizes

- **Full Dataset**: 8,792 problems from GSM8k (test + train splits)
- **Classified Dataset**: 600 problems (200 per category)
- **Prototype Dataset**: 60 problems (20 per category) for quick testing

### Analysis Pipeline

1. **Extraction**: Extract continuous thought representations from CODI model
   - Model: LLaMA-3.2-1B-Instruct fine-tuned with CODI
   - Layers: 3 layers (early=4, middle=8, late=14 out of 16 total)
   - Tokens: 6 [THINK] tokens per problem
   - Hidden dim: 2048

2. **Analysis**:
   - PCA clustering visualization
   - Classification (Logistic Regression, Random Forest, Neural Network)
   - Feature importance across tokens and layers
   - Within-group vs between-group similarity

## Files

### Python Scripts

- `download_gsm8k.py` - Download GSM8k dataset
- `classify_operations.py` - Classify problems by operation type
- `create_prototype_dataset.py` - Create 60-sample prototype dataset
- `extract_continuous_thoughts.py` - Extract CODI continuous thoughts
- `analyze_continuous_thoughts.py` - Comprehensive analysis
- `run_experiment.py` - Master script to run full pipeline

### Data Files

- `gsm8k_full.json` - Full GSM8k dataset (8,792 problems)
- `operation_samples_200.json` - Classified dataset (600 problems)
- `operation_samples_prototype_60.json` - Prototype dataset (60 problems)

### Results

- `results/continuous_thoughts_prototype_60.json` - Extracted thoughts (prototype)
- `results/continuous_thoughts_full_600.json` - Extracted thoughts (full) [to be generated]
- `results/analysis/` - Analysis visualizations and reports

## Usage

### Quick Start (Prototype)

Run the prototype experiment (60 samples, ~10 minutes):

```bash
python run_experiment.py --mode prototype
```

### Full Experiment

Run the full experiment (600 samples, ~60 minutes):

```bash
python run_experiment.py --mode full
```

### Analysis Only

Re-run analysis on existing data:

```bash
python run_experiment.py --mode analysis_only --data_path results/continuous_thoughts_prototype_60.json
```

### Individual Steps

```bash
# 1. Download GSM8k
python download_gsm8k.py

# 2. Classify by operation type (200 per category)
python classify_operations.py

# 3. Create prototype dataset (20 per category)
python create_prototype_dataset.py

# 4. Extract continuous thoughts
python extract_continuous_thoughts.py --test  # Test on single problem
# (Use run_experiment.py for full extraction)

# 5. Analyze
python analyze_continuous_thoughts.py --data_path results/continuous_thoughts_prototype_60.json
```

## Results (Prototype - 60 samples)

### Classification Accuracy

| Classifier | Accuracy | Precision | Recall | F1 |
|------------|----------|-----------|--------|-----|
| Logistic Regression | 0.667 | 0.722 | 0.667 | 0.656 |
| Random Forest | 0.667 | 0.722 | 0.667 | 0.656 |
| Neural Network | 0.583 | 0.690 | 0.583 | 0.542 |

**Baseline (chance)**: 33.3%

### Key Findings

1. **Above-Chance Classification**: All classifiers achieve ~60-67% accuracy, significantly above the 33% baseline, suggesting operation-specific information is encoded.

2. **PCA Clustering**: Visualizations show partial separation between operation types, with pure_addition and pure_multiplication forming distinct clusters while mixed problems overlap both.

3. **Layer Importance**: Middle layers (L8) show strongest discriminative power, consistent with the hypothesis that abstract reasoning happens in middle layers.

4. **Token Importance**: Later tokens (positions 5-6) show higher importance, suggesting operation-specific information accumulates during the thinking process.

5. **Similarity Patterns**: Within-group similarity is higher than between-group similarity, validating that problems of the same operation type have more similar representations.

## Model Details

- **Base Model**: meta-llama/Llama-3.2-1B-Instruct
- **Training**: CODI fine-tuning with 6 latent tokens
- **Architecture**: 16 layers, 2048 hidden dim
- **LoRA**: r=128, alpha=32
- **Projection**: 2048-dim projection layer

## Dependencies

```
torch
transformers
datasets
peft
numpy
matplotlib
seaborn
scikit-learn
```

## Citation

This experiment is based on the CODI framework:

```
@article{codi2024,
  title={CODI: Continuous Chain-of-Thought via Self-Distillation},
  author={...},
  year={2024}
}
```

## Related Experiments

- **Activation Patching**: `/src/experiments/activation_patching/` - Studies causal role of continuous thoughts
- **Token Threshold**: `/src/experiments/token_threshold/` - Analyzes information flow across tokens
- **CODI Attention**: `/src/experiments/codi_attention_interp/` - Interprets attention patterns in continuous space

## Future Directions

1. **Scaling**: Run full 600-sample experiment for more robust statistics
2. **Layer Analysis**: Examine all 16 layers to identify operation-specific circuits
3. **Intervention**: Use activation patching to test causal role of identified circuits
4. **Transfer**: Test if circuits generalize to other mathematical reasoning tasks
5. **Mechanistic Analysis**: Identify specific neurons/attention heads involved in operation detection
