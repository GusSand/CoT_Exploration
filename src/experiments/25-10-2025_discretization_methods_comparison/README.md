# Chain-of-Thought Discretization Methods Comparison

## Overview

This experiment evaluates three different discretization approaches for Chain-of-Thought (CoT) in CODI models:
1. **Vanilla (Continuous)**: Standard CODI with continuous latent thought representations
2. **Full Discretization**: Discretize all thought tokens during the reasoning process
3. **Post-hoc Discretization**: Generate continuous CoT first, then discretize each position and recalculate activations

## Models Evaluated

- **CODI-GPT2**: [zen-E/CODI-gpt2](https://huggingface.co/zen-E/CODI-gpt2)
- **CODI-Llama3.2-1B**: [zen-E/CODI-llama3.2-1b-Instruct](https://huggingface.co/zen-E/CODI-llama3.2-1b-Instruct)

## Dataset

- **Source**: CoT-dependent problems from GSM8K
- **Path**: `/workspace/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_all.json`
- **Total examples**: 285 (208 with valid ground truth)
- **Description**: Subset of GSM8K math problems identified as requiring chain-of-thought reasoning

## Results Summary

### Accuracy Comparison

| Method | CODI-GPT2 | CODI-Llama3.2-1B |
|--------|-----------|------------------|
| **Vanilla (Continuous)** | **39.42%** | **55.29%** |
| Full Discretization | 11.06% | 30.77% |
| Post-hoc Discretization | 14.42% | 37.50% |

### Key Findings

1. **Continuous CoT Outperforms Discretization**: Vanilla continuous reasoning achieves the highest accuracy for both models
   - GPT-2: 39.42% (baseline)
   - Llama: 55.29% (baseline)

2. **Llama Superior to GPT-2**: CODI-Llama3.2-1B significantly outperforms CODI-GPT2 across all discretization methods
   - Average improvement: ~17-20 percentage points

3. **Post-hoc Better Than Full Discretization**: For both models, post-hoc discretization outperforms full discretization
   - GPT-2: 14.42% vs 11.06% (+30% relative)
   - Llama: 37.50% vs 30.77% (+22% relative)

4. **Performance Degradation from Discretization**:
   - GPT-2 Full: -71.9% relative performance loss
   - GPT-2 Post-hoc: -63.4% relative performance loss
   - Llama Full: -44.3% relative performance loss
   - Llama Post-hoc: -32.2% relative performance loss

5. **Inference Speed**: GPT-2 is ~4-8x faster than Llama
   - GPT-2: 11-18ms per example
   - Llama: 55-114ms per example

## Methodology

### Discretization Approaches

#### 1. Vanilla (Continuous)
Standard CODI inference with continuous latent embeddings throughout the thought chain.

#### 2. Full Discretization
At each thought position:
1. Compute token probabilities from continuous embedding
2. Select argmax token
3. Replace continuous embedding with scaled token embedding
4. Norm-controlled: Scale token embedding to match L2 norm of original continuous vector
5. Continue forward pass with discretized representation

#### 3. Post-hoc Discretization
1. **Phase 1**: Generate complete vanilla continuous CoT, save all hidden states
2. **Phase 2**: Starting from question encoding, discretize each saved hidden state position:
   - Use original continuous hidden state
   - Discretize to nearest token (norm-controlled)
   - Update KV cache with discretized version
   - Move to next position (using original continuous state, not discretized output)
3. **Phase 3**: Generate answer using recalculated KV cache

### Configuration

#### CODI-GPT2
- Base model: `openai-community/gpt2`
- Checkpoint: `/workspace/CoT_Exploration/models/CODI-gpt2`
- LoRA rank: 128, alpha: 32
- Projection dimension: 768
- Thought iterations: 6

#### CODI-Llama3.2-1B
- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- Checkpoint: `/workspace/CoT_Exploration/models/CODI-llama3.2-1b`
- LoRA rank: 128, alpha: 32
- Projection dimension: 2048
- Thought iterations: 6

### Batch Processing
- Batch size: 16 examples
- Checkpoint saving: Every 10 batches
- Device: CUDA with bfloat16 precision

## Files and Structure

```
src/experiments/discretization_methods_comparison/
├── README.md                                    # This file
├── run_discretization_custom_dataset.py         # Main evaluation script
├── visualize_discretization_comparison.py       # Visualization script
├── results/
│   ├── gpt2/
│   │   ├── final_results.json                   # Full GPT-2 results
│   │   └── run.log                              # Execution log
│   ├── llama/
│   │   ├── final_results.json                   # Full Llama results
│   │   └── run.log                              # Execution log
└── figures/
    ├── discretization_comparison.png            # Multi-panel comparison
    ├── discretization_comparison_simple.png     # Simple bar chart
    └── timing_comparison.png                    # Inference speed comparison
```

## Usage

### Run Evaluation

```bash
# GPT-2 evaluation
python run_discretization_custom_dataset.py \
    --model_type gpt2 \
    --batch_size 16 \
    --output_dir ./results/gpt2

# Llama evaluation
python run_discretization_custom_dataset.py \
    --model_type llama \
    --batch_size 16 \
    --output_dir ./results/llama
```

### Generate Visualizations

```bash
python visualize_discretization_comparison.py \
    --gpt2_results ./results/gpt2/final_results.json \
    --llama_results ./results/llama/final_results.json \
    --output_dir ./figures
```

## Implications

1. **Continuous representations preserve critical reasoning information**: The substantial performance drop with discretization suggests continuous embeddings capture nuances important for multi-step reasoning

2. **Post-hoc discretization is more interpretable**: While still degrading performance, post-hoc discretization maintains the continuous reasoning chain for analysis while providing discrete interpretations

3. **Model capacity matters**: Llama's superior performance and smaller relative degradation from discretization suggests larger models may be more robust to discretization

4. **Speed-accuracy tradeoff**: While GPT-2 is much faster, the accuracy gap makes Llama more suitable for reasoning tasks

## Related Work

- Original dataset from activation patching experiments investigating CoT necessity
- Builds on norm-controlled discretization methodology
- Extends previous work on understanding CODI's latent reasoning process

## Citation

If you use this experiment or methodology, please cite:
- CODI paper: [Add citation]
- This experiment: Discretization Methods Comparison on CoT-Dependent GSM8K Problems

## Experiment Metadata

- **Date**: October 25, 2025
- **Total runtime**:
  - GPT-2: ~0.2 minutes
  - Llama: ~1.3 minutes
- **Hardware**: NVIDIA GPU (CUDA-enabled)
- **Dataset version**: llama_cot_all.json (285 examples, 208 valid)
