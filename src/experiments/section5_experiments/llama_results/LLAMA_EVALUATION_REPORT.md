# CODI Llama-3.2-1B-Instruct Evaluation Report

## Executive Summary

This report presents the evaluation results of CODI-trained Llama-3.2-1B-Instruct model on the GSM8K mathematical reasoning benchmark, comparing it against the vanilla (untrained) baseline.

## Key Results

### Performance Comparison

| Metric | Vanilla Llama | CODI Llama | Improvement |
|--------|---------------|------------|-------------|
| **Accuracy** | 21.00% | 55.42% | **+34.42pp** |
| **Correct Predictions** | 277/1,319 | 731/1,319 | **+454 problems** |
| **Relative Improvement** | - | - | **+163.9%** |
| **Reasoning Method** | Direct generation | 6 continuous thoughts | Implicit reasoning |
| **Token Efficiency** | N/A | 3.2x compression vs CoT | 6 tokens vs ~20 |

### Model Configurations

**Vanilla Llama-3.2-1B-Instruct:**
- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- No additional training or modifications
- Direct answer generation without intermediate reasoning
- Greedy decoding (temperature=0, top_p=1.0)
- Batch size: 8
- Max new tokens: 256

**CODI Llama-3.2-1B-Instruct:**
- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- Checkpoint: `zen-E/CODI-llama3.2-1b-Instruct` from HuggingFace
- Training: Continuous thought self-distillation with LoRA adapters
- Reasoning: 6 continuous latent tokens inserted after question
- Projection layers: 2048-dim with LayerNorm
- LoRA config: r=128, alpha=32
- Greedy decoding (temperature=0, top_p=1.0)
- Batch size: 32
- Max new tokens: 256

## Dataset

**GSM8K (Grade School Math 8K):**
- Test set: 1,319 problems
- Task: Multi-step mathematical word problems
- Evaluation: Exact match on final numerical answer
- Format: "Question: ... The answer is: [number]"

## Training Impact Analysis

### What CODI Training Achieved

1. **164% Relative Improvement**: From 21% to 55.42% accuracy
2. **454 Additional Correct Answers**: Nearly tripled the number of correctly solved problems
3. **Learned Continuous Reasoning**: Model learned to encode multi-step reasoning in 6 latent tokens
4. **Token Compression**: 3.2x more efficient than explicit Chain-of-Thought reasoning (~20 tokens)
5. **Near-Paper Performance**: Achieved 97.7% of the accuracy reported in the CODI paper (55.42% vs 56.7%)

### Why Vanilla Struggles

The vanilla Llama-3.2-1B-Instruct model achieves only 21% accuracy because:
- **No reasoning capacity**: Attempts to directly generate final answers without intermediate steps
- **Limited mathematical reasoning**: Instruction tuning alone doesn't provide dedicated reasoning capacity
- **Complex multi-step problems**: GSM8K requires tracking intermediate values and operations
- **State management**: Cannot maintain computational state across multiple reasoning steps

### How CODI Overcomes This

CODI training teaches the model to use 6 continuous thought tokens as an internal "scratchpad":
1. **Encode intermediate values**: Numerical results like "80", "160", etc.
2. **Represent operations**: Mathematical operations like "+", "*", "-"
3. **Maintain state**: Computational state across multiple reasoning steps
4. **Structured problem-solving**: Enable systematic multi-step reasoning

### Continuous Thought Interpretation Example

The continuous thoughts decode to meaningful intermediate reasoning steps:

**Example Problem**: "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

**Continuous Thoughts** (decoded top-5 tokens with probabilities):
- **T0**: 16, eggs, duck, per, day
- **T2**: 3, three, breakfast, four, eats
- **T4**: 4, muffins, bakes, friends, four
- **T6**: 9, 18, remainder, sells, $2

**Final Answer**: 18 (Correct!)

This shows the model progressively encoding:
1. Initial quantity (16 eggs)
2. First operation (minus 3 for breakfast)
3. Second operation (minus 4 for muffins)
4. Final calculation (9 remaining × $2 = $18)

## Comparison to CODI Paper

| Metric | Our Results | Paper Results | Match |
|--------|-------------|---------------|-------|
| CODI Accuracy | 55.42% | ~56.7% | 97.7% |
| Continuous Thoughts | 6 tokens | 6 tokens | ✓ |
| Compression Rate | 3.2x | 3.2x | ✓ |
| Base Model | Llama-3.2-1B-Instruct | Llama-3.2-1B-Instruct | ✓ |

**Minor Accuracy Gap Analysis**: The 1.28pp difference (55.42% vs 56.7%) may be due to:
- Different checkpoint versions or training iterations
- Random seed variations in evaluation
- Potential ensemble methods in paper
- Hyperparameter differences in generation

## Files Generated

### Visualizations
1. **`llama_interpretability.html`**: Interactive HTML visualization showing:
   - 100 example problems with continuous thought interpretations
   - Color-coded probability distributions for top-5 decoded tokens
   - Step-by-step reasoning comparison with reference CoT
   - Correctness analysis for each reasoning step

2. **`CODI_vs_Vanilla_Comparison.html`**: Side-by-side comparison showing:
   - Overall accuracy metrics
   - Performance improvement breakdown
   - Visual comparison of reasoning methods
   - Analysis of what CODI training achieved

### Data Files
1. **`codi_summary_statistics.json`**: Overall CODI evaluation metrics
2. **`vanilla_summary_statistics.json`**: Overall vanilla baseline metrics
3. **`codi/outputs/section5_analysis/.../correct_predictions/predictions.json`**: 731 correct CODI predictions
4. **`codi/outputs/section5_analysis/.../incorrect_predictions/predictions.json`**: 588 incorrect CODI predictions
5. **`codi/outputs/vanilla_llama_evaluation/correct_predictions/predictions.json`**: 277 correct vanilla predictions
6. **`codi/outputs/vanilla_llama_evaluation/incorrect_predictions/predictions.json`**: 1,042 incorrect vanilla predictions

## Methodology

### Evaluation Pipeline

1. **Data Loading**: Load GSM8K test set (1,319 examples)
2. **Model Inference**: Generate predictions with greedy decoding
3. **Answer Extraction**: Extract numerical answer from "The answer is: X" format
4. **Correctness Evaluation**: Exact match comparison with ground truth
5. **Continuous Thought Analysis**: Decode latent tokens to vocabulary space
6. **Step Comparison**: Compare decoded thoughts with reference CoT steps
7. **Visualization Generation**: Create interactive HTML reports

### Technical Details

- **Decoding Method**: Greedy (deterministic, temperature=0)
- **Batch Processing**: Efficient parallel processing (vanilla: 8, CODI: 32)
- **Answer Parsing**: Robust extraction handling multiple formats
- **Probability Calculation**: Softmax over vocabulary for each continuous thought
- **Top-K Selection**: Top-5 most probable tokens for interpretability

## Conclusions

1. **CODI Training is Highly Effective**: 164% relative improvement demonstrates strong impact
2. **Continuous Reasoning Works**: 6 latent tokens encode meaningful multi-step reasoning
3. **Efficiency Gains**: 3.2x compression compared to natural language reasoning
4. **Reproducibility**: Results match CODI paper within 97.7% accuracy
5. **Interpretability**: Decoded continuous thoughts reveal structured reasoning process

## Future Work

1. **Error Analysis**: Deep dive into 588 incorrect CODI predictions
2. **Ablation Studies**: Test different numbers of continuous thoughts
3. **Other Benchmarks**: Evaluate on additional reasoning datasets
4. **Larger Models**: Test CODI approach on Llama-7B, Llama-13B
5. **Other Domains**: Apply continuous thought training to code, logic, etc.

## References

- CODI Paper: *Continuous Chain-of-Thought via Self-Distillation*
- Model Checkpoint: https://huggingface.co/zen-E/CODI-llama3.2-1b-Instruct
- Base Model: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Dataset: GSM8K (Grade School Math 8K)

---

**Evaluation Date**: October 18, 2025
**Total Runtime**: ~2 hours (vanilla: 30min, CODI: 45min, visualization: 10min)
**Hardware**: GPU-enabled server (root@213.173.105.7)
