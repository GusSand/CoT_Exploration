# Tuned Lens for CODI

Decode continuous thought tokens into interpretable text using learned affine transformations.

## Overview

This experiment implements Tuned Lens to decode CODI's continuous thought representations into human-readable tokens. Unlike raw logit lens (direct unembedding), Tuned Lens learns layer-specific affine transformations that improve decoding accuracy and semantic quality.

## Quick Start

```bash
# Run full pipeline (LLaMA, post-MLP representation)
python run.py full-pipeline --config config.yaml

# View results
cat results/evaluation_summary.json
python run.py decode --problem-ids 42
```

## Project Structure

```
tuned_lens/
├── config.yaml                      # Configuration file
├── run.py                           # CLI entry point
├── collect_data.py                  # Story 1: Extract activations
├── model.py                         # Story 2: TunedLens & LogitLens models
├── train.py                         # Story 3: Training pipeline
├── evaluate.py                      # Story 4: Evaluation & metrics
├── decode_problems.py               # Story 5: Interpretability analysis
├── verify_operation_encoding.py    # Story 6: Operation verification
├── causal_intervention.py          # Story 7: Causal validation
├── position_specific_model.py      # Story 8: Position-specific extension
├── utils.py                         # Shared utilities
├── visualize.py                     # Visualization functions
├── data/                            # Training/test data (excluded from git)
├── models/                          # Trained models (excluded from git)
└── results/                         # Outputs (excluded from git)
    ├── decoded_problems/
    └── figures/
```

## Individual Steps

```bash
# 1. Collect training data (800 train / 200 test problems)
python run.py collect --num-problems 1000

# 2. Train Tuned Lens
python run.py train --config config.yaml

# 3. Evaluate
python run.py evaluate

# 4. Decode sample problems
python run.py decode --problem-ids 1,2,3

# 5. Verify operation encoding (Token 1 @ Layer 8)
python run.py verify-op

# 6. Causal intervention test
python run.py intervene --positions 1,4,5
```

## Configuration

Edit `config.yaml` to customize:
- **Model**: llama or gpt2
- **Representation**: pre_mlp or post_mlp
- **Training hyperparameters**: learning rate, batch size, epochs
- **Evaluation settings**: metrics, number of samples
- **W&B integration**: project name, tags

## Expected Results

### Success Criteria
- **Top-1 accuracy**: 60-70% (vs ~30% baseline)
- **KL divergence**: Significant reduction
- **Causal intervention**: <20% accuracy degradation
- **Semantic quality**: Meaningful decoded tokens

### Example Output
```
TUNED LENS PIPELINE COMPLETE

Logit Lens Baseline:
  Top-1 Accuracy: 31.2% ± 2.8%
  KL Divergence: 4.52

Tuned Lens:
  Top-1 Accuracy: 67.8% ± 3.1%  ✅ TARGET MET!
  KL Divergence: 2.14 (53% reduction)
  Improvement: +36.6% (p < 0.001, d = 2.41)

Operation Verification (Token 1 @ Layer 8):
  Addition detection: 78% accuracy
  Multiplication detection: 82% accuracy

Causal Intervention:
  Accuracy Drop (Tuned): 15.3%  ✅ ACCEPTABLE!
  Accuracy Drop (Logit): 58.7%
  Accuracy Drop (Random): 89.2%
```

## Troubleshooting

### Training not improving
- Check layer norm scale (drastically different between layers?)
- Visualize with PCA (is transform doing something meaningful?)
- Try without position-specific transforms first
- Check for collapse (all decode to same token?)
- Add L2 regularization if needed

### Memory issues
- Reduce batch size in `config.yaml`
- Process fewer layers at once
- Clear CUDA cache: `torch.cuda.empty_cache()`

### Model loading errors
- Verify checkpoint path in config
- Check CODI model is available at `~/codi_ckpt/llama_gsm8k/`
- Ensure correct model type (llama vs gpt2)

## Data Format

### Training Data (`data/train_data_llama_post_mlp.pt`)
```python
{
    'hidden_states': torch.Tensor,     # (N, 2048) - continuous thought activations
    'target_token_ids': torch.LongTensor,  # (N,) - ground truth tokens
    'metadata': {
        'problem_ids': List[str],       # Problem identifiers
        'layers': List[int],            # Layer indices (0-15)
        'positions': List[int],         # Token positions (0-5)
        'difficulties': List[str]       # Problem difficulty
    }
}
```

## Dependencies

```bash
pip install torch>=2.0.0 transformers>=4.30.0 wandb scikit-learn plotly scipy tqdm pyyaml
```

## Documentation

- **Experiment Report**: `docs/experiments/MM-DD_llama_gsm8k_tuned_lens.md`
- **Architecture**: `docs/architecture/tuned_lens_adr.md`
- **API Reference**: `docs/code/tuned_lens_api.md`

## References

- **Tuned Lens Paper**: [Seeing Transformers from the Inside](https://arxiv.org/abs/2303.08112)
- **CODI Paper**: [Continuous Chain-of-Thought via Self-Distillation](https://arxiv.org/abs/2502.21074)

## Status

**Branch**: `experiment/tuned-lens`
**Implementation**: In progress
**Estimated Completion**: 4-5 days (~36.5 hours)
