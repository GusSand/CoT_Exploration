# Section 5 Analysis - Extended Similarity Metrics

## Overview

This extension adds geometric similarity metrics to the Section 5 interpretability analysis, allowing us to quantify the relationship between continuous thought activations and token embeddings beyond just decoding probabilities.

## Motivation

**Key Question**: Does an activation that decodes to token X actually resemble token X's embedding?

The original analysis only computed **decoding probabilities** via softmax:
- P(token | activation) = softmax(W @ activation)

However, high probability doesn't guarantee geometric similarity! An activation can:
- Decode to token X (high probability)
- While being geometrically distant from X's embedding

## New Metrics

### 1. Cosine Similarity
```
cos_sim = (activation · token_embedding) / (||activation|| × ||token_embedding||)
```
- **Range**: -1 to 1 (typically 0 to 1 for embeddings)
- **Interpretation**: Directional alignment, independent of magnitude
- **High value**: Activation and token embedding point in similar directions

### 2. Normalized Euclidean Distance
```
norm_l2_dist = ||activation_norm - token_embedding_norm||₂
```
where both vectors are normalized to unit length first.

- **Range**: 0 to 2 (on unit sphere)
- **Interpretation**: Geometric distance after normalization
- **Low value**: Activation and token embedding are geometrically close

### Mathematical Relationship

These metrics are related but provide complementary perspectives:
```
norm_l2_dist ≈ sqrt(2 - 2×cos_sim)
```

While mathematically dependent, both are useful:
- **Cosine similarity**: Natural interpretation as directional alignment
- **Normalized L2 distance**: Intuitive geometric distance measure

## Files

### Extended Analysis Script
**`section5_analysis_extended.py`**

Key changes from original:
1. New function `compute_similarity_metrics()` - computes cosine sim and L2 distance
2. Extended `decode_continuous_thought_extended()` - adds similarity metrics to output
3. Modified `PredictionOutput` dataclass - stores extended metrics
4. New `analyze_similarity_metrics()` - aggregate statistics

Output structure (per continuous thought):
```json
{
  "iteration": 0,
  "type": "continuous_thought",
  "topk_indices": [245, 532, ...],
  "topk_probs": [0.342, 0.187, ...],
  "topk_decoded": [" 10", " 5", ...],
  "cosine_similarities": [0.856, 0.743, ...],  // NEW
  "norm_l2_distances": [0.537, 0.713, ...]     // NEW
}
```

### Extended Visualization Script
**`visualize_interpretability_extended.py`**

Generates HTML visualizations showing:
- **Blue gradient**: Decoding probability (darker = higher)
- **Green gradient**: Cosine similarity (darker = more similar)
- **Red gradient**: Normalized L2 distance (darker = closer)

Allows visual comparison of:
- What token is most likely (probability)
- What token is directionally aligned (cosine)
- What token is geometrically closest (L2 distance)

## Usage

### Running Extended Analysis

```bash
cd /workspace/CoT_Exploration

# For Llama model
python section5_experiments/section5_analysis_extended.py \
    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
    --lora_init True \
    --lora_r 128 \
    --lora_alpha 32 \
    --ckpt_dir ./models/CODI-llama3.2-1b \
    --data_name zen-E/GSM8k-Aug \
    --batch_size 16 \
    --inf_latent_iterations 6 \
    --use_prj True \
    --prj_dim 2048 \
    --remove_eos True \
    --greedy True \
    --bf16 True

# For GPT-2 model
python section5_experiments/section5_analysis_extended.py \
    --model_name_or_path openai-community/gpt2 \
    --lora_init True \
    --lora_r 128 \
    --lora_alpha 32 \
    --ckpt_dir ./models/CODI-gpt2 \
    --data_name zen-E/GSM8k-Aug \
    --batch_size 16 \
    --inf_latent_iterations 6 \
    --use_prj True \
    --prj_dim 768 \
    --remove_eos True \
    --greedy True \
    --bf16 True
```

### Generating Visualizations

```bash
# Visualize extended results
python section5_experiments/visualize_interpretability_extended.py \
    --input_dir outputs/section5_analysis_extended/section5_extended_YYYYMMDD_HHMMSS \
    --max_examples 50 \
    --output_name interpretability_visualization_extended
```

## Output Files

```
outputs/section5_analysis_extended/section5_extended_YYYYMMDD_HHMMSS/
├── correct_predictions/
│   └── predictions.json              # Correct predictions with extended metrics
├── incorrect_predictions/
│   └── predictions.json              # Incorrect predictions with extended metrics
├── summary_statistics.json           # Aggregate statistics including similarity metrics
├── interpretability_analysis_extended.csv  # Per-example metrics for analysis
├── interpretability_visualization_extended.html  # Interactive HTML visualization
└── interpretability_visualization_extended.txt   # Text report
```

## Example Analysis Questions

With these extended metrics, you can now investigate:

1. **Probability vs. Similarity Discrepancy**
   - Are high-probability tokens also high-similarity?
   - Do correct predictions have higher geometric similarity?

2. **Embedding Space Structure**
   - How far are activations from token embeddings on average?
   - Does cosine similarity correlate with task accuracy?

3. **Model Interpretability**
   - When model gets answer correct, are continuous thoughts geometrically close to meaningful tokens?
   - Can we identify "reasoning artifacts" where activation decodes to one token but is closer to another?

## CSV Output Format

The extended CSV includes average similarity metrics for easy analysis:

```csv
question_id,is_correct,ground_truth,predicted,step_accuracy,avg_cosine_sim_top1,avg_norm_l2_dist_top1,top1_continuous_thought_0
0,True,18,18,1.0,0.8567,0.5432," 10"
1,False,24,32,0.5,0.7234,0.6789," 8"
...
```

## Performance Notes

- **Additional computation**: ~10-15% overhead vs original analysis
- **Memory**: Stores 2 additional float arrays per continuous thought (topk × 2)
- **Recommended batch size**: 8-16 depending on GPU memory

## Implementation Details

### Token Embedding Extraction

```python
embedding_layer = model.get_embd(model.codi, model.model_name)
token_embeddings = embedding_layer(topk_token_ids)  # [topk, hidden_dim]
```

### Normalization

```python
# L2 normalization to unit sphere
activation_norm = F.normalize(activation.unsqueeze(0), p=2, dim=1)
token_embeddings_norm = F.normalize(token_embeddings, p=2, dim=1)
```

### Cosine Similarity

```python
# Dot product of normalized vectors
cosine_similarities = torch.mm(activation_norm, token_embeddings_norm.t()).squeeze(0)
```

### Normalized L2 Distance

```python
# Euclidean distance on unit sphere
differences = activation_norm - token_embeddings_norm
norm_l2_distances = torch.norm(differences, p=2, dim=1)
```

## Comparison with Original

| Feature | Original | Extended |
|---------|----------|----------|
| Decoding Probabilities | ✓ | ✓ |
| Top-K Tokens | ✓ | ✓ |
| Cosine Similarity | ✗ | ✓ |
| Normalized L2 Distance | ✗ | ✓ |
| Step Validation | ✓ | ✓ |
| HTML Visualization | Basic | Multi-metric |
| CSV Export | Basic | Enhanced |

## Future Extensions

Potential additions:
1. **Un-normalized L2 distance**: Compare magnitude differences
2. **Angular distance**: acos(cos_sim) for interpretable angle
3. **Rank correlation**: Spearman correlation between probability and similarity rankings
4. **Per-layer analysis**: Compute metrics at different transformer layers

## References

- Original CODI paper Section 5: Interpretability Analysis
- Cosine similarity in NLP: https://en.wikipedia.org/wiki/Cosine_similarity
- Vector space models: Embedding geometry and semantics

## Citation

If you use this extension in your research:

```bibtex
@misc{codi_similarity_extension,
  title={Extended Similarity Metrics for CODI Interpretability Analysis},
  author={},
  year={2025},
  note={Extension of CODI Section 5 analysis with cosine similarity and normalized L2 distance}
}
```
