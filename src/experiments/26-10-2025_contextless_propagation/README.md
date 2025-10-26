# Contextless Layer Propagation Decoding

**Date:** October 26, 2025
**Experiment:** Layer-wise contextless propagation for understanding intermediate layer outputs

## Overview

This experiment implements a novel approach to decoding intermediate transformer layer activations. Instead of directly decoding layer k hidden states via unembedding (logit lensing), we propagate activations through subsequent layers **in isolation**, without contextual influence from other token positions.

## Motivation

### Previous Approach: Direct Logit Lensing

In the CoT layer decoding experiment (`25-10-2025_cot_layer_decoding`), we applied direct logit lensing:
- Take hidden state at layer k
- Pass through language model head (unembedding)
- Decode top-k vocabulary tokens

**Limitation:** This shows what layer k immediately represents, but not what it ultimately leads to.

### New Approach: Contextless Propagation

Now we ask: **What output do layer k activations lead to when processed through remaining layers, absent contextual influence?**

Process:
1. Extract hidden state from layer k
2. Pass through layers k+1, k+2, ..., final layer
3. **Critical:** Process token in isolation (no attention to other positions)
4. Decode final layer output via cosine similarity to vocabulary embeddings

### Why This Matters

- **Residual stream analysis:** Understand transformation through layer stack
- **Context-free semantics:** What does layer k encode independent of surrounding tokens?
- **Comparison to direct decoding:** How does intermediate processing change representation?

## Methodology

### Architecture Implementation

```
Hidden state at layer k [1, hidden_dim]
         ↓
[Isolated forward pass through layers k+1 to final]
    - No attention mask (no positional context)
    - Single token processing
    - No KV cache from other positions
         ↓
Final layer hidden state [1, hidden_dim]
         ↓
[Cosine similarity with vocabulary embeddings]
         ↓
Top-k tokens by similarity
```

### Key Differences from Logit Lensing

| Aspect | Logit Lensing | Contextless Propagation |
|--------|---------------|-------------------------|
| Decoding method | Direct unembedding | Full layer propagation |
| Context | N/A (single layer) | Explicitly removed |
| Output | Immediate representation | Final transformed output |
| Metric | Softmax probabilities | Cosine similarities |

### Models Analyzed

- **GPT-2** (124M parameters, 13 layers)
  - Status: ✓ Completed (10 examples)
- **LLaMA-3.2-1B** (1B parameters, 23 layers)
  - Status: ⧗ Pending (requires HF authentication)

### Data

- Dataset: GSM8K test set
- Examples per model: 10
- Thought iterations: 7 (BOT + 6 continuous iterations)
- Top-k tokens per layer: 5

## Implementation Details

### Propagating Through Layers Without Context

**GPT-2:**
```python
def propagate_through_layers_contextless(hidden_state, model, start_layer, model_type="gpt2"):
    blocks = model.transformer.h
    h = hidden_state.unsqueeze(0)  # [1, 1, hidden_dim]

    for layer_idx in range(start_layer, len(blocks)):
        block = blocks[layer_idx]
        h = block(h, attention_mask=None, use_cache=False)[0]  # No context!

    return h[:, -1, :]
```

**Key:** `attention_mask=None` means no attention to other positions - truly isolated processing.

### Vocabulary Similarity Decoding

Instead of softmax over logits, we use cosine similarity:

```python
def decode_via_vocab_similarity(hidden_state, embedding_layer, topk=10):
    # Normalize both hidden state and vocabulary embeddings
    hidden_norm = F.normalize(hidden_state, p=2, dim=-1)
    vocab_norm = F.normalize(embedding_layer.weight, p=2, dim=-1)

    # Compute similarities
    similarities = torch.matmul(hidden_norm, vocab_norm.T)

    # Get top-k
    topk_vals, topk_idx = torch.topk(similarities, k=topk)
    return topk_idx, topk_vals
```

This measures semantic similarity rather than likelihood.

## Results Structure

```
results/
├── gpt2/
│   └── contextless_propagation_run_20251026_160813/
│       └── contextless_propagation_results.json
└── llama/
    └── (pending)
```

### Output Format

```json
{
  "example_index": 0,
  "question": "...",
  "ground_truth_answer": "...",
  "generated_text": "...",
  "predicted_answer": 18.0,
  "bot_position": 65,
  "num_continuous_thoughts": 7,
  "continuous_thoughts": [
    {
      "iteration": 0,
      "type": "initial_bot",
      "position": 65,
      "layers": [
        {
          "layer_index": 0,
          "topk_indices": [262, 257, 290, 13, 12],
          "topk_similarities": [0.055, 0.048, 0.036, 0.031, 0.030],
          "topk_tokens": [" the", " a", " and", ".", "-"]
        },
        ...
      ]
    },
    ...
  ]
}
```

Note: Uses `topk_similarities` (cosine similarity) instead of `topk_probs` (softmax probabilities).

## Usage

### Running Contextless Propagation

```bash
python scripts/decode_contextless_propagation.py \
    --model_name_or_path gpt2 \
    --ckpt_dir /workspace/CoT_Exploration/models/CODI-gpt2 \
    --data_name gsm8k \
    --output_dir ./results/gpt2 \
    --batch_size 1 \
    --topk 5 \
    --max_examples 10 \
    --num_thought_iterations 6
```

### For LLaMA (requires HF token):

```bash
export HF_TOKEN=your_token_here
python scripts/decode_contextless_propagation.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --ckpt_dir /workspace/CoT_Exploration/models/CODI-llama3.2-1b \
    --data_name gsm8k \
    --output_dir ./results/llama \
    --batch_size 1 \
    --topk 5 \
    --max_examples 10 \
    --num_thought_iterations 6
```

## Comparison to Previous Experiment

| Experiment | Decoding Method | Output Type | Measures |
|------------|----------------|-------------|----------|
| **25-10-2025 CoT Layer Decoding** | Direct unembedding | Softmax probabilities | What each layer immediately represents |
| **26-10-2025 Contextless Propagation** | Propagated + vocab similarity | Cosine similarities | What each layer leads to without context |

### Analysis Questions

1. **How does propagation change token predictions?**
   - Compare top-k tokens between direct and propagated decoding
   - Are early layers dramatically different after propagation?

2. **What is the role of context?**
   - Contextless propagation removes positional influence
   - How much do representations depend on surrounding tokens?

3. **Layer-wise evolution:**
   - Do predictions converge across layers?
   - Which layers show most dramatic transformation?

## Files

- `scripts/decode_contextless_propagation.py` - Main analysis script
- `results/gpt2/contextless_propagation_run_*/contextless_propagation_results.json` - GPT-2 results

## Technical Notes

### Why Cosine Similarity Instead of Softmax?

After propagating through layers without context, the hidden states may have different scale properties than normal forward passes. Cosine similarity:
- Normalizes magnitude differences
- Measures pure directional similarity
- More robust to scale variations

### Isolated Token Processing

The key to "contextless" propagation is ensuring no information flows between positions:
- No attention mask → each position attends only to itself
- No past_key_values from other tokens
- Single token input `[1, 1, hidden_dim]`

This reveals what the hidden state encodes **intrinsically**, separate from contextual modulation.

## Future Work

1. **Complete LLaMA experiment** with HF authentication
2. **Create comparative visualization** showing:
   - Direct decoding (previous experiment)
   - Contextless propagation (this experiment)
   - Side-by-side comparison per layer/position
3. **Analyze differences quantitatively:**
   - Token overlap between methods
   - Similarity correlation across layers
   - Position-specific patterns

## References

- Previous experiment: `25-10-2025_cot_layer_decoding`
- Logit lensing: Interpretability research on transformer internals
- CODI architecture: `/workspace/CoT_Exploration/discretization_experiment/src/model.py`
