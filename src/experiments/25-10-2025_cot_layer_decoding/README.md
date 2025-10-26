# CoT Layer Decoding Analysis

**Date:** October 25, 2025
**Experiment:** Layer-wise logit lensing of continuous chain-of-thought hidden states

## Overview

This experiment performs layer-wise decoding of continuous chain-of-thought (CoT) hidden states during CODI generation. By capturing and decoding hidden states at each transformer layer during thought iterations, we can understand how different layers represent and process implicit reasoning steps.

## Motivation

CODI (Continuous Chain-of-Thought Distillation) uses continuous latent representations for chain-of-thought reasoning. Unlike discrete CoT which generates explicit reasoning tokens, CODI's thought process occurs in continuous hidden states. This experiment applies "logit lensing" - projecting hidden states from intermediate layers through the language model head - to decode what vocabulary tokens each layer is predicting during continuous thought iterations.

## Methodology

### Architecture Details

The experiment follows proper CODI architecture:

1. **BOT Token Handling**: Appends special BOT (beginning of thought) token to input sequences
2. **Continuous Thought Generation**: Uses `inputs_embeds` (not `input_ids`) for thought iterations
3. **Pre-Projection Decoding**: Decodes hidden states BEFORE the projection layer is applied
4. **Efficient Generation**: Utilizes KV caching via `past_key_values`
5. **Device/Dtype**: Runs on CUDA with bfloat16 precision

### Generation Flow

```
Input Question + BOT Token
    ↓
[Initial Forward Pass - capture all layer hidden states]
    ↓
Iteration 0: Decode BOT position across all layers
    ↓
[Apply Projection Layer]
    ↓
Iteration 1-6: Continuous thought iterations
    - Forward pass with inputs_embeds
    - Decode hidden states (pre-projection) across all layers
    - Apply projection for next iteration
    ↓
EOT Token → Answer Generation
```

### Models Analyzed

- **GPT-2** (124M parameters, 13 layers including embedding layer)
- **LLaMA-3.2-1B** (1B parameters, 23 layers including embedding layer)

### Data

- Dataset: GSM8K test set
- Examples per model: 10
- Thought iterations: 7 (initial BOT + 6 continuous iterations)
- Top-k tokens per layer: 5

## Results Structure

```
results/
├── gpt2/
│   ├── cot_layer_decoding_results.json    # Raw decoding results
│   └── gpt2_cot_visualization.html         # Interactive visualization
└── llama/
    ├── cot_layer_decoding_results.json
    └── llama_cot_visualization.html
```

### Visualization Format

The HTML visualizations present data in a table format:
- **Rows**: Transformer layers (L0 to L12 for GPT-2, L0 to L22 for LLaMA)
- **Columns**: Thought iteration positions (BOT, T1-T6)
- **Cells**: Top-5 decoded tokens with probabilities
- **Color Coding**: Darker blue = higher probability

## Usage

### Running Layer Decoding

```bash
python scripts/decode_cot_layers_corrected.py \
    --model_name_or_path gpt2 \
    --ckpt_dir /workspace/CoT_Exploration/models/CODI-gpt2 \
    --data_name gsm8k \
    --output_dir ./outputs \
    --batch_size 1 \
    --topk 5 \
    --max_examples 10 \
    --num_thought_iterations 6
```

For LLaMA:
```bash
python scripts/decode_cot_layers_corrected.py \
    --model_name_or_path meta-llama/Llama-3.2-1B \
    --ckpt_dir /workspace/CoT_Exploration/models/CODI-llama3.2-1b \
    --data_name gsm8k \
    --output_dir ./outputs \
    --batch_size 1 \
    --topk 5 \
    --max_examples 10 \
    --num_thought_iterations 6
```

### Generating Visualizations

```bash
python scripts/visualize_cot_layers.py \
    --input_json path/to/cot_layer_decoding_results.json \
    --output_html output_visualization.html \
    --max_examples 10 \
    --topk 5
```

## Key Implementation Notes

### Critical Architecture Requirements

1. **BOT Token Addition**:
   ```python
   if training_args.remove_eos:
       bot_tensor = torch.tensor([codi_model.bot_id], dtype=torch.long)
   else:
       bot_tensor = torch.tensor([tokenizer.eos_token_id, codi_model.bot_id], dtype=torch.long)
   input_ids = torch.cat((input_ids, bot_tensor), dim=1)
   ```

2. **Continuous Thought Iteration**:
   ```python
   outputs = codi_model.codi(
       inputs_embeds=latent_embd,  # CRITICAL: use inputs_embeds
       use_cache=True,
       output_hidden_states=True,
       past_key_values=past_key_values
   )
   ```

3. **Pre-Projection Decoding**:
   ```python
   latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
   # Decode BEFORE applying projection
   layers_decoded = decode_hiddenstates_all_layers(...)
   # Then apply projection for next iteration
   if training_args.use_prj:
       latent_embd = codi_model.prj(latent_embd_pre_proj)
   ```

4. **Model Setup**:
   ```python
   codi_model = codi_model.to('cuda')
   codi_model.to(torch.bfloat16)
   codi_model.eval()
   ```

## Files

- `scripts/decode_cot_layers_corrected.py` - Main analysis script implementing proper CODI architecture
- `scripts/visualize_cot_layers.py` - HTML table visualization generator
- `results/gpt2/` - GPT-2 results and visualization
- `results/llama/` - LLaMA results and visualization

## References

- Based on CODI architecture from `/workspace/CoT_Exploration/section5_experiments/section5_analysis.py`
- Logit lensing methodology similar to interpretability research on transformer internals

## Notes

- The experiment requires CUDA and sufficient GPU memory
- Results are deterministic with fixed seed (seed=11)
- Batch size must be 1 for hidden state capture during generation
