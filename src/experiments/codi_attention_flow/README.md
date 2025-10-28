# CODI Attention Flow Analysis

Extract and analyze 6×6 attention patterns between continuous thought token positions to understand information flow during CODI's compressed reasoning.

**Status**: ✅ Phase 1 & 2 Complete (LLaMA + GPT-2 comparison)

---

## Quick Start

### Phase 1: LLaMA Analysis (100 problems)

```bash
cd /home/paperspace/dev/CoT_Exploration

# 1. Sample dataset
python src/experiments/codi_attention_flow/scripts/1_sample_dataset.py

# 2. Extract attention (15 seconds)
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/codi_attention_flow/scripts/2_extract_attention_6x6.py --model llama

# 3. Aggregate patterns
python src/experiments/codi_attention_flow/scripts/3_aggregate_attention.py --model llama

# 4. Create visualizations
python src/experiments/codi_attention_flow/scripts/4_visualize_heatmaps.py --model llama

# 5. Analyze hubs and flow
python src/experiments/codi_attention_flow/scripts/5_analyze_hubs_and_flow.py --model llama
```

**Total time**: ~2 minutes (mostly extraction)

### Phase 2: GPT-2 Analysis & Model Comparison

```bash
# Run GPT-2 pipeline
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/codi_attention_flow/scripts/2_extract_attention_6x6.py --model gpt2
python src/experiments/codi_attention_flow/scripts/3_aggregate_attention.py --model gpt2
python src/experiments/codi_attention_flow/scripts/4_visualize_heatmaps.py --model gpt2
python src/experiments/codi_attention_flow/scripts/5_analyze_hubs_and_flow.py --model gpt2

# Compute head metrics for both models
python src/experiments/codi_attention_flow/scripts/6_compute_head_metrics.py --model llama
python src/experiments/codi_attention_flow/scripts/6_compute_head_metrics.py --model gpt2

# Visualize critical heads
python src/experiments/codi_attention_flow/scripts/7_visualize_critical_heads.py --model llama
python src/experiments/codi_attention_flow/scripts/7_visualize_critical_heads.py --model gpt2

# Compare models
python src/experiments/codi_attention_flow/scripts/8_compare_models.py
```

**Total time**: ~3 minutes

---

## Key Findings

### Phase 1: LLaMA Hub-Centric Architecture

**Finding**: Position 0 acts as a **working memory hub** that accumulates information from all other positions.

### Phase 2: Model Comparison - Different Hub Strategies

**Key Discovery**: LLaMA and GPT-2 use **different hub positions** for continuous thought aggregation:

- **LLaMA (1B)**: Hub at Position 0 (1.18× uniform baseline)
  - Critical head: L4H5 (Hub Aggregator, composite=0.528)
  - Prefers middle layers (13/20 top heads)
  - Hub Aggregators: 9/20, Skip Connections: 11/20

- **GPT-2 (124M)**: Hub at Position 1 (1.63× uniform baseline)
  - Critical head: L0H3 (Multi-Purpose, composite=0.600)
  - Balanced early/middle (9/9 in top heads)
  - Hub Aggregators: 10/20, Skip Connections: 9/20

**Interpretation**: Both models use hub-centric reasoning, but GPT-2 shows stronger hub concentration (1.63× vs 1.18×) despite being a smaller model. This suggests different architectural strategies for information aggregation during compressed reasoning.

### Original Phase 1 Details

### Hub-Centric Architecture Discovered

**Finding**: Position 0 acts as a **working memory hub** that accumulates information from all other positions.

```
NOT Sequential:              BUT Hub-Centric:

CT5 → CT4 → CT3              CT1 ↘
              ↓                CT2 → CT0 (HUB)
           CT2 → CT1         CT3 ↗
                   ↓          CT4 ↗
                 CT0         CT5 (output)
```

**Evidence**:
- Position 0 receives most incoming attention (0.197 vs 0.167 uniform)
- ALL top 3 heads have max attention at [1→0]
- No sequential flow (avg 0.038 << 0.3 threshold)
- No skip connections (avg 0.029 << 0.1 threshold)

**Top Attention Heads**:
1. L0H9: max=0.804 at [1→0]
2. L4H26: max=0.781 at [1→0]
3. L6H2: max=0.756 at [1→0]

---

## Project Structure

```
codi_attention_flow/
├── README.md                       # This file
├── scripts/
│   ├── 1_sample_dataset.py        # Sample 100 GSM8K training problems
│   ├── 2_extract_attention_6x6.py # Extract 6×6 attention matrices
│   ├── 3_aggregate_attention.py   # Aggregate across problems
│   ├── 4_visualize_heatmaps.py    # Create visualizations
│   └── 5_analyze_hubs_and_flow.py # Hub & flow analysis
├── data/
│   └── attention_dataset_100_train.json  # 100 sampled problems
├── results/
│   ├── llama/
│   │   ├── attention_patterns_raw.npy    # [100, 16, 32, 6, 6]
│   │   ├── attention_patterns_avg.npy    # [16, 32, 6, 6]
│   │   ├── attention_stats.json          # Top heads, statistics
│   │   └── attention_summary.json        # Hub/flow/skip analysis
│   └── gpt2/                             # (Phase 2)
└── figures/
    ├── llama/
    │   ├── 1_top_heads_attention.png     # Top 20 heads heatmaps
    │   ├── 2_attention_by_layer.png      # Attention by layer
    │   └── 3_hub_analysis.png            # Hub/flow/skip charts
    └── gpt2/                             # (Phase 2)
```

---

## Technical Details

### Attention Matrix Interpretation

**6×6 Matrix**:
- **Rows (i)**: Destination position (which token is attending)
- **Columns (j)**: Source position (which token is being attended to)
- **Value [i, j]**: Attention weight from position i to position j

**Example**:
```
attention[1, 0] = 0.804  means  Position 1 attends to Position 0 with weight 0.804
```

**Causal Masking**: Position i can only attend to positions 0..i-1 (not future)

### Extraction Process

Unlike prior work that extracts attention FROM answer token TO continuous thoughts, we extract attention **DURING** continuous thought generation:

```python
# For each continuous thought step (0-5):
for step in range(6):
    outputs = model.codi(..., output_attentions=True)

    # Extract attention from current CT to all previous CTs
    attn = outputs.attentions[layer_idx][:, :, -1, :]  # [heads, seq_len]

    # Store attention to previous CT positions
    for prev_step in range(step):
        attention_matrix[layer, head, step, prev_step] = attn[head, ct_positions[prev_step]]
```

This captures **thought-to-thought** information flow, not answer-to-thought.

---

## Scripts Usage

### 1. Dataset Sampling

```bash
python scripts/1_sample_dataset.py [--seed SEED] [--n_samples N]
```

**Options**:
- `--seed`: Random seed (default: 42)
- `--n_samples`: Number of problems (default: 100)

**Output**: `data/attention_dataset_100_train.json`

**Validation**:
- No duplicates
- All answers present
- Source is training set only

---

### 2. Attention Extraction

```bash
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python scripts/2_extract_attention_6x6.py [--model MODEL] [--n_problems N]
```

**Options**:
- `--model`: llama or gpt2 (default: llama)
- `--n_problems`: Limit number (default: all)

**Output**:
- `results/{model}/attention_patterns_raw.npy`: [N, L, H, 6, 6]
- `results/{model}/attention_metadata.json`: Model info

**Runtime**: ~15 seconds for 100 problems (6.3 problems/second)

---

### 3. Aggregation

```bash
python scripts/3_aggregate_attention.py [--model MODEL]
```

**Output**:
- `results/{model}/attention_patterns_avg.npy`: [L, H, 6, 6] averaged
- `results/{model}/attention_stats.json`: Top 20 heads, consistency metrics

**Metrics Computed**:
- Mean attention across problems
- Standard deviation (consistency check)
- Top 20 heads by max attention
- Per-position statistics

---

### 4. Visualization

```bash
python scripts/4_visualize_heatmaps.py [--model MODEL]
```

**Output**:
- `figures/{model}/1_top_heads_attention.png`: Top 20 heads (4×5 grid)
- `figures/{model}/2_attention_by_layer.png`: 5 representative layers

**Visualization Details**:
- Heatmaps use 0-1.0 color scale
- Annotations show exact values
- Consistent scale across all subplots

---

### 5. Hub & Flow Analysis

```bash
python scripts/5_analyze_hubs_and_flow.py [--model MODEL]
```

**Output**:
- `results/{model}/attention_summary.json`: Hub scores, flow metrics, answers
- `figures/{model}/3_hub_analysis.png`: 3-panel chart

**Metrics**:
1. **Hub Score**: Average incoming attention per position
2. **Sequential Flow**: Attention from i → i-1
3. **Skip Connections**: Attention from pos 5 → pos 0,1,2

**Thresholds**:
- Strong hub: > 2.0× uniform baseline (0.167)
- Sequential flow: avg > 0.3
- Skip connections: avg > 0.1

---

## Dependencies

**Python Packages**:
```
torch >= 2.0
transformers >= 4.30
datasets >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
```

**Model Checkpoints**:
- LLaMA: `~/codi_ckpt/llama_gsm8k/` (3.2 GB)
- GPT-2: `~/codi_ckpt/gpt2_gsm8k/` (388 MB)

**Existing Code** (imported):
- `src/experiments/activation_patching/core/cache_activations_llama.py`

---

## Validation & Quality Checks

### Data Quality (Story 1.1)

```python
# Automatic checks in 1_sample_dataset.py:
assert len(set(ids)) == len(ids)  # No duplicates
assert all(len(q) > 0 for q in questions)  # Non-empty questions
assert all(a for a in answers)  # All answers present
assert all(id.startswith('train_') for id in ids)  # Training set only
```

### Attention Quality (Story 1.2)

```python
# Checks in 2_extract_attention_6x6.py:
# 1. Attention normalization (row sums close to 1.0)
row_sums = attention.sum(axis=-1)
print(f"Row sums: min={row_sums.min()}, max={row_sums.max()}")
# Expected: < 1.0 (we only extract attention to 6 CTs, not full sequence)

# 2. Non-zero patterns (at least 10% heads have max > 0.4)
max_attention = attention.max(axis=-1)
n_strong_heads = (max_attention > 0.4).sum()
print(f"Strong heads: {n_strong_heads}/{total_heads}")

# 3. Success rate
print(f"Extraction success: {n_success}/{total} ({100*n_success/total:.1f}%)")
```

### Consistency (Story 1.3)

```python
# Check in 3_aggregate_attention.py:
mean_std = attention_std.mean()
assert mean_std < 0.2, "Patterns inconsistent across problems"
# Phase 1 result: mean_std = 0.0113 ✓
```

---

## Performance

### Computational Cost

| Operation | Time | GPU Memory |
|-----------|------|------------|
| Dataset sampling | 2s | CPU only |
| Attention extraction (100 problems) | 15s | ~4 GB |
| Aggregation | 1s | CPU only |
| Visualization | 3s | CPU only |
| Analysis | 1s | CPU only |
| **Total** | **~25s** | **4 GB peak** |

### Storage

| File | Size | Format |
|------|------|--------|
| Dataset | 63 KB | JSON |
| Raw attention (100 problems) | 3.5 MB | .npy (float16) |
| Aggregated attention | 36 KB | .npy (float16) |
| Statistics | 10 KB | JSON |
| Figures (3 files) | 1.8 MB | PNG (300 DPI) |
| **Total** | **~5.4 MB** | |

---

## Troubleshooting

### Issue: Module not found error

```
ModuleNotFoundError: No module named 'cache_activations_llama'
```

**Solution**: Set PYTHONPATH:
```bash
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH python ...
```

### Issue: Extraction fails with "index out of bounds"

**Cause**: Trying to extract attention before token is added to sequence

**Solution**: Use `attention[:, -1, :]` to get last token, not `attention[:, current_pos, :]`

### Issue: JSON serialization error

```
TypeError: Object of type bool_ is not JSON serializable
```

**Solution**: Cast numpy types: `bool(x)`, `int(x)`, `float(x)`

---

## Citation

If you use this code or findings, please cite:

```bibtex
@misc{codi_attention_flow_2025,
  title={CODI Attention Flow Analysis: Hub-Centric Architecture in Continuous Thought Reasoning},
  author={Research Team},
  year={2025},
  note={Internal research, see docs/experiments/10-28_llama_gsm8k_attention_flow_analysis.md}
}
```

---

## Related Work

- **CODI Paper**: [docs/codi.pdf](../../../docs/codi.pdf)
- **Prior Analysis**: [docs/experiments/10-25_gpt2_gsm8k_attention_visualization.md](../../../docs/experiments/10-25_gpt2_gsm8k_attention_visualization.md)
- **Architecture Doc**: [docs/architecture/attention_flow_analysis_architecture.md](../../../docs/architecture/attention_flow_analysis_architecture.md)
- **User Stories**: [docs/project/user_stories_attention_flow_analysis.md](../../../docs/project/user_stories_attention_flow_analysis.md)

---

## Future Work

### Phase 2: Critical Heads Analysis (Planned)

Compute flow/hub/skip scores **per head** and compare LLaMA vs GPT-2:
- Stories 2.1-2.4: Metrics per head
- Story 2.6: Model comparison

**Estimate**: 7.0 hours

### Phase 4: Full Dataset Validation (Optional)

Scale to all 7,473 training problems:
- Story 4.1-4.4: Full-scale extraction & validation
- Check if hub ratio strengthens with more data

**Estimate**: 3.5 hours (2.5h GPU time)

### Beyond Scope

1. **Causal Intervention**: Ablate Position 0, test if hub is causal
2. **Cross-Dataset**: Test on MATH, StrategyQA
3. **Head Ablation**: Remove top hub heads, measure accuracy impact

---

## Contact

For questions or issues, see project documentation in `docs/` or create an issue in the repository.

**Last Updated**: 2025-10-28 (Phase 1 Complete)
