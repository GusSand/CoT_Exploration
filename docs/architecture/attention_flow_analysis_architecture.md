# Architecture: CODI Attention Flow Analysis

**Created**: 2025-10-27
**Architect**: Claude Code
**Status**: Approved
**Related User Stories**: `docs/project/user_stories_attention_flow_analysis.md`

---

## Overview

This document defines the architecture for extracting and analyzing attention patterns between the 6 continuous thought token positions in CODI models, comparing GPT-2 and LLaMA.

**Goal**: Extract 6×6 attention matrices (continuous thought position i → position j) across all layers and heads, then identify critical reasoning heads.

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────┐
│ GSM8K Training  │
│   (7,473 probs) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dataset Sampler │ ← Random sample with seed=42
│   (100 problems)│
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Attention Extractor               │
│   ┌──────────────────────────────┐  │
│   │ For each problem:            │  │
│   │  1. Load CODI model          │  │
│   │  2. Run inference            │  │
│   │  3. Extract attention at     │  │
│   │     6 continuous thought     │  │
│   │     positions                │  │
│   │  4. Output: [L, H, 6, 6]    │  │
│   └──────────────────────────────┘  │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Attention Aggregator            │
│  - Average across problems      │
│  - Compute statistics           │
│  - Identify top heads           │
│  Output: [L, H, 6, 6] averaged  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Metric Computer                 │
│  - Flow scores (forward flow)   │
│  - Hub scores (variance)        │
│  - Skip scores (long-range)     │
│  - Composite rankings           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Visualizer                      │
│  - Heatmaps (6×6 matrices)      │
│  - Hub analysis charts          │
│  - Critical head comparison     │
└─────────────────────────────────┘
```

---

## Key Architectural Decisions

### 1. Attention Extraction Strategy

**Problem**: Existing code (`2_extract_attention.py`) extracts attention **FROM answer token TO continuous thoughts**, but we need attention **BETWEEN continuous thought positions** (6×6 matrix).

**Solution**: Modify extraction to capture attention weights during the continuous thought generation phase:

```python
# For each continuous thought step (0-5):
outputs = model.codi(
    inputs_embeds=latent_embd,
    output_attentions=True,  # Enable attention extraction
    past_key_values=past_key_values
)

# Extract attention at current position attending to all previous positions
# This gives us the 6×6 attention matrix incrementally
```

**Key Insight**: We need to extract attention at **each of the 6 continuous thought token positions**, not just at the answer token.

### 2. Attention Matrix Interpretation

**6×6 Matrix Structure**:
- **Rows** (i): Destination position (which token is attending)
- **Columns** (j): Source position (which token is being attended to)
- **Value** [i, j]: Attention weight from position i to position j

**Important**: Due to causal masking, position i can only attend to positions 0..i (not future positions).

**Example**:
```
     Attend TO →
     j=0  j=1  j=2  j=3  j=4  j=5
i=0 [1.0  0.0  0.0  0.0  0.0  0.0]  ← Pos 0 attends only to itself
i=1 [0.3  0.7  0.0  0.0  0.0  0.0]  ← Pos 1 attends to 0,1
i=2 [0.2  0.3  0.5  0.0  0.0  0.0]
i=3 [0.1  0.2  0.3  0.4  0.0  0.0]
i=4 [0.1  0.1  0.2  0.3  0.3  0.0]
i=5 [0.1  0.1  0.1  0.2  0.2  0.3]  ← Pos 5 can attend to all
```

**Metrics**:
- **Sequential flow**: Sum of diagonal+1 elements (i → i-1)
- **Hub detection**: Variance across columns (which source is heavily attended?)
- **Skip connections**: Attention from i=5 to j={0,1,2} (skipping middle)

### 3. Data Pipeline Design

**Dataset Preparation**:
```python
# Story 1.1: Dataset Sampler
from datasets import load_dataset
import random

train_dataset = load_dataset('gsm8k', 'main', split='train')
random.seed(42)
sampled = random.sample(range(len(train_dataset)), 100)

output = [{
    'gsm8k_id': f'train_{idx}',
    'question': train_dataset[idx]['question'],
    'answer': extract_answer(train_dataset[idx]['answer']),
    'full_solution': train_dataset[idx]['answer']
} for idx in sampled]
```

**Attention Extraction**:
```python
# Story 1.2: Extract 6×6 attention per head/layer
attention_data = {
    'problem_id': problem_id,
    'attention': np.zeros((n_layers, n_heads, 6, 6)),  # [L, H, 6, 6]
}

# During continuous thought generation:
for step in range(6):
    outputs = model.codi(..., output_attentions=True)

    for layer_idx in range(n_layers):
        # Extract attention for this layer
        attn = outputs.attentions[layer_idx]  # [batch, heads, seq_len, seq_len]

        # Get attention from current position (step) to all previous positions
        current_pos = question_len + 1 + step  # BOT token + continuous thoughts
        continuous_start = question_len + 1

        for head_idx in range(n_heads):
            # Extract attention to continuous thought positions
            attn_to_continuous = attn[0, head_idx, current_pos, continuous_start:continuous_start+step+1]

            # Store in matrix (row = current step, cols = previous steps)
            attention_data['attention'][layer_idx, head_idx, step, :step+1] = attn_to_continuous.cpu().numpy()
```

**Aggregation**:
```python
# Story 1.3: Average across problems
all_attention = []  # List of [L, H, 6, 6] arrays
for problem in problems:
    all_attention.append(problem['attention'])

avg_attention = np.mean(all_attention, axis=0)  # [L, H, 6, 6]
std_attention = np.std(all_attention, axis=0)   # Check stability
```

### 4. Metric Computation

**Information Flow Score** (Story 2.1):
```python
def compute_flow_score(attention_matrix):
    """
    Measures forward information flow (later → earlier positions).

    Args:
        attention_matrix: [6, 6] attention weights

    Returns:
        flow_score: 0-1 (higher = more forward flow)
    """
    forward_flow = 0.0
    total_attention = 0.0

    for i in range(6):
        for j in range(i):  # j < i (attending to earlier positions)
            forward_flow += attention_matrix[i, j]
        total_attention += attention_matrix[i, :i+1].sum()

    return forward_flow / total_attention if total_attention > 0 else 0.0
```

**Hub Connectivity Score** (Story 2.2):
```python
def compute_hub_score(attention_matrix):
    """
    Measures concentration of attention (hub detection).

    Args:
        attention_matrix: [6, 6] attention weights

    Returns:
        hub_score: variance of column sums (higher = stronger hub)
        hub_position: which position is the hub (0-5)
    """
    # Sum attention TO each position (column sums)
    incoming_attention = attention_matrix.sum(axis=0)  # [6]

    hub_score = np.var(incoming_attention)
    hub_position = np.argmax(incoming_attention)

    return hub_score, hub_position
```

**Skip Connection Score** (Story 2.3):
```python
def compute_skip_score(attention_matrix):
    """
    Measures long-range dependencies (position 5 → early positions).

    Args:
        attention_matrix: [6, 6] attention weights

    Returns:
        skip_score: average attention from pos 5 to pos 0-2
    """
    # Attention from last position to first 3 positions
    skip_attention = attention_matrix[5, 0:3]  # [3]

    return np.mean(skip_attention)
```

**Composite Score** (Story 2.4):
```python
composite_score = (
    0.4 * flow_score +
    0.4 * normalized_hub_score +
    0.2 * normalized_skip_score
)
```

### 5. Model Comparison Strategy

**GPT-2 vs LLaMA Comparison** (Story 2.6):
- Run **identical pipeline** on both models
- Use **same 100 problems** (from Story 1.1)
- Store results in separate directories:
  - `results/llama/`
  - `results/gpt2/`
- Compare:
  - Hub positions (do they differ?)
  - Number of critical heads (GPT-2 should have fewer)
  - Attention concentration (LLaMA should be more distributed)

### 6. Scalability for Full Dataset

**Checkpointing Strategy** (Story 4.2):
```python
# For ~7,473 problems, checkpoint every 1,000
CHECKPOINT_INTERVAL = 1000

for i, problem in enumerate(all_problems):
    # Process problem
    results.append(extract_attention(problem))

    # Checkpoint
    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_file = f'results/checkpoints/attention_checkpoint_{i+1}.npy'
        np.save(checkpoint_file, np.array([r['attention'] for r in results]))
        print(f"Checkpoint saved: {i+1}/{len(all_problems)}")

# Resume from checkpoint if interrupted
def load_checkpoint(checkpoint_dir):
    checkpoints = sorted(checkpoint_dir.glob('attention_checkpoint_*.npy'))
    if checkpoints:
        last_checkpoint = checkpoints[-1]
        return np.load(last_checkpoint), int(last_checkpoint.stem.split('_')[-1])
    return None, 0
```

---

## Tools & Libraries

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Model inference, attention extraction |
| `transformers` | ≥4.30 | CODI model loading |
| `datasets` | ≥2.0 | GSM8K data loading |
| `numpy` | ≥1.24 | Array operations, metric computation |
| `matplotlib` | ≥3.7 | Heatmap visualizations |
| `seaborn` | ≥0.12 | Enhanced visualizations |
| `pandas` | ≥2.0 | Data manipulation, rankings |
| `wandb` | ≥0.15 | Experiment tracking (optional) |
| `tqdm` | ≥4.65 | Progress bars |

### Existing Code to Reuse

| File | Purpose | Modifications Needed |
|------|---------|---------------------|
| `cache_activations_llama.py` | Model loading, inference | ✅ No changes - wrapper works |
| `2_extract_attention.py` | Attention extraction skeleton | ⚠️ Modify to extract 6×6 matrices |
| `3_analyze_and_visualize.py` | Visualization patterns | ✅ Reuse heatmap code |

### New Code to Write

| Module | Purpose | Estimated LOC |
|--------|---------|---------------|
| `dataset_sampler.py` | Story 1.1 - Sample 100 problems | ~50 |
| `attention_extractor_6x6.py` | Story 1.2 - Extract 6×6 matrices | ~200 |
| `attention_aggregator.py` | Story 1.3 - Average & statistics | ~100 |
| `metric_computer.py` | Stories 2.1-2.4 - Flow/hub/skip scores | ~150 |
| `visualizer.py` | Stories 1.4, 2.5 - Heatmaps & charts | ~200 |
| `model_comparator.py` | Story 2.6 - GPT-2 vs LLaMA | ~100 |
| **Total** | | **~800 LOC** |

---

## Directory Structure

```
src/experiments/codi_attention_flow/
├── README.md                          # Quick start guide
├── scripts/
│   ├── 1_sample_dataset.py           # Story 1.1
│   ├── 2_extract_attention_6x6.py    # Story 1.2 (LLaMA)
│   ├── 3_aggregate_attention.py      # Story 1.3
│   ├── 4_compute_metrics.py          # Stories 2.1-2.4
│   ├── 5_visualize.py                # Stories 1.4, 2.5
│   ├── 6_run_gpt2.py                 # Story 2.6 (GPT-2 pipeline)
│   ├── 7_compare_models.py           # Story 2.6 (comparison)
│   └── 8_full_dataset.py             # Story 4.2 (full scale)
├── data/
│   ├── attention_dataset_100_train.json
│   └── attention_dataset_full_train.json
├── results/
│   ├── llama/
│   │   ├── attention_patterns_raw.npy        # [100, 16, 32, 6, 6]
│   │   ├── attention_patterns_avg.npy        # [16, 32, 6, 6]
│   │   ├── attention_stats.json
│   │   ├── heads_ranked_by_flow.csv
│   │   ├── heads_ranked_by_hub.csv
│   │   ├── heads_ranked_by_skip.csv
│   │   ├── ranked_heads.csv
│   │   └── attention_summary.json
│   ├── gpt2/
│   │   └── (same structure as llama/)
│   ├── comparison/
│   │   └── gpt2_vs_llama_comparison.json
│   └── llama_full/                           # Phase 4
│       ├── checkpoints/
│       │   └── attention_checkpoint_*.npy
│       └── (same structure as llama/)
└── figures/
    ├── llama/
    │   ├── 1_top_heads_attention.png
    │   ├── 2_attention_by_layer.png
    │   ├── 3_hub_analysis.png
    │   ├── 4_top_heads_visualization.png
    │   └── 5_critical_head_detail_L{layer}H{head}.png
    ├── gpt2/
    │   └── (same structure as llama/)
    └── comparison/
        ├── 6_model_comparison_hubs.png
        ├── 7_model_comparison_flow.png
        └── 8_model_comparison_critical_heads.png
```

---

## Performance Considerations

### Memory Management

**Problem**: Storing full attention matrices for all problems:
- LLaMA: `[100, 16, 32, 6, 6]` = ~18 MB (float32)
- Full dataset: `[7473, 16, 32, 6, 6]` = ~1.3 GB

**Solution**: Use float16 for storage, process in batches:
```python
# Store as float16 to save space
attention_raw = np.zeros((n_problems, n_layers, n_heads, 6, 6), dtype=np.float16)

# Convert to float32 only during computation
attention_f32 = attention_raw.astype(np.float32)
```

### GPU Utilization

**Batch Size**: 1 problem at a time (CODI generates sequentially)
**Expected Runtime**:
- 100 problems (LLaMA): ~15 minutes (0.9s per problem)
- 100 problems (GPT-2): ~10 minutes (0.6s per problem)
- 7,473 problems (LLaMA): ~2.5 hours (1.2s per problem with overhead)

**Optimization**: Process problems in parallel if multiple GPUs available (not implemented in Phase 1).

---

## Validation & Quality Checks

### Data Quality Checks

**Story 1.1 - Dataset Preparation**:
- [ ] Verify no duplicates: `assert len(set(ids)) == len(ids)`
- [ ] Verify all questions non-empty: `assert all(len(q) > 0 for q in questions)`
- [ ] Verify all answers present: `assert all(a is not None for a in answers)`
- [ ] Verify source is training set: `assert all(id.startswith('train_') for id in ids)`

**Story 1.2 - Attention Extraction**:
- [ ] Verify attention sums to 1: `assert np.allclose(attn.sum(axis=-1), 1.0, atol=0.01)`
- [ ] Verify causal masking: `assert np.allclose(attn[i, j], 0) for i < j` (upper triangle is zero)
- [ ] Verify shape: `assert attn.shape == (n_layers, n_heads, 6, 6)`

**Story 1.3 - Aggregation**:
- [ ] Verify consistency: `assert std_attention.mean() < 0.2` (patterns stable across problems)
- [ ] Verify non-uniform: `assert (avg_attention.max(axis=-1) > 0.4).sum() > 0.1 * n_heads` (at least 10% heads show structure)

### Success Criteria Validation

**Phase 1 Pass Criteria**:
```python
def validate_phase1(attention_avg, attention_std):
    """Validate Phase 1 results meet success criteria."""

    # Criterion 1: Non-random patterns
    max_attention = attention_avg.max(axis=-1)  # [L, H]
    n_strong_heads = (max_attention > 0.4).sum()
    assert n_strong_heads > 0.1 * max_attention.size, "Too few heads with strong patterns"

    # Criterion 2: Patterns consistent
    assert attention_std.mean() < 0.2, "Attention patterns too variable across problems"

    # Criterion 3: Can identify hubs
    hub_scores = attention_avg.sum(axis=2).var(axis=-1)  # Variance of incoming attention
    assert hub_scores.max() > 0.2, "No clear hub positions found"

    print("✓ Phase 1 success criteria met!")
    return True
```

**Phase 2 Pass Criteria**:
```python
def validate_phase2(ranked_heads):
    """Validate Phase 2 results meet success criteria."""

    # Criterion 1: Top heads clearly identified
    assert ranked_heads['composite_score'].iloc[0] > 0.6, "Top head composite score too low"

    # Criterion 2: Clear gap between top and random
    top_10_mean = ranked_heads['composite_score'].iloc[:10].mean()
    random_mean = ranked_heads['composite_score'].iloc[-50:].mean()
    assert top_10_mean > 2 * random_mean, "No clear separation between top and random heads"

    # Criterion 3: Functional specialization
    top_types = ranked_heads['functional_type'].iloc[:10].unique()
    assert len(top_types) > 1, "No functional specialization detected"

    print("✓ Phase 2 success criteria met!")
    return True
```

---

## Error Handling

### Extraction Failures

**Strategy**: Continue on error, log failures:
```python
failed_problems = []

for problem in problems:
    try:
        attention = extract_attention(problem)
        results.append(attention)
    except Exception as e:
        print(f"Error on {problem['id']}: {e}")
        failed_problems.append({
            'id': problem['id'],
            'error': str(e)
        })

        # Skip this problem, continue with rest
        continue

# Report failures
if failed_problems:
    print(f"⚠️  {len(failed_problems)}/{len(problems)} problems failed")
    with open('failed_problems.json', 'w') as f:
        json.dump(failed_problems, f, indent=2)
```

### Model Loading Failures

**Strategy**: Fail fast with clear error message:
```python
def load_model(model_path):
    """Load CODI model with validation."""
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}\n"
            f"Expected checkpoint at ~/codi_ckpt/<model_name>/"
        )

    try:
        cacher = ActivationCacherLLaMA(model_path)
        print(f"✓ Loaded model from {model_path}")
        return cacher
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path}\n"
            f"Error: {e}\n"
            f"Check that checkpoint is valid CODI model"
        )
```

---

## Testing Strategy

### Unit Tests

**Module**: `tests/test_attention_extractor.py`
```python
def test_attention_shape():
    """Test attention matrix has correct shape."""
    attention = extract_dummy_attention()
    assert attention.shape == (16, 32, 6, 6)  # LLaMA: 16 layers, 32 heads

def test_attention_normalization():
    """Test attention weights sum to 1."""
    attention = extract_dummy_attention()
    row_sums = attention.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=0.01)

def test_causal_masking():
    """Test upper triangle is zero (causal attention)."""
    attention = extract_dummy_attention()
    for i in range(6):
        for j in range(i+1, 6):
            assert np.allclose(attention[:, :, i, j], 0), f"Non-zero attention from {i} to {j}"
```

**Module**: `tests/test_metrics.py`
```python
def test_flow_score():
    """Test flow score computation."""
    # Sequential attention (strong forward flow)
    sequential_attn = np.eye(6)
    sequential_attn[range(1, 6), range(5)] = 1.0
    flow = compute_flow_score(sequential_attn)
    assert flow > 0.8, "Sequential pattern should have high flow score"

    # Uniform attention (no forward flow)
    uniform_attn = np.ones((6, 6)) / 6
    flow = compute_flow_score(uniform_attn)
    assert flow < 0.3, "Uniform pattern should have low flow score"
```

### Integration Tests

**Test End-to-End Pipeline**:
```python
def test_pipeline_on_single_problem():
    """Test full pipeline on one problem."""
    # Sample one problem
    dataset = [{'question': '...', 'answer': '...'}]

    # Extract attention
    attention = extract_attention_6x6(dataset[0])
    assert attention.shape == (16, 32, 6, 6)

    # Compute metrics
    flow_score = compute_flow_score(attention[8, 5])  # Layer 8, Head 5
    assert 0 <= flow_score <= 1

    # Generate visualization (should not crash)
    visualize_heatmap(attention[8, 5])
```

---

## Rollback Plan

**If Phase 1 fails** (patterns are noise):
1. Check attention extraction code - verify we're extracting correct positions
2. Increase sample size from 100 to 1000
3. Try different layers (maybe early/late layers have more structure)
4. Fall back to existing analysis (attention from answer token)

**If GPT-2 comparison fails** (no differences):
1. Still valuable negative result - document thoroughly
2. Analyze why models converge on same strategy
3. Check if dataset is too easy (try harder problems)

**If full-scale extraction fails** (out of memory):
1. Process in smaller batches (e.g., 500 at a time)
2. Use checkpointing more aggressively (every 100 problems)
3. Store only averaged attention (discard per-problem data after aggregation)

---

## Monitoring & Observability

### WandB Integration

```python
import wandb

# Initialize run
wandb.init(
    project="codi-attention-flow",
    config={
        "model": "llama-3.2-1b",
        "n_problems": 100,
        "seed": 42,
        "layers": 16,
        "heads": 32
    },
    tags=["attention-flow", "gsm8k", "llama", "phase1"]
)

# Log during extraction
for i, problem in enumerate(problems):
    attention = extract_attention(problem)

    # Log progress
    wandb.log({
        "problems_processed": i + 1,
        "mean_attention": attention.mean(),
        "max_attention": attention.max(),
    })

# Log final metrics
wandb.log({
    "top_flow_score": flow_scores.max(),
    "n_critical_heads": (composite_scores > 0.7).sum(),
    "hub_position": hub_analysis['position']
})

# Log visualizations
wandb.log({"attention_heatmap": wandb.Image("figures/1_top_heads_attention.png")})
```

### Progress Tracking

```python
from tqdm import tqdm

with tqdm(total=len(problems), desc="Extracting attention") as pbar:
    for problem in problems:
        attention = extract_attention(problem)
        pbar.update(1)
        pbar.set_postfix({
            'avg_attn': f"{attention.mean():.3f}",
            'max_attn': f"{attention.max():.3f}"
        })
```

---

## Security & Privacy

**Data Privacy**: ✅ No issues
- GSM8K is public dataset
- No PII or sensitive data

**Model Security**: ✅ No issues
- Models are local checkpoints
- No external API calls

**Compute Resources**: ⚠️ Monitor
- A100 GPU time: ~3 hours total
- Disk space: ~2 GB for results
- Set wandb to log-only (no artifact uploads for large files)

---

## Documentation Requirements

### Code Documentation

**Every module must have**:
- Docstring with purpose, inputs, outputs
- Type hints for all functions
- Example usage in docstring

**Example**:
```python
def extract_attention_6x6(
    problem: dict,
    model: ActivationCacherLLaMA,
    layers: list[int]
) -> np.ndarray:
    """
    Extract 6×6 attention matrix between continuous thought positions.

    Args:
        problem: Dict with 'question', 'answer' keys
        model: Loaded CODI model wrapper
        layers: List of layer indices to extract (e.g., [4, 8, 14])

    Returns:
        attention: Array of shape [n_layers, n_heads, 6, 6]
            - attention[l, h, i, j] = attention from position i to j at layer l, head h
            - Row sums to 1.0 (normalized)
            - Upper triangle is 0 (causal masking)

    Example:
        >>> problem = {'question': '2+2=?', 'answer': '4'}
        >>> attn = extract_attention_6x6(problem, model, [8])
        >>> attn.shape
        (1, 32, 6, 6)
        >>> attn[0, 5, 3, :4].sum()  # Pos 3 attention sums to 1
        1.0
    """
    ...
```

### Experiment Documentation

**After each phase**, update:
1. `docs/research_journal.md` - TLDR summary
2. `docs/experiments/10-27_<model>_gsm8k_attention_flow.md` - Detailed report
3. `docs/DATA_INVENTORY.md` - New datasets created
4. `src/experiments/codi_attention_flow/README.md` - How to reproduce

---

## Success Metrics Summary

### Phase 1 Success
- ✅ Attention patterns are non-random (max > 0.4 in 10%+ heads)
- ✅ Patterns are consistent across problems (std < 0.2)
- ✅ Can identify hub positions (variance > 0.2)
- ✅ Can answer: sequential flow? skip connections? hub positions?

### Phase 2 Success
- ✅ Top 10 heads identified (composite score > 0.6)
- ✅ Clear separation from random heads (2× gap)
- ✅ Functional specialization visible
- ✅ Can name 2-3 most critical heads per model

### Comparison Success
- ✅ Clear differences between GPT-2 and LLaMA
- ✅ Findings align with capacity hypothesis
- ✅ Side-by-side visualizations show distinct patterns

---

## Next Steps After Architecture Approval

1. **Review this document** with stakeholders
2. **Transition to Developer role** to implement
3. **Start with Story 1.1** (dataset preparation)
4. **Follow user stories sequentially** (dependencies respected)
5. **Commit after each story** (incremental progress)

---

**Status**: Ready for implementation
**Estimated Implementation Time**: 14.5 dev hours + 3.0 GPU hours
**Risk Level**: Low (data validated, existing code patterns available)
**Dependencies**: All verified ✅
