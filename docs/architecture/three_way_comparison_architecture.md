# Three-Way CODI Mechanistic Comparison - Architecture

**Date**: 2025-10-30
**Architect**: Claude
**Status**: ✅ APPROVED - Ready for Implementation
**Estimated Effort**: 16-20 hours

---

## Executive Summary

Architecture for mechanistic comparison across three reasoning types: **Personal Relations** (graph traversal), **GSM8K** (arithmetic), and **CommonsenseQA** (semantic reasoning). All models are LLaMA-3.2-1B CODI with 6 continuous thought tokens.

### Key Architectural Decisions

1. **Sequential Processing**: Load one model at a time to avoid memory issues (A100 80GB can handle 1B models easily)
2. **Standardized Data Format**: NPZ format with consistent schema across all three tasks
3. **Reusable Components**: Leverage existing `ActivationCacherLLaMA` and `NTokenPatcher` infrastructure
4. **Incremental Execution**: Each story produces standalone outputs; can pause/resume

---

## 1. System Architecture

### 1.1 Model Inventory

| Task | Model Path | Checkpoint | Accuracy | Status |
|------|------------|------------|----------|--------|
| **Personal Relations** | `/home/paperspace/dev/CoT_Exploration/models/personal_relations_1b_codi_v2/.../checkpoint-270/` | ep_10, lr_0.0008, seed_42 | 43.7% (328/750) | ✅ Verified |
| **GSM8K** | `/home/paperspace/codi_ckpt/llama_gsm8k/pytorch_model.bin` | Single file | ~50-60% | ✅ Verified |
| **CommonsenseQA** | `/home/paperspace/codi_ckpt/llama_commonsense/gsm8k_llama1b_latent_baseline/.../seed_11/` | ep_3, lr_0.0008, seed_11 | 75% (75/100) | ✅ Verified |

**Model Loading Requirements**:
- **Personal Relations**: Requires LoRA (`use_lora=True`), `full_precision=True`
- **GSM8K**: Standard loading
- **CommonsenseQA**: Standard loading (already has `NTokenPatcher` working)

---

### 1.2 Data Architecture

#### Test Dataset Sizes
- **Personal Relations**: 750 examples (use 100 stratified)
- **GSM8K**: 1,319 examples (use 100 stratified)
- **CommonsenseQA**: 1,221 validation examples (use 100 stratified)

#### Stratification Strategy
For each task, select 100 examples stratified by:
- **Baseline correctness**: Include both correct and incorrect predictions
- **Task-specific complexity**:
  - Personal Relations: Relationship chain length (1-hop, 2-hop, 3+ hop)
  - GSM8K: Operation count (1-2, 3-4, 5+ operations)
  - CommonsenseQA: Question type (causal, spatial, temporal, etc.)

---

### 1.3 Output Data Schema

**File**: `activations_{task}.npz`

```python
{
    'metadata': {
        'task': str,  # 'personal_relations', 'gsm8k', 'commonsense'
        'model_path': str,
        'n_examples': int,
        'n_layers': int,  # 16 for LLaMA-1B
        'n_tokens': int,  # 6 CT tokens
        'hidden_dim': int,  # 2048 for LLaMA-1B
        'timestamp': str
    },
    'examples': [
        {
            'example_id': int,
            'question': str,
            'answer': str,  # Ground truth
            'predicted': str,
            'correct': bool,
            'hidden_states': np.array,  # Shape: [16 layers, 6 tokens, 2048 dims]
            'attention_to_ct': np.array,  # Shape: [6 tokens] - avg attention from all positions
        }
    ]
}
```

**Total size estimate**: 300 examples × 16 layers × 6 tokens × 2048 × 4 bytes ≈ **225 MB**

---

## 2. Component Architecture

### 2.1 Unified Model Loader (Story 1)

**File**: `src/experiments/three_way_comparison/model_loader.py`

**Key Functions**:
```python
def load_codi_model(task: str, device: str = 'cuda:0') -> tuple:
    """
    Load CODI model for specified task.

    Args:
        task: 'personal_relations', 'gsm8k', 'commonsense'
        device: CUDA device

    Returns:
        (model, tokenizer, config_dict)
    """

def format_input(task: str, example: dict) -> str:
    """Format input according to task requirements."""

def extract_answer(task: str, output: str, example: dict) -> str:
    """Extract answer from model output."""
```

**Critical Design Decisions**:
1. **Single model at a time**: Load → extract → unload → next task
2. **Task-specific formatters**: Each task has unique prompt format
3. **Validation step**: Test forward pass before extraction
4. **Error handling**: Graceful degradation if checkpoint issues

---

### 2.2 Activation Extraction Pipeline (Story 2)

**File**: `src/experiments/three_way_comparison/extract_activations.py`

**Architecture**:
```
┌─────────────────────────────────────────────────────┐
│  Load Model (task='personal_relations')             │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│  Load & Stratify Test Dataset (100 examples)       │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│  Register Forward Hooks (CT positions only)         │
│  - Hook on CT token hidden states                   │
│  - Hook on attention weights to CT tokens           │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│  For each example:                                   │
│    1. Format input                                   │
│    2. Run forward pass (capture activations)        │
│    3. Extract answer                                 │
│    4. Store: hidden_states, attention, correctness  │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│  Save to NPZ: activations_personal_relations.npz    │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│  Unload model, clear cache                          │
└─────────────────────────────────────────────────────┘
                 │
                 └──► Repeat for GSM8K, then CommonsenseQA
```

**Performance Optimization**:
- Batch size: 1 (each problem is variable length)
- Checkpoint saving: Every 25 examples
- Memory clearing: After each task completion
- ETA: ~1 hour per task = **3 hours total**

---

### 2.3 Analysis Components (Stories 3-6)

#### Story 3: Embedding Visualization
**File**: `src/experiments/three_way_comparison/visualize_embeddings.py`

**Approach**:
- PCA for initial exploration (fast, interpretable)
- t-SNE for final publication figures (slower, better separation)
- Focus on **CT0** (most critical) and **CT5** (final reasoning)
- Layers: 0 (input), 8 (middle), 15 (output)

#### Story 4: Attention Hub Analysis
**File**: `src/experiments/three_way_comparison/analyze_attention_hubs.py`

**Hypothesis**:
- GSM8K: CT0 = hub, CT5 = critical (**dissociated**)
- CommonsenseQA: CT0 = hub + critical (**unified**)
- Personal Relations: **TO DISCOVER** (expect hub at CT0 for graph encoding)

**Metrics**:
- Hub score: Average incoming attention per token
- Layer progression: How hub strength evolves
- Statistical tests: ANOVA across tasks, pairwise t-tests

#### Story 5: Token Importance (Ablation)
**File**: `src/experiments/three_way_comparison/ablate_tokens.py`

**Method**: Hidden state zeroing at each CT position
- Run on 50 examples per task (computational constraint)
- Measure accuracy drop per ablation
- Compare critical token across tasks

**Expected**:
- GSM8K: CT5 most critical
- CommonsenseQA: CT0 most critical
- Personal Relations: CT0 or CT1 (graph encoding)

#### Story 6: Layer Divergence
**File**: `src/experiments/three_way_comparison/analyze_divergence.py`

**Research Question**: At which layer do task representations diverge?

**Method**:
- Cosine similarity between task pairs (PR-GSM, PR-CQA, GSM-CQA)
- Track similarity across 16 layers
- Identify divergence point

**Expected**: Early layers similar, divergence at layers 4-8

---

## 3. Technology Stack

### 3.1 Dependencies

**Required** (already installed):
```
torch==2.4.0+cu121
transformers
peft (for LoRA)
datasets (for CommonsenseQA)
numpy
scipy
matplotlib
seaborn
scikit-learn
```

**Optional**:
```
wandb (Story 8)
plotly (interactive visualizations)
```

### 3.2 Compute Requirements

**GPU**: NVIDIA A100-SXM4-80GB ✅ Available
**RAM**: 32GB+ (current system: 80GB VRAM + host RAM)
**Storage**: 500MB for outputs

**Estimated Runtime**:
- Story 1 (Model Loading): 1 hour (testing all 3 models)
- Story 2 (Extraction): 3 hours (1 hour per task)
- Story 3 (Visualization): 1 hour
- Story 4 (Attention): 1 hour
- Story 5 (Ablation): 3 hours (50 examples × 6 tokens × 3 tasks)
- Story 6 (Divergence): 1 hour
- Story 7 (Report): 2 hours
**Total**: ~12 hours compute + 2 hours writing = **14 hours**

---

## 4. Data Quality Assurance

### 4.1 Pre-Execution Checks

**Data Availability**:
- ✅ Personal Relations test: 750 examples at `data/personal_relations/personal_relations_test_codi_v2.json`
- ✅ GSM8K test: Available via Hugging Face datasets
- ✅ CommonsenseQA validation: Available via Hugging Face datasets

**Model Checkpoint Integrity**:
1. Verify all 3 model paths exist
2. Test forward pass on 1 example per task
3. Verify output format matches expected

### 4.2 Stratification Validation

For each task, ensure 100-example sample includes:
- 50% baseline correct, 50% incorrect (or closest available)
- Balanced complexity distribution
- No duplicate examples

### 4.3 Output Validation

After each story:
- File size checks (NPZ files ~75 MB each)
- Schema validation (correct shapes, no NaNs)
- Metadata completeness
- Reproducibility: Save random seeds

---

## 5. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| **Personal Relations model load fails** | Low | High | Already verified in `eval_personal_relations_v2_1b_SIMPLE.py` |
| **Memory overflow** | Very Low | Medium | Sequential loading, single model at a time |
| **Activation extraction slow** | Medium | Low | Checkpointing every 25 examples, can resume |
| **No task separation in visualizations** | Medium | Medium | Use multiple metrics (PCA, t-SNE, cosine sim); qualitative analysis still valuable |
| **Ablations too computationally expensive** | Low | Low | Reduce to 30 examples if needed |

---

## 6. Testing Strategy

### 6.1 Unit Tests

**File**: `src/experiments/three_way_comparison/test_components.py`

```python
def test_model_loading():
    """Test all 3 models load without errors."""

def test_data_formatting():
    """Test input formatting for all 3 tasks."""

def test_answer_extraction():
    """Test answer extraction for all 3 tasks."""

def test_activation_shapes():
    """Test extracted activations have correct shapes."""
```

### 6.2 Integration Test

**Smoke test before full run**:
1. Load each model
2. Extract 5 examples per task
3. Verify NPZ output schema
4. Generate 1 visualization

**ETA**: 15 minutes

---

## 7. Output Specifications

### 7.1 Deliverables

**Story 1**: `model_loader.py` + validation log
**Story 2**: 3 NPZ files (225 MB total) + extraction logs
**Story 3**: 10-15 publication-quality figures (PNG + PDF)
**Story 4**: Attention analysis results + 3-5 heatmaps
**Story 5**: Token importance rankings + comparison table
**Story 6**: Layer divergence plots + divergence points
**Story 7**: `docs/experiments/10-30_three_way_codi_comparison.md` (comprehensive report)
**Story 8** (optional): WandB dashboard with all metrics

### 7.2 Documentation Requirements

Each component must include:
- Docstrings (Google style)
- Usage examples
- Expected input/output
- Error handling documentation

---

## 8. Reproducibility

### 8.1 Seeds and Configuration

**Random seeds**:
- NumPy: 42
- PyTorch: 42
- Stratification: 42

**Configuration file**: `three_way_config.json`
```json
{
    "tasks": ["personal_relations", "gsm8k", "commonsense"],
    "n_examples_per_task": 100,
    "n_ablation_examples": 50,
    "device": "cuda:0",
    "batch_size": 1,
    "checkpoint_interval": 25,
    "random_seed": 42
}
```

### 8.2 Version Control

All scripts and configs will be committed with:
- Clear commit messages
- Experiment timestamp
- Hardware specifications
- Dependency versions

---

## 9. Success Criteria

### Must Have ✅
- All 3 models successfully loaded and extracted
- 300 examples (100 per task) with activations
- At least 3 distinct mechanistic findings
- Reproducible pipeline documented

### Nice to Have 🎯
- Publication-ready visualizations
- WandB tracking
- Ablation results for all tokens
- Statistical significance tests

---

## 10. Implementation Timeline

### Phase 1: Infrastructure (4 hours)
- Story 1: Model loading (2h)
- Story 2: Extraction pipeline setup (2h)

### Phase 2: Data Collection (3 hours)
- Story 2: Run extraction for all 3 tasks (3h)

### Phase 3: Analysis (5 hours)
- Story 3: Visualization (1h)
- Story 4: Attention hubs (1h)
- Story 5: Ablation (3h)

### Phase 4: Synthesis (4 hours)
- Story 6: Divergence (1h)
- Story 7: Report writing (2h)
- Story 8: WandB (1h, optional)

**Total**: 16 hours (can compress to 12h by skipping Story 8 and reducing ablation examples)

---

## 11. Approval Sign-Off

**Architecture Review**:
- ✅ Feasibility: Confirmed (models exist, GPU available)
- ✅ Data Quality: Verified (test sets available)
- ✅ Risk Assessment: Acceptable (low-medium risks, all mitigated)
- ✅ Timeline: Realistic (16 hours over 2-3 days)
- ✅ Deliverables: Clear and achievable

**Ready for Implementation**: ✅ **YES**

**Architect Approval**: Claude (Architect Role)
**Date**: 2025-10-30

---

## Appendix: Code Structure

```
src/experiments/three_way_comparison/
├── config.json                          # Configuration
├── model_loader.py                      # Story 1
├── extract_activations.py               # Story 2
├── visualize_embeddings.py              # Story 3
├── analyze_attention_hubs.py            # Story 4
├── ablate_tokens.py                     # Story 5
├── analyze_divergence.py                # Story 6
├── generate_report.py                   # Story 7
├── setup_wandb.py                       # Story 8 (optional)
├── test_components.py                   # Unit tests
└── utils/
    ├── data_loading.py
    ├── stratification.py
    └── plotting.py

results/three_way_comparison/
├── activations_personal_relations.npz   # 75 MB
├── activations_gsm8k.npz                # 75 MB
├── activations_commonsense.npz          # 75 MB
├── figures/                             # Visualizations
├── attention_analysis/                  # Attention results
├── ablation_results/                    # Token importance
└── divergence_analysis/                 # Layer divergence

docs/experiments/
└── 10-30_three_way_codi_comparison.md   # Final report
```
