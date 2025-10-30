# Experiment: Pre-Compression Deception Signal Analysis (LLaMA-1B)

**Date**: 2025-10-30
**Model**: LLaMA-3.2-1B (1B parameters, 16 layers)
**Dataset**: LIARS-BENCH
**Research Question**: WHERE is deception information lost during continuous thought compression?
**Status**: PLANNED

---

## Executive Summary

**Objective**: Identify the specific layers where deception detection signal degrades during CODI's compression process.

**Hypothesis**: Deception signals exist in early/mid layers (0-9) but disappear during compression to continuous thought tokens (CT0-CT5), while remaining detectable in final response tokens.

**Method**: Train logistic regression probes at multiple layers × token positions to create a signal degradation map.

**Expected Result**:
- Early layers (0, 3, 6): 55-65% accuracy (weak signal exists)
- Mid layers (9, 12): Signal degradation begins
- Late layers (15): CT positions ~50% (signal lost), response positions 60-70% (signal preserved)

---

## Architecture Verification

**LLaMA-3.2-1B Architecture**:
- Layers: 16 (indexed 0-15)
- Hidden size: 2048
- Attention heads: 32
- Parameters: ~1B

**Layers to Probe**: [0, 3, 6, 9, 12, 15] (6 layers)
**Positions to Probe**: ['question_last', 'ct0', 'ct1', 'ct2', 'ct3', 'ct4', 'ct5', 'answer_first'] (8 positions)
**Total Probes**: 6 × 8 = 48

---

## Data Verification

**LIARS-BENCH Data Status** (verified 2025-10-30):

**Training Set**:
- File: `src/experiments/liars_bench_codi/data/processed/probe_train_proper.json`
- Samples: 288 (144 honest, 144 deceptive)
- Balance: 50/50 (perfectly balanced)

**Test Set**:
- File: `src/experiments/liars_bench_codi/data/processed/probe_test_proper.json`
- Samples: 288 (144 honest, 144 deceptive)
- Balance: 50/50 (perfectly balanced)

**Data Quality**:
- Zero overlap between train/test (verified)
- Question-level split methodology (proper held-out)
- Created: 2025-10-28

**CODI Training Set**:
- File: `src/experiments/liars_bench_codi/data/processed/train_proper.json`
- Samples: 6,405
- Purpose: Train CODI model to answer honestly

---

## Missing Resources

**CRITICAL**: LLaMA-1B CODI checkpoint for LIARS-BENCH does NOT exist yet.

**Available Checkpoints**:
- LLaMA-3.2-3B LIARS-BENCH: `/home/paperspace/codi_ckpt/llama3b_liars_bench_proper/`
- LLaMA-1B GSM8K: `/home/paperspace/codi_ckpt/llama_commonsense/gsm8k_llama1b_latent_baseline`
- LLaMA-1B Personal Relations: `/home/paperspace/dev/CoT_Exploration/models/personal_relations_1b_codi_v2/`

**Action Required**: Train LLaMA-1B CODI on LIARS-BENCH (Phase 1)

---

## User Stories

### **PHASE 1: LLaMA-1B CODI Training**

#### **Story 1.1: Create LLaMA-1B Training Script**
**Priority**: CRITICAL
**Estimated Time**: 30 minutes
**Status**: PENDING

**As a** researcher
**I want** a training script for LLaMA-1B on LIARS-BENCH
**So that** I can train the CODI model needed for the experiment

**Acceptance Criteria**:
- [ ] Training script created at `src/experiments/liars_bench_codi/scripts/train_llama1b.sh`
- [ ] Script based on existing templates: `codi/scripts/train_llama1b_personal_relations_v2.sh`
- [ ] Uses proper training data: `train_proper.json` (6,405 samples)
- [ ] Model: `meta-llama/Llama-3.2-1B-Instruct`
- [ ] Configuration:
  - Learning rate: 8e-4 (conservative to avoid divergence)
  - Batch size: 8 per device
  - Gradient accumulation: 16 steps (effective batch: 128)
  - Epochs: 10-15
  - LoRA: r=128, alpha=32
  - Num latent tokens: 6 (ct0-ct5)
- [ ] WandB logging configured:
  - Project: `deception-detection-llama1b`
  - Tags: `['llama-1b', 'liars-bench', 'codi', 'precompression-analysis']`
- [ ] Script tested with dry-run (prints config, no training)

**Implementation Notes**:
- Use `train_liars_bench.py` (already exists, handles LIARS-BENCH data)
- Lower learning rate than 3B (3e-3 caused divergence in LLaMA-3B)
- Monitor for divergence, stop if loss > 5.0 after initial descent

**Cost**: None (script creation only)

---

#### **Story 1.2: Train LLaMA-1B CODI Model**
**Priority**: CRITICAL
**Estimated Time**: 4-6 hours (GPU time)
**Estimated Cost**: $10-15
**Status**: PENDING
**Dependencies**: Story 1.1

**As a** researcher
**I want** to train LLaMA-1B CODI on LIARS-BENCH
**So that** I have the checkpoint needed for activation extraction

**Acceptance Criteria**:
- [ ] Training launched successfully using `train_llama1b.sh`
- [ ] Training completes 10-15 epochs without divergence
- [ ] Final loss < 1.5
- [ ] Loss trajectory shows stable descent (no spikes)
- [ ] Checkpoint saved to: `~/codi_ckpt/llama1b_liars_bench_proper/`
- [ ] WandB logs captured with metrics:
  - Training loss (explicit + distillation)
  - CE loss, distillation loss, ref CE loss
  - Learning rate schedule
  - Gradient norms

**Success Criteria**:
- Loss drops from ~4-5 to < 1.5
- No divergence (loss spikes > 5.0)
- Model learns task (can generate coherent answers)

**Rollback Plan**:
- If divergence occurs: Lower learning rate to 5e-4 and restart
- If memory issues: Reduce batch size or increase gradient accumulation

**Cost**: $10-15 (GPU hours on A100)

---

#### **Story 1.3: Validate LLaMA-1B Checkpoint**
**Priority**: HIGH
**Estimated Time**: 15 minutes
**Status**: PENDING
**Dependencies**: Story 1.2

**As a** researcher
**I want** to verify the trained checkpoint loads correctly
**So that** I can proceed with confidence to activation extraction

**Acceptance Criteria**:
- [ ] Checkpoint loads without errors
- [ ] Model architecture verified:
  - 16 layers (0-15)
  - Hidden size: 2048
  - Has CODI modifications (bot_id, eot_id)
- [ ] Can perform inference on 3-5 test samples
- [ ] Output includes continuous thought tokens (ct0-ct5)
- [ ] Output format matches expected CODI structure
- [ ] Final loss from checkpoint matches WandB logs

**Validation Script**:
```python
# Quick checkpoint validation
from transformers import AutoModelForCausalLM
import torch

checkpoint_path = "~/codi_ckpt/llama1b_liars_bench_proper/.../checkpoint-XXX"
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)

print(f"Layers: {model.config.num_hidden_layers}")
print(f"Hidden: {model.config.hidden_size}")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# Test inference on sample
sample = "Question: What is 2+2? Answer:"
# ... run inference ...
```

**Cost**: None (quick validation)

---

### **PHASE 2: Pre-Compression Deception Signal Analysis**

#### **Story 2.1: Extract Multi-Layer Activations**
**Priority**: CRITICAL
**Estimated Time**: 1.5 hours
**Status**: PENDING
**Dependencies**: Story 1.3

**As a** researcher
**I want** to extract activations from layers [0, 3, 6, 9, 12, 15] at all token positions
**So that** I can analyze signal degradation across model depth

**Acceptance Criteria**:
- [ ] Script created: `src/experiments/liars_bench_codi/scripts/extract_activations_llama1b_multilayer.py`
- [ ] Based on: `extract_activations_llama3b_proper.py` (existing)
- [ ] Extracts from layers: [0, 3, 6, 9, 12, 15]
- [ ] Extracts from positions: ['question_last', 'ct0', 'ct1', 'ct2', 'ct3', 'ct4', 'ct5', 'answer_first']
- [ ] Processes both train and test splits:
  - Train: 288 samples (probe_train_proper.json)
  - Test: 288 samples (probe_test_proper.json)
- [ ] Output format:
  ```python
  {
    "activations": {
      "layer_0": {"question_last": [...], "ct0": [...], ...},
      "layer_3": {"question_last": [...], "ct0": [...], ...},
      ...
    },
    "labels": [0, 1, 1, 0, ...],  # 0=honest, 1=deceptive
    "metadata": {
      "model": "Llama-3.2-1B",
      "layers": [0, 3, 6, 9, 12, 15],
      "positions": ["question_last", "ct0", ...],
      "n_samples": 288,
      "split": "train"
    }
  }
  ```
- [ ] Saved to:
  - `data/processed/multilayer_activations_llama1b_train.json`
  - `data/processed/multilayer_activations_llama1b_test.json`
- [ ] Activations shape verified: `[n_samples, hidden_dim]` per layer-position
- [ ] Script completes in < 30 minutes

**Technical Details**:
- Use `model.model.layers[i]` to access layer outputs
- Extract hidden states after layer norm
- Use torch.no_grad() for efficiency
- Save as float32 (not float16) for probe training

**Cost**: Minimal (inference only)

---

#### **Story 2.2: Train Probes for All Layer-Position Combinations**
**Priority**: CRITICAL
**Estimated Time**: 2 hours
**Status**: PENDING
**Dependencies**: Story 2.1

**As a** researcher
**I want** to train logistic regression probes for each layer-position combination
**So that** I can measure deception signal strength across the model

**Acceptance Criteria**:
- [ ] Script created: `src/experiments/liars_bench_codi/scripts/train_multilayer_probes_llama1b.py`
- [ ] Trains 48 probes: 6 layers × 8 positions
- [ ] Each probe configuration:
  - Model: Logistic Regression (sklearn)
  - Max iterations: 1000
  - Solver: 'lbfgs'
  - Regularization: C=1.0
- [ ] Proper train/test split maintained:
  - Train: 288 samples (144 honest, 144 deceptive)
  - Test: 288 samples (144 honest, 144 deceptive)
- [ ] Metrics computed per probe:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Confusion matrix
- [ ] Results saved to: `results/multilayer_probe_results_llama1b.json`
- [ ] Output format:
  ```python
  {
    "layer_0": {
      "question_last": {"accuracy": 0.52, "precision": 0.51, ...},
      "ct0": {"accuracy": 0.50, ...},
      ...
    },
    "layer_3": {...},
    ...
    "metadata": {
      "model": "Llama-3.2-1B",
      "train_samples": 288,
      "test_samples": 288,
      "layers": [0, 3, 6, 9, 12, 15],
      "positions": ["question_last", "ct0", ...]
    }
  }
  ```
- [ ] Progress bar shows training progress
- [ ] Script completes in < 2 hours

**Data Quality Check**:
- Verify no data leakage (train/test separate)
- Verify balanced classes (50/50 split maintained)
- Check for NaN values in activations

**Cost**: None (CPU only)

---

#### **Story 2.3: Generate Heatmap Visualization**
**Priority**: HIGH
**Estimated Time**: 1 hour
**Status**: PENDING
**Dependencies**: Story 2.2

**As a** researcher
**I want** a heatmap showing probe accuracy by layer (y-axis) and position (x-axis)
**So that** I can visualize where deception signal is lost

**Acceptance Criteria**:
- [ ] Script created: `src/experiments/liars_bench_codi/scripts/visualize_multilayer_results.py`
- [ ] Heatmap configuration:
  - Y-axis: Layers [0, 3, 6, 9, 12, 15]
  - X-axis: Positions ['question_last', 'ct0', 'ct1', 'ct2', 'ct3', 'ct4', 'ct5', 'answer_first']
  - Color scale: 0.45-0.75 (centered at 0.50 = random chance)
  - Colormap: 'RdYlGn' (red=poor, yellow=random, green=good)
- [ ] Annotations:
  - Accuracy percentage in each cell
  - Bold/highlight cells > 55% (signal present)
  - Grid lines between cells
- [ ] Labels:
  - Clear axis labels
  - Title: "Deception Detection Accuracy by Layer and Position (LLaMA-1B)"
  - Colorbar with "Accuracy" label
- [ ] Saved to: `results/figures/multilayer_heatmap_llama1b.png`
- [ ] High resolution: 300 DPI, 12×8 inches
- [ ] Readable font sizes (axis: 12pt, title: 14pt, annotations: 10pt)

**Design Notes**:
- Use seaborn.heatmap for clean visualization
- Add horizontal line separating early/mid/late layers
- Add vertical line separating CT positions from question/answer

**Cost**: None

---

#### **Story 2.4: Generate Line Plot by Position**
**Priority**: HIGH
**Estimated Time**: 45 minutes
**Status**: PENDING
**Dependencies**: Story 2.2

**As a** researcher
**I want** line plots showing accuracy by layer for each position type
**So that** I can identify signal degradation patterns

**Acceptance Criteria**:
- [ ] Line plot configuration:
  - X-axis: Layers [0, 3, 6, 9, 12, 15]
  - Y-axis: Accuracy (0.45 - 0.75 range)
  - 8 lines (one per position)
  - Different colors/markers per position
- [ ] Plot elements:
  - Horizontal line at 50% (random baseline) - dashed, gray
  - Horizontal line at 55% (signal threshold) - dotted, gray
  - Grid for easier reading
  - Legend clearly identifying each position
- [ ] Labels:
  - Title: "Deception Detection Signal Across Layers (LLaMA-1B)"
  - X-axis: "Layer"
  - Y-axis: "Probe Accuracy"
- [ ] Saved to: `results/figures/multilayer_lineplot_llama1b.png`
- [ ] High resolution: 300 DPI, 10×6 inches

**Additional Analysis**:
- [ ] Create second plot grouping positions:
  - Group 1: CT positions (ct0-ct5)
  - Group 2: Question/Answer positions
  - Shows average accuracy per group by layer

**Cost**: None

---

#### **Story 2.5: Statistical Analysis & Key Findings**
**Priority**: HIGH
**Estimated Time**: 1 hour
**Status**: PENDING
**Dependencies**: Stories 2.2, 2.3, 2.4

**As a** researcher
**I want** statistical analysis of signal degradation patterns
**So that** I can make evidence-based conclusions

**Acceptance Criteria**:
- [ ] Analysis script created: `src/experiments/liars_bench_codi/scripts/analyze_multilayer_patterns.py`
- [ ] Key metrics computed:
  - **Signal threshold**: Layer where CT positions drop below 55% accuracy
  - **Signal loss**: Layer where CT positions reach 50% ± 2%
  - **Early layer signal**: Average accuracy layers 0-6 for CT positions
  - **Late layer signal**: Average accuracy layers 9-15 for CT positions
  - **Response preservation**: Accuracy of answer_first across all layers
- [ ] Statistical tests:
  - T-test: Early vs late layer accuracy for CT positions
  - T-test: CT positions vs question/answer positions (layer-matched)
  - Effect size (Cohen's d) for significant differences
- [ ] Pattern identification:
  - Does signal degrade monotonically or suddenly?
  - Which layer shows largest drop?
  - Do all CT positions degrade equally?
- [ ] Key finding statement generated:
  - Format: "Signal exists at layer X (accuracy Y%) but disappears by layer Z (accuracy W%)"
  - Must be specific with numbers and statistical significance
- [ ] Results saved to: `results/multilayer_statistical_analysis_llama1b.json`

**Expected Patterns**:
- Early layers (0, 3, 6): 55-65% accuracy (weak signal)
- Mid layers (9, 12): Degradation begins
- Late layers (15): CT ~50%, response 60-70%

**Cost**: None

---

#### **Story 2.6: Document Results**
**Priority**: HIGH
**Estimated Time**: 1.5 hours
**Status**: PENDING
**Dependencies**: Story 2.5

**As a** researcher
**I want** comprehensive documentation of the experiment
**So that** results are reproducible and findings are clear

**Acceptance Criteria**:
- [ ] Update this file with:
  - [ ] Status changed to "COMPLETE"
  - [ ] Results section added with:
    - Key findings
    - Statistical analysis summary
    - Heatmap and line plot embedded
    - Comparison to hypothesis
  - [ ] Configuration section added with:
    - All hyperparameters used
    - Checkpoint paths
    - Data splits
  - [ ] Reproducibility section added with:
    - Exact commands to reproduce
    - Environment requirements
    - Compute resources used
- [ ] Update `docs/research_journal.md`:
  - [ ] Add entry dated 2025-10-30
  - [ ] High-level summary (3-4 sentences)
  - [ ] Key finding highlighted
  - [ ] Link to this detailed report
- [ ] Update `docs/DATA_INVENTORY.md`:
  - [ ] Add entry for multi-layer activation dataset
  - [ ] Include: location, size, creation method
  - [ ] Link to this experiment
  - [ ] Document stratification (6 layers × 8 positions)

**Documentation Standards** (per CLAUDE.md):
- Use descriptive headers
- Include all hyperlinks to data/code
- Provide recreation instructions
- Document how data was stratified
- Include sample counts and splits

**Cost**: None

---

#### **Story 2.7: Commit & Push Results**
**Priority**: HIGH
**Estimated Time**: 15 minutes
**Status**: PENDING
**Dependencies**: Story 2.6

**As a** researcher
**I want** all code, data, and documentation committed to GitHub
**So that** work is preserved and teammates have access

**Acceptance Criteria**:
- [ ] Stage new scripts:
  - `scripts/train_llama1b.sh`
  - `scripts/extract_activations_llama1b_multilayer.py`
  - `scripts/train_multilayer_probes_llama1b.py`
  - `scripts/visualize_multilayer_results.py`
  - `scripts/analyze_multilayer_patterns.py`
- [ ] Stage documentation:
  - `docs/experiments/10-30_llama1b_liars_bench_precompression_signal_analysis.md` (this file)
  - `docs/research_journal.md`
  - `docs/DATA_INVENTORY.md`
- [ ] Stage visualizations:
  - `results/figures/multilayer_heatmap_llama1b.png`
  - `results/figures/multilayer_lineplot_llama1b.png`
- [ ] Stage results (JSON files are typically small enough):
  - `results/multilayer_probe_results_llama1b.json`
  - `results/multilayer_statistical_analysis_llama1b.json`
- [ ] Create descriptive commit message:
  ```
  feat: Pre-compression deception signal analysis (LLaMA-1B)

  Complete analysis of deception signal degradation across model layers.
  Train LLaMA-1B CODI on LIARS-BENCH and probe 6 layers × 8 positions.

  Key finding: [Insert key finding from analysis]

  Results:
  - LLaMA-1B CODI trained on LIARS-BENCH (6,405 samples)
  - 48 probes trained (6 layers × 8 positions)
  - Heatmap and line plot visualizations
  - Statistical analysis of signal degradation

  Time: ~10-14 hours
  Cost: ~$10-15

  Generated with Claude Code (https://claude.com/claude-code)

  Co-Authored-By: Claude <noreply@anthropic.com>
  ```
- [ ] Push to GitHub
- [ ] Verify push successful
- [ ] Check .gitignore excludes large files (checkpoints, activations > 100MB)

**Cost**: None

---

## Summary

**Total Stories**: 10
**Total Estimated Time**: 10-14 hours
- Phase 1 (Training): 4-6 hours
- Phase 2 (Analysis): 6-8 hours

**Total Estimated Cost**: $10-15 (GPU training only)

**Key Deliverables**:
1. LLaMA-1B CODI checkpoint trained on LIARS-BENCH
2. Multi-layer activation dataset (6 layers × 8 positions × 288 samples)
3. 48 trained probes with performance metrics
4. Heatmap visualization (layer × position accuracy)
5. Line plot visualization (accuracy by layer per position)
6. Statistical analysis report
7. Complete documentation in research journal and DATA_INVENTORY

---

## Expected Outcome

**Hypothesis Validation**:

We expect to find:
1. **Early layers (0, 3, 6)**: Weak deception signal present (55-65% accuracy)
2. **Mid layers (9, 12)**: Signal degradation begins
3. **Late layers (15)**:
   - CT positions: ~50% (signal lost during compression)
   - Response positions: 60-70% (signal preserved in language space)

**Key Mechanistic Insight**:
This will reveal WHERE the information bottleneck occurs during CODI's compression process, explaining WHY continuous thoughts fail at deception detection despite response tokens succeeding.

**Impact**:
- Identifies the compression bottleneck layer
- Guides future work on improving continuous thought representations
- Provides evidence for information loss hypothesis vs representation quality hypothesis

---

## Reproducibility

**Environment Requirements**:
- Python 3.9+
- PyTorch 2.0+
- Transformers 4.40+
- scikit-learn 1.3+
- matplotlib, seaborn
- WandB account

**Compute Requirements**:
- Training: 1× A100 80GB (or 2× A100 40GB)
- Inference/Probes: CPU sufficient (or 1× T4/V100)

**Commands to Reproduce**:
```bash
# Phase 1: Train LLaMA-1B CODI
cd /home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi
bash scripts/train_llama1b.sh

# Phase 2: Extract activations and train probes
python scripts/extract_activations_llama1b_multilayer.py
python scripts/train_multilayer_probes_llama1b.py
python scripts/visualize_multilayer_results.py
python scripts/analyze_multilayer_patterns.py
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-30 | Initial experiment plan created |

---

## Notes

- This experiment is CRITICAL for understanding the mechanistic failure mode of CODI on deception detection
- Results will inform whether the issue is compression (information loss) vs representation (wrong features)
- If signal exists in early layers but disappears, it's compression; if never exists, it's representation
- LLaMA-1B chosen for consistency with other experiments in the project (per user requirement)
