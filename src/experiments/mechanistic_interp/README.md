# Mechanistic Interpretability of Continuous Chain-of-Thought

**Objective:** Understand how CODI (Continuous Chain-of-Thought with Distillation) performs reasoning through mechanistic interpretability experiments.

**Status:** ✅ MECH-02 Complete | ⏸️ Paused (awaiting improved SAE model)

---

## Quick Start

### Current Status
- **MECH-01** (Data Prep): ✅ Complete
- **MECH-02** (Step Importance): ✅ Complete - **Progressive refinement discovered!**
- **MECH-03** (Feature Extraction): ⏸️ Blocked (waiting for better SAE)
- **MECH-04** (Correlation Analysis): ⏸️ Waiting on MECH-03
- **MECH-06** (Interventions): ⏸️ Waiting on MECH-04

### Key Finding (MECH-02)
**Late continuous thought positions (4,5) are most critical, not early positions!**
- Position 5: 86.8% importance (final commitment)
- Position 1: 14.5% importance (exploration)
- Interpretation: CODI uses **progressive refinement** strategy

See: `PROJECT_STATUS.md` for full details

---

## Directory Structure

```
mechanistic_interp/
├── README.md                           ← You are here
├── PROJECT_STATUS.md                   ← Full project status & roadmap
│
├── scripts/                            ← Executable experiments
│   ├── 01_validate_data.py            ← Data validation (MECH-01)
│   └── 02_measure_step_importance.py  ← Step importance (MECH-02)
│
├── utils/                              ← Reusable infrastructure
│   ├── codi_interface.py              ← CODIInterface, StepImportanceMeasurer
│   ├── test_step_importance.py        ← Basic validation tests
│   ├── test_harder_problems.py        ← Sensitivity validation
│   └── test_debug_zeroing.py          ← Hook verification
│
├── data/                               ← Experimental data
│   ├── stratified_test_problems.json  ← 1,000 test problems (MECH-01)
│   ├── step_importance_scores.json    ← Full results (MECH-02, 1.6 MB)
│   ├── step_importance_summary_stats.json  ← Statistics (MECH-02)
│   └── step_importance_validation.json     ← 87-problem validation
│
└── [Status Documents]                  ← Experiment tracking
    ├── MECH-01_COMPLETION_SUMMARY.md
    ├── MECH-02_COMPLETION_SUMMARY.md
    └── FULL_SWEEP_IN_PROGRESS.md
```

---

## Running Experiments

### MECH-02: Step Importance Analysis

```bash
cd /home/paperspace/dev/CoT_Exploration/src/experiments/mechanistic_interp/scripts
export PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH
python 02_measure_step_importance.py
```

**What it does:**
1. Loads CODI model (LLaMA-3.2-1B-Instruct + 6 latent tokens)
2. For each of 1,000 problems:
   - Generate baseline answer (full continuous thoughts)
   - Ablate positions 0-4 one at a time (zero via forward hook)
   - Measure importance (did ablation break reasoning?)
3. Compute statistics by position and difficulty level

**Expected output:**
- Validation: ~2.5 minutes (87 problems)
- Full sweep: ~46 minutes (1,000 problems)
- Results: `data/step_importance_*.json`

---

## Reusable Code

### CODIInterface

```python
from utils.codi_interface import CODIInterface

# Load CODI
interface = CODIInterface('~/codi_ckpt/llama_gsm8k')

# Generate answer
answer = interface.generate_answer("What is 15 * 3?")

# Extract continuous thoughts (layer 8, all 6 positions)
thoughts = interface.extract_continuous_thoughts("What is 15 * 3?", layer_idx=8)
# Returns: List[Tensor(1, 2048)] - 6 positions
```

### StepImportanceMeasurer

```python
from utils.codi_interface import CODIInterface, StepImportanceMeasurer

interface = CODIInterface('~/codi_ckpt/llama_gsm8k')
measurer = StepImportanceMeasurer(interface, layer_idx=8, debug=False)

# Measure importance of position 3
result = measurer.measure_position_importance(
    problem_text="What is 15 * 3?",
    position=3  # Ablate positions [0,1,2], keep [3,4,5]
)

# Returns:
# {
#   'baseline_answer': '45',
#   'baseline_correct': True,
#   'ablated_answer': '42',
#   'ablated_correct': False,
#   'importance_score': 1.0  # Ablation broke reasoning
# }
```

---

## Key Results (MECH-02)

### Position-wise Importance

| Position | Importance | Interpretation |
|----------|-----------|----------------|
| 0 | 0.000 | Baseline (no ablation) |
| 1 | 0.145 | Exploration (low importance) |
| 2 | 0.468 | Solution space narrowing |
| 3 | 0.528 | Refinement begins |
| 4 | 0.556 | Solution converging |
| 5 | **0.868** | **Final commitment (critical!)** |

**Pattern:** Monotonic increase (Spearman ρ=0.99, p<0.001)

### Progressive Refinement Strategy

```
Position 0-2: Exploration Phase
              - Low importance (14-47%)
              - Can recover from errors
              - Rough context establishment

Position 3-4: Refinement Phase
              - Moderate importance (53-56%)
              - Solution space narrowing
              - Approach focusing

Position 5:   Commitment Phase
              - HIGH importance (87%)
              - Final decision locked in
              - Errors are fatal
```

**Key Insight:** Opposite of explicit CoT where "if step 1 is wrong, everything fails"

---

## Documentation

### Detailed Reports
- **Complete methodology & results:** `docs/experiments/10-26_llama_gsm8k_step_importance.md`
- **High-level summary:** `docs/research_journal.md` (entry: 2025-10-26d)

### Internal Documentation
- **Project status & roadmap:** `PROJECT_STATUS.md` (this directory)
- **Completion summaries:** `MECH-*_COMPLETION_SUMMARY.md` files

### Git History
- **Latest commit:** `68d0fa0` - feat: Complete MECH-02 step importance analysis
- **Branch:** master
- **Remote:** origin/master (pushed)

---

## Dependencies

### Model
- **CODI checkpoint:** `~/codi_ckpt/llama_gsm8k/`
- **Base model:** LLaMA-3.2-1B-Instruct
- **CODI config:** 6 latent tokens, LoRA fine-tuned, 2048-dim projection

### Python Packages
- PyTorch (model execution)
- Transformers (LLaMA loading)
- PEFT (LoRA)
- NumPy, tqdm, json (utilities)

### Environment
```bash
export PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH
```

---

## Future Work

### When SAE Model is Ready:

1. **MECH-03: Feature Extraction**
   - Load improved SAE (≥70% EV, <30% death)
   - Extract 2048 features × 6 positions
   - Focus on position 5 (most important)

2. **MECH-04: Correlation Analysis**
   - Rank features by correlation with correctness
   - Identify "commitment features" vs "exploration features"
   - Validate position 5 features are most discriminative

3. **MECH-06: Intervention Framework**
   - Test late-stage steering (positions 4-5)
   - Feature ablation experiments
   - Feature enhancement experiments

**Estimated total time:** 6-8 hours computation + 2-3 hours documentation

---

## Troubleshooting

### CODI Model Not Found
```bash
# Check checkpoint location
ls ~/codi_ckpt/llama_gsm8k/

# Expected files:
# - adapter_config.json
# - adapter_model.safetensors
# - model_config.json
```

### PYTHONPATH Issues
```bash
# Must include codi directory in PYTHONPATH
export PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH

# Verify
python -c "import codi; print('CODI loaded successfully')"
```

### Memory Issues
- MECH-02 requires ~8 GB GPU memory
- Use `nvidia-smi` to check availability
- Consider reducing batch size if needed

---

## Contact

For questions about this work:
- See `PROJECT_STATUS.md` for full context
- Check `docs/experiments/10-26_llama_gsm8k_step_importance.md` for complete methodology
- Review git history: `git log --oneline src/experiments/mechanistic_interp/`

**Status:** Ready to resume when improved SAE model is available
