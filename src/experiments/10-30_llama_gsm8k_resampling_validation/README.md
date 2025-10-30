# Resampling Validation Experiment

**Date:** 2025-10-30
**Model:** LLaMA-1B GSM8K CODI
**Purpose:** Validate thought anchor findings through convergent evidence

---

## Overview

This experiment validates whether CT positions contain localized, problem-specific information by swapping individual CT tokens between problems and measuring accuracy impact. Results are compared with ablation studies to provide convergent evidence for thought anchor identification.

---

## Research Question

**Do CT positions contain localized, problem-specific information (i.e., are they true 'thought anchors')?**

Success criteria:
- At least one position shows >15% resampling impact
- Positive correlation with ablation (r > 0.3, p < 0.05)
- CT0 or CT5 (ideally both) in top 3 positions

---

## Architecture

### Phase 1: Pilot (20 problems)
1. **Story 1.1:** Project setup
2. **Story 1.2:** Extract CT hidden states
3. **Story 1.3:** Implement CT swapping function
4. **Story 1.4:** Run pilot resampling (600 generations)
5. **Story 1.5:** Analyze pilot & Go/No-Go decision

### Phase 2: Full (100 problems) - Only if pilot succeeds
6. **Story 2.1:** Extract CT hidden states (full)
7. **Story 2.2:** Run full resampling (6,000 generations)
8. **Story 2.3:** Create convergence analysis visualization
9. **Story 2.4:** Generate summary statistics table
10. **Story 2.5:** Statistical significance testing
11. **Story 2.6:** Final experiment documentation

---

## Directory Structure

```
10-30_llama_gsm8k_resampling_validation/
├── scripts/
│   ├── utils.py                     # Shared utilities
│   ├── 1_extract_ct_states.py       # Extract & cache CT hidden states
│   ├── 2_implement_swapping.py      # CT token swapping function + validation
│   ├── 3_run_resampling.py          # Main resampling experiment
│   └── 4_analyze_pilot.py           # Pilot analysis & Go/No-Go decision
├── data/
│   ├── ct_hidden_states_cache_pilot.pkl   # Pilot cache (20 problems)
│   └── ct_hidden_states_cache_full.pkl    # Full cache (100 problems) - if GO
├── results/
│   ├── resampling_pilot_results.json      # Pilot resampling results
│   ├── pilot_resampling_vs_ablation.png   # Pilot visualization
│   ├── pilot_analysis.md                  # Go/No-Go analysis report
│   └── resampling_full_results.json       # Full results - if GO
└── README.md                         # This file
```

---

## Usage

### Phase 1: Pilot Experiment

```bash
cd src/experiments/10-30_llama_gsm8k_resampling_validation/scripts

# Story 1.2: Extract CT hidden states (pilot)
python 1_extract_ct_states.py --phase pilot --n_problems 20

# Story 1.3: Validate swapping function
python 2_implement_swapping.py

# Story 1.4: Run pilot resampling
python 3_run_resampling.py --phase pilot --n_samples 5

# Story 1.5: Analyze pilot & make Go/No-Go decision
python 4_analyze_pilot.py
```

### Phase 2: Full Experiment (if GO)

```bash
# Story 2.1: Extract CT hidden states (full)
python 1_extract_ct_states.py --phase full --n_problems 100

# Story 2.2: Run full resampling
python 3_run_resampling.py --phase full --n_samples 10

# Story 2.3-2.6: Analysis (TBD)
```

---

## Reference Data

### Ablation Impact (from research journal)
- CT0: 18.7% drop
- CT1: 12.8% drop
- CT2: 14.6% drop
- CT3: 15.0% drop
- CT4: 3.5% drop
- CT5: 26.0% drop

### Attention Weights (CT → CT0)
- CT1: 4.77% (±1.07%)
- CT2: 4.21% (±1.00%)
- CT3: 3.53% (±0.79%)
- CT4: 2.76% (±0.82%)
- CT5: 3.04% (±0.93%)

---

## Implementation Notes

### CT Hidden State Extraction
- **Layer:** Final layer output (`outputs.hidden_states[-1]`)
- **Shape:** `[6, 2048]` for LLaMA-1B
- **Validation:** Checks for NaN, Inf, abnormal values

### CT Token Swapping
- **Method:** KV cache manipulation (not hooks)
- **Strategy:** Replace hidden state at `swap_position` with cached state from different problem
- **Validation:** Self-swap test, reproducibility test, position variance

### Random Seed Management
- **Seed:** 42 (fixed across all scripts)
- **Scope:** Python, NumPy, PyTorch (CPU+GPU), CUDNN, PYTHONHASHSEED

---

## W&B Tracking

Project: [`codi-resampling`](https://wandb.ai/gussand/codi-resampling)

Runs:
- `pilot_extraction` - Story 1.2
- `pilot_resampling` - Story 1.4
- `pilot_analysis` - Story 1.5
- `full_extraction` - Story 2.1 (if GO)
- `full_resampling` - Story 2.2 (if GO)
- `full_analysis` - Story 2.6 (if GO)

---

## Time Estimates

### Phase 1 (Pilot)
- Story 1.1: 1h (setup)
- Story 1.2: 2.5h (extraction: 2h coding + 10min runtime)
- Story 1.3: 3.5h (swapping: 3h coding + 30min validation)
- Story 1.4: 2.5h (resampling: 2h coding + 30min runtime)
- Story 1.5: 1.5h (analysis)
- **Total:** 11 hours

### Phase 2 (Full) - if GO
- Story 2.1: 1.5h (extraction: 30min coding + 50min runtime)
- Story 2.2: 6h (resampling: 1h coding + 5h runtime)
- Story 2.3: 1.5h (visualization)
- Story 2.4: 1h (summary table)
- Story 2.5: 1.5h (statistical testing)
- Story 2.6: 2h (documentation)
- **Total:** 13.5 hours

**Grand Total:** 24.5 hours (pilot + full)

---

## Success Scenarios

### Strong Success (r > 0.7, p < 0.05)
- CT0: 15-20% resampling impact (ablation: 18.7%)
- CT5: 22-28% resampling impact (ablation: 26%)
- **Claim:** "Convergent evidence from three methods identifies CT0 and CT5 as thought anchors"

### Moderate Success (0.4 < r < 0.7)
- General agreement but some dissociation
- **Claim:** "Convergent evidence with some task-specific variation"

### Interesting Dissociation (r < 0.3)
- Ablation and resampling identify different positions
- **Interpretation:** Computational importance ≠ information storage
- **Claim:** "Task execution and information storage are mechanistically separate"

---

## Documentation

### Architecture Specification
- `docs/architecture/2025-10-30_resampling_architecture.md`

### Experiment Results (after completion)
- `docs/experiments/10-30_llama_gsm8k_resampling_validation.md`

### Research Journal Entry (after completion)
- `docs/research_journal.md` - High-level summary

### Data Inventory (after completion)
- `docs/DATA_INVENTORY.md` - Cache file documentation

---

## Contact

For questions about this experiment, see:
- PM handoff document (conversation history)
- Architecture specification (`docs/architecture/`)
- Research journal (`docs/research_journal.md`)
