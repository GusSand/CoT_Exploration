# Resampling Experiment Progress

**Date:** 2025-10-30
**Experiment:** LLaMA-1B GSM8K Resampling Validation
**Status:** IN PROGRESS (Story 2.2)

---

## Phase 1: Pilot Experiment (20 problems) ✓ COMPLETE

### Story 1.1: Setup ✓
- Created directory structure
- Set up utilities and W&B integration
- Time: ~15 min

### Story 1.2: Extract CT States (Pilot) ✓
- Extracted 20 problems from GSM8K test set
- Baseline accuracy: 70% (14/20)
- Cache size: 1.0 MB
- Time: ~10 min

### Story 1.3: Implement Swapping ✓
- **CRITICAL BUG DISCOVERED AND FIXED**
- Original implementation had off-by-one error
- CT5 showed 0% impact due to bug
- Fixed by checking swap BEFORE forward pass
- Time: ~3.5 hours (including debugging)

### Story 1.4: Run Pilot Resampling ✓
- 600 generations (20 problems × 5 samples × 6 positions)
- Completed with both buggy and fixed versions
- Time: ~30 min each run

### Story 1.5: Pilot Analysis ✓
**Buggy Results:**
- CT5: 0% impact (impossible!)
- CT3: -1% impact (improved performance - suspicious)
- Correlation: r = -0.751 (strongly negative - wrong direction)

**Fixed Results:**
- CT0: 10% impact
- CT1: 13% impact
- CT2: 17% impact
- CT3: 0% impact
- CT4: 19% impact
- CT5: 19% impact
- Correlation: r = -0.058, p = 0.9125 (near zero, not significant)
- **Decision: GO** ✓

---

## Phase 2: Full Experiment (100 problems) - IN PROGRESS

### Story 2.1: Extract 100 Problems ✓
- Extracted 100 problems from GSM8K test set
- Baseline accuracy: 60% (60/100)
- Cache size: 4.8 MB
- Time: ~30 seconds
- W&B run: https://wandb.ai/gussand/codi-resampling/runs/sshn7qme

### Story 2.2: Full Resampling ⏳ IN PROGRESS
- **Status:** Running in background (process 0f6bd0)
- **Configuration:**
  - 6,000 generations (100 problems × 10 samples × 6 positions)
  - Estimated time: ~5 hours
  - Started: 2025-10-30 05:20:51
  - Expected completion: ~10:20 (5 hours runtime)
- **W&B run:** https://wandb.ai/gussand/codi-resampling/runs/55hmkxb2
- **Current progress:** CT0 at 4% (~4.8 hours remaining)

### Story 2.3: Per-Problem Analysis 📋 PENDING
- Awaiting Story 2.2 completion
- Will analyze per-problem impact variance
- Identify which problems are most sensitive to swapping

### Story 2.4: Statistical Analysis 📋 PENDING
- Awaiting Story 2.2 completion
- Correlation analysis (resampling vs ablation)
- Significance testing
- Effect size calculations

### Story 2.5: Visualizations 📋 PENDING
- Awaiting Story 2.2 completion
- Per-position bar charts
- Correlation scatter plots
- Per-problem heatmaps

### Story 2.6: Final Report 📋 PENDING
- Awaiting Story 2.5 completion
- Update DATA_INVENTORY.md
- Update research journal
- Create final experiment report

---

## Key Findings So Far

### Bug Discovery
The original swapping implementation had a critical off-by-one error:
- Swapped state was set AFTER forward pass instead of BEFORE
- For CT5, the swapped value was set but never used (loop ended)
- This caused CT5 to show 0% impact and created negative correlation

**Root Cause:**
```python
# BUGGY (original):
for step in range(6):
    outputs = model.codi(inputs_embeds=latent_embd, ...)
    if step == swap_position:
        latent_embd = problem_B[step]  # Set AFTER use!

# FIXED:
for step in range(6):
    if step == swap_position:
        latent_embd = problem_B[step]  # Set BEFORE use!
    outputs = model.codi(inputs_embeds=latent_embd, ...)
```

### Pilot Results (Fixed Version)
- Non-uniform impact distribution across positions
- CT4 and CT5 show highest impact (19% each)
- CT3 shows lowest impact (0%)
- Near-zero correlation with ablation (r = -0.058)
- Suggests resampling and ablation measure different aspects

---

## Time Tracking

**Phase 1 (Pilot):**
- Estimated: 6.5 hours
- Actual: ~10 hours (including bug discovery/fix)
- Variance: +3.5 hours (debugging time)

**Phase 2 (Full) - In Progress:**
- Estimated: 8 hours total
- Completed: Story 2.1 (~0.5 hours)
- In Progress: Story 2.2 (~5 hours, 4.8 hours remaining)
- Remaining: Stories 2.3-2.6 (~2.5 hours estimated)

**Total Project:**
- Estimated: 14.5 hours
- Projected: ~18 hours (with debugging)

---

## Next Steps

1. ⏳ Wait for Story 2.2 (full resampling) to complete (~4.8 hours)
2. 📊 Run Story 2.3 (per-problem analysis)
3. 📈 Run Story 2.4 (statistical analysis)
4. 🎨 Create Story 2.5 (visualizations)
5. 📝 Write Story 2.6 (final report)
6. 🔍 Document results in research journal
7. 📦 Update DATA_INVENTORY.md

---

## Monitoring Commands

Check resampling progress:
```bash
# Check if still running
ps aux | grep "3_run_resampling.py --phase full" | grep -v grep

# View latest output
python -c "import sys; sys.path.insert(0, '/home/paperspace/dev/CoT_Exploration/src/experiments/10-30_llama_gsm8k_resampling_validation/scripts'); from utils import *; import pickle; results = pickle.load(open('../results/resampling_full_results.json', 'rb')) if os.path.exists('../results/resampling_full_results.json') else None; print('Not started yet' if results is None else f'Progress: {len(results)} positions complete')"

# Check W&B
# https://wandb.ai/gussand/codi-resampling/runs/55hmkxb2
```

---

## Files Created

**Scripts:**
- `utils.py` - Shared utilities
- `1_extract_ct_states.py` - CT hidden state extraction
- `2_implement_swapping.py` - CT swapping function (FIXED)
- `3_run_resampling.py` - Resampling experiment runner
- `4_analyze_pilot.py` - Pilot analysis and Go/No-Go decision
- `5_debug_swapping.py` - Diagnostic tests for swapping

**Data:**
- `data/ct_hidden_states_cache_pilot.pkl` - 20 problems (1.0 MB)
- `data/ct_hidden_states_cache_full.pkl` - 100 problems (4.8 MB)

**Results:**
- `results/resampling_pilot_results_BUGGY.json` - Pilot with bug
- `results/resampling_pilot_results.json` - Pilot fixed
- `results/pilot_analysis.md` - GO decision report
- `results/pilot_resampling_vs_ablation.png` - Correlation plot
- `results/resampling_full_results.json` - IN PROGRESS

**Documentation:**
- `README.md` - Experiment overview
- `BUGFIX_REPORT.md` - Bug analysis and fix
- `PROGRESS.md` - This file

---

Last updated: 2025-10-30 05:22:00 UTC
