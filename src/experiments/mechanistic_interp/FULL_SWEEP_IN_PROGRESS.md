# MECH-02: Full Sweep In Progress

**Started:** 2025-10-26 16:33:16 UTC
**Process ID:** 15122
**Status:** 🟢 RUNNING

---

## Progress Tracking

### Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Model Loading | ~10 sec | ✅ Complete |
| Validation (87 problems) | ~2.5 min | 🔄 Running |
| Full Sweep (1000 problems) | ~50 min | ⏳ Pending |
| **Total Estimated** | **~53 min** | 🔄 **In Progress** |

### Checkpoints

The script will create checkpoints every 500 problems:
- `checkpoint_500.json` - After 500 problems (~25 min)
- `checkpoint_1000.json` - After 1000 problems (~50 min)

### Current Status (Live)

```bash
# Check process
ps aux | grep 02_measure_step_importance.py | grep -v grep

# Check for new files
ls -lht /home/paperspace/dev/CoT_Exploration/src/experiments/mechanistic_interp/data/

# Monitor memory usage
free -h
```

---

## What's Being Computed

### For Each Problem (1000 total):

1. **Generate Baseline Answer**
   - Use full continuous thoughts [0,1,2,3,4,5]
   - Check correctness vs expected answer

2. **Measure Position Importance** (6 ablations per problem)
   - Position 0: No ablation (baseline)
   - Position 1: Ablate [0], keep [1,2,3,4,5]
   - Position 2: Ablate [0,1], keep [2,3,4,5]
   - Position 3: Ablate [0,1,2], keep [3,4,5]
   - Position 4: Ablate [0,1,2,3], keep [4,5]
   - Position 5: Ablate [0,1,2,3,4], keep [5]

3. **Record Results**
   - Importance = Did ablation cause error?
   - Store all answers for analysis

**Total Forward Passes:** 1000 problems × 7 forward passes = 7000 forward passes

---

## Expected Results

Based on validation (87 problems):

### Position-wise Importance Pattern

```
Position 0: 0.000  (baseline - no ablation)
Position 1: 0.345
Position 2: 0.644
Position 3: 0.667
Position 4: 0.701
Position 5: 0.897  ← MOST CRITICAL
```

We expect the full sweep to confirm this pattern:
- **Late positions (4, 5) most important**
- **Progressive refinement strategy**
- **Monotonic increase across positions**

### By Difficulty

| Difficulty | Early (0-2) | Late (3-5) | Expected Ratio |
|-----------|-------------|------------|----------------|
| 1-2 step  | 0.10-0.17   | 0.47-0.58  | ~3-5x          |
| 3-4 step  | 0.33-0.42   | 0.86-0.94  | ~2-3x          |
| 5+ step   | 0.40-0.47   | 0.78-0.92  | ~2x            |

---

## Output Files

When complete, the following files will be created:

### 1. step_importance_scores.json (~2-3 MB)
**Content:** Detailed results for all 1000 problems
```json
{
  "problem_id": "gsm8k_123",
  "question": "...",
  "baseline_answer": "42",
  "baseline_correct": true,
  "difficulty": 3,
  "position_results": [
    {
      "position": 0,
      "ablated_answer": "42",
      "ablated_correct": true,
      "importance_score": 0.0
    },
    ...
  ]
}
```

### 2. step_importance_summary_stats.json (~10 KB)
**Content:** Aggregate statistics
```json
{
  "n_problems": 1000,
  "baseline_accuracy": 0.98,
  "position_importance": {
    "0": 0.000,
    "1": 0.345,
    ...
  },
  "by_difficulty": {
    "2-step": {
      "n_problems": 250,
      "baseline_accuracy": 0.99,
      "early_importance": 0.15,
      "late_importance": 0.55
    },
    ...
  }
}
```

### 3. step_importance_validation.json (152 KB) ✅ Already saved
**Content:** Validation results from 87 problems

---

## Performance Metrics

Based on validation run:

- **Throughput:** ~1200 problems/hour
- **Time per problem:** ~3 seconds (with 6 ablations)
- **GPU Memory:** 8 GB peak
- **CPU Usage:** 110-125%

### Resource Usage (Live)

```bash
# GPU
nvidia-smi

# CPU & Memory
top -p 15122

# Disk space
df -h /home/paperspace/dev/CoT_Exploration
```

---

## Monitoring Commands

### Check if still running:
```bash
ps aux | grep 02_measure_step_importance.py | grep -v grep
```

### Check progress (look for checkpoint files):
```bash
ls -lh /home/paperspace/dev/CoT_Exploration/src/experiments/mechanistic_interp/data/checkpoint_*.json
```

### Monitor output (if any):
```bash
tail -f /home/paperspace/dev/CoT_Exploration/src/experiments/mechanistic_interp/data/*.log
```

### Check completion:
```bash
ls -lh /home/paperspace/dev/CoT_Exploration/src/experiments/mechanistic_interp/data/step_importance_scores.json
```

---

## Estimated Completion Time

**Start Time:** 16:33:16 UTC
**Estimated End:** 17:26:00 UTC (~53 min later)

### Milestones:

- ✅ **16:33** - Process started
- 🔄 **16:35** - Validation in progress
- ⏳ **16:36** - Full sweep starts
- ⏳ **17:01** - Checkpoint 500 (50% complete)
- ⏳ **17:26** - Complete (checkpoint 1000 + final files)

---

## What Happens When Complete?

1. **Automatic file generation:**
   - `step_importance_scores.json`
   - `step_importance_summary_stats.json`

2. **Console output:**
   - Summary statistics
   - Position-wise importance
   - Key findings

3. **Ready for next steps:**
   - MECH-03: Feature Extraction
   - MECH-04: Correlation Analysis
   - Documentation of results

---

## Troubleshooting

### If process dies:
```bash
# Check for checkpoint files
ls -lh /home/paperspace/dev/CoT_Exploration/src/experiments/mechanistic_interp/data/checkpoint_*.json

# If checkpoint_500.json exists, can resume from there
# (would need to modify script to resume from checkpoint)
```

### If taking longer than expected:
- Check GPU availability: `nvidia-smi`
- Check CPU usage: `top`
- Expected variance: ±10 minutes from estimate

### If encountering errors:
- Check log files (if any)
- Check GPU memory: `nvidia-smi`
- Check disk space: `df -h`

---

## After Completion

### Next Tasks:

1. **Analyze Results** (30 min)
   - Review summary statistics
   - Compare to validation results
   - Check for unexpected patterns

2. **Create Experiment Report** (30 min)
   - `docs/experiments/10-26_llama_gsm8k_step_importance.md`
   - Full methodology
   - Complete results
   - Interpretation

3. **Update Documentation** (15 min)
   - Research journal entry
   - Status files
   - Dependency updates

4. **Plan Next Steps** (15 min)
   - MECH-03 scope
   - MECH-04 dependencies
   - Timeline adjustments

**Total post-processing:** ~1.5 hours

---

**Status:** 🟢 RUNNING
**Monitor:** Check this file for updates
**ETA:** ~17:26 UTC (53 min from start)
