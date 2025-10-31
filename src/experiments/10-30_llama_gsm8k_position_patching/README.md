# Position-wise CoT Activation Patching Experiment

**Status:** ✅ Ready for Testing
**Created:** 2025-10-30

## Overview

This experiment tests all (layer, position) combinations individually to create a fine-grained map of position-specific importance in continuous thought (CoT) tokens.

**Dataset:** 57 filtered pairs (removed 9 same-answer edge cases)
**Combinations tested:** 16 layers × 5 positions = 80 per pair
**Total forward passes:** 5,529 (57 pairs × 97 passes per pair)
**Compute time:** ~60 minutes on A100

## Directory Structure

```
10-30_llama_gsm8k_position_patching/
├── README.md (this file)
├── config.py                    # Experiment configuration
├── core/
│   ├── model_loader.py         # CODI model loading (copied from layer_patching)
│   ├── single_position_patcher.py  # Single-position patching infrastructure
│   └── metrics.py              # Metrics including new answer_logit_difference
├── scripts/
│   └── run_position_patching.py   # Main experiment script
└── results/                    # Output directory (created on run)
    ├── position_patching_results.json
    └── aggregate_statistics.json
```

## Implementation Details

### 1. Single Position Patching (`core/single_position_patcher.py`)

**Class:** `SinglePositionPatcher`
- Patches activation at a single (layer, position) combination
- Uses context manager pattern for clean hook management
- Input: layer_idx (0-15), position (single int), replacement_activation [batch, hidden_dim]

### 2. New Metric (`core/metrics.py`)

**Function:** `compute_answer_logit_difference()`
- Measures how much patching shifts logits toward clean vs corrupted answer
- Returns:
  - `mean_diff`: positive = recovery toward clean, negative = shift toward corrupted
  - `clean_score`: mean logit for clean answer tokens
  - `corrupted_score`: mean logit for corrupted answer tokens

### 3. Main Script (`scripts/run_position_patching.py`)

**Process for each pair:**
1. Extract clean activations at all 16 layers × 5 positions
2. Run baseline corrupted (no patching)
3. For each of 80 (layer, position) combinations:
   - Patch ONLY that single position
   - Run forward pass
   - Compute 4 metrics: KL div, L2 diff, pred change, answer logit diff
4. Save results and log to W&B

## Usage

### Test Mode (5 pairs)

```bash
cd /home/paperspace/dev/CoT_Exploration/src/experiments/10-30_llama_gsm8k_position_patching

# Enable test mode
python3 -c "
import config
config.TEST_MODE = True
"

# Run
python3 scripts/run_position_patching.py
```

### Full Run (57 pairs)

```bash
# Ensure TEST_MODE = False in config.py
python3 scripts/run_position_patching.py
```

## Expected Outputs

### 1. Position Patching Results
**File:** `results/position_patching_results.json`

**Structure:**
```json
[
  {
    "pair_id": 0,
    "clean_question": "...",
    "corrupted_question": "...",
    "clean_answer": 18,
    "corrupted_answer": 20,
    "layer_position_results": {
      "0": {  // layer 0
        "0": {  // position 0
          "kl_divergence": 0.0234,
          "l2_difference": 1.234,
          "prediction_change_rate": 0.15,
          "answer_logit_diff": 0.567,
          "clean_answer_score": 5.2,
          "corrupted_answer_score": 4.6
        },
        "1": { ... },  // position 1
        ...
      },
      "1": { ... },  // layer 1
      ...
    }
  },
  ...
]
```

### 2. Aggregate Statistics
**File:** `results/aggregate_statistics.json`

**Structure:**
```json
{
  "layer_position_kl_means": {
    "0": {"0": 0.0234, "1": 0.0456, ...},
    "1": {...},
    ...
  },
  "layer_position_ans_diff_means": {
    "0": {"0": 0.567, "1": 0.789, ...},
    ...
  },
  "top_5_critical": [
    {"layer": 12, "position": 3, "kl_div": 0.234, "ans_diff": 1.234},
    ...
  ]
}
```

## Validation Checklist

Before full run, verify in TEST_MODE:

- [ ] All 5 pairs complete without errors
- [ ] Results structure is correct
- [ ] All 4 metrics compute successfully
- [ ] Answer logit diff is reasonable (not NaN, not all zeros)
- [ ] W&B logging works
- [ ] Memory stays under 10 GB (check with `nvidia-smi`)
- [ ] Each pair takes ~60-90 seconds

**Expected behavior:**
- Pair 0: clean_answer=18, corrupted_answer=20
- Answer logit diff should be positive after effective patching (recovery toward clean)
- Some (layer, position) combinations should show higher KL than others (not uniform)

## Next Steps

After successful run:

1. **Create visualizations:** Heatmap script for layer×position KL divergence
2. **Save activation cache:** For reuse in Experiment 2
3. **Analyze results:** Identify critical positions and patterns
4. **Documentation:** Update research journal and create detailed report

## Notes

- Uses filtered dataset: `prepared_pairs_filtered.json` (57 pairs, not 66)
- Single-position patching differs from layer-patching (patches one position at a time vs all 5)
- Answer logit difference is new metric specific to this experiment
- Results will feed into Experiment 2 (iterative patching)
