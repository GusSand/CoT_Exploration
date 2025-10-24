# GSM8K CoT Dataset Expansion Guide

## Overview

This guide explains how to expand the LLaMA CoT dataset from 132 to 1,000-1,500 problems using the end-to-end pipeline script.

## Quick Start

### Run Full Pipeline

```bash
cd /home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching

# Run full pipeline (will take ~8-12 hours with 5 GPUs)
python expand_gsm8k_cot_dataset.py --num_samples 5000
```

### Resume from Checkpoint

If the script is interrupted:

```bash
# Resume from last checkpoint
python expand_gsm8k_cot_dataset.py --resume
```

### Use Existing Checkpoint (Skip Testing)

If you already have CoT necessity results:

```bash
# Skip testing, just stratify and filter
python expand_gsm8k_cot_dataset.py --skip_testing
```

## Pipeline Steps

The script performs 5 steps automatically:

### 1. Load Original GSM8K (1-2 minutes)
- Loads GSM8K test set (1,319 problems)
- Loads GSM8K train set (7,473 problems)
- Excludes 532 already-tested problems
- Collects ~5,000 new candidate problems

### 2. Test CoT Necessity (8-10 hours)
- For each problem:
  - Run baseline inference (with 6 CoT tokens)
  - Run ablated inference (CoT tokens = 0)
  - Mark as "needs CoT" if baseline correct but ablated wrong
- Expected yield: ~1,200 CoT-needed problems (24.8% rate)
- **Checkpoints every 100 problems** (can resume if interrupted)

### 3. Calculate Difficulty Metrics (1-2 minutes)
- Parse GSM8K solutions to count reasoning steps
- Classify: 2-step, 3-step, 4-step, 5+step
- Add difficulty labels to each problem

### 4. Load Existing 132 Problems (< 1 minute)
- Loads existing problems from `data/llama_cot_all.json`
- Filters to original (clean) variants only
- Marks as "existing" to preserve them

### 5. Stratify and Filter (< 1 minute)
- Groups problems by difficulty
- Samples to meet targets:
  - 2-step: ≥150 problems
  - 3-step: ≥150 problems
  - 4-step: ≥100 problems
  - 5+step: ≥50 problems
- Prioritizes keeping all existing 132 problems
- Randomly samples new problems to fill gaps
- Total: 450-1,500 problems

## Output Files

### Checkpoint File (created during testing)
**Location**: `data/gsm8k_expansion_checkpoint.json`

Contains:
- All problems tested so far
- Baseline and ablated predictions
- CoT necessity flags
- Used for resuming if interrupted

### Final Dataset
**Location**: `data/llama_cot_original_stratified_final.json`

Format:
```json
[
  {
    "gsm8k_id": "test_123",
    "question": "...",
    "answer": 42,
    "full_solution": "...",
    "source": "test",
    "baseline_correct": true,
    "ablated_correct": false,
    "needs_cot": true,
    "reasoning_steps": 3,
    "difficulty": "3-step",
    "is_existing": false
  }
]
```

## Command Line Options

```bash
python expand_gsm8k_cot_dataset.py [OPTIONS]

Options:
  --num_samples N         Number of new problems to test (default: 5000)
  --model_path PATH       Path to LLaMA model (default: ~/codi_ckpt/llama_gsm8k)
  --checkpoint_file PATH  Checkpoint file (default: data/gsm8k_expansion_checkpoint.json)
  --output PATH           Output file (default: data/llama_cot_original_stratified_final.json)
  --resume                Resume from checkpoint if available
  --skip_testing          Skip testing, use existing checkpoint
```

## Computational Requirements

### Resources
- **GPU**: A100 with 80GB (1 GPU sufficient, but slow)
- **RAM**: ~16GB
- **Storage**: ~1GB for datasets

### Time Estimates
- Single A100: ~40 GPU-hours = ~40 wall-clock hours
- 5× A100s (parallel): ~40 GPU-hours = ~8 wall-clock hours

### Cost
- **GPU compute**: 40 GPU-hours (A100)
- **API costs**: $0 (no GPT-4 needed)

## Monitoring Progress

The script prints progress every 100 problems:

```
Testing CoT necessity: 42%|████▏     | 421/1000 [2:05:30<2:52:40, 17.89s/it]
```

Check checkpoint file for intermediate results:

```bash
jq '.problems_tested, .needs_cot_count' data/gsm8k_expansion_checkpoint.json
```

## Troubleshooting

### Issue: Not enough 5+ step problems

If you don't get 50+ problems with ≥5 steps:

```bash
# Generate more samples from train set
python expand_gsm8k_cot_dataset.py --num_samples 10000
```

### Issue: Script crashes during testing

Resume from checkpoint:

```bash
python expand_gsm8k_cot_dataset.py --resume
```

### Issue: GPU out of memory

The script uses batch size 1, but if OOM occurs:
- Check that no other processes are using GPU
- Reduce max_new_tokens in the script (currently 200)

### Issue: Want to see intermediate results

Use the checkpoint:

```bash
# See how many CoT-needed so far
jq '.needs_cot_count' data/gsm8k_expansion_checkpoint.json

# See difficulty distribution so far
jq '[.results[] | select(.needs_cot) | .difficulty] | group_by(.) | map({difficulty: .[0], count: length})' data/gsm8k_expansion_checkpoint.json
```

## Next Steps After Completion

After the pipeline completes:

1. **Validate the dataset**:
   ```bash
   python validate_expanded_dataset.py --input data/llama_cot_original_stratified_final.json
   ```

2. **Update DATA_INVENTORY.md**:
   - Document the new dataset
   - Record final counts and distribution
   - Note generation date and parameters

3. **Commit to version control**:
   ```bash
   git add data/llama_cot_original_stratified_final.json
   git add docs/DATA_INVENTORY.md
   git commit -m "feat: Expand LLaMA CoT dataset to 1000+ problems"
   git push
   ```

4. **Update research journal**:
   - Document in `docs/research_journal.md`
   - Create detailed report in `docs/experiments/gsm8k_expansion_YYYY-MM-DD.md`

## Tips for Efficiency

### Parallel Testing (Advanced)

If you have multiple GPUs, you can split testing:

```bash
# GPU 0: Test problems 0-999
CUDA_VISIBLE_DEVICES=0 python expand_gsm8k_cot_dataset.py --num_samples 1000 --checkpoint_file data/checkpoint_0.json &

# GPU 1: Test problems 1000-1999
# (Modify script to skip first 1000)
CUDA_VISIBLE_DEVICES=1 python expand_gsm8k_cot_dataset.py --num_samples 1000 --checkpoint_file data/checkpoint_1.json &

# Then merge checkpoints manually
```

### Monitor GPU Usage

```bash
watch -n 1 nvidia-smi
```

### Run in Background

```bash
nohup python expand_gsm8k_cot_dataset.py --num_samples 5000 > expansion.log 2>&1 &

# Monitor progress
tail -f expansion.log
```

## Expected Results

Based on current data (132 problems from 532 tested = 24.8% CoT rate):

| Metric | Expected |
|--------|----------|
| New problems tested | 5,000 |
| New CoT-needed | ~1,200 (24.8%) |
| Existing kept | 132 |
| **Total available** | **~1,332** |
| Final dataset size | 450-1,500 |

Distribution should meet:
- ✓ 2-step: ≥150
- ✓ 3-step: ≥150
- ✓ 4-step: ≥100
- ⚠️ 5+step: ≥50 (may need more samples)

## Questions?

If you encounter issues:
1. Check the checkpoint file for progress
2. Review the logs for error messages
3. Ensure model path is correct: `~/codi_ckpt/llama_gsm8k`
4. Verify datasets are accessible (huggingface `gsm8k` dataset)
