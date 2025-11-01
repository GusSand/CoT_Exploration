# Top-K Projection Intervention Experiment - Status

## Experiment Overview

**Purpose:** Test how projecting continuous CoT activations onto the top-k vocabulary embedding subspace affects reasoning accuracy.

**Location:** `/workspace/CoT_Exploration/src/experiments/01-11-2025-topk-projection/`

**Main Script:** `test_topk_projection_corrected.py`

## What This Tests

The experiment compares different intervention methods during CODI's latent chain-of-thought computation:

### Intervention Conditions (12 total)
1. **Baseline** - No intervention
2. **Average ablation** - Replace with position-specific means from 100 GSM8K training examples (numbers only)
3. **Projection@k** - Project onto top-k vocabulary embeddings for k = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50] (numbers only)

### Datasets (132 examples each)
1. Clean dataset - Examples that LLAMA solves correctly
2. GSM8K train - First 132 training examples
3. GSM8K test - First 132 test examples

### Key Function: `project_onto_topk_vocab()`

For k=1: Projects onto single vocabulary embedding direction
For k>1: Projects onto subspace spanned by top-k vocabulary embeddings using least squares

The projection is L2-normalized to preserve the magnitude of the continuous activation.

## Current Status: CUDA Out of Memory

### Issue
Cannot run because GPU memory is fully utilized:
- **Total GPU memory:** 19.67 GiB
- **Used by learned-mapping-residual experiment:** 18.87 GiB
- **Free:** ~815 MB (need ~10 GB to load CODI-LLAMA)

### Error Log
See: `topk_test_corrected.log`

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB.
GPU 0 has a total capacity of 19.67 GiB of which 4.06 MiB is free.
Process 1875353 has 18.87 GiB memory in use.
```

### Attempted Solutions
1. ✗ Environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - No effect (not a fragmentation issue)
2. ✓ **Automatic monitoring solution** (currently active)

## Solution: Automatic Monitoring

Created a monitoring script that checks GPU memory every 2 minutes and automatically starts the experiment when sufficient memory becomes available (>= 10 GB free).

### Monitor Script
- **Location:** `monitor_and_run_topk.sh`
- **Log:** `monitor.log`
- **Status:** Running in background (nohup)
- **Trigger:** Will auto-start `test_topk_projection_corrected.py` when GPU memory >= 10 GB

### Check Monitor Status
```bash
# On server
cd /workspace/CoT_Exploration/src/experiments/01-11-2025-topk-projection
tail -f monitor.log

# Check if still running
ps aux | grep monitor_and_run_topk.sh

# Check GPU memory
nvidia-smi
```

### When Experiment Runs
Results will be saved to:
- Output log: `topk_test_auto_run.log`
- Results: `intervention_comparison_results/full_results_<dataset>_132_examples.json`

## Expected Experiment Duration

Once started:
1. Load model: ~30 seconds
2. Compute average activations (100 GSM8K train): ~5 minutes
3. Run interventions:
   - 3 datasets × 132 examples × 12 conditions = 4,752 total runs
   - Estimated: 6-8 hours for full experiment

## Next Steps

The experiment will automatically start when:
1. The learned-mapping-residual experiment completes, OR
2. GPU memory is manually freed by killing that process

Monitor progress:
```bash
ssh root@213.173.108.19 -p 13325 -i ~/.ssh/id_ed25519
cd /workspace/CoT_Exploration/src/experiments/01-11-2025-topk-projection
tail -f monitor.log  # Shows monitoring status
# When running:
tail -f topk_test_auto_run.log  # Shows experiment progress
```

## Manual Override

To start immediately (requires killing learned-mapping-residual experiment):
```bash
# Find and kill the other experiment
ps aux | grep run_experiment.py
kill <PID>

# Wait for GPU memory to free, then:
python test_topk_projection_corrected.py
```
