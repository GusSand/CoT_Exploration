# Activation Patching Causal Analysis

**Research Question**: Are CODI's continuous thought representations causally involved in reasoning, or merely epiphenomenal correlates?

**Approach**: Direct activation patching - inject clean activations into corrupted problems and measure accuracy recovery across multiple layers.

---

## Quick Start

```bash
# 1. Generate problem pairs
python generate_pairs.py --num_candidates 70 --show_samples

# 2. Review and approve pairs (manual step - edit the JSON)
# Open problem_pairs_for_review.json, fill in corrupted answers, mark as approved

# 3. Filter to final 50 pairs
python generate_pairs.py --filter problem_pairs_for_review.json

# 4. Run experiment
python run_experiment.py \
    --model_path ~/codi_ckpt/gpt2_gsm8k_6latent/ \
    --problem_pairs problem_pairs.json \
    --output_dir results/

# 5. Generate visualizations
cd ../../viz
python plot_results.py \
    --results ../experiments/activation_patching/results/experiment_results.json \
    --output plots/
```

---

## Experiment Overview

### Conditions Tested (5 per problem pair)

1. **Clean Baseline**: Original problem → Expected answer
2. **Corrupted Baseline**: Modified problem (one number changed) → Different answer
3. **Patched Early (L3)**: Corrupted problem + clean activation from layer 3 → Test recovery
4. **Patched Middle (L6)**: Corrupted problem + clean activation from layer 6 → Test recovery
5. **Patched Late (L11)**: Corrupted problem + clean activation from layer 11 → Test recovery

### Hypothesis

**If causal**: Patching clean activations should restore accuracy (recovery rate >50%)

**If epiphenomenal**: Patching should have minimal effect (recovery rate <10%)

---

## File Structure

```
src/experiments/activation_patching/
├── README.md                          # This file
├── generate_pairs.py                  # Generate problem pairs from GSM8K
├── cache_activations.py               # Extract and cache activations
├── patch_and_eval.py                  # Patch activations during inference
├── run_experiment.py                  # Main experiment runner with WandB
├── problem_pairs.json                 # Final problem pairs (50)
└── results/
    ├── experiment_results.json        # Full results
    ├── checkpoint_*.json              # Checkpoints every 10 problems
    └── activations/                   # Cached activations (if saved)

src/viz/
├── plot_results.py                    # Create visualizations
└── plots/
    ├── accuracy_by_layer.png          # Accuracy comparison
    ├── recovery_by_layer.png          # Recovery rates
    └── layer_importance.png           # Layer ranking
```

---

## Detailed Usage

### Step 1: Generate Problem Pairs

```bash
python generate_pairs.py \
    --num_candidates 70 \
    --output problem_pairs_for_review.json \
    --show_samples
```

This generates 70 candidate pairs from GSM8K. Each pair has:
- **Clean**: Original problem with original answer
- **Corrupted**: Modified problem (one number changed)

**Output**: `problem_pairs_for_review.json`

---

### Step 2: Manual Review (IMPORTANT!)

Open `problem_pairs_for_review.json` and for each pair:

1. **Check corrupted question makes sense**
   - Does the number change create a valid problem?
   - Is it still solvable?

2. **Calculate corrupted answer**
   - Work out what the answer should be with the changed number
   - Fill in `corrupted.answer` field

3. **Mark status**
   - Set `review_status` to `"approved"` or `"rejected"`
   - Add notes in `notes` field if needed

**Example**:
```json
{
  "pair_id": 0,
  "clean": {
    "question": "John has 3 bags with 7 apples each. How many total?",
    "answer": 21
  },
  "corrupted": {
    "question": "John has 4 bags with 7 apples each. How many total?",
    "answer": 28,  # <-- Fill this in!
    "changed_number": "3 -> 4"
  },
  "review_status": "approved",  # <-- Change from "pending"
  "notes": ""
}
```

---

### Step 3: Filter to Final Pairs

```bash
python generate_pairs.py \
    --filter problem_pairs_for_review.json \
    --final_output problem_pairs.json
```

This filters to only approved pairs and validates all have corrupted answers filled in.

**Output**: `problem_pairs.json` (50 final pairs)

---

### Step 4: Run Experiment

```bash
python run_experiment.py \
    --model_path ~/codi_ckpt/gpt2_gsm8k_6latent/ \
    --problem_pairs problem_pairs.json \
    --output_dir results/ \
    --wandb_project codi-activation-patching
```

**Runtime**: ~1-2 hours for 50 pairs (250 forward passes total)

**Monitoring Options**:

1. **WandB Dashboard** (Primary):
   - Visit `https://wandb.ai/your-username/codi-activation-patching`
   - See real-time metrics, progress tracking

2. **Console** (Local):
   - Progress bar shows X/50 pairs complete
   - Checkpoints saved every 10 problems

3. **Checkpoints**:
   - Check `results/checkpoint_40.json` for current results

**Output**:
- `results/experiment_results.json` - Full results with all metrics
- `results/checkpoint_*.json` - Intermediate checkpoints
- WandB run URL printed at end

---

### Step 5: Visualize Results

```bash
cd ../../viz

python plot_results.py \
    --results ../experiments/activation_patching/results/experiment_results.json \
    --output plots/
```

**Creates 3 plots**:
1. `accuracy_by_layer.png` - Bar chart comparing all 5 conditions
2. `recovery_by_layer.png` - Recovery rate by layer with 50% threshold
3. `layer_importance.png` - Ranking of layers by causal importance

**Also logs to WandB** automatically.

---

## Interpreting Results

### Key Metrics

1. **Clean Accuracy**: Model's baseline performance
2. **Corrupted Accuracy**: How much accuracy drops with corrupted input
3. **Patched Accuracy**: Accuracy after patching clean activation
4. **Recovery Rate**: `(Patched - Corrupted) / (Clean - Corrupted)`

### Interpretation Guide

**Strong Causal Effect** (Recovery > 50%):
- ✅ Continuous thoughts are causally involved
- ✅ Decoded intermediates are trustworthy
- ✅ Latent reasoning is interpretable

**Moderate Effect** (30% < Recovery < 50%):
- ⚠️ Partial causal involvement
- ⚠️ May depend on problem type
- ⚠️ Further investigation needed

**Weak/No Effect** (Recovery < 30%):
- ❌ Likely epiphenomenal
- ❌ Reasoning may occur through different mechanisms
- ❌ Decoded values are correlates, not causes

### Layer Analysis

**Early layer (L3) strongest**:
- Reasoning decisions made early
- Initial encoding is critical

**Middle layer (L6) strongest**:
- Core reasoning happens mid-network
- Balanced position optimal

**Late layer (L11) strongest**:
- Final reasoning refinement matters most
- Output preparation layer

---

## Troubleshooting

### Model Loading Errors

**Error**: `FileNotFoundError: model checkpoint not found`

**Fix**: Ensure correct path to CODI checkpoint:
```bash
ls ~/codi_ckpt/gpt2_gsm8k_6latent/
# Should show: adapter_config.json, adapter_model.bin, ...
```

---

### WandB Login Issues

**Error**: `wandb: ERROR Not logged in`

**Fix**: Login to WandB:
```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

### CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Fix**: Reduce batch size or use CPU:
```python
# In cache_activations.py, line 18:
cacher = ActivationCacher(args.model_path, device='cpu')
```

---

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src.model'`

**Fix**: Ensure CODI is in path:
```bash
export PYTHONPATH=/home/paperspace/dev/CoT_Exploration:$PYTHONPATH
```

---

## Testing Individual Components

### Test Activation Caching

```bash
python cache_activations.py \
    --model_path ~/codi_ckpt/gpt2_gsm8k_6latent/ \
    --test
```

Expected output:
```
Loading CODI model...
✓ Model loaded successfully!

Testing activation caching...
Cached activations:
  early   : shape=torch.Size([1, 768]), device=cpu
  middle  : shape=torch.Size([1, 768]), device=cpu
  late    : shape=torch.Size([1, 768]), device=cpu

✓ Save/load test passed!
```

---

### Test Activation Patching

```bash
python patch_and_eval.py \
    --model_path ~/codi_ckpt/gpt2_gsm8k_6latent/
```

Expected output:
```
============================================================
TEST: Activation Patching
============================================================

1. Running clean problem...
   Answer: 21

2. Running corrupted problem...
   Answer: 28

3. Patching with clean activation...
   Early layer: Answer = 21 (or closer to 21)
   Middle layer: Answer = 21 (or closer to 21)
   Late layer: Answer = 21 (or closer to 21)

Expected: Patched answers should shift toward clean answer (21)
============================================================
```

---

## Expected Runtime

| Step | Time | GPU |
|------|------|-----|
| Generate pairs | 5 min | No |
| Manual review | 30 min | No |
| Run experiment (50 pairs) | 1-2 hrs | Yes |
| Visualization | 1 min | No |
| **Total** | **~2-2.5 hrs** | |

---

## Next Steps After Completion

### If Positive Results (Recovery > 50%)

1. **Scale up**: Run on 500 pairs for statistical power
2. **Add Experiment 2**: Counterfactual patching
3. **Add Experiment 3**: Ablation study
4. **Test all layers**: Full 12-layer analysis
5. **Compare to explicit CoT**: Does it show similar causal structure?

### If Negative Results (Recovery < 30%)

1. **Debug**: Verify activation extraction is correct
2. **Try different layers**: Test all 12 layers
3. **Test explicit CoT**: Should show causal effects as control
4. **Investigate**: Alternative mechanisms for CODI's success
5. **Publish**: Negative results are valuable for the field!

---

## Citation

If you use this code, please cite:

```bibtex
@misc{activation_patching_2025,
  title={Causal Analysis of Continuous Thought Representations in CODI},
  author={CoT Exploration Project},
  year={2025},
  howpublished={\\url{https://github.com/your-repo/CoT_Exploration}}
}
```

---

## Contact & Support

- **Documentation**: `/home/paperspace/dev/CoT_Exploration/docs/experiments/10-18_gpt2_gsm8k_activation_patching_causal.md`
- **User Stories**: `/home/paperspace/dev/CoT_Exploration/docs/project/activation_patching_user_stories_REVISED.md`
- **Research Journal**: `/home/paperspace/dev/CoT_Exploration/docs/research_journal.md`

---

**Status**: ✅ Ready to run
**Estimated Time**: 1-2 days
**Last Updated**: 2025-10-18
