# User Stories: Activation Patching Causal Analysis (REVISED)

**Project**: Chain of Thought Exploration - CODI Interpretability
**Epic**: Causal Analysis of Continuous Thought Representations
**Created**: 2025-10-18 (Revised)
**Status**: Planning

---

## Overview

**REVISED for realistic 1-2 day implementation**

This is a lean, focused experiment to answer one core question: **Are CODI's continuous thoughts causally involved in reasoning?**

**Approach**: Minimal viable experiment using existing CODI infrastructure with targeted additions.

**Total Estimated Cost**: 1-2 developer-days

---

## Code Organization

### Directory Structure
```
/home/paperspace/dev/CoT_Exploration/
├── codi/                                    # Original CODI (submodule - don't modify)
├── src/                                     # Our source code
│   ├── experiments/                         # NEW - Experiment scripts
│   │   └── activation_patching/
│   │       ├── cache_activations.py         # Cache clean & corrupted runs
│   │       ├── patch_and_eval.py            # Patch & evaluate
│   │       ├── generate_pairs.py            # Create problem pairs
│   │       ├── run_experiment.py            # Run full experiment
│   │       ├── problem_pairs.json           # Problem data
│   │       └── results/                     # Outputs (metrics, logs)
│   └── viz/                                 # NEW - Visualization code
│       ├── plot_results.py                  # Bar charts, recovery rates
│       └── plots/                           # Generated plots
└── docs/experiments/                        # Documentation
```

### Strategy
- Keep CODI pristine (it's a submodule)
- Write experiment scripts in `src/experiments/activation_patching/`
- Separate visualization code in `src/viz/`
- Import from CODI when possible: `from codi.src.model import CODI`
- Copy/modify specific files only if necessary

---

## User Stories (REVISED - 5 Stories)

### Story 1: Activation Caching System

**As a** researcher
**I want** to run problems and cache [THINK] token activations
**So that** I can reuse them for patching experiments

#### Acceptance Criteria
- [ ] Script runs CODI on GSM8K problems
- [ ] Caches activations at [THINK] token positions
- [ ] Saves to disk (pickle or torch.save)
- [ ] Handles both clean and corrupted problems
- [ ] Stores metadata (problem ID, position, layer)

#### Technical Approach
```python
# cache_activations.py
import torch
from codi.src.model import CODI

def cache_run(model, problem, save_path):
    """Run problem and cache [THINK] token activations"""
    # Based on existing probe_latent_token.py
    with torch.no_grad():
        outputs = model.generate(...)
        # Extract: outputs.hidden_states[-1][:, -1, :]
        latent_embd = outputs.hidden_states[-1][:, -1, :]
        torch.save({
            'activation': latent_embd,
            'problem': problem,
            'position': token_position
        }, save_path)
```

#### Implementation Notes
- Build on existing `probe_latent_token.py` (lines 200-250)
- Already has activation extraction: `latent_embd = outputs.hidden_states[-1][:, -1, :]`
- Just add save to disk
- Run on 50 problem pairs = 100 cache files

**Estimated Cost**: 2-3 hours
**Priority**: P0

---

### Story 2: Activation Patching Script

**As a** researcher
**I want** to inject cached activations during forward passes at different layers
**So that** I can test causal effects across early, middle, and late layers

#### Acceptance Criteria
- [ ] Loads cached activation from disk
- [ ] Injects into [THINK] token position during generation
- [ ] Supports patching at different layers (early, middle, late)
- [ ] Tests 3 layers: early (layer 3), middle (layer 6), late (layer 11/final)
- [ ] Runs patched forward pass to completion
- [ ] Returns final answer
- [ ] Works without modifying CODI source code

#### Technical Approach
```python
# src/experiments/activation_patching/patch_and_eval.py
import torch

def run_with_patch(model, problem, cached_activation, patch_layer, patch_position):
    """Run problem with activation patched at specific layer and position

    Args:
        model: CODI model
        problem: Problem text
        cached_activation: Tensor to inject
        patch_layer: Layer index to patch (e.g., 3, 6, 11)
        patch_position: Token position to patch (e.g., THINK token position)
    """

    # Option 1: Use forward hooks (cleanest)
    current_pos = 0

    def patch_hook(module, input, output):
        nonlocal current_pos
        if current_pos == patch_position:
            # Patch at the specified layer
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            hidden_states[:, -1, :] = cached_activation
            current_pos += 1
            return output
        current_pos += 1
        return output

    # Register hook at target layer
    target_layer = model.base_model.h[patch_layer]  # GPT-2 layer access
    hook = target_layer.register_forward_hook(patch_hook)

    result = model.generate(problem)
    hook.remove()
    return result

    # Option 2: Modify generation loop (if hooks don't work)
    # Copy relevant code from codi/test.py, insert patch at right step

# Test on multiple layers
LAYER_CONFIG = {
    'early': 3,    # Early layer (1/4 through model)
    'middle': 6,   # Middle layer (1/2 through model)
    'late': 11     # Late layer (near final, GPT-2 has 12 layers)
}

def test_all_layers(model, problem, cached_activation, position):
    """Test patching at all three layer depths"""
    results = {}
    for layer_name, layer_idx in LAYER_CONFIG.items():
        result = run_with_patch(model, problem, cached_activation, layer_idx, position)
        results[layer_name] = result
    return results
```

#### Implementation Notes
- Two approaches:
  1. **PyTorch hooks** (cleaner, no code copying)
  2. **Modified generation** (copy test.py, insert patch)
- Start with hooks, fall back to modified generation if needed
- Test on 5 examples before full run
- Layer indices assume GPT-2 (12 layers):
  - Early: layer 3 (after initial processing)
  - Middle: layer 6 (halfway through)
  - Late: layer 11 (pre-output)
- Adjust if using different model architecture

**Estimated Cost**: 3-4 hours
**Priority**: P0

---

### Story 3: Problem Pair Generation

**As a** researcher
**I want** 50 clean/corrupted GSM8K problem pairs
**So that** I can run patching experiments

#### Acceptance Criteria
- [ ] Generates 50 problem pairs from GSM8K (scripted)
- [ ] Clean and corrupted differ by exactly one number
- [ ] Script outputs pairs for manual review
- [ ] After review, saves to JSON with ground truth answers
- [ ] Includes expected intermediate values (for analysis)
- [ ] Total time: ~1 hour (30 min script + 30 min review)

#### Technical Approach (Scripted + Review)
```python
# src/experiments/activation_patching/generate_pairs.py
import json
import re
from datasets import load_dataset

def create_pair(problem, idx):
    """Create clean/corrupted pair by changing one number"""
    question = problem['question']

    # Find numbers in question
    numbers = re.findall(r'\d+', question)

    if len(numbers) < 2:
        return None  # Skip problems with too few numbers

    # Change first operand (simple heuristic)
    original_num = numbers[0]
    corrupted_num = str(int(original_num) + 1)
    corrupted_question = question.replace(original_num, corrupted_num, 1)

    return {
        'pair_id': idx,
        'clean': {
            'question': question,
            'answer': problem['answer'],
            'solution': problem.get('answer', '')  # Include solution if available
        },
        'corrupted': {
            'question': corrupted_question,
            'answer': None,  # Will calculate during review
            'changed_number': f"{original_num} -> {corrupted_num}"
        },
        'review_status': 'pending'
    }

# Generate 70 candidates (to get 50 good ones after review)
dataset = load_dataset('gsm8k', 'main', split='test[:100]')
pairs = []
for idx, problem in enumerate(dataset):
    pair = create_pair(problem, idx)
    if pair:
        pairs.append(pair)
        if len(pairs) >= 70:
            break

# Save for review
with open('problem_pairs_for_review.json', 'w') as f:
    json.dump(pairs, f, indent=2)

print(f"Generated {len(pairs)} pairs for review")
print("Please review and:")
print("1. Mark 'review_status' as 'approved' or 'rejected'")
print("2. Fill in 'corrupted.answer' for approved pairs")
print("3. Keep 50 best pairs")
```

#### Review Process
1. Run script to generate 70 candidate pairs
2. Manually review JSON file:
   - Check that corruption makes sense
   - Calculate expected corrupted answer
   - Mark quality pairs as 'approved'
   - Reject ambiguous or problematic pairs
3. Keep top 50 approved pairs
4. Save final version as `problem_pairs.json`

**Estimated Cost**: 1 hour (30 min script + 30 min review)
**Priority**: P0

---

### Story 4: Run Experiment & Collect Metrics

**As a** researcher
**I want** to run all conditions across 3 layers and collect accuracy metrics with WandB tracking
**So that** I can measure causal effects, identify which layers matter, and monitor progress in real-time

#### Acceptance Criteria
- [ ] Initializes WandB experiment tracking
- [ ] Runs 5 conditions on 50 pairs:
  1. Clean baseline
  2. Corrupted baseline
  3. Patched at early layer (layer 3)
  4. Patched at middle layer (layer 6)
  5. Patched at late layer (layer 11)
- [ ] Logs progress to WandB in real-time (per-problem metrics)
- [ ] Displays tqdm progress bar in console
- [ ] Calculates accuracy for each condition
- [ ] Computes recovery rate per layer: (Patched - Corrupted) / (Clean - Corrupted)
- [ ] Identifies which layer shows strongest causal effect
- [ ] Saves results to JSON + WandB artifacts
- [ ] Saves checkpoints every 10 problems
- [ ] Prints summary statistics

#### Technical Approach
```python
# src/experiments/activation_patching/run_experiment.py
import json
import wandb
from tqdm import tqdm
from cache_activations import cache_run
from patch_and_eval import run_with_patch, LAYER_CONFIG

# Initialize WandB
wandb.init(
    project="codi-activation-patching",
    name="direct-patching-3layers",
    config={
        "experiment": "direct_activation_patching",
        "num_pairs": 50,
        "layers": ["early-L3", "middle-L6", "late-L11"],
        "model": "gpt2-codi",
        "dataset": "gsm8k"
    },
    tags=["causal-analysis", "mechanistic-interpretability"]
)

pairs = json.load(open('problem_pairs.json'))
results = []

# Progress tracking
for pair in tqdm(pairs, desc="Processing problem pairs"):
    # 1. Run clean, cache activation (at all layers)
    clean_activations = {}
    for layer_name, layer_idx in LAYER_CONFIG.items():
        clean_activations[layer_name] = cache_run(model, pair['clean'], layer_idx)

    clean_answer = get_answer(model, pair['clean'])

    # 2. Run corrupted
    corrupted_answer = get_answer(model, pair['corrupted'])

    # 3. Run patched at each layer
    patched_answers = {}
    for layer_name, layer_idx in LAYER_CONFIG.items():
        patched_answers[layer_name] = run_with_patch(
            model, pair['corrupted'], clean_activations[layer_name], layer_idx
        )

    # Store results
    pair_result = {
        'pair_id': pair['pair_id'],
        'clean_correct': clean_answer == pair['clean']['answer'],
        'corrupted_correct': corrupted_answer == pair['corrupted']['answer'],
        'patched_correct': {
            layer: (ans == pair['clean']['answer'])
            for layer, ans in patched_answers.items()
        }
    }
    results.append(pair_result)

    # Log to WandB (per-problem)
    wandb.log({
        'pair_id': pair['pair_id'],
        'clean_correct': int(pair_result['clean_correct']),
        'corrupted_correct': int(pair_result['corrupted_correct']),
        'early_correct': int(pair_result['patched_correct']['early']),
        'middle_correct': int(pair_result['patched_correct']['middle']),
        'late_correct': int(pair_result['patched_correct']['late']),
    })

    # Checkpoint every 10 problems
    if pair['pair_id'] % 10 == 0:
        checkpoint = {'results': results}
        with open(f'results/checkpoint_{pair["pair_id"]}.json', 'w') as f:
            json.dump(checkpoint, f, indent=2)
        tqdm.write(f"✓ Checkpoint saved at problem {pair['pair_id']}")

# Calculate metrics per layer
clean_acc = mean([r['clean_correct'] for r in results])
corrupted_acc = mean([r['corrupted_correct'] for r in results])

layer_metrics = {}
for layer_name in LAYER_CONFIG.keys():
    patched_acc = mean([r['patched_correct'][layer_name] for r in results])
    recovery_rate = (patched_acc - corrupted_acc) / (clean_acc - corrupted_acc) if (clean_acc - corrupted_acc) > 0 else 0
    layer_metrics[layer_name] = {
        'accuracy': patched_acc,
        'recovery_rate': recovery_rate
    }

# Log summary to WandB
wandb.log({
    'summary/clean_accuracy': clean_acc,
    'summary/corrupted_accuracy': corrupted_acc,
    'summary/early_accuracy': layer_metrics['early']['accuracy'],
    'summary/middle_accuracy': layer_metrics['middle']['accuracy'],
    'summary/late_accuracy': layer_metrics['late']['accuracy'],
    'summary/early_recovery': layer_metrics['early']['recovery_rate'],
    'summary/middle_recovery': layer_metrics['middle']['recovery_rate'],
    'summary/late_recovery': layer_metrics['late']['recovery_rate'],
})

# Print results
print(f"\n{'='*50}")
print(f"FINAL RESULTS")
print(f"{'='*50}")
print(f"Clean: {clean_acc:.2%}")
print(f"Corrupted: {corrupted_acc:.2%}")
print(f"\nPer-Layer Results:")
for layer_name, metrics in layer_metrics.items():
    print(f"  {layer_name:8s} - Acc: {metrics['accuracy']:.2%}, Recovery: {metrics['recovery_rate']:.2%}")

# Save to JSON
output = {
    'summary': {
        'clean_accuracy': clean_acc,
        'corrupted_accuracy': corrupted_acc,
        'layer_results': layer_metrics
    },
    'per_problem': results
}
with open('results/experiment_results.json', 'w') as f:
    json.dump(output, f, indent=2)

# Save results as WandB artifact
artifact = wandb.Artifact('experiment_results', type='results')
artifact.add_file('results/experiment_results.json')
wandb.log_artifact(artifact)

print(f"\n✓ Results logged to WandB: {wandb.run.url}")
wandb.finish()
```

#### Metrics
- **Primary**: Accuracy recovery rate per layer
- **Secondary**:
  - Which layer shows strongest effect?
  - Per-problem analysis (which problems benefit from patching?)
  - Effect size distribution

**Estimated Cost**: 2-3 hours (coding + runtime)
**Priority**: P0

---

### Story 5: Visualization & Documentation

**As a** researcher
**I want** visualizations showing layer-wise effects and clear documentation
**So that** I can interpret which layers are causally important and share findings

#### Acceptance Criteria
- [ ] Bar chart: Clean vs Corrupted vs Patched (per layer) accuracy
- [ ] Recovery rate by layer visualization
- [ ] Layer importance ranking plot
- [ ] All plots saved to `src/viz/plots/`
- [ ] Results documented in `docs/experiments/`
- [ ] Code committed to GitHub with clear README

#### Visualizations

**1. Accuracy Comparison by Layer**
```python
# src/viz/plot_results.py
import matplotlib.pyplot as plt
import json

# Load results
with open('../experiments/activation_patching/results/experiment_results.json') as f:
    data = json.load(f)

summary = data['summary']
layer_results = summary['layer_results']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

conditions = ['Clean', 'Corrupted', 'Early\n(L3)', 'Middle\n(L6)', 'Late\n(L11)']
accuracies = [
    summary['clean_accuracy'],
    summary['corrupted_accuracy'],
    layer_results['early']['accuracy'],
    layer_results['middle']['accuracy'],
    layer_results['late']['accuracy']
]

bars = ax.bar(conditions, accuracies, color=['green', 'red', 'blue', 'blue', 'blue'])
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim(0, 1)
ax.set_title('Activation Patching: Accuracy by Layer', fontsize=14)
ax.axhline(summary['clean_accuracy'], color='green', linestyle='--', alpha=0.3, label='Clean baseline')
ax.legend()

plt.tight_layout()
plt.savefig('plots/accuracy_by_layer.png', dpi=300)
print("Saved: plots/accuracy_by_layer.png")
```

**2. Recovery Rate by Layer**
```python
# Recovery rate comparison
fig, ax = plt.subplots(figsize=(8, 6))

layers = ['Early\n(L3)', 'Middle\n(L6)', 'Late\n(L11)']
recovery_rates = [
    layer_results['early']['recovery_rate'],
    layer_results['middle']['recovery_rate'],
    layer_results['late']['recovery_rate']
]

bars = ax.bar(layers, recovery_rates, color=['skyblue', 'steelblue', 'darkblue'])
ax.set_ylabel('Recovery Rate', fontsize=12)
ax.set_ylim(0, 1)
ax.set_title('Recovery Rate by Layer\n(Higher = Stronger Causal Effect)', fontsize=14)
ax.axhline(0.5, color='red', linestyle='--', linewidth=2, label='50% threshold')
ax.legend()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, recovery_rates)):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/recovery_by_layer.png', dpi=300)
print("Saved: plots/recovery_by_layer.png")
```

**3. Layer Importance Ranking**
```python
# Simple ranking visualization
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))

layer_names = list(layer_results.keys())
importance = [layer_results[l]['recovery_rate'] for l in layer_names]

# Sort by importance
sorted_idx = np.argsort(importance)[::-1]
sorted_layers = [layer_names[i].capitalize() for i in sorted_idx]
sorted_importance = [importance[i] for i in sorted_idx]

bars = ax.barh(sorted_layers, sorted_importance, color=['gold', 'silver', 'chocolate'])
ax.set_xlabel('Recovery Rate (Causal Importance)', fontsize=12)
ax.set_title('Layer Importance Ranking', fontsize=14)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('plots/layer_importance.png', dpi=300)
print("Saved: plots/layer_importance.png")
```

**4. Log to WandB**
```python
# Log all plots to WandB
wandb.log({
    'plots/accuracy_by_layer': wandb.Image('plots/accuracy_by_layer.png'),
    'plots/recovery_by_layer': wandb.Image('plots/recovery_by_layer.png'),
    'plots/layer_importance': wandb.Image('plots/layer_importance.png')
})
```

**5. Documentation**
- Update `docs/experiments/activation_patching_causal_analysis_2025-10-18.md` with results
- Add summary to `docs/research_journal.md`
- Create `src/experiments/activation_patching/README.md` with usage instructions
- Include all plots in documentation
- Include WandB run URL in documentation

**Estimated Cost**: 1-1.5 hours (includes WandB plot logging)
**Priority**: P0

---

## Revised Timeline

### Day 1 (4-5 hours)
**Morning (2-3 hours)**:
- Story 3: Generate 70 problem pairs (scripted), manually review and select best 50
- Story 1: Write activation caching script (support multiple layers)
- Test on 5 examples

**Afternoon (2 hours)**:
- Story 2: Write patching script with multi-layer support
- Test patching on 5 examples at all 3 layers
- Validate it works as expected

### Day 2 (3-4 hours)
**Morning (2-3 hours)**:
- Story 4: Run full experiment (50 pairs × 5 conditions: clean, corrupted, 3 patched layers)
- Collect all metrics per layer
- Debug any issues

**Afternoon (1 hour)**:
- Story 5: Generate 3 plots (accuracy by layer, recovery rate, importance ranking)
- Document results
- Commit to GitHub

**Total: 7-9 hours = 1-2 developer days**

---

## Story Summary

| Story | Description | Time | Priority |
|-------|-------------|------|----------|
| 1 | Activation caching (multi-layer) | 2-3 hrs | P0 |
| 2 | Activation patching (3 layers: early, middle, late) | 3-4 hrs | P0 |
| 3 | Problem pairs - scripted + review (50 final) | 1 hr | P0 |
| 4 | Run experiment (5 conditions × 50 pairs) & WandB tracking | 2-3 hrs | P0 |
| 5 | Visualization (3 plots), WandB logging & docs | 1-1.5 hrs | P0 |
| **TOTAL** | **9-12.5 hours** | **1-2 days** | |

---

## What We're NOT Doing (vs Original Plan)

### Removed Features
- ❌ Full hook management system with cleanup (simple hooks only)
- ❌ Generic intervention framework class (focused patching only)
- ❌ Comprehensive visualization suite (just 3 key matplotlib plots)
- ❌ Enhanced decoder with confidence scores (use existing probe)
- ❌ 500 problem pairs (just 50)
- ❌ All 12 layers (just 3: early, middle, late)
- ❌ Statistical significance tests (just descriptive stats)
- ❌ Experiments 2-3 (counterfactual, ablation) - **save for later if Exp 1 works**
- ❌ Control conditions (random patching, token position) - **add later if needed**
- ❌ Explicit CoT baseline comparison - **future work**

### Included Features (Updated)
- ✅ **WandB integration** - Real-time monitoring, metrics tracking, artifact storage
- ✅ **3-layer analysis** - Early (L3), Middle (L6), Late (L11)
- ✅ **Scripted + review** problem pairs
- ✅ **Progress tracking** - tqdm + checkpoints

### Why This Is Enough
This MVP answers the **core question**:
- ✅ Does patching clean activations into corrupted problems restore accuracy?
- ✅ If yes → continuous thoughts are causally involved
- ✅ If no → they're epiphenomenal

**If successful**, we can expand to:
- More problems (50 → 500)
- More experiments (counterfactual, ablation)
- More layers (test early vs late)
- Better infrastructure (WandB, fancy viz)
- Statistical rigor (significance tests)

---

## Risk Assessment

### Technical Risks
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Hooks don't work | Medium | Fall back to copying test.py and modifying |
| CODI import issues | Low | Use sys.path or relative imports |
| Cache files too large | Low | 50 pairs × small tensors = <100MB total |
| Runtime too long | Low | 50 pairs should run in <1 hour |

### Scientific Risks
| Risk | Probability | Mitigation |
|------|-------------|------------|
| No causal effect found | Medium | Still valuable negative result! |
| Effects too small to measure | Medium | Try 100 pairs instead of 50 |
| Problem pairs not well-matched | Medium | Manual review ensures quality |

---

## Success Criteria

### Minimum Success (Experiment Runs)
- ✅ All scripts execute without errors
- ✅ Results collected for all 50 pairs
- ✅ Basic plots generated

### Scientific Success (Positive Result)
- ✅ Patched accuracy > Corrupted accuracy (some recovery)
- ✅ Recovery rate > 30% (meaningful effect)
- ✅ Clear interpretation possible

### Scientific Success (Negative Result)
- ✅ Patched accuracy ≈ Corrupted accuracy (no recovery)
- ✅ Conclusion: Continuous thoughts are epiphenomenal
- ✅ Still valuable for the field!

---

## Next Steps After Completion

### If Positive Results (Causal Effects Found):
1. Expand to 500 problem pairs for statistical power
2. Add Experiment 2 (counterfactual patching)
3. Add Experiment 3 (ablation study)
4. Test across layers
5. Compare to explicit CoT baseline
6. Write full paper

### If Negative Results (No Causal Effects):
1. Debug: Check activation extraction is correct
2. Try different patch positions/layers
3. Test on explicit CoT baseline (should show effects)
4. If still null, publish negative result (important for field!)
5. Investigate alternative mechanisms for CODI's success

### If Mixed/Unclear Results:
1. Increase sample size (50 → 200)
2. Add statistical tests
3. Analyze per-problem patterns
4. Try different problem types

---

## Documentation Deliverables

### Code
- `experiments/activation_patching/README.md` - Usage instructions
- Inline code comments explaining key steps
- Requirements file if needed

### Results
- `docs/experiments/activation_patching_results_2025-10-18.md` - Detailed findings
- `docs/research_journal.md` - Summary entry
- Plots saved to `experiments/activation_patching/results/`

### Version Control
- Commit all code with message: "Add activation patching experiment"
- Push to GitHub
- Update .gitignore if needed (cache files, results)

---

## Questions - ALL ANSWERED ✅

1. ✅ **Scope confirmed?** Just Experiment 1 (direct patching) - YES
2. ✅ **Problem pairs**: Scripted + review - YES
3. ✅ **Layers to test**: Early (L3), middle (L6), late (L11) - YES
4. ✅ **Patching positions**: One key [THINK] token per problem - YES
5. ✅ **Directory structure**: `src/experiments/activation_patching/` and `src/viz/` - YES
6. ✅ **Monitoring**: WandB integration (user already logged in) - YES

---

**Status**: ✅ READY TO START - All questions answered
**Estimated Start**: Immediately
**Estimated Completion**: 1-2 days from start
**Next Review**: After Day 1 to check progress
**Owner**: Product Manager → **Ready to hand off to Developer**
