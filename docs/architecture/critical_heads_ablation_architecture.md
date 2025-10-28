# Critical Heads Ablation - Architecture Specification

**Status**: 🎯 Ready for Implementation
**Estimated Time**: 7-8 hours total
**Priority**: HIGH - Validates causal importance of critical heads findings

---

## Executive Summary

We've identified critical attention heads for continuous thought reasoning (L4H5 for LLaMA, L0H3 for GPT-2). This architecture specifies how to test if these heads are **causally necessary** for correct reasoning.

**Scope**: 3 stories implementing ablation experiments
**Goal**: Answer "Are critical heads functionally important or just correlational?"

---

## Context & Prior Work

### What We Know
From [10-28_both_gsm8k_critical_heads_comparison.md](../experiments/10-28_both_gsm8k_critical_heads_comparison.md):

**LLaMA Critical Heads (Top 3)**:
- L4H5: Hub Aggregator, composite=0.528, creates hub at Position 0
- L5H30: Hub Aggregator, composite=0.449
- L5H28: Skip Connection, composite=0.434

**GPT-2 Critical Heads (Top 3)**:
- L0H3: Multi-Purpose, composite=0.600, creates hub at Position 1
- L5H3: Hub Aggregator, composite=0.425
- L6H0: Hub Aggregator, composite=0.417

**Hub Architecture**:
- LLaMA: Hub at Position 0 (CT0), middle layers (L4-L5)
- GPT-2: Hub at Position 1 (CT1), early layers (L0-L1)

### Open Question
Are these attention patterns **causally necessary** for reasoning, or just observational artifacts?

---

## Recommended Implementation Path

### Story 0: Sanity Check (30 minutes)
**Goal**: Validate baseline infrastructure before ablation

### Story 1: Critical Head Ablation with Baseline (3-4 hours)
**Goal**: Test if top 3 heads are causally important vs random heads

### Story 2: Hub Position Patching (3-4 hours)
**Goal**: Test if hub position is a bottleneck for information flow

**Total**: 7-8 hours

**Decision Points**:
- After Story 1: If <5% accuracy drop → Stop (hub is distributed)
- After Story 1: If >10% accuracy drop → Continue to Story 2
- After Story 2: If hub position critical → Consider Phase 2 (cross-model)

---

## Story 0: Sanity Check & Baseline Establishment

### Objective
Validate that inference pipeline works and establish baseline accuracy before ablation experiments.

### Requirements

**Functional**:
1. Load CODI models (LLaMA 1B, GPT-2 124M)
2. Load GSM8K test set (100-problem subset for development)
3. Run inference and compute accuracy
4. Verify results match expected baseline (~80-85% for CODI models)

**Non-Functional**:
- Execution time: <10 minutes
- Should work on existing GPU (4GB VRAM sufficient)

### Implementation Approach

```python
# File: src/experiments/codi_attention_flow/ablation/0_sanity_check.py

from pathlib import Path
import json
from datasets import load_dataset
from cache_activations_llama import ActivationCacherLLaMA
from cache_activations import ActivationCacher

def sanity_check(model_name='llama', n_problems=100):
    """
    Validate baseline inference pipeline.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems (default 100)

    Returns:
        baseline_accuracy: float (0-1)
    """
    # Load model
    model_paths = {
        'llama': str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
        'gpt2': str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    }

    if model_name == 'llama':
        cacher = ActivationCacherLLaMA(model_paths[model_name])
    else:
        cacher = ActivationCacher(model_paths[model_name])

    model = cacher.model
    tokenizer = cacher.tokenizer

    # Load test set
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = dataset.select(range(n_problems))

    # Run inference
    n_correct = 0
    for problem in test_problems:
        question = problem['question']
        gold_answer = extract_answer(problem['answer'])

        # Generate answer
        pred_answer = model.solve(question)

        if pred_answer == gold_answer:
            n_correct += 1

    accuracy = n_correct / n_problems

    # Save baseline
    results = {
        'model': model_name,
        'n_problems': n_problems,
        'accuracy': accuracy,
        'n_correct': n_correct
    }

    output_path = Path(__file__).parent.parent / 'results' / f'{model_name}_baseline.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{model_name.upper()} Baseline Accuracy: {accuracy:.2%} ({n_correct}/{n_problems})")

    return accuracy

def extract_answer(answer_str):
    """Extract numeric answer from GSM8K format."""
    # Answer format: "#### 42"
    return int(answer_str.split('####')[1].strip())
```

### Data Flow

```
GSM8K Test Set (HuggingFace)
    ↓
Select 100 problems
    ↓
Load CODI Model
    ↓
For each problem:
    - Tokenize question
    - Generate answer (6 CT tokens + answer)
    - Extract numeric answer
    - Compare to gold answer
    ↓
Compute accuracy
    ↓
Save baseline.json
```

### Success Criteria
- ✅ LLaMA baseline accuracy: 75-85%
- ✅ GPT-2 baseline accuracy: 70-80%
- ✅ Inference completes in <10 minutes
- ✅ Baseline saved to `results/{model}_baseline.json`

### Deliverables
1. Script: `ablation/0_sanity_check.py`
2. Baseline results: `results/llama_baseline.json`, `results/gpt2_baseline.json`
3. Console output showing accuracy

---

## Story 1: Critical Head Ablation with Random Baseline

### Objective
Test if top 3 critical heads are causally important by comparing ablation effects vs random heads.

### Requirements

**Functional**:
1. Implement attention head ablation mechanism (zero out specific head outputs)
2. Ablate 4 conditions:
   - **Condition A**: Top 3 critical heads
   - **Condition B**: 3 random heads (average of 5 random samples)
   - **Condition C**: Bottom 3 heads (lowest composite scores)
   - **Condition D**: Baseline (no ablation) - from Story 0
3. Measure accuracy for each condition
4. Compute accuracy drop: `baseline_acc - ablated_acc`
5. Test on both LLaMA and GPT-2

**Non-Functional**:
- Use 100 problems for development, 1,000 for final results
- Execution time: ~10 min (100 problems), ~1.5 hours (1,000 problems)
- GPU memory: <6GB VRAM

### Implementation Approach

#### Architecture Design

**Ablation Mechanism**:
```python
# Hook-based ablation (cleanest approach)
class AttentionAblator:
    """Zero out specific attention heads during forward pass."""

    def __init__(self, model, heads_to_ablate):
        """
        Args:
            model: CODI model
            heads_to_ablate: List of (layer, head) tuples
        """
        self.model = model
        self.heads_to_ablate = set(heads_to_ablate)
        self.hooks = []

    def ablation_hook(self, layer_idx):
        """Create hook function for specific layer."""
        def hook(module, input, output):
            # output is attention output: [batch, seq_len, hidden_dim]
            # Need to zero out specific heads

            # Get attention head outputs
            # LLaMA: 32 heads × 64 dim = 2048 hidden
            # GPT-2: 12 heads × 64 dim = 768 hidden

            batch_size, seq_len, hidden_dim = output.shape
            n_heads = self.model.config.num_attention_heads
            head_dim = hidden_dim // n_heads

            # Reshape to separate heads
            output_heads = output.view(batch_size, seq_len, n_heads, head_dim)

            # Zero out ablated heads
            for head_idx in range(n_heads):
                if (layer_idx, head_idx) in self.heads_to_ablate:
                    output_heads[:, :, head_idx, :] = 0.0

            # Reshape back
            output = output_heads.view(batch_size, seq_len, hidden_dim)

            return output

        return hook

    def __enter__(self):
        """Register hooks on entry."""
        # Find attention output layers
        for layer_idx, layer in enumerate(self.model.layers):
            # Hook into attention output projection
            hook = layer.self_attn.o_proj.register_forward_hook(
                self.ablation_hook(layer_idx)
            )
            self.hooks.append(hook)

        return self

    def __exit__(self, *args):
        """Remove hooks on exit."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

#### Main Ablation Script

```python
# File: src/experiments/codi_attention_flow/ablation/1_ablate_critical_heads.py

import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
import pandas as pd

def run_ablation_experiment(
    model_name='llama',
    n_problems=100,
    n_random_samples=5
):
    """
    Run head ablation experiment with 4 conditions.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems
        n_random_samples: Number of random head samples to average

    Returns:
        results: Dict with accuracy for each condition
    """
    # Load baseline
    baseline_path = Path(__file__).parent.parent / 'results' / f'{model_name}_baseline.json'
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    baseline_acc = baseline['accuracy']

    # Load critical heads rankings
    rankings_path = Path(__file__).parent.parent / 'results' / model_name / 'ranked_heads.csv'
    df = pd.read_csv(rankings_path)

    # Define ablation conditions
    top_3_heads = [
        (int(df.iloc[0]['layer']), int(df.iloc[0]['head'])),
        (int(df.iloc[1]['layer']), int(df.iloc[1]['head'])),
        (int(df.iloc[2]['layer']), int(df.iloc[2]['head']))
    ]

    bottom_3_heads = [
        (int(df.iloc[-1]['layer']), int(df.iloc[-1]['head'])),
        (int(df.iloc[-2]['layer']), int(df.iloc[-2]['head'])),
        (int(df.iloc[-3]['layer']), int(df.iloc[-3]['head']))
    ]

    # Load model
    model = load_model(model_name)

    # Load test set
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = dataset.select(range(n_problems))

    results = {
        'model': model_name,
        'n_problems': n_problems,
        'baseline_accuracy': baseline_acc,
        'conditions': {}
    }

    # Condition A: Ablate top 3 critical heads
    print(f"\nCondition A: Ablating top 3 critical heads...")
    print(f"  Heads: {top_3_heads}")

    with AttentionAblator(model, top_3_heads):
        top3_acc = evaluate_model(model, test_problems)

    results['conditions']['top_3_critical'] = {
        'heads': top_3_heads,
        'accuracy': top3_acc,
        'drop': baseline_acc - top3_acc,
        'drop_pct': 100 * (baseline_acc - top3_acc) / baseline_acc
    }

    print(f"  Accuracy: {top3_acc:.2%} (drop: {results['conditions']['top_3_critical']['drop']:.2%})")

    # Condition B: Ablate 3 random heads (average of 5 samples)
    print(f"\nCondition B: Ablating 3 random heads (average of {n_random_samples} samples)...")

    random_accuracies = []
    all_heads = [(int(row['layer']), int(row['head'])) for _, row in df.iterrows()]

    for sample_idx in range(n_random_samples):
        random_heads = random.sample(all_heads, 3)
        print(f"  Sample {sample_idx+1}: {random_heads}")

        with AttentionAblator(model, random_heads):
            random_acc = evaluate_model(model, test_problems)

        random_accuracies.append(random_acc)

    random_acc_mean = np.mean(random_accuracies)
    random_acc_std = np.std(random_accuracies)

    results['conditions']['random_3'] = {
        'n_samples': n_random_samples,
        'accuracies': random_accuracies,
        'accuracy_mean': random_acc_mean,
        'accuracy_std': random_acc_std,
        'drop_mean': baseline_acc - random_acc_mean,
        'drop_pct_mean': 100 * (baseline_acc - random_acc_mean) / baseline_acc
    }

    print(f"  Accuracy: {random_acc_mean:.2%} ± {random_acc_std:.2%}")
    print(f"  Drop: {results['conditions']['random_3']['drop_mean']:.2%}")

    # Condition C: Ablate bottom 3 heads
    print(f"\nCondition C: Ablating bottom 3 heads...")
    print(f"  Heads: {bottom_3_heads}")

    with AttentionAblator(model, bottom_3_heads):
        bottom3_acc = evaluate_model(model, test_problems)

    results['conditions']['bottom_3'] = {
        'heads': bottom_3_heads,
        'accuracy': bottom3_acc,
        'drop': baseline_acc - bottom3_acc,
        'drop_pct': 100 * (baseline_acc - bottom3_acc) / baseline_acc
    }

    print(f"  Accuracy: {bottom3_acc:.2%} (drop: {results['conditions']['bottom_3']['drop']:.2%})")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    top_drop = results['conditions']['top_3_critical']['drop']
    random_drop = results['conditions']['random_3']['drop_mean']
    bottom_drop = results['conditions']['bottom_3']['drop']

    print(f"\nAccuracy Drops:")
    print(f"  Top 3 critical heads:  {top_drop:.2%}")
    print(f"  Random 3 heads:        {random_drop:.2%}")
    print(f"  Bottom 3 heads:        {bottom_drop:.2%}")

    # Statistical test
    if top_drop > random_drop + 0.05:  # 5% threshold
        print(f"\n✅ CRITICAL HEADS ARE CAUSALLY IMPORTANT")
        print(f"   Top heads drop ({top_drop:.2%}) >> Random drop ({random_drop:.2%})")
        decision = "continue_to_story_2"
    elif top_drop > 0.10:  # 10% absolute drop
        print(f"\n⚠️  HEADS MATTER BUT NOT SPECIAL")
        print(f"   Top heads drop similar to random (network fragility)")
        decision = "investigate_distributed_hub"
    else:
        print(f"\n❌ CRITICAL HEADS NOT CAUSALLY IMPORTANT")
        print(f"   All drops <10% - hub is distributed or redundant")
        decision = "stop_ablation_experiments"

    results['decision'] = decision

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / f'{model_name}_ablation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved: {output_path}")

    return results

def evaluate_model(model, test_problems):
    """Evaluate model accuracy on test problems."""
    n_correct = 0
    for problem in test_problems:
        question = problem['question']
        gold_answer = extract_answer(problem['answer'])

        pred_answer = model.solve(question)

        if pred_answer == gold_answer:
            n_correct += 1

    return n_correct / len(test_problems)

def load_model(model_name):
    """Load CODI model."""
    # Implementation from Story 0
    pass

def extract_answer(answer_str):
    """Extract numeric answer from GSM8K format."""
    return int(answer_str.split('####')[1].strip())
```

### Data Flow

```
Load baseline accuracy
    ↓
Load ranked heads (from critical heads analysis)
    ↓
Define 4 conditions:
    - Top 3 critical heads
    - 3 random heads (5 samples, averaged)
    - Bottom 3 heads
    - Baseline (no ablation)
    ↓
For each condition:
    - Register ablation hooks
    - Run inference on test set
    - Compute accuracy
    - Remove hooks
    ↓
Compare accuracy drops
    ↓
Decision: Continue to Story 2? Stop?
    ↓
Save results + decision
```

### Success Criteria

**If top 3 critical heads are causally important**:
- ✅ Top 3 drop >10% accuracy
- ✅ Top 3 drop > Random drop + 5%
- ✅ Decision: Continue to Story 2

**If critical heads are NOT special**:
- ❌ Top 3 drop ≈ Random drop (within 5%)
- ❌ All drops <10% (network fragility or distributed hub)
- ❌ Decision: Stop experiments or pivot

### Deliverables

1. **Script**: `ablation/1_ablate_critical_heads.py`
2. **Results**: `results/{model}_ablation_results.json`
3. **Visualization**: Bar chart showing accuracy drops per condition
4. **Experiment Report**: Documenting findings and decision

---

## Story 2: Hub Position Patching

### Objective
Test if hub position (Position 0 for LLaMA, Position 1 for GPT-2) is a causal bottleneck for information flow.

### Requirements

**Functional**:
1. Implement activation patching mechanism (replace activations with noise)
2. Test all 6 continuous thought positions (CT0-CT5)
3. Measure accuracy drop per position
4. Compare hub position vs non-hub positions
5. Test on both LLaMA and GPT-2

**Non-Functional**:
- Use 100 problems for development
- Execution time: ~15 minutes (100 problems × 6 positions)
- GPU memory: <6GB VRAM

**Decision Dependency**:
- **Only run if Story 1 shows >10% drop** from ablating critical heads
- If Story 1 shows <5% drop, skip this story

### Implementation Approach

#### Activation Patching Mechanism

```python
# File: src/experiments/codi_attention_flow/ablation/2_patch_hub_position.py

import torch
import numpy as np

class ActivationPatcher:
    """Replace activations at specific continuous thought positions with noise."""

    def __init__(self, model, position_to_patch, noise_type='gaussian'):
        """
        Args:
            model: CODI model
            position_to_patch: Which CT position to patch (0-5)
            noise_type: 'gaussian' or 'uniform'
        """
        self.model = model
        self.position_to_patch = position_to_patch
        self.noise_type = noise_type
        self.hooks = []
        self.ct_positions = None  # Will be set during forward pass

    def patch_hook(self, layer_idx):
        """Create hook to replace activations at CT position."""
        def hook(module, input, output):
            # output: [batch, seq_len, hidden_dim]

            if self.ct_positions is None:
                return output  # Not ready yet

            # Get position of continuous thought token to patch
            patch_pos = self.ct_positions[self.position_to_patch]

            # Replace with noise
            if self.noise_type == 'gaussian':
                noise = torch.randn_like(output[:, patch_pos, :])
            else:  # uniform
                noise = torch.rand_like(output[:, patch_pos, :]) * 2 - 1

            output[:, patch_pos, :] = noise

            return output

        return hook

    def set_ct_positions(self, ct_positions):
        """Set continuous thought token positions for current problem."""
        self.ct_positions = ct_positions

    def __enter__(self):
        """Register hooks."""
        for layer_idx, layer in enumerate(self.model.layers):
            hook = layer.register_forward_hook(self.patch_hook(layer_idx))
            self.hooks.append(hook)
        return self

    def __exit__(self, *args):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
```

#### Main Patching Experiment

```python
def run_position_patching_experiment(
    model_name='llama',
    n_problems=100
):
    """
    Test if hub position is a bottleneck by patching each position.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems

    Returns:
        results: Dict with accuracy for each position
    """
    # Load baseline
    baseline_path = Path(__file__).parent.parent / 'results' / f'{model_name}_baseline.json'
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    baseline_acc = baseline['accuracy']

    # Load hub position from summary
    summary_path = Path(__file__).parent.parent / 'results' / model_name / 'attention_summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    hub_position = summary['hub_analysis']['hub_position']

    print(f"\n{model_name.upper()} Hub Position: {hub_position}")

    # Load model
    model = load_model(model_name)

    # Load test set
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = dataset.select(range(n_problems))

    results = {
        'model': model_name,
        'n_problems': n_problems,
        'baseline_accuracy': baseline_acc,
        'hub_position': hub_position,
        'positions': {}
    }

    # Test each position
    for position in range(6):
        print(f"\nPatching Position {position}...")

        with ActivationPatcher(model, position):
            # Run inference with patching
            n_correct = 0
            for problem in test_problems:
                question = problem['question']
                gold_answer = extract_answer(problem['answer'])

                # Track CT positions during generation
                # (model needs to expose this or we track it)
                pred_answer = model.solve(question)

                if pred_answer == gold_answer:
                    n_correct += 1

            patched_acc = n_correct / n_problems

        drop = baseline_acc - patched_acc
        is_hub = (position == hub_position)

        results['positions'][position] = {
            'accuracy': patched_acc,
            'drop': drop,
            'drop_pct': 100 * drop / baseline_acc,
            'is_hub': is_hub
        }

        marker = "🎯 HUB" if is_hub else ""
        print(f"  Position {position} {marker}: {patched_acc:.2%} (drop: {drop:.2%})")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    hub_drop = results['positions'][hub_position]['drop']

    # Average drop for non-hub positions
    non_hub_drops = [
        results['positions'][pos]['drop']
        for pos in range(6)
        if pos != hub_position
    ]
    avg_non_hub_drop = np.mean(non_hub_drops)

    print(f"\nHub Position {hub_position} drop:     {hub_drop:.2%}")
    print(f"Non-hub positions avg drop: {avg_non_hub_drop:.2%}")
    print(f"Delta:                      {hub_drop - avg_non_hub_drop:.2%}")

    # Decision
    if hub_drop > avg_non_hub_drop + 0.10:  # 10% threshold
        print(f"\n✅ HUB POSITION IS CAUSALLY CRITICAL")
        print(f"   Hub drop significantly larger than non-hub")
        decision = "hub_is_bottleneck"
    elif hub_drop > 0.15:  # 15% absolute drop
        print(f"\n⚠️  HUB POSITION MATTERS BUT NOT UNIQUE")
        print(f"   All positions show large drops (distributed processing)")
        decision = "distributed_importance"
    else:
        print(f"\n❌ HUB POSITION NOT CRITICAL")
        print(f"   Similar drops across all positions")
        decision = "hub_not_bottleneck"

    results['decision'] = decision
    results['hub_drop'] = hub_drop
    results['avg_non_hub_drop'] = avg_non_hub_drop

    # Save results
    output_path = Path(__file__).parent.parent / 'results' / f'{model_name}_position_patching_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved: {output_path}")

    return results
```

### Data Flow

```
Load baseline accuracy
    ↓
Load hub position (from attention summary)
    ↓
For each CT position (0-5):
    - Register patching hooks
    - Run inference on test set
    - Replace position activations with noise
    - Compute accuracy
    - Remove hooks
    ↓
Compare hub position drop vs non-hub drops
    ↓
Decision: Is hub a bottleneck?
    ↓
Save results
```

### Success Criteria

**If hub position is a bottleneck**:
- ✅ Hub position drop >15% accuracy
- ✅ Hub drop > Non-hub average + 10%
- ✅ Decision: Hub-centric architecture is causally critical

**If hub position is NOT special**:
- ❌ Hub drop ≈ Non-hub drops (within 5%)
- ❌ All positions show large drops (distributed processing)
- ❌ Decision: Hub is not a unique bottleneck

### Deliverables

1. **Script**: `ablation/2_patch_hub_position.py`
2. **Results**: `results/{model}_position_patching_results.json`
3. **Visualization**: Bar chart showing accuracy drop per position (highlight hub)
4. **Experiment Report**: Documenting findings

---

## Technical Architecture

### Directory Structure

```
src/experiments/codi_attention_flow/
├── ablation/                          # NEW
│   ├── __init__.py
│   ├── 0_sanity_check.py             # Story 0
│   ├── 1_ablate_critical_heads.py    # Story 1
│   ├── 2_patch_hub_position.py       # Story 2
│   └── utils.py                       # Shared utilities
├── results/
│   ├── llama_baseline.json           # From Story 0
│   ├── gpt2_baseline.json
│   ├── llama_ablation_results.json   # From Story 1
│   ├── gpt2_ablation_results.json
│   ├── llama_position_patching_results.json  # From Story 2
│   └── gpt2_position_patching_results.json
└── figures/
    ├── ablation_comparison.png        # From Story 1
    └── position_patching_comparison.png  # From Story 2
```

### Shared Utilities

```python
# File: ablation/utils.py

def load_model(model_name):
    """Load CODI model (LLaMA or GPT-2)."""
    model_paths = {
        'llama': str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
        'gpt2': str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    }

    if model_name == 'llama':
        cacher = ActivationCacherLLaMA(model_paths[model_name])
    else:
        cacher = ActivationCacher(model_paths[model_name])

    return cacher.model

def extract_answer(answer_str):
    """Extract numeric answer from GSM8K format."""
    return int(answer_str.split('####')[1].strip())

def create_comparison_chart(results, output_path):
    """Create bar chart comparing accuracy drops."""
    import matplotlib.pyplot as plt

    # Implementation details...
    pass
```

### Dependencies

**Existing Infrastructure** (✅ Available):
- CODI models: `~/codi_ckpt/llama_gsm8k/`, `~/codi_ckpt/gpt2_gsm8k/`
- Activation cachers: `cache_activations.py`, `cache_activations_llama.py`
- GSM8K dataset: HuggingFace `datasets` library
- Critical heads rankings: `results/{model}/ranked_heads.csv`
- Hub position data: `results/{model}/attention_summary.json`

**New Infrastructure** (⚠️ To Build):
- `AttentionAblator` class (Story 1)
- `ActivationPatcher` class (Story 2)
- Evaluation utilities

**Python Packages**:
- PyTorch 2.0+ (already installed)
- Transformers 4.30+ (already installed)
- Datasets 2.0+ (already installed)
- NumPy, Pandas, Matplotlib (already installed)

---

## Data Requirements

### Test Set

**Source**: GSM8K test split (HuggingFace)
- Total available: 1,319 problems
- Development: Use 100 problems (fast iteration)
- Final validation: Use 1,000 problems (robust statistics)

**Format**:
```json
{
  "question": "Natalia sold clips to 48 of her friends...",
  "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 24"
}
```

### Critical Heads Data

**Source**: From prior experiment
- File: `results/{model}/ranked_heads.csv`
- Contains: 512 LLaMA heads, 144 GPT-2 heads
- Fields: layer, head, composite_score, functional_type

**Top 3 Heads**:
```csv
layer,head,composite_score,functional_type
4,5,0.528,Hub Aggregator
5,30,0.449,Hub Aggregator
5,28,0.434,Skip Connection
```

### Hub Position Data

**Source**: From prior experiment
- File: `results/{model}/attention_summary.json`
- Contains: Hub position, hub score, hub ratio

```json
{
  "hub_analysis": {
    "hub_position": 0,
    "hub_score": 0.197,
    "hub_ratio": 1.18
  }
}
```

---

## Performance Considerations

### Inference Speed

**100 problems**:
- LLaMA: ~8-10 minutes
- GPT-2: ~5-7 minutes
- Total: ~15 minutes per model

**1,000 problems**:
- LLaMA: ~80-100 minutes
- GPT-2: ~50-70 minutes
- Total: ~2.5 hours per model

**Optimization**: Use batching if memory permits (likely not - CODI is autoregressive)

### Memory Requirements

**GPU Memory**:
- LLaMA: ~3.5 GB VRAM
- GPT-2: ~1.5 GB VRAM
- Safe with 4GB GPU

**Disk Space**:
- Results files: ~50 KB per experiment
- Figures: ~500 KB per chart
- Total: <5 MB

---

## Error Handling

### Known Failure Modes

1. **Model Loading Failure**
   - Cause: Checkpoint not found or corrupted
   - Mitigation: Check path before loading, validate checkpoint

2. **OOM (Out of Memory)**
   - Cause: GPU memory exhausted
   - Mitigation: Use CPU fallback, reduce batch size

3. **Attention Hook Failure**
   - Cause: Model architecture changed or hook registered incorrectly
   - Mitigation: Validate hook registration, test on single example first

4. **Answer Extraction Failure**
   - Cause: GSM8K answer format variation
   - Mitigation: Robust parsing with fallback

### Validation Checks

**Before Running**:
- ✅ Models exist at expected paths
- ✅ Test set loads successfully
- ✅ Critical heads file exists
- ✅ GPU available (or CPU fallback ready)

**During Execution**:
- ✅ Hooks register successfully
- ✅ Inference produces valid outputs
- ✅ Accuracy is in reasonable range (20-90%)

**After Completion**:
- ✅ Results file saved
- ✅ All conditions executed
- ✅ Decision criteria clear

---

## Testing Strategy

### Unit Tests

**Test Ablation Mechanism**:
```python
def test_attention_ablator():
    """Test that ablation zeros out correct heads."""
    model = load_test_model()
    heads_to_ablate = [(0, 0), (1, 5)]

    with AttentionAblator(model, heads_to_ablate):
        # Run forward pass
        output = model(test_input)

        # Verify heads are zeroed
        # (requires inspection of intermediate activations)
        assert output is not None
```

**Test Patching Mechanism**:
```python
def test_activation_patcher():
    """Test that patching replaces activations with noise."""
    model = load_test_model()

    with ActivationPatcher(model, position=0):
        output1 = model(test_input)
        output2 = model(test_input)

        # Outputs should differ (stochastic noise)
        assert not torch.allclose(output1, output2)
```

### Integration Tests

**End-to-End Sanity Check**:
```python
def test_story_0_sanity_check():
    """Test baseline inference works."""
    accuracy = sanity_check('llama', n_problems=10)
    assert 0.5 < accuracy < 0.95  # Reasonable range
```

**Ablation Pipeline**:
```python
def test_story_1_ablation():
    """Test ablation experiment runs."""
    results = run_ablation_experiment('llama', n_problems=10)
    assert 'conditions' in results
    assert 'top_3_critical' in results['conditions']
```

---

## Success Metrics & Decision Criteria

### Story 0: Sanity Check
- ✅ Baseline accuracy 70-85%
- ✅ Completes in <10 minutes
- ✅ Results saved

### Story 1: Critical Head Ablation

**Success (Continue to Story 2)**:
- ✅ Top 3 drop >10%
- ✅ Top 3 drop > Random + 5%
- ✅ Statistically significant

**Failure (Stop or Pivot)**:
- ❌ Top 3 drop <5%
- ❌ Top 3 ≈ Random (within 3%)
- ❌ All conditions <10% drop

### Story 2: Hub Position Patching

**Success (Hub is bottleneck)**:
- ✅ Hub drop >15%
- ✅ Hub > Non-hub + 10%

**Failure (Hub not special)**:
- ❌ Hub ≈ Non-hub (within 5%)

---

## Timeline & Effort Estimate

| Story | Task | Time |
|-------|------|------|
| **Story 0** | Sanity Check | 30 min |
| | - Write sanity check script | 15 min |
| | - Run baseline (both models) | 10 min |
| | - Validate results | 5 min |
| **Story 1** | Critical Head Ablation | 3-4 hours |
| | - Implement AttentionAblator | 1h |
| | - Write ablation script | 1h |
| | - Run experiments (100 problems) | 30 min |
| | - Analysis & visualization | 45 min |
| | - Documentation | 30 min |
| **Story 2** | Hub Position Patching | 3-4 hours |
| | - Implement ActivationPatcher | 1h |
| | - Write patching script | 1h |
| | - Run experiments (100 problems) | 30 min |
| | - Analysis & visualization | 45 min |
| | - Documentation | 30 min |
| **Total** | | **7-8.5 hours** |

---

## Risk Assessment

### High Risk
None identified

### Medium Risk

**Risk 1: Ablation doesn't work as expected**
- Probability: Low (20%)
- Impact: Medium (need to debug hooks)
- Mitigation: Test on single example first, validate hook registration

**Risk 2: All ablation drops are small (<5%)**
- Probability: Medium (40%)
- Impact: Low (still valuable negative result)
- Mitigation: Document as "hub is distributed, not localized"

### Low Risk

**Risk 3: Inference is slower than expected**
- Probability: Low (20%)
- Impact: Low (just takes longer)
- Mitigation: Use smaller test set (100 instead of 1,000)

---

## Deliverables Summary

### Code
1. `ablation/0_sanity_check.py` - Baseline validation
2. `ablation/1_ablate_critical_heads.py` - Head ablation
3. `ablation/2_patch_hub_position.py` - Position patching
4. `ablation/utils.py` - Shared utilities

### Data
1. `results/llama_baseline.json` - LLaMA baseline
2. `results/gpt2_baseline.json` - GPT-2 baseline
3. `results/llama_ablation_results.json` - LLaMA ablation
4. `results/gpt2_ablation_results.json` - GPT-2 ablation
5. `results/llama_position_patching_results.json` - LLaMA patching
6. `results/gpt2_position_patching_results.json` - GPT-2 patching

### Visualizations
1. `figures/ablation_comparison.png` - Bar chart of accuracy drops
2. `figures/position_patching_comparison.png` - Position patching results

### Documentation
1. Experiment report: `docs/experiments/MM-DD_{model}_ablation_analysis.md`
2. Architecture review (this document)

---

## Open Questions for Architect

1. **Hook Implementation**: Should we use `register_forward_hook` or custom forward pass modification?
   - Recommendation: Use hooks (cleaner, less invasive)

2. **Noise Type**: Gaussian or uniform noise for patching?
   - Recommendation: Gaussian (matches typical activation distributions)

3. **CT Position Tracking**: How to track continuous thought token positions during generation?
   - Options:
     - A) Modify model to expose positions
     - B) Track based on sequence length (question_len + 1 + step)
   - Recommendation: Option B (non-invasive)

4. **Batch Size**: Run one problem at a time or batch?
   - Recommendation: One at a time (CODI is autoregressive, hard to batch)

5. **Final Test Set Size**: 100 or 1,000 problems?
   - Recommendation: 100 for development, 1,000 for publication

---

## Next Steps

1. **Architect reviews this specification**
2. **Architect addresses open questions**
3. **Architect validates technical approach**
4. **PM creates user stories from this spec**
5. **Developer implements Story 0 → Story 1 → Story 2**

---

**Status**: 🎯 Ready for Architect Review
**Created**: 2025-10-28
**Estimated Total Time**: 7-8.5 hours
**Priority**: HIGH

---

## References

- **Prior Work**: [10-28_both_gsm8k_critical_heads_comparison.md](../experiments/10-28_both_gsm8k_critical_heads_comparison.md)
- **Critical Heads Data**: `src/experiments/codi_attention_flow/results/{model}/ranked_heads.csv`
- **Hub Position Data**: `src/experiments/codi_attention_flow/results/{model}/attention_summary.json`
- **PM Handoff**: [mechanistic_analysis_follow_up_ideas.md](../project/mechanistic_analysis_follow_up_ideas.md)
