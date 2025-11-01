# CODI Chain-of-Thought Circuit Analysis

Analysis of computational circuits in CODI-LLaMA's latent chain-of-thought reasoning using intervention-based methods.

## Overview

This experiment applies circuit analysis techniques (inspired by IOI circuit analysis in transformers) to understand information flow in CODI's continuous chain-of-thought computation. CODI uses 7 latent reasoning steps (BoT + 6 iterations) to solve arithmetic problems.

**Model:** CODI-LLaMA 3.2-1B
**Dataset:** GSM8K (math word problems)
**Method:** Average activation intervention across all CoT positions

## Key Findings

### Intervention Cascade Effects

Using average activation replacement (computed from 100 training examples), we measured how interventions propagate through the 7-position reasoning chain:

**Strongest Cascades:**
- Position 2 → Position 3: **80% token change rate** (dominant cascade)
- Position 4 → Position 5: 50% token change
- Position 1 → Position 2: 50% token change

**Robust Positions:**
- Position 0: Completely robust (0% affected by any intervention)
- Position 6: Terminal position, 0% self-intervention effect

**Downstream Impact:**
- Position 4 has highest average downstream impact: 52%
- Most other positions: ~26-30% average downstream impact

### Computational Specialization

Analysis of token decoding patterns reveals functional specialization:

**Operator Positions (5-10% decode to numbers):**
- Position 3: 5% numbers - generates arithmetic operators (+, -, >>)
- Position 6: 10% numbers - terminal answer position

**Arithmetic Positions (90% decode to numbers):**
- Position 1: 90% numbers
- Position 4: 90% numbers

**Mixed Positions (55-65% decode to numbers):**
- Positions 0, 2, 5: Hybrid numerical/operator roles

### Comparison: Number-Only vs Average Activation Intervention

Previous intervention methods only modified number tokens, skipping operators. This created artifacts:

- **Position 3 appeared robust** with number-only (0% change)
- **Position 3 actually vulnerable** with average activation (80% change from position 2)
- **3.4x more cascade effects detected** with average activation (11.2% vs 3.3% overall)

The number-only approach severely underestimated cascade effects because position 3 generates operators that were skipped.

## Methodology

### Average Activation Intervention

1. **Compute Reference Activations:**
   - Run 100 GSM8K training examples through model
   - For each CoT position, compute average hidden state across examples
   - Store 7 average activation vectors

2. **Intervention Analysis:**
   - For each test example:
     - Run baseline (no intervention)
     - Run 7 intervention conditions (one per position)
     - Intervention: Replace hidden state with average activation
   - Measure cascade effects: token changes, norm differences

3. **Aggregate Statistics:**
   - 20 test examples × 8 conditions = 160 total runs
   - Compute token change rates, hidden state perturbations
   - Analyze position-specific patterns

### Dual Pathway Architecture

CODI processes information through two pathways at each iteration:
- **Attention pathway:** Self-attention over previous latent states
- **Direct projection:** Learned projection transformation

Early analysis showed roughly equal contribution (1.06:1 ratio).

## Files

### Analysis Scripts

**Core Intervention Analysis:**
- `codi_intervention_average_activation.py` - Main average activation intervention (final method)
- `codi_intervention_cascade_enhanced.py` - Enhanced cascade with continuous measures (20 examples)
- `codi_intervention_propagation.py` - Single-example intervention cascade

**Pathway Analysis:**
- `codi_pathway_decomposition.py` - Attention vs projection pathway decomposition
- `codi_quick_analysis.py` - Quick pathway contribution measurement
- `codi_attention_analysis.py` - Detailed attention pattern analysis

**Full Analysis:**
- `codi_full_circuit_analysis.py` - Comprehensive circuit analysis combining all methods
- `codi_circuit_analysis.py` - Initial circuit analysis implementation

**Utilities:**
- `download_model.py` - Download CODI-LLaMA from HuggingFace

### Visualization Scripts

- `visualize_avg_intervention_flat.py` - Flattened bar chart (3 metrics per position)
- `visualize_avg_intervention_simple.py` - 3-panel heatmap (token change, norm diff, number decode)
- `visualize_intervention_comparison.py` - Number-only vs average activation comparison
- `visualize_enhanced_cascade.py` - Enhanced cascade visualization (4 panels)
- `create_visualizations.py` - Initial visualization generation

### Results

**Average Activation Intervention (Final):**
- `avg_intervention_cascade_statistics.json` - Aggregated statistics (11KB)
- `avg_intervention_cascade_raw.json` - Raw data from 160 runs (352KB)
- `average_activations_train100.json` - Computed average activations (289KB)
- `avg_intervention_flat.png` - Flattened bar chart visualization
- `avg_intervention_simple.png` - 3-panel heatmap

**Enhanced Cascade (Number-only, for comparison):**
- `enhanced_cascade_statistics.json` - Aggregated statistics (14KB)
- `enhanced_cascade_raw.json` - Raw data (343KB)
- `enhanced_cascade_visualization.png` - 4-panel visualization
- `enhanced_cascade_summary.png` - Summary findings

**Method Comparison:**
- `intervention_method_comparison.png` - Side-by-side comparison

**Pathway Analysis:**
- `pathway_contributions.png` - Attention vs projection contributions
- `intervention_cascade.png` - Intervention cascade diagram
- `circuit_diagram.png` - Full circuit diagram

**Initial Analysis:**
- `circuit_analysis.json` - Initial pathway decomposition data
- `intervention_propagation.json` - Single-example cascade data

## Quantitative Results

### Per-Position Metrics

| Position | Next Impact | Avg Downstream | Number Decode |
|----------|-------------|----------------|---------------|
| P0 | 25% | 27% | 55% |
| P1 | 50% | 26% | 90% |
| P2 | 80% | 26% | 65% |
| P3 | 0% | 7% | 5% |
| P4 | 50% | 52% | 90% |
| P5 | 30% | 30% | 65% |
| P6 | 0% | 0% | 10% |

- **Next Impact:** Token change rate when intervening on this position → next position
- **Avg Downstream:** Average token change rate across all following positions
- **Number Decode:** Percentage of tokens that decode to numbers at this position

### Overall Statistics

- **Average cascade strength:** 11.2% token change across all position pairs
- **Average norm perturbation:** 0.94 L2 norm difference
- **Number decoding rate:** 57% of tokens are numbers

## Technical Details

**Model Configuration:**
- Base: LLaMA 3.2-1B
- LoRA: r=128, alpha=32
- Projection dimension: 2048
- Latent iterations: 6
- Total positions: 7 (BoT + 6 iterations)

**Intervention Method:**
- Replace hidden state with average activation from training set
- Applied BEFORE projection at each iteration
- No restriction on token type (numbers, operators, text)

**Datasets:**
- Training: GSM8K train split (100 examples for averaging)
- Test: GSM8K test split (20 examples for intervention analysis)

## Interpretation

The analysis reveals a structured computational pipeline:

1. **Position 0-2: Setup Phase**
   - Mixed number/operator processing
   - Moderate cascade effects

2. **Position 3: Operator Generation**
   - Specialized for operators (+, -, >>)
   - Highly vulnerable to position 2 interventions (80%)
   - But position 4 is robust to position 3 (0% impact)

3. **Position 4-5: Computation Phase**
   - High number decoding (90%)
   - Position 4 has strongest downstream impact (52%)

4. **Position 6: Terminal Answer**
   - Low number decoding (10%)
   - Completely robust to interventions

The alternating pattern of arithmetic (high number %) and operator (low number %) positions suggests CODI implements a step-by-step calculation process internally.

## Related Work

This analysis applies techniques from:
- Wang et al. (2022): Interpretability in the Wild - IOI circuit analysis
- Conmy et al. (2023): Automated circuit discovery in transformers
- Elhage et al. (2021): Mathematical framework for transformer circuits

Adapted for continuous latent reasoning in CODI's non-standard architecture.
