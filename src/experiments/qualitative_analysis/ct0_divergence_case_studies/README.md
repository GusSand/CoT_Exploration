# CT0 Hidden State Divergence - Qualitative Case Studies

## Overview

This directory contains qualitative case studies demonstrating how intermediate continuous thought (CT) tokens diverge when CT0 attention is blocked in CODI reasoning chains.

## Key Finding

When CT0 attention is blocked, **intermediate CT tokens change immediately** - not just the final answer. Divergence starts at CT1 (30% diverged) and cascades through the reasoning chain (up to 74% diverged by CT4).

## Directory Structure

```
ct0_divergence_case_studies/
├── README.md                              # This file
├── 1_select_divergence_cases.py           # Selection script (12 diverse cases)
├── 2_generate_divergence_narratives.py    # Narrative generation script
└── results/
    ├── selected_divergence_cases.json     # Selected case metadata
    └── case_study_narratives/             # Generated narratives and visualizations
        ├── README.md                      # Index of all cases
        ├── case_01_problem_9.md           # Individual case narratives
        ├── case_01_problem_9_divergence.png
        ├── case_02_problem_10.md
        ├── case_02_problem_10_divergence.png
        └── ...                            # 12 cases total
```

## Scripts

### 1. Case Selection (`1_select_divergence_cases.py`)

**Purpose**: Select 12 diverse case studies from 100 analyzed problems.

**Selection Criteria**:
- Top 3 highest divergence cases
- Early divergence patterns (CT1 < 0.6 similarity)
- Late divergence patterns (CT1 ok, CT4 very low)
- Steep cascading (large negative slope)
- Degradation cases (baseline correct, blocked wrong)
- No-change cases (both correct despite divergence)
- Diverse remaining profiles

**Key Metrics Computed**:
- `ct1_similarity`: How similar CT1 is between baseline and blocked
- `ct4_similarity`: How similar CT4 is between baseline and blocked
- `total_divergence`: Mean divergence across all CT steps
- `divergence_slope`: Rate of divergence accumulation
- `pattern`: Classification (early_divergence, late_divergence, cascading, stable)

**Output**: `results/selected_divergence_cases.json`

**Usage**:
```bash
python 1_select_divergence_cases.py
```

### 2. Narrative Generation (`2_generate_divergence_narratives.py`)

**Purpose**: Generate detailed narratives and visualizations for each selected case.

**Generates**:
- **Per-case markdown narratives** with:
  - Problem details (question, answers, gold)
  - Divergence profile (overall metrics)
  - Step-by-step analysis (CT0-CT5)
  - Per-layer analysis (which layers diverge most)
  - Interpretation and key takeaways

- **4-panel visualizations** showing:
  1. Mean similarity across steps (with ±1 std band)
  2. Per-layer similarity heatmap
  3. L2 distance trajectory
  4. Divergence range (min/max across layers)

**Output**: `results/case_study_narratives/` (12 markdown files + 12 PNG visualizations + README)

**Usage**:
```bash
python 2_generate_divergence_narratives.py
```

## Case Study Structure

Each case study narrative includes:

1. **Problem Information**
   - Question text
   - Gold answer, baseline prediction, CT0-blocked prediction
   - Impact type (degradation, no_change, improvement)

2. **Divergence Profile**
   - Total divergence score
   - CT1 and CT4 similarities
   - Divergence slope
   - Pattern classification

3. **Step-by-Step Analysis** (CT0-CT5)
   - Mean cosine similarity
   - L2 distance
   - Interpretation (identical, similar, moderately/heavily diverged)
   - Most/least diverged layers
   - Layer variance (heterogeneity)

4. **Interpretation**
   - Pattern explanation (early, late, cascading)
   - Impact on answer (correct vs incorrect)

5. **Key Takeaways**
   - Cascading effects
   - Layer specialization
   - Critical layers identified

6. **Visualization**
   - 4-panel divergence trajectory plot

## Key Insights from Case Studies

### Cascading Divergence
- Divergence starts **immediately** at CT1 (typically 30% diverged)
- Accumulates through the chain (slope: -0.06 to -0.10 per step)
- CT4 shows maximum divergence (40-75% diverged)
- CT5 sometimes shows slight recovery

### Layer Heterogeneity
- Different layers diverge at different rates
- Early layers (0-4) remain relatively stable
- Mid layers (5-10) show moderate divergence
- Late layers (11-15) diverge most (related to output preparation)

### Impact Patterns
- **Degradation cases**: High divergence → incorrect answer
- **No-change cases**: High divergence but still correct (robustness)
- Late divergence more harmful than early divergence

## Data Source

These case studies use data from the CT0 mechanistic analysis experiment:

- **Experiment**: `10-29_llama_gsm8k_ct0_mechanistic_analysis.md`
- **Source Data**: `src/experiments/codi_attention_flow/ablation/results/attention_data/`
- **Hidden States**: `llama_problem_XXXX_baseline_hidden.h5` and `llama_problem_XXXX_ct0blocked_hidden.h5`
- **Metadata**: `llama_metadata_final.json` (100 problems)
- **Model**: LLaMA-3.2-1B-Instruct with CODI
- **Dataset**: GSM8K (math reasoning)

## Reproducibility

To regenerate case studies:

1. Ensure hidden state data exists in `src/experiments/codi_attention_flow/ablation/results/attention_data/`
2. Run case selection: `python 1_select_divergence_cases.py`
3. Generate narratives: `python 2_generate_divergence_narratives.py`

**Dependencies**: numpy, matplotlib, h5py, json, pathlib

## Related Documentation

- **Main Experiment**: `docs/experiments/10-29_llama_gsm8k_ct0_mechanistic_analysis.md`
- **Research Journal**: `docs/research_journal.md` (section: 2025-10-29)
- **Conversation**: `docs/conversations/2025-10/2025-10-29-1942-ct0-mechanistic-analysis-passive-hub.md`

## Next Steps

Potential follow-up analyses:
- Compare divergence patterns across problem types (arithmetic, word problems, multi-step)
- Analyze which semantic information is lost at each divergence step
- Test if intervening on specific diverged layers can recover performance
- Extend to other datasets (MMLU, CommonsenseQA)
