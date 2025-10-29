# CT0 Mechanistic Analysis: Passive Hub with Cascading Divergence

**Date**: October 29, 2025
**Model**: LLaMA-3.2-1B-Instruct CODI
**Dataset**: GSM8K test set (100 problems)
**Task**: Understand CT0's mechanistic role in CODI reasoning
**Status**: ✅ Complete

---

## Executive Summary

Through three complementary experiments (bidirectional attention blocking, attention flow analysis, and hidden state divergence tracking), we established that **CT0 functions as a passive information hub** that encodes question information which later CT tokens read from via attention. When CT0's attention output is blocked, reasoning diverges **immediately at CT1** and cascades through the entire chain, with hidden states showing only 38% similarity by CT4.

**Key Finding**: CT0 is not an active coordinator but a **question encoding cache** - it doesn't need to see the question after initial encoding, but other tokens critically depend on reading from it.

---

## Research Questions

1. **Is CT0 a passive hub or active coordinator?**
   - Answer: **Passive hub** - doesn't need question input after encoding, others read from it

2. **Who writes TO CT0 and who reads FROM CT0?**
   - Answer: **CT0 reads from question (100% attention)**, **CT1-CT5 read from CT0 (3-5% attention each)**

3. **When CT0 is blocked, do only final answers change or do intermediate CT tokens diverge?**
   - Answer: **Intermediate tokens diverge immediately** - CT1 shows 30% divergence, cascading to 62% by CT4

---

## Experiment 1: Bidirectional Attention Blocking

### Method

Tested 4 conditions on 100 GSM8K problems:
1. **Baseline**: Normal generation (control)
2. **Output blocked**: Other positions can't attend TO CT0 (mask CT0 with -10000)
3. **Input blocked**: CT0 can't attend TO other positions (blind CT0 during generation)
4. **Both blocked**: Full isolation

### Results

| Condition | Accuracy | Drop from Baseline | Interpretation |
|-----------|----------|-------------------|----------------|
| **Baseline** | 58% | 0% | Control |
| **Output blocked** | 42% | **-16%** | Strong impact |
| **Input blocked** | 58% | **0%** | No impact |
| **Both blocked** | 42% | -16% | Same as output-only |

### Key Findings

**Finding 1: CT0 is a PASSIVE HUB**
- Input/output ratio: 0.0 (input blocking has no effect)
- CT0 doesn't need to "see" the question during generation
- Others critically need to "see" CT0 (read from it)

**Finding 2: Additive interaction**
- Expected drop (input + output): 16%
- Actual drop: 16%
- Interaction effect: 0%
- No synergistic or compensatory effects

**Mechanistic Interpretation**:
```
CT0 functions as "storage/cache":
- Encodes question information at step 0
- Stores it in hidden state
- Later positions READ from CT0 via attention
- CT0 itself doesn't need to process question repeatedly
```

### Script
- Location: `src/experiments/codi_attention_flow/ablation/11_ct0_bidirectional_blocking.py`
- Runtime: ~3 minutes (100 problems × 4 conditions)
- Results: `src/experiments/codi_attention_flow/results/llama_ct0_bidirectional_blocking.json`

---

## Experiment 2: CT0 Writers and Readers Analysis

### Method

Analyzed baseline attention patterns (no blocking) to identify:
1. **Writers TO CT0**: Who attends to CT0?
2. **Readers FROM CT0**: What does CT0 attend to?

Loaded attention data for 100 problems across all 6 CT generation steps.

### Results

**CT0 Reads (Step 0 only)**:
- **Question tokens → CT0**: 100% attention weight (1.0000)
- **Other CT tokens → CT0**: 0% attention weight (0.0000)

**CT1-CT5 Write TO CT0 (Steps 1-5)**:

| CT Token | Attention TO CT0 | Std Dev |
|----------|------------------|---------|
| CT1 | 4.77% | ±1.07% |
| CT2 | 4.21% | ±1.00% |
| CT3 | 3.53% | ±0.79% |
| CT4 | 2.76% | ±0.82% |
| CT5 | 3.04% | ±0.93% |
| **Average** | **3.66%** | - |

### Key Findings

**Finding 1: CT0 fully attends to question during encoding**
- At step 0, CT0 allocates 100% attention to question tokens
- No attention to other CT tokens (they don't exist yet)
- This is the "encoding" phase

**Finding 2: Later CT tokens read from CT0**
- Each CT token (CT1-CT5) attends ~3-5% to CT0
- Decreasing pattern: CT1 (highest) → CT4 (lowest)
- Suggests early tokens need more question context

**Finding 3: Complete information flow**
```
Question Tokens (100% attention)
      ↓
     CT0  [Encodes question into hidden state]
      ↓
   CT1-CT5 [Read from CT0 via 3-5% attention]
      ↓
   Answer Generation
```

### Script
- Location: `src/experiments/codi_attention_flow/ablation/12_identify_ct0_writers_readers.py`
- Runtime: ~3 seconds (reads pre-computed attention data)
- Results: `src/experiments/codi_attention_flow/results/ct0_writers_readers_analysis.json`
- Visualizations: `ct0_writers_by_step.png`, `ct0_readers_step0.png`

---

## Experiment 3: Hidden State Divergence Analysis

### Method

For 100 problems, computed cosine similarity between baseline and CT0-blocked hidden states at each CT generation step (CT0-CT5) across all 16 layers.

**Metrics**:
- **Cosine similarity**: 1.0 = identical, 0.0 = orthogonal
- **L2 distance**: Euclidean distance between hidden vectors
- **Per-layer analysis**: Track which layers diverge most

### Results

**Aggregate Divergence Pattern**:

| CT Token | Cosine Similarity | Interpretation | L2 Distance |
|----------|-------------------|----------------|-------------|
| **CT0** | 1.0000 ± 0.0000 | Identical | 0.00 ± 0.00 |
| **CT1** | 0.6962 ± 0.0440 | **Diverged** | 18.19 ± 1.08 |
| **CT2** | 0.5352 ± 0.0866 | Diverged | 24.46 ± 2.67 |
| **CT3** | 0.4714 ± 0.0666 | Heavily diverged | 22.58 ± 1.99 |
| **CT4** | 0.3775 ± 0.0693 | **Most diverged** | 22.15 ± 2.33 |
| **CT5** | 0.5159 ± 0.0740 | Recovering | 22.02 ± 1.59 |

**Divergence Pattern**:
- **Trend slope**: -0.0983 per step (similarity decreases ~10% per step)
- **Type**: Accumulating divergence (cascading effect)
- **Onset**: Immediate (CT1 already at 0.70 similarity)

### Key Findings

**Finding 1: EARLY DIVERGENCE**
- CT1 similarity: 0.6962 (< 0.85 threshold)
- 30% divergence at the **FIRST step**
- Not a late-stage final-answer effect

**Finding 2: ACCUMULATING DIVERGENCE**
- Similarity drops ~9.8% per step
- CT1 (0.70) → CT2 (0.54) → CT3 (0.47) → CT4 (0.38)
- Cascading failure through reasoning chain

**Finding 3: Problems with degradation show more divergence**
| Impact Type | Avg Similarity (CT1-CT5) |
|-------------|--------------------------|
| Degradation (baseline ✓ → blocked ✗) | 0.5749 |
| No change (both correct) | 0.6049 |

**Mechanistic Interpretation**:
```
Why divergence cascades:

Step 0: Both generate identical CT0 (1.00 similarity)

Step 1:
  Baseline: CT1 reads from CT0 (4.77% attention)
  Blocked: CT1 can't read from CT0 → generates different CT1 (0.70 similarity)

Step 2:
  Baseline: CT2 reads from CT0 (4.21% attention) + original CT1
  Blocked: CT2 can't read from CT0 + reads from diverged CT1
          → generates even more different CT2 (0.54 similarity)

Steps 3-5: Divergence continues to accumulate
```

### Script
- Location: `src/experiments/codi_attention_flow/ablation/13_analyze_ct_hidden_state_divergence.py`
- Runtime: ~5 seconds (100 problems)
- Results: `src/experiments/codi_attention_flow/results/ct_hidden_state_divergence.json`
- Visualizations: `ct_hidden_state_divergence.png`, `ct_divergence_by_impact.png`

---

## Integrated Mechanistic Model

### Normal Operation (Baseline)

```
┌─────────────────────────────────────────────────┐
│ Question Tokens                                  │
│ "Janet's ducks lay 16 eggs per day..."          │
└────────────────┬────────────────────────────────┘
                 │ 100% attention
                 ↓
┌────────────────────────────────────────────────┐
│ CT0 Generation (Step 0)                        │
│ - Encodes question information                 │
│ - Stores numerical values, context            │
│ - Hidden state: [2048-dim vector]             │
└────────────────┬───────────────────────────────┘
                 │ Stored in sequence
                 ↓
┌────────────────────────────────────────────────┐
│ CT1 Generation (Step 1)                        │
│ - Reads from CT0 (4.77% attention)            │
│ - Begins calculation/reasoning                │
│ - Hidden state: [2048-dim vector]             │
└────────────────┬───────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────┐
│ CT2-CT5 Generation (Steps 2-5)                │
│ - Continue reading from CT0 (3-4% attention)  │
│ - Build on previous CT tokens                 │
│ - Execute multi-step reasoning                │
└────────────────┬───────────────────────────────┘
                 │
                 ↓
┌────────────────────────────────────────────────┐
│ Answer Generation                              │
│ - Uses accumulated reasoning from CT0-CT5     │
│ - Produces final answer: "18"                 │
└────────────────────────────────────────────────┘
```

### CT0 Attention Blocked

```
┌─────────────────────────────────────────────────┐
│ Question Tokens                                  │
│ "Janet's ducks lay 16 eggs per day..."          │
└────────────────┬────────────────────────────────┘
                 │ 100% attention
                 ↓
┌────────────────────────────────────────────────┐
│ CT0 Generation (Step 0)                        │
│ ✓ Identical to baseline (similarity = 1.00)   │
│ - Still encodes question                       │
│ - Hidden state exists in residual stream      │
└────────────────┬───────────────────────────────┘
                 │ Stored but ATTENTION BLOCKED
                 ↓
┌────────────────────────────────────────────────┐
│ CT1 Generation (Step 1)                        │
│ ✗ CAN'T read from CT0 via attention (blocked) │
│ ✗ DIVERGENCE BEGINS (similarity = 0.70)       │
│ - Uses residual stream only (partial info)    │
│ - Generates different CT1                     │
└────────────────┬───────────────────────────────┘
                 │ Diverged state
                 ↓
┌────────────────────────────────────────────────┐
│ CT2 Generation (Step 2)                        │
│ ✗ CAN'T read from CT0 (still blocked)         │
│ ✗ Reads from DIVERGED CT1                     │
│ ✗ DIVERGENCE ACCUMULATES (similarity = 0.54)  │
└────────────────┬───────────────────────────────┘
                 │ More diverged
                 ↓
┌────────────────────────────────────────────────┐
│ CT3-CT5 Generation (Steps 3-5)                │
│ ✗ Cascading divergence continues              │
│ ✗ CT4 most diverged (similarity = 0.38)       │
│ ✗ Each step amplifies the difference          │
└────────────────┬───────────────────────────────┘
                 │ Heavily diverged reasoning
                 ↓
┌────────────────────────────────────────────────┐
│ Answer Generation                              │
│ ✗ Based on diverged reasoning chain           │
│ ✗ Produces wrong answer: "15" (gold: "18")    │
│ → 16% accuracy drop overall                   │
└────────────────────────────────────────────────┘
```

---

## Key Insights

### 1. CT0 is a Passive Information Hub

**Evidence**:
- Input blocking: 0% accuracy drop
- Output blocking: -16% accuracy drop
- Input/output ratio: 0.0

**Interpretation**: CT0 doesn't actively coordinate by processing question repeatedly. It encodes question once, then serves as a cache that others read from.

### 2. Attention is Critical for Information Flow

**Evidence**:
- CT0's hidden state still exists in residual stream when attention blocked
- Yet CT1 immediately diverges (0.70 similarity)
- Divergence cascades through chain

**Interpretation**: Residual connections alone are insufficient. The attention mechanism is critical for CT tokens to "read" from CT0's encoded representation.

### 3. Reasoning Failures Cascade

**Evidence**:
- CT1: 30% diverged
- CT2: 46% diverged
- CT4: 62% diverged
- Trend: -9.8% similarity per step

**Interpretation**: CODI reasoning is sequential and dependent. Each CT token builds on previous ones, so early divergence amplifies through the chain.

### 4. Decreasing Attention Pattern

**Evidence**:
- CT1 → CT0: 4.77%
- CT2 → CT0: 4.21%
- CT3 → CT0: 3.53%
- CT4 → CT0: 2.76%

**Interpretation**: Early CT tokens need more direct access to question encoding (CT0), while later tokens may rely more on earlier CT tokens or have already extracted needed information.

---

## Implications for CODI Architecture

### Strengths

1. **Efficient encoding**: Question encoded once in CT0, reused throughout
2. **Modular reasoning**: Each CT token can focus on specific sub-task while accessing shared context
3. **Compression**: 6 tokens encode entire reasoning process

### Vulnerabilities

1. **Single point of failure**: CT0 is critical - blocking causes 16% accuracy drop
2. **Cascading errors**: Early divergence amplifies through chain
3. **Fragile information flow**: Attention mechanism is bottleneck

### Design Implications

1. **Redundancy needed**: Could benefit from multiple "hub" positions
2. **Attention robustness**: Critical paths need protection/redundancy
3. **Error correction**: Later CT tokens should detect/correct early divergence

---

## Comparison to Prior Work

### Previous CT0 Case Studies (10-29)

**Previous focus**: Final answer errors when CT0 blocked
- Identified CT0 causes +13.4% calculation errors
- Showed 10 case studies with error types (precision, sequencing, aggregation)
- Established CT0 as "calculation coordination hub"

**This work adds**: Mechanistic understanding of HOW and WHY
- CT0 is passive hub, not active coordinator
- Complete information flow: Question → CT0 → CT1-CT5 → Answer
- Hidden states diverge immediately and cascade
- Quantified: 30% divergence at CT1, 62% by CT4

**Complementary findings**: Both confirm CT0's critical role, this work explains the mechanism

---

## Limitations

1. **Attention blocking only**: Didn't test hidden state zeroing (stronger intervention)
2. **GSM8K specific**: May not generalize to other reasoning types
3. **100 problems**: Smaller than full test set (1,319 problems)
4. **Layer-agnostic analysis**: Computed average across layers, didn't identify which layers diverge most
5. **No interpretation of CT semantics**: Can't decode what each CT token "means"

---

## Future Work

### Immediate (High Priority)

1. **Qualitative case studies**: Show step-by-step divergence with specific examples
2. **Layer-specific analysis**: Which layers diverge most? Early vs late layers?
3. **Hidden state zeroing**: Test stronger intervention (zero CT0 hidden state, not just attention)

### Medium Priority

4. **CT1-CT5 interdependence**: Do CT2-CT5 also read from CT1, or only from CT0?
5. **Cross-dataset validation**: Test on MMLU, CommonsenseQA
6. **Multi-position blocking**: What happens if CT0+CT1 both blocked?

### Long-term

7. **Causal tracing**: Restore specific CT0 components, measure recovery
8. **Semantic decoding**: Probe what information each CT token encodes
9. **Architecture improvements**: Design more robust CODI variants

---

## Reproducibility

### Data Sources
- Attention data: `src/experiments/codi_attention_flow/ablation/results/attention_data/`
- Hidden states: `src/experiments/codi_attention_flow/ablation/results/attention_data/`
- Metadata: `llama_metadata_final.json`

### Scripts
1. `11_ct0_bidirectional_blocking.py` - Bidirectional blocking experiment
2. `12_identify_ct0_writers_readers.py` - Attention flow analysis
3. `13_analyze_ct_hidden_state_divergence.py` - Hidden state divergence

### Runtime
- Total: ~10 minutes (bidirectional blocking takes longest)
- Can run in parallel if needed

### Dependencies
- h5py (for reading attention/hidden state data)
- scipy (for cosine similarity)
- matplotlib, seaborn (for visualizations)

---

## Conclusion

Through three complementary experiments, we established that **CT0 functions as a passive information hub** in CODI's reasoning architecture. It encodes question information at step 0 (allocating 100% attention to question tokens), then serves as a cache that later CT tokens read from via attention (3-5% attention each). When this attention pathway is blocked, reasoning diverges immediately at CT1 (30% divergence) and cascades through the chain (62% divergence by CT4), demonstrating that:

1. **CT0 is passive, not active** - doesn't need to re-process question, just stores encoded info
2. **Attention is critical** - residual connections alone insufficient for information flow
3. **Reasoning is sequential and fragile** - early errors cascade through dependent steps

This provides a mechanistic explanation for CT0's critical role in CODI reasoning and highlights both the efficiency (centralized encoding) and vulnerability (single point of failure) of the architecture.

---

## Acknowledgments

Analysis built on prior work:
- CT0 case studies (10-29): Established CT0's role in calculation coordination
- Attention masking experiments (earlier): Identified CT0 as most critical position
- CODI paper: Original architecture and training methodology
