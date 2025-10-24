# Token Threshold & Criticality Experiments

**Objective**: Determine minimum token thresholds for reasoning and identify which continuous thought tokens are most critical in LLaMA CODI.

**Research Questions**:
1. **RQ1 (Threshold)**: What is the degradation curve as we corrupt 1→6 tokens? Does 4/6 corruption (67%) cause catastrophic failure?
2. **RQ2 (Criticality)**: Which token position(s) are most critical? Are any tokens significantly more important than others?
3. **RQ3 (Enhancement)**: Can enhancing specific tokens improve performance? Which positions are most enhancement-responsive?

**Key Innovation**: Data-driven identification of critical tokens using both corruption (removal) and enhancement (amplification) approaches.

---

## Experiment Design

### Experiment 1: Threshold Degradation Test

**Goal**: Map accuracy degradation as function of # corrupted tokens

**Corruption Strategy**:
- **1 token**: All 6 positions individually (reuse CCTA results)
- **2 tokens**: 3 strategic samples - [0,1], [2,3], [4,5]
- **3 tokens**: 3 samples - [0,1,2], [3,4,5], [0,2,4]
- **4 tokens**: 6 samples - skip each token individually (test which single token rescues)
- **5 tokens**: 6 samples - keep each token individually (test minimum viable token)
- **6 tokens**: Complete ablation

**Corruption Methods**:
- Zero ablation (set to zeros)
- Gaussian noise (σ=1.0)

**Total**: 25 combinations × 2 methods = **50 experiments per problem**

---

### Experiment 2: Token Enhancement Test

**Goal**: Test if amplifying specific tokens improves reasoning

**Enhancement Strategy**:
- **Positions**: All 6 tokens individually
- **Multipliers**: [0.5x, 1.0x (baseline), 1.5x, 2.0x, 3.0x]
- **Mode**: Standalone (no corruption, pure enhancement)

**Total**: 6 positions × 5 multipliers = **30 experiments per problem**

---

### Experiment 3: Combined Analysis

**Goal**: Synthesize corruption + enhancement to rank token criticality

**Analyses**:
1. Correlation between corruption vulnerability and enhancement responsiveness
2. Comprehensive token criticality ranking
3. Comparison to CCTA importance scores
4. Validation of paper claims (z₃/z₄ special)

---

## Quick Start: Pilot (10 Problems)

```bash
cd /home/paperspace/dev/CoT_Exploration

# 1. Run threshold test (~8 minutes)
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/token_threshold/scripts/1_run_threshold_test.py

# 2. Run enhancement test (~5 minutes)
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/token_threshold/scripts/3_run_enhancement_test.py

# 3. Analyze threshold results
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/token_threshold/scripts/2_analyze_threshold.py

# 4. Analyze enhancement results
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/token_threshold/scripts/4_analyze_enhancement.py

# 5. Combined analysis
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python src/experiments/token_threshold/scripts/5_combined_analysis.py
```

**Total runtime**: ~15 minutes (10 problems)

---

## File Structure

```
token_threshold/
├── README.md                           # This file
├── scripts/
│   ├── 1_run_threshold_test.py        # Threshold degradation experiment
│   ├── 2_analyze_threshold.py         # Threshold analysis & visualization
│   ├── 3_run_enhancement_test.py      # Token enhancement experiment
│   ├── 4_analyze_enhancement.py       # Enhancement analysis & visualization
│   └── 5_combined_analysis.py         # Combined token criticality ranking
├── results/
│   ├── test_dataset_10.json           # Test dataset (10 problems)
│   ├── threshold_test_10.json         # Threshold experiment results
│   ├── enhancement_test_10.json       # Enhancement experiment results
│   ├── threshold_analysis.json        # Threshold statistics
│   ├── enhancement_analysis.json      # Enhancement statistics
│   └── combined_analysis.json         # Combined criticality ranking
└── figures/
    ├── degradation_curve.{pdf,png}    # Accuracy vs # corrupted tokens
    ├── enhancement_heatmap.{pdf,png}  # Position × multiplier heatmap
    └── combined_ranking.{pdf,png}     # Token criticality comparison
```

---

## Expected Outputs

### Threshold Test
- **Degradation curve**: Shows how accuracy drops with increasing corruption
- **Critical threshold**: Identifies point of catastrophic failure
- **Token ranking**: Which single tokens are most critical (from 4/6 skip tests)

### Enhancement Test
- **Enhancement heatmap**: Position × multiplier → accuracy
- **Optimal multipliers**: Best amplification factor per position
- **Critical positions**: Which tokens are most enhancement-responsive

### Combined Analysis
- **Criticality ranking**: Data-driven ordering of all 6 tokens
- **Convergent validity**: Do corruption and enhancement agree?
- **Paper validation**: Are z₃/z₄ indeed special in LLaMA CODI?

---

## Success Criteria

1. ✅ Clear degradation curve showing accuracy vs # corrupted tokens
2. ✅ Statistical evidence for/against 67% threshold claim
3. ✅ Data-driven identification of which token(s) are most critical
4. ✅ Enhancement responsiveness mapped for all positions
5. ✅ Comprehensive token criticality ranking with statistical support
6. ✅ Publication-ready figures (PDF + PNG)

---

## Configuration

- **Model**: LLaMA-3.2-1B CODI (`~/codi_ckpt/llama_gsm8k/`)
- **Dataset**: 10 problems (pilot) from stratified GSM8K
- **Ablation layer**: Middle (L8)
- **WandB project**: `codi-token-threshold`

---

## Next Steps After Pilot

If pilot shows promising results:
1. Expand to 100 problems (~2 hours runtime)
2. Test combined scenarios (enhance middle + corrupt edges)
3. Stratify by difficulty (easy/medium/hard)
4. Compare to CCTA multi-method results

---

## References

- CODI paper: Continuous Chain-of-Thought via Self-Distillation
- CCTA experiment: `src/experiments/codi_attention_interp/`
- Related work: Activation patching, token ablation studies
