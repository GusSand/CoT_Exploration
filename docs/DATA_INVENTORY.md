# Data Inventory - CoT Exploration Project

**Last Updated**: 2025-10-23

This document provides a complete breakdown of all datasets in the project, organized by experiment type and model.

**Status Indicators**:
- ✅ File exists and is ready to use
- ⚠️ File needs to be generated (see notes for generation commands)

---

## Quick Reference

| Dataset | Purpose | Size | Models | Location |
|---------|---------|------|--------|----------|
| **All 532 Pairs** | Base dataset | 532 pairs | Both | [`problem_pairs_gpt4_answers.json`](../src/experiments/activation_patching/problem_pairs_gpt4_answers.json) |
| **Matched Pairs** | Both models correct | 101 pairs | Both | [`data/problem_pairs_matched.json`](../src/experiments/activation_patching/data/problem_pairs_matched.json) |
| **CoT-Dependent** | Fair comparison | 43 pairs | Both | [`data/problem_pairs_cot_dependent.json`](../src/experiments/activation_patching/data/problem_pairs_cot_dependent.json) |
| **LLaMA CoT All** | All LLaMA CoT problems | 285 problems | LLaMA | [`data/llama_cot_all.json`](../src/experiments/activation_patching/data/llama_cot_all.json) |
| **LLaMA CoT Stratified (450)** | Original GSM8K problems stratified by difficulty | 450 problems | LLaMA | [`data/llama_cot_original_stratified_final.json`](../src/experiments/activation_patching/data/llama_cot_original_stratified_final.json) |
| **LLaMA CoT Stratified (1000)** | Expanded stratified dataset with strong statistical power | **1000 problems** | LLaMA | [`data/llama_cot_original_stratified_1000.json`](../src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json) |
| **GPT-2 Steering** | Activation steering | 344 train / 86 test | GPT-2 | [`results/steering_dataset_gpt2.json`](../src/experiments/activation_patching/results/steering_dataset_gpt2.json) |
| **LLaMA Steering (Full)** | Activation steering | 425 train / 107 test | LLaMA | [`results/steering_dataset_llama_full.json`](../src/experiments/activation_patching/results/steering_dataset_llama_full.json) |

---

## 1. Problem Pair Datasets (Activation Patching)

### 1.1 Base Dataset: All 532 Pairs
**File**: [`src/experiments/activation_patching/problem_pairs_gpt4_answers.json`](../src/experiments/activation_patching/problem_pairs_gpt4_answers.json)

**Purpose**: Complete set of matched problem pairs (clean + corrupted versions) with GPT-4 calculated answers

**Size**: 532 pairs

**Structure**:
```json
[
  {
    "clean": {
      "question": "Problem with original numbers",
      "answer": 42,
      "cot": "Step-by-step reasoning"
    },
    "corrupted": {
      "question": "Problem with slightly changed numbers",
      "answer": 45,
      "cot": "Step-by-step reasoning"
    },
    "clean_answer": 42,
    "corrupted_answer": 45,
    "difficulty_metrics": {...}
  }
]
```

**Key Stats**:
- Source: GSM8K dataset
- All answers calculated by GPT-4
- Used as starting point for filtering pipelines

**Used By**:
- CoT necessity testing (both models)
- Filtering to matched pairs
- Difficulty analysis

---

### 1.2 Matched Pairs (Both Models Correct)
**File**: [`src/experiments/activation_patching/data/problem_pairs_matched.json`](../src/experiments/activation_patching/data/problem_pairs_matched.json)

**Purpose**: Pairs where BOTH GPT-2 and LLaMA answer BOTH clean and corrupted correctly

**Size**: 101 pairs (19% of 532)

**Filtering Pipeline**:
```
532 All Pairs
  ↓
101 Matched Pairs (both models both-correct)
  ↓
43 CoT-Dependent Pairs (both models need CoT)
```

**Used By**:
- CoT necessity testing
- Further filtering to CoT-dependent pairs

---

### 1.3 CoT-Dependent Pairs (Fair Comparison)
**File**: [`src/experiments/activation_patching/data/problem_pairs_cot_dependent.json`](../src/experiments/activation_patching/data/problem_pairs_cot_dependent.json)

**Purpose**: Pairs where BOTH models demonstrably NEED continuous thought tokens to solve correctly

**Size**: 43 pairs (8% of 532, 43% of matched)

**Filtering Criteria**:
1. Both models answer both variants correctly (matched)
2. LLaMA needs CoT for at least one variant (clean or corrupted)
3. GPT-2 needs CoT for at least one variant (clean or corrupted)

**Key Stats**:
- Difficulty distribution: 19 easy (≤2 steps), 19 medium (3 steps), 5 hard (≥4 steps)
- Mean reasoning steps: 2.6
- Ensures fair cross-model comparison (no "direct computation" bypass)

**Used By**:
- N-token ablation experiments
- Fair LLaMA vs GPT-2 comparisons
- Analysis by difficulty strata

**Documentation**: [`docs/experiments/cot_necessity_and_ablation_2025-10-21.md`](experiments/cot_necessity_and_ablation_2025-10-21.md)

---

### 1.4 LLaMA CoT All Problems
**File**: [`src/experiments/activation_patching/data/llama_cot_all.json`](../src/experiments/activation_patching/data/llama_cot_all.json)

**Purpose**: Complete list of all individual problems where LLaMA needs CoT to solve correctly (regardless of whether it got them right)

**Size**: 285 problems from 219 unique pairs

**Structure**:
```json
{
  "pair_id": 0,
  "variant": "clean",
  "question": "Problem text...",
  "answer": 18,
  "reasoning_steps": 2,
  "baseline_correct": true,
  "ablated_correct": false,
  "needs_cot": true
}
```

**Key Stats**:
- Total problems: 285 individual problems from 219 unique pairs
- Clean variants: 132 problems (46.3%)
- Corrupted variants: 153 problems (53.7%)

**Pair Type Breakdown**:
- 66 pairs (30%): BOTH clean AND corrupted need CoT → contributes 132 problems
- 66 pairs (30%): ONLY clean needs CoT → contributes 66 problems
- 87 pairs (40%): ONLY corrupted needs CoT → contributes 87 problems
- **Why 285 vs 219?** Individual problems are stored separately. Some pairs contribute 1 problem, others contribute 2.

**Reasoning Step Distribution**:
- 1 step: 10 problems (3.5%)
- 2 steps: 115 problems (40.4%)
- 3 steps: 81 problems (28.4%)
- 4 steps: 47 problems (16.5%)
- 5+ steps: 32 problems (11.2%)

**Fields**:
- `pair_id`: Original pair identifier
- `variant`: "clean" or "corrupted"
- `question`: The problem text
- `answer`: Correct answer
- `reasoning_steps`: Number of computational steps (based on calculation blocks in CoT)
- `baseline_correct`: Whether LLaMA got it right WITH CoT tokens
- `ablated_correct`: Whether LLaMA got it right WITHOUT CoT tokens
- `needs_cot`: Always true (filter criterion)

**Used By**:
- Analyzing all problems that require latent reasoning for LLaMA
- Difficulty stratification by reasoning steps
- Understanding what types of problems need CoT vs direct computation

**Documentation**: Created 2025-10-22

---

### 1.5 LLaMA CoT Stratified Dataset (Original GSM8K Problems)
**File**: [`src/experiments/activation_patching/data/llama_cot_original_stratified_final.json`](../src/experiments/activation_patching/data/llama_cot_original_stratified_final.json)

**Purpose**: Stratified dataset of **original GSM8K problems** (not generated pairs) where LLaMA needs CoT tokens, with exact target distribution across difficulty levels

**Size**: 450 problems

**Structure**:
```json
{
  "gsm8k_id": "test_86",
  "question": "Problem text...",
  "answer": 22,
  "full_solution": "Step by step... #### 22",
  "source": "test",
  "baseline_correct": true,
  "ablated_correct": false,
  "needs_cot": true,
  "reasoning_steps": 3,
  "difficulty": "3-step",
  "is_existing": false
}
```

**Target Distribution (ALL TARGETS MET)**:

| Difficulty | Target | Existing | New | Total |
|-----------|--------|----------|-----|-------|
| 2-step    | ≥150   | 42       | 108 | **150** |
| 3-step    | ≥150   | 37       | 113 | **150** |
| 4-step    | ≥100   | 32       | 68  | **100** |
| 5+step    | ≥50    | 21       | 29  | **50**  |
| **TOTAL** |        | 132      | 318 | **450** |

**Key Stats**:
- Source: Original GSM8K problems (test + train sets)
- Testing: 7,500 problems tested, 3,080 CoT-needed found (41.1% rate)
- Stratification: Exact targets achieved for all difficulty buckets
- Existing problems: 132 from previous work marked with `is_existing=True`
- New problems: 318 from expansion marked with `is_existing=false`
- Buffer: 2,762 additional CoT-needed problems available in checkpoint

**Difficulty Classification**:
- Based on counting `<<calculation>>` blocks in GSM8K solutions
- 2-step: 1-2 calculation blocks (easy)
- 3-step: 3 calculation blocks (medium)
- 4-step: 4 calculation blocks (hard)
- 5+step: 5+ calculation blocks (very hard)

**Fields**:
- `gsm8k_id`: Original GSM8K identifier (test_N or train_N or existing_N)
- `question`: Problem text from GSM8K
- `answer`: Correct answer extracted from `#### N` format
- `full_solution`: Complete GSM8K solution with calculation blocks
- `source`: "test" or "train" (GSM8K split)
- `baseline_correct`: LLaMA correct WITH 6 CoT tokens (always true)
- `ablated_correct`: LLaMA correct WITHOUT CoT tokens (always false)
- `needs_cot`: Always true (filter criterion)
- `reasoning_steps`: Count of calculation blocks
- `difficulty`: Classified bucket (2-step, 3-step, 4-step, 5+step)
- `is_existing`: True if from original 132, false if newly added

**Advantages Over Pair-Based Dataset**:
1. **Scale**: 450 vs 285 problems (58% increase)
2. **Distribution Control**: Exact stratification by difficulty
3. **Statistical Power**: n≥50 per difficulty bucket for robust analysis
4. **Original Problems**: Uses GSM8K directly, no generation artifacts
5. **Future Expansion**: 2,762 additional problems available for scaling to 1,500+

**Used By**:
- Difficulty-stratified ablation experiments
- N-token analysis by reasoning complexity
- Statistical power analysis with adequate sample sizes
- Cross-difficulty performance comparisons
- Robustness testing across problem types

**Generation Pipeline**:
1. Test CoT necessity on 7,500 GSM8K problems (baseline vs ablated)
2. Calculate difficulty from solution calculation blocks
3. Load existing 132 problems from `llama_cot_all.json`
4. Stratify by difficulty and sample to exact targets
5. Validate all targets met

**Checkpoint File**: [`data/gsm8k_expansion_checkpoint.json`](../src/experiments/activation_patching/data/gsm8k_expansion_checkpoint.json)
- Contains all 7,500 tested problems
- 3,080 CoT-needed problems total
- Available for future expansion or alternative stratifications

**Documentation**:
- Detailed report: [`docs/experiments/gsm8k_expansion_2025-10-23.md`](experiments/gsm8k_expansion_2025-10-23.md)
- Usage guide: [`src/experiments/activation_patching/GSM8K_EXPANSION_GUIDE.md`](../src/experiments/activation_patching/GSM8K_EXPANSION_GUIDE.md)
- Research journal: [`docs/research_journal.md`](research_journal.md) (2025-10-23 entry)

**Created**: 2025-10-23

**Performance Metrics**:
- Testing time: 94 minutes for 7,500 problems
- Rate: 1.33 problems/second
- 25× faster than conservative estimate

---

### 1.6 LLaMA CoT Stratified Dataset - 1000 Problems (RECOMMENDED)
**File**: [`src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json`](../src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json)

**Purpose**: **Expanded stratified dataset with strong statistical power** - 1,000 original GSM8K problems where LLaMA needs CoT tokens, with perfectly balanced distribution across all difficulty levels

**Size**: 1,000 problems

**Structure**: Same as section 1.5 above

**Distribution (PERFECTLY BALANCED FOR STRONG STATISTICAL POWER)**:

| Difficulty | Target | Existing | New (from 450) | Additional (from checkpoint) | Total |
|-----------|--------|----------|----------------|------------------------------|-------|
| 2-step    | 250    | 42       | 108            | 100                          | **250** |
| 3-step    | 250    | 37       | 113            | 100                          | **250** |
| 4-step    | 250    | 32       | 68             | 150                          | **250** |
| 5+step    | 250    | 21       | 29             | 200                          | **250** |
| **TOTAL** |        | 132      | 318            | 550                          | **1000** |

**Key Stats**:
- Source: Original GSM8K problems (test + train sets)
- Perfectly balanced: 250 problems per difficulty level
- Strong statistical power: Enables robust subgroup analyses
- All 132 existing problems preserved with `is_existing=True`
- 318 problems from initial expansion (450 dataset)
- 550 additional problems sampled from checkpoint
- Buffer remaining: 2,080 additional CoT-needed problems still available

**Advantages Over 450-Problem Dataset**:
1. **Balanced Distribution**: Equal samples across all difficulty levels (250 each)
2. **Strong Statistical Power**: n=250 per group enables:
   - Detection of small effect sizes
   - Robust subgroup analyses
   - Cross-difficulty comparisons with high confidence
3. **Flexibility**: Equal group sizes simplify statistical analyses
4. **Scale**: 122% larger than 450-problem dataset

**Statistical Power**:
- **n=250 per difficulty** enables detection of:
  - Small effect sizes (Cohen's d ≥ 0.25)
  - Interactions between difficulty and treatment
  - Meaningful subgroup differences
- Power analysis: >90% power for medium effects (d=0.5) at α=0.05

**Used By**:
- Difficulty-stratified ablation experiments requiring strong power
- N-token analysis across complexity levels
- Statistical modeling with difficulty as covariate
- Cross-difficulty performance comparisons
- Subgroup analyses and interaction effects

**Generation Process**:
1. Started with 450-problem dataset (section 1.5)
2. Loaded checkpoint with 3,080 CoT-needed problems
3. Filtered to available problems not in 450 dataset
4. Random sampled to reach 250 per difficulty level
5. Verified perfect balance

**Files**:
- Primary: `data/llama_cot_original_stratified_1000.json` (1,000 problems)
- Source: `data/llama_cot_original_stratified_final.json` (450 problems)
- Checkpoint: `data/gsm8k_expansion_checkpoint.json` (7,500 tested, 3,080 CoT-needed)

**Documentation**:
- Detailed report: [`docs/experiments/gsm8k_expansion_2025-10-23.md`](experiments/gsm8k_expansion_2025-10-23.md)
- Usage guide: [`src/experiments/activation_patching/GSM8K_EXPANSION_GUIDE.md`](../src/experiments/activation_patching/GSM8K_EXPANSION_GUIDE.md)
- Research journal: [`docs/research_journal.md`](research_journal.md) (2025-10-23 entry)

**Created**: 2025-10-23 (expanded from 450 to 1,000)

**Recommendation**: **Use this dataset as default** for all difficulty-stratified analyses requiring robust statistical conclusions.

---

## 2. CoT Necessity Test Results

### 2.1 LLaMA CoT Necessity (101 Matched Pairs)
**File**: [`src/experiments/activation_patching/results/cot_necessity_llama_simple.json`](../src/experiments/activation_patching/results/cot_necessity_llama_simple.json)

**Purpose**: Test which problems LLaMA needs continuous thought tokens to solve

**Method**: Replace all 6 continuous tokens with zeros, compare baseline vs ablated predictions

**Size**: 101 pairs tested

**Key Results**:
```json
{
  "model_name": "llama",
  "num_pairs_tested": 101,
  "statistics": {
    "needs_cot_clean": 28,        // 27.7%
    "needs_cot_corrupted": 38,    // 37.6%
    "needs_cot_either": 44,       // 43.6% ← Used for filtering
    "needs_cot_both": 22          // 21.8%
  }
}
```

**Critical Finding**: LLaMA only needs CoT for 44% of problems (can solve 56% via direct computation)

**Used By**:
- Filtering to CoT-dependent pairs (43 pairs)
- Understanding LLaMA's reasoning pathways
- Difficulty analysis

---

### 2.2 GPT-2 CoT Necessity (101 Matched Pairs)
**File**: [`src/experiments/activation_patching/results/cot_necessity_gpt2_simple.json`](../src/experiments/activation_patching/results/cot_necessity_gpt2_simple.json)

**Purpose**: Test which problems GPT-2 needs continuous thought tokens to solve

**Method**: Replace all 6 continuous tokens with zeros, compare baseline vs ablated predictions

**Size**: 101 pairs tested

**Key Results**:
```json
{
  "model_name": "gpt2",
  "num_pairs_tested": 101,
  "statistics": {
    "needs_cot_clean": 101,       // 100%
    "needs_cot_corrupted": 101,   // 100%
    "needs_cot_either": 101,      // 100% ← All problems need CoT
    "needs_cot_both": 101         // 100%
  }
}
```

**Critical Finding**: GPT-2 needs CoT for 100% of problems (never uses direct computation)

**Insight**: This 100% vs 44% gap validates the need for CoT-dependent filtering to ensure fair comparison

---

### 2.3 LLaMA CoT Necessity (All 532 Pairs)
**File**: [`src/experiments/activation_patching/results/llama_cot_necessity_532.json`](../src/experiments/activation_patching/results/llama_cot_necessity_532.json)

**Purpose**: Test CoT necessity on entire dataset (not just matched pairs)

**Size**: 532 pairs tested

**Key Results**:
- Needs CoT (either clean or corrupted): **229 pairs (43.0%)**
- Can skip CoT: **303 pairs (57.0%)**

**Insight**: Confirms 43% CoT dependency holds across full dataset, not just matched pairs

---

## 3. Activation Steering Datasets

### 3.1 GPT-2 Steering Dataset
**File**: [`src/experiments/activation_patching/results/steering_dataset_gpt2.json`](../src/experiments/activation_patching/results/steering_dataset_gpt2.json)

**Purpose**: Balanced dataset for computing steering directions (correct_mean - wrong_mean)

**Size**:
- **Train**: 344 samples (172 correct + 172 wrong)
- **Test**: 86 samples (43 correct + 43 wrong)
- **Total**: 430 samples

**Structure**:
```json
{
  "model": "gpt2",
  "train_correct": [/* 172 samples */],
  "train_wrong": [/* 172 samples */],
  "test_correct": [/* 43 samples */],
  "test_wrong": [/* 43 samples */]
}
```

**Used By**:
- Computing steering direction (train set)
- Evaluating steering effectiveness (test set)
- Random direction control validation

**Results**:
- Suppression: -12.8 pts @ α=-3.0 ✅
- Amplification: +2.3 pts @ α=+1.0 (limited by ceiling)
- Random control: -6.7 pts @ α=-3.0 (validates steering is meaningful)

**Documentation**: [`docs/experiments/activation_steering_gpt2_2025-10-21.md`](experiments/activation_steering_gpt2_2025-10-21.md)

---

### 3.2 LLaMA Steering Dataset (Full - 532 Pairs)
**File**: [`src/experiments/activation_patching/results/steering_dataset_llama_full.json`](../src/experiments/activation_patching/results/steering_dataset_llama_full.json)

**Purpose**: Maximum dataset for LLaMA steering (using all 229 CoT-dependent pairs from 532)

**Size**:
- **Train**: 425 samples (230 correct + 195 wrong) - 80% split
- **Test**: 107 samples (58 correct + 49 wrong) - 20% split
- **Total**: 532 samples (from CoT-dependent subset)

**Structure**:
```json
{
  "train_correct": [/* 230 samples */],
  "train_wrong": [/* 195 samples */],
  "test_correct": [/* 58 samples */],
  "test_wrong": [/* 49 samples */]
}
```

**Key Difference from GPT-2**:
- 26.6× more training data (425 vs 16 in pilot)
- 17.8× more test data (107 vs 6 in pilot)
- Unbalanced by design (reflects LLaMA's 54% baseline accuracy)

**Used By**:
- Computing steering direction (train set)
- Testing steering at 3 layers (early/middle/late)
- Testing 13 alpha values (-3.0 to +3.0)

**Results**:
- **Complete invariance**: 0.0 pts change at early/middle layers across ALL alphas
- Late layer: -0.9 pts maximum effect
- Conclusion: LLaMA is fundamentally immune to linear steering

**Documentation**: [`src/experiments/activation_patching/docs/experiments/activation_steering_llama_full_2025-10-21.md`](../src/experiments/activation_patching/docs/experiments/activation_steering_llama_full_2025-10-21.md)

---

### 3.3 LLaMA Steering Dataset (Pilot - Small)
**File**: [`src/experiments/activation_patching/results/steering_dataset_llama.json`](../src/experiments/activation_patching/results/steering_dataset_llama.json)

**Purpose**: Original small pilot steering dataset (before full 532 expansion)

**Size**: ~22 samples (from 101 matched pairs)
- Train: 16 samples (8 correct + 8 wrong)
- Test: 6 samples (3 correct + 3 wrong)

**Status**: Archived/superseded by full dataset

**Note**: This was the pilot that showed no steering effect, leading to the hypothesis that more data was needed. The full dataset (532 pairs) confirmed steering doesn't work on LLaMA regardless of dataset size.

---

## 4. Dataset Relationships

### 4.1 The Filtering Pipeline
```
┌─────────────────────────────────────────────────────────┐
│  All 532 Pairs (GPT-4 calculated)                       │
│  problem_pairs_gpt4_answers.json                        │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ├─► LLaMA Necessity Test (532 pairs)
                   │   → 229 need CoT (43.0%)
                   │   → 303 can skip CoT (57.0%)
                   │   ├─► LLaMA Steering Dataset (Full)
                   │       425 train / 107 test
                   │
                   ↓
┌──────────────────────────────────────────────────────────┐
│  Matched Pairs (both models both-correct)                │
│  data/problem_pairs_matched.json                         │
│  101 pairs (19% of 532)                                  │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ├─► LLaMA Necessity Test (101 pairs)
                   │   → 44 need CoT (43.6%)
                   │   → 57 can skip CoT (56.4%)
                   │
                   ├─► GPT-2 Necessity Test (101 pairs)
                   │   → 101 need CoT (100%)
                   │   → 0 can skip CoT (0%)
                   │
                   ↓
┌──────────────────────────────────────────────────────────┐
│  CoT-Dependent Pairs (both models need CoT)              │
│  data/problem_pairs_cot_dependent.json                   │
│  43 pairs (8% of 532, 43% of matched)                    │
└──────────────────┬───────────────────────────────────────┘
                   │
                   └─► N-Token Ablation Experiments
                       ├─ LLaMA: 1, 2, 4 tokens
                       └─ GPT-2: 1, 2, 4 tokens
```

### 4.2 Dataset Usage by Experiment Type

| Experiment | Dataset Used | Size | Models |
|------------|-------------|------|--------|
| **CoT Necessity Testing** | All 532 Pairs | 532 | LLaMA |
| **CoT Necessity Testing** | Matched Pairs | 101 | Both |
| **N-Token Ablation** | CoT-Dependent | 43 | Both |
| **Activation Steering** | GPT-2 Steering | 344/86 | GPT-2 |
| **Activation Steering** | LLaMA Steering Full | 425/107 | LLaMA |
| **Difficulty Analysis** | Matched + Necessity | 96 | LLaMA |
| **LLaMA CoT Analysis** | LLaMA CoT All | 285 | LLaMA |

---

## 5. Model-Specific Breakdowns

### 5.1 GPT-2 (117M parameters)
**CoT Dependency**: 100% (always needs continuous thought tokens)

**Datasets**:
1. **Matched Pairs** (101) - Both-correct baseline
2. **CoT-Dependent** (43) - Fair comparison subset
3. **Steering Dataset** (344 train / 86 test) - Balanced correct/wrong

**Key Findings**:
- Cannot solve ANY problems via direct computation
- Needs all 6 continuous tokens for reasoning
- N-token ablation: Linear improvement (1→2→4 tokens)
- Steering: Suppression works (-12.8 pts), amplification limited (+2.3 pts)

### 5.2 LLaMA-3.2-1B (1B parameters)
**CoT Dependency**: 43% (only needs CoT for hard problems)

**Datasets**:
1. **All 532 Pairs** - Complete test set
2. **Matched Pairs** (101) - Both-correct baseline
3. **CoT-Dependent** (43) - Fair comparison subset
4. **LLaMA CoT All** (285) - All problems requiring CoT reasoning
5. **Steering Dataset Full** (425 train / 107 test) - Maximum data

**Key Findings**:
- Can solve 57% of problems via direct computation (bypass latent reasoning)
- Strong phase transition at 2-3 reasoning steps
- N-token ablation: Non-linear jump (30% → 70% at 4 tokens)
- Steering: Complete invariance (0.0 pts across all alphas)

---

## 6. File Locations

### Main Data Directory
```
src/experiments/activation_patching/
├── problem_pairs_gpt4_answers.json         # Base dataset (532 pairs) ✅
├── validation_results_llama_gpt4_532.json  # LLaMA validation results ✅
├── validation_results_gpt2_gpt4_532.json   # GPT-2 validation results ✅
├── data/
│   ├── problem_pairs_matched.json          # 101 matched pairs ✅
│   ├── problem_pairs_cot_dependent.json    # 43 CoT-dependent pairs ✅
│   └── llama_cot_all.json                  # 285 LLaMA CoT problems ✅
└── results/ (⚠️ Directory needs to be created)
    ├── cot_necessity_llama_simple.json     # LLaMA necessity (101) - To generate
    ├── cot_necessity_gpt2_simple.json      # GPT-2 necessity (101) - To generate
    ├── llama_cot_necessity_532.json        # LLaMA necessity (all 532) - To generate
    ├── steering_dataset_gpt2.json          # GPT-2 steering data - To generate
    └── steering_dataset_llama_full.json    # LLaMA steering data (full) - To generate
```

---

## 7. Critical Insights

### 7.1 Why Filter to CoT-Dependent Pairs?
**Problem**: LLaMA solves 57% of problems via direct computation (no latent reasoning), while GPT-2 always uses latent reasoning.

**Solution**: Filter to 43 pairs where BOTH models demonstrably need CoT tokens.

**Impact**: Ensures fair "apples to apples" comparison of latent reasoning efficiency.

### 7.2 Why the Size Difference in Steering Datasets?
- **GPT-2**: 344 train / 86 test (from matched pairs only)
- **LLaMA**: 425 train / 107 test (from all 532 CoT-dependent pairs)

**Reason**: Initial hypothesis was that small dataset (16 train) caused steering failure. Full dataset (425 train) proved steering fails regardless of size - it's a fundamental property of LLaMA.

### 7.3 Dataset Quality Hierarchy
```
Highest Quality (Most Filtered):
  CoT-Dependent (43 pairs)
    ↑ Both models need CoT
    ↑ Both models both-correct
    ↑ GPT-4 calculated answers

Medium Quality (Moderate Filtering):
  Matched Pairs (101 pairs)
    ↑ Both models both-correct
    ↑ GPT-4 calculated answers

Base Quality (No Filtering):
  All 532 Pairs
    ↑ GPT-4 calculated answers
```

---

## 8. Usage Guidelines

### When to Use Each Dataset

**Use All 532 Pairs when**:
- Testing single-model performance (e.g., LLaMA necessity across full dataset)
- Analyzing difficulty distributions
- Maximum statistical power for single-model analysis

**Use Matched Pairs (101) when**:
- Both models solve problems correctly (but may use different pathways)
- Initial cross-model comparison
- CoT necessity testing

**Use CoT-Dependent (43) when**:
- Fair cross-model comparison (both use latent reasoning)
- N-token ablation experiments
- Testing latent reasoning efficiency

**Use Steering Datasets when**:
- Activation steering experiments
- Computing reasoning directions (correct_mean - wrong_mean)
- Testing causal interventions

---

## 9. Data Integrity Checks

### Verification Commands
```bash
# Count all datasets
cd src/experiments/activation_patching

# Base datasets (✅ These files exist)
jq '. | length' problem_pairs_gpt4_answers.json           # Should be 532
jq '. | length' data/problem_pairs_matched.json           # Should be 101
jq '. | length' data/problem_pairs_cot_dependent.json     # Should be 43
jq '. | length' data/llama_cot_all.json                   # Should be 285

# Validation results (✅ These files exist)
jq '.num_pairs' validation_results_llama_gpt4_532.json    # Should be 532
jq '.num_pairs' validation_results_gpt2_gpt4_532.json     # Should be 532

# CoT necessity results (⚠️ Files need to be generated first)
jq '.num_pairs_tested' results/cot_necessity_llama_simple.json  # Should be 101
jq '.num_pairs_tested' results/cot_necessity_gpt2_simple.json   # Should be 101
jq '. | length' results/llama_cot_necessity_532.json            # Should be 532

# Steering datasets (⚠️ Files need to be generated first)
jq '.train_correct | length' results/steering_dataset_gpt2.json       # Should be 172
jq '.train_correct | length' results/steering_dataset_llama_full.json # Should be 230
```

### Expected Relationships
- Matched ⊂ All: `101 ≤ 532` ✓
- CoT-Dependent ⊂ Matched: `43 ≤ 101` ✓
- LLaMA CoT-dependent %: `44/101 = 43.6%` ✓
- GPT-2 CoT-dependent %: `101/101 = 100%` ✓

---

## 10. Generating Missing Files

Several result files are referenced in this document but need to be generated. Here's how to create them:

### Create the Results Directory
```bash
cd src/experiments/activation_patching
mkdir -p results
```

### Generate CoT Necessity Files

**LLaMA CoT Necessity (101 matched pairs)**:
```bash
python manual_cot_necessity_test.py
# Output: results/cot_necessity_llama_simple.json
```

**GPT-2 CoT Necessity (101 matched pairs)**:
```bash
python manual_cot_necessity_test_gpt2.py
# Output: results/cot_necessity_gpt2_simple.json
```

**LLaMA CoT Necessity (All 532 pairs)**:
```bash
python prepare_llama_steering_dataset_full.py
# Output: results/llama_cot_necessity_532.json
# Also generates: results/steering_dataset_llama_full.json
```

### Generate Steering Datasets

**GPT-2 Steering Dataset**:
```bash
python prepare_steering_dataset.py
# Output: results/steering_dataset_gpt2.json
```

**LLaMA Steering Dataset (Full)**:
```bash
python prepare_llama_steering_dataset_full.py
# Output: results/steering_dataset_llama_full.json
# Also generates: results/llama_cot_necessity_532.json
```

**LLaMA Steering Dataset (Pilot)**:
```bash
python prepare_llama_steering_dataset_fast.py
# Output: results/steering_dataset_llama.json
# Also generates: results/llama_cot_necessity_532.json
```

### Notes
- All scripts should be run from the `src/experiments/activation_patching/` directory
- Make sure models are available at `~/codi_ckpt/gpt2_gsm8k/` and `~/codi_ckpt/llama_gsm8k/`
- Some scripts may take significant time to run (especially full dataset processing)
- Files are used by downstream analysis and steering experiment scripts

---

## 11. Future Datasets

### Planned
- [ ] Multi-step reasoning dataset (≥5 steps)
- [ ] Out-of-distribution test set (MATH, StrategyQA)
- [ ] Ablation results organized by difficulty strata

### Ideas
- [ ] Cross-model transfer pairs (correct on one, wrong on other)
- [ ] Difficulty-stratified steering datasets
- [ ] Token-position-specific ablation results

---

**Document Status**: Living document, update when new datasets are created or experiments are run.

**Last Reviewed**: 2025-10-22
