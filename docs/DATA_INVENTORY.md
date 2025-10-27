# Data Inventory - CoT Exploration Project

**Last Updated**: 2025-10-27

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
| **SAE Error Analysis** | Error classification | 914 solutions (1.07 GB) | LLaMA | [`sae_error_analysis/data/error_analysis_dataset.json`](../src/experiments/sae_error_analysis/data/error_analysis_dataset.json) ⚠️ |

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

**Documentation**: [`docs/experiments/10-21_both_gsm8k_cot_necessity_ablation.md`](experiments/10-21_both_gsm8k_cot_necessity_ablation.md)

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
- Detailed report: [`docs/experiments/10-23_llama_gsm8k_dataset_expansion_1000.md`](experiments/10-23_llama_gsm8k_dataset_expansion_1000.md)
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
- Detailed report: [`docs/experiments/10-23_llama_gsm8k_dataset_expansion_1000.md`](experiments/10-23_llama_gsm8k_dataset_expansion_1000.md)
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

**Documentation**: [`docs/experiments/10-21_gpt2_gsm8k_activation_steering.md`](experiments/10-21_gpt2_gsm8k_activation_steering.md)

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

## 4. SAE Error Analysis Datasets

### 4.1 Error Analysis Dataset with Continuous Thoughts
**File**: [`src/experiments/sae_error_analysis/data/error_analysis_dataset.json`](../src/experiments/sae_error_analysis/data/error_analysis_dataset.json)

**Purpose**: Continuous thought activations extracted for correct and incorrect LLaMA solutions, used to train SAE-based error classifier

**Size**: 914 solutions (1.07 GB)

**Composition**:
- **Incorrect solutions**: 462 (50.5%)
- **Correct solutions**: 452 (49.5%)
- Balanced dataset for binary classification

**Structure**:
```json
{
  "metadata": {
    "n_correct": 452,
    "n_incorrect": 462,
    "total": 914,
    "layers": ["early", "middle", "late"],
    "layer_indices": {"early": 4, "middle": 8, "late": 14},
    "n_latent_tokens": 6,
    "source": "532 problem pairs validation results"
  },
  "correct_solutions": [
    {
      "pair_id": 0,
      "variant": "clean" | "corrupted",
      "question": "Problem text",
      "ground_truth": "Expected answer",
      "predicted": "Model prediction",
      "is_correct": true,
      "continuous_thoughts": {
        "early": [[2048 floats], ...],   // 6 vectors
        "middle": [[2048 floats], ...],  // 6 vectors
        "late": [[2048 floats], ...]     // 6 vectors
      }
    }
  ],
  "incorrect_solutions": [...]  // Same structure, is_correct: false
}
```

**Extraction Details**:
- **Layers extracted**: L4 (early), L8 (middle), L14 (late)
- **Tokens per layer**: 6 continuous thought tokens
- **Vectors per solution**: 18 (3 layers × 6 tokens)
- **Dimensions per vector**: 2048 (LLaMA-3.2-1B hidden size)
- **Total dimensions per solution**: 36,864 (after concatenation)

**Source Data**:
- Base: [`validation_results_llama_gpt4_532.json`](../src/experiments/activation_patching/validation_results_llama_gpt4_532.json)
- Problems: [`problem_pairs_gpt4_answers.json`](../src/experiments/activation_patching/problem_pairs_gpt4_answers.json)
- Both clean and corrupted variants included

**Selection Criteria**:
- Sampled 462 incorrect solutions (82% of 566 available)
- Sampled 452 correct solutions (for balance)
- Random seed: 42 for reproducibility

**Key Stats**:
- Model: LLaMA-3.2-1B-Instruct with CODI (6 latent tokens)
- Extraction time: ~3.5 minutes (4.6 solutions/second)
- File size: 1.07 GB (excluded from git via .gitignore)

**Used By**:
- SAE error classification experiment (65.57% test accuracy)
- Error pattern analysis (layer/token localization)
- Feature importance analysis (Cohen's d)

**Generation Command**:
```bash
python src/experiments/sae_error_analysis/extract_error_thoughts_simple.py \
  --n_wrong 462 --n_correct 462
```

**Created**: 2025-10-24

**Status**: ✅ File exists locally (not in git due to size)

**Documentation**: [`docs/experiments/10-24_llama_gsm8k_sae_error_analysis.md`](experiments/10-24_llama_gsm8k_sae_error_analysis.md)

---

## 5. Dataset Relationships

### 5.1 The Filtering Pipeline
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

## 6. Model-Specific Breakdowns

### 6.1 GPT-2 (117M parameters)
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

## 7. File Locations

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

## 8. Critical Insights

### 8.1 Why Filter to CoT-Dependent Pairs?
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

## 9. Usage Guidelines

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

## 10. Data Integrity Checks

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

## 11. Generating Missing Files

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

## 12. Future Datasets

### Planned
- [ ] Multi-step reasoning dataset (≥5 steps)
- [ ] Out-of-distribution test set (MATH, StrategyQA)
- [ ] Ablation results organized by difficulty strata

### Ideas
- [ ] Cross-model transfer pairs (correct on one, wrong on other)
- [ ] Difficulty-stratified steering datasets
- [ ] Token-position-specific ablation results

---

## 13. Position-wise Token Ablation Datasets

### 13.1 LLaMA CoT-Dependent Activations (Filtered)
**File**: [`src/experiments/gpt2_token_ablation/data/llama_activations_cot_dependent.json`](../src/experiments/gpt2_token_ablation/data/llama_activations_cot_dependent.json)

**Purpose**: LLaMA activations filtered to only CoT-dependent problems for position ablation experiments

**Size**: 424 samples (362 correct, 62 incorrect)

**Source**: Filtered from `sae_error_analysis/data/error_analysis_dataset_l12_l16.json` using CoT necessity results

**Layers**: L12-L16 (0-indexed layers 12-15)

**Generation Command**:
```bash
cd src/experiments/gpt2_token_ablation
python scripts/1_filter_cot_dependent.py
```

**Used By**: Position-type ablation experiment (2025-10-24)

---

### 13.2 GPT-2 Final Layer Token Decoding
**File**: [`src/experiments/gpt2_token_ablation/results/gpt2_final_layer_decoding.json`](../src/experiments/gpt2_token_ablation/results/gpt2_final_layer_decoding.json)

**Purpose**: Final layer (L11) token decodings for all 6 continuous thought positions

**Size**: 1000 samples

**Structure**:
```json
{
  "model": "GPT-2",
  "layer": 11,
  "samples": [
    {
      "id": 0,
      "decoded_positions": [
        {"position": 0, "token_text": "42", "is_number": true},
        ...
      ]
    }
  ]
}
```

**Generation Command**:
```bash
cd src/experiments/gpt2_token_ablation
python scripts/2_decode_final_layer_simple.py
```

**Key Stats**:
- Position 0: 23.8% decode to numbers
- Positions 1, 3, 5: 0.0% decode to numbers (alternating pattern)
- Positions 2, 4: 29.2%, 14.6% decode to numbers

**Used By**: Position ablation analysis (2025-10-24)

---

### 13.3 LLaMA Final Layer Token Decoding
**File**: [`src/experiments/gpt2_token_ablation/results/llama_final_layer_decoding.json`](../src/experiments/gpt2_token_ablation/results/llama_final_layer_decoding.json)

**Purpose**: Final layer (L15) token decodings for all 6 continuous thought positions

**Size**: 424 CoT-dependent samples

**Generation Command**:
```bash
cd src/experiments/gpt2_token_ablation
python scripts/2_decode_final_layer_simple.py
```

**Key Stats**:
- Positions 1 & 4: 85.8%, 83.3% decode to numbers (strong specialization)
- Positions 0, 2, 5: 54-62% decode to numbers
- Position 3: Only 4.7% decode to numbers

**Used By**: Position ablation analysis (2025-10-24)

---

### 13.4 Position Ablation Results

**Files**:
- [`src/experiments/gpt2_token_ablation/results/gpt2_position_ablation.json`](../src/experiments/gpt2_token_ablation/results/gpt2_position_ablation.json)
- [`src/experiments/gpt2_token_ablation/results/llama_position_ablation.json`](../src/experiments/gpt2_token_ablation/results/llama_position_ablation.json)

**Purpose**: Ablation experiment results testing causal importance of number vs non-number positions

**Size**:
- GPT-2: 1000 samples
- LLaMA: 424 samples

**Generation Commands**:
```bash
cd src/experiments/activation_patching
python run_position_type_ablation.py          # GPT-2
python run_position_type_ablation_llama.py    # LLaMA
```

**Key Results**:
- **GPT-2**: Ablating ANY positions causes 100% accuracy drop (43.2% → 0.0%)
- **LLaMA**: Ablating ANY positions causes ~98% accuracy drop (85.4% → 2-4%)
- Conclusion: Both models use collective reasoning despite position specialization

**Used By**: Cross-model comparison (2025-10-24)

---

**Document Status**: Living document, update when new datasets are created or experiments are run.

**Last Reviewed**: 2025-10-26

## 14. Liars-Bench Deception Detection Datasets

### 14.1 Preprocessed CODI Training/Validation/Test Data
**Location**: [`src/experiments/liars_bench_codi/data/processed/`](../src/experiments/liars_bench_codi/data/processed/)

**Source**: [Cadenza-Labs/liars-bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench) - Instructed Deception (ID) subset

**Purpose**: CODI-formatted liars-bench data for training GPT-2 to answer factual questions honestly

**Files**:
- `train.json` - Training set (honest examples only)
- `val.json` - Validation set (honest examples only)
- `test_honest.json` - Test set (honest examples)
- `deceptive_for_probes.json` - Deceptive examples for probe training

**Sizes**:
- Train: 7,074 examples (honest)
- Validation: 1,515 examples (honest)
- Test: 1,517 examples (honest)
- Deceptive: 10,692 examples (for probe training)

**Structure**:
```json
{
  "question": "Is the following statement correct or incorrect? [statement]",
  "answer": "The statement is [correct/incorrect]; [explanation]",
  "cot": "",
  "meta": {
    "is_honest": true/false,
    "original_index": 0
  }
}
```

**Question Formats**:
- "Is the following statement true or false?"
- "Is the following statement correct or incorrect?"

**Answer Formats**:
- "True/False"
- "Correct/Incorrect"
- "The statement is [label]"
- "That statement is [label]"

**Generation Command**:
```bash
cd src/experiments/liars_bench_codi/scripts
python 1_download_dataset.py  # Requires HuggingFace token
python 2_preprocess_data.py
```

**Used By**:
- GPT-2 CODI training (train.json, val.json)
- Honest example evaluation (test_honest.json)
- Deception probe training (deceptive_for_probes.json + test_honest.json)

**Created**: 2025-10-25

---

### 14.2 GPT-2 Continuous Thought Activations for Probes
**Location**: [`src/experiments/liars_bench_codi/data/processed/`](../src/experiments/liars_bench_codi/data/processed/)

**Purpose**: Extracted continuous thought activations from GPT-2 CODI for training deception detection probes

**Files**:
- `activations_gpt2_500.json` - Initial extraction (500+500)
- `activations_gpt2_1000.json` - Final extraction (1000+1000)

**Size**:
- Initial: 500 honest + 500 deceptive = 1,000 samples
- Final: 1,000 honest + 1,000 deceptive = 2,000 samples
- After deduplication: 622 honest + 236 deceptive = 858 final samples

**Structure**:
```json
{
  "model": "gpt2",
  "layers": ["layer_4", "layer_8", "layer_11"],
  "layer_indices": {"layer_4": 4, "layer_8": 8, "layer_11": 11},
  "n_latent_tokens": 6,
  "samples": [
    {
      "question": "Is the following statement correct or incorrect? [statement]",
      "answer": "The statement is [correct/incorrect]...",
      "is_honest": true/false,
      "thoughts": {
        "layer_4": [[768 floats], ...],   // 6 vectors
        "layer_8": [[768 floats], ...],   // 6 vectors
        "layer_11": [[768 floats], ...]   // 6 vectors
      }
    }
  ]
}
```

**Extraction Details**:
- **Layers**: L4 (early), L8 (middle), L11 (late)
- **Tokens per layer**: 6 continuous thought tokens
- **Dimensions per vector**: 768 (GPT-2 hidden size)
- **Total dimensions per sample**: 13,824 (3 layers × 6 tokens × 768 dims)

**Generation Command**:
```bash
cd src/experiments/liars_bench_codi/scripts
python extract_activations.py --num_samples 500   # Initial
python extract_activations.py --num_samples 1000  # Final
```

**Class Distribution** (after deduplication):
- Honest: 622 (72.49%)
- Deceptive: 236 (27.51%)
- Imbalance ratio: ~3:1

**Used By**:
- Deception detection probe training
- Per-token probe analysis (18 probes: 3 layers × 6 tokens)
- Mean pooling probe analysis

**Created**: 2025-10-25

---

### 14.3 Deception Detection Probe Results
**Location**: [`src/experiments/liars_bench_codi/results/`](../src/experiments/liars_bench_codi/results/)

**Purpose**: Trained linear probe results for honest vs deceptive classification

**Files**:
- `probe_results_gpt2.json` - Per-token probe results
- `probe_results_pooled_gpt2.json` - Mean-pooled probe results
- `probe_heatmap_gpt2.png` - Visualization of probe accuracy matrix

**Probe Configuration**: 18 total probes (3 layers × 6 tokens)

**Key Results**:
```json
{
  "model": "gpt2",
  "summary": {
    "mean_accuracy": 0.7249417249417249,
    "std_accuracy": 0.0,  // Perfect uniformity
    "min_accuracy": 0.7249417249417249,
    "max_accuracy": 0.7249417249417249
  }
}
```

**Accuracy Matrix** (all values 72.49%):
```
            Token 0  Token 1  Token 2  Token 3  Token 4  Token 5
Layer 4     72.49%   72.49%   72.49%   72.49%   72.49%   72.49%
Layer 8     72.49%   72.49%   72.49%   72.49%   72.49%   72.49%
Layer 11    72.49%   72.49%   72.49%   72.49%   72.49%   72.49%
```

**Critical Finding**: All probes achieve exactly 72.49% accuracy (= majority class baseline). Confusion matrices show probes learned "always predict honest" instead of detecting deception signal.

**Generation Command**:
```bash
cd src/experiments/liars_bench_codi/scripts
python train_probes.py         # Per-token probes
python train_probes_pooled.py  # Mean-pooled probes
python visualize_probes.py     # Heatmap visualization
```

**Created**: 2025-10-25

---

### 14.4 GPT-2 Honest Example Evaluation Results
**Location**: [`src/experiments/liars_bench_codi/results/gpt2_honest_eval.json`](../src/experiments/liars_bench_codi/results/gpt2_honest_eval.json)

**Purpose**: Evaluation of trained GPT-2 CODI model on honest test examples

**Size**: 1,517 test examples

**Key Results**:
```json
{
  "accuracy": 91.36453526697429,
  "correct": 1386,
  "total": 1517,
  "target_met": true
}
```

**Sample Predictions**: First 100 predictions stored with:
- Question
- Expected answer
- Predicted answer
- Extracted labels (true/false/correct/incorrect)
- Match status

**Generation Command**:
```bash
cd src/experiments/liars_bench_codi/scripts
python eval_gpt2.py
```

**Created**: 2025-10-25

---

### 14.5 Dataset Relationships - Liars-Bench Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Raw Liars-Bench Dataset (HuggingFace)                  │
│  Cadenza-Labs/liars-bench (Instructed Deception)        │
│  20,798 total examples                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓ Preprocessing (2_preprocess_data.py)
                   │
                   ├─► train.json (7,074 honest)
                   │   └─► GPT-2 CODI Training
                   │       ├─► Checkpoint: ~/codi_ckpt/gpt2_liars_bench/
                   │       └─► Training time: 22.5 minutes
                   │
                   ├─► val.json (1,515 honest)
                   │   └─► Validation during training
                   │
                   ├─► test_honest.json (1,517 honest)
                   │   ├─► Evaluation (eval_gpt2.py)
                   │   │   └─► gpt2_honest_eval.json (91.36% accuracy)
                   │   └─► Activation extraction (honest examples)
                   │       └─► activations_gpt2_1000.json (1000 honest)
                   │
                   └─► deceptive_for_probes.json (10,692 deceptive)
                       └─► Activation extraction (deceptive examples)
                           └─► activations_gpt2_1000.json (1000 deceptive)
                                   │
                                   ↓ After deduplication: 858 samples
                                   │
                                   ├─► train_probes.py
                                   │   └─► probe_results_gpt2.json
                                   │       (18 probes: 3 layers × 6 tokens)
                                   │       Mean accuracy: 72.49%
                                   │
                                   ├─► train_probes_pooled.py
                                   │   └─► probe_results_pooled_gpt2.json
                                   │       Pooled accuracy: 72.49%
                                   │
                                   └─► visualize_probes.py
                                       └─► probe_heatmap_gpt2.png
```

---

### 14.6 Key Statistics Summary

**Dataset Distribution**:
| Split | Honest | Deceptive | Total |
|-------|--------|-----------|-------|
| Train | 7,074 | 0 | 7,074 |
| Val | 1,515 | 0 | 1,515 |
| Test | 1,517 | 0 | 1,517 |
| Probes (raw) | 1,000 | 1,000 | 2,000 |
| Probes (deduplicated) | 622 | 236 | 858 |

**Model Performance**:
| Task | Metric | Target | Achieved | Status |
|------|--------|--------|----------|--------|
| Task Accuracy | Accuracy on honest examples | ≥90% | **91.36%** | ✅ Exceeded |
| Probe Accuracy | Deception detection | ≥70% | **72.49%** | ⚠️ Majority baseline |

**Critical Note**: Despite 72.49% probe accuracy exceeding the 70% target, this is a **majority class baseline artifact**. Confusion matrices reveal probes learned "always predict honest" (100% recall for honest, 0% for deceptive). True deception detection capability is ~59% (from cross-validation), below the target.

---

### 14.7 Experiment Documentation

**Research Journal**: [`docs/research_journal.md`](research_journal.md) (2025-10-25 entry)

**Detailed Report**: [`docs/experiments/10-25_gpt2_liars_bench_deception_detection.md`](experiments/10-25_gpt2_liars_bench_deception_detection.md)

**Reference Paper**: [Measuring Deceptive Alignment in Language Models](https://arxiv.org/pdf/2502.03407) (Apollo Research)

**Code Location**: [`src/experiments/liars_bench_codi/`](../src/experiments/liars_bench_codi/)

**Last Updated**: 2025-10-26

---

## 15. SAE CoT Decoder Datasets (Tuned Lens Expansion)

### 15.1 Enriched CoT-Aligned Training Data
**File**: [`src/experiments/sae_cot_decoder/data/enriched_train_data_with_cot.pt`](../src/experiments/sae_cot_decoder/data/enriched_train_data_with_cot.pt)

**Purpose**: Tuned Lens activation data enriched with GSM8K chain-of-thought token sequences for training SAEs to decode continuous thoughts

**Size**: 76,800 samples (603.0 MB)

**Composition**:
- **Source**: 1,000 GSM8K problems × 16 layers × 6 positions × 0.8 train split
- **Layers**: L0-L15 (all LLaMA-1B layers)
- **Positions**: 6 continuous thought token positions (0-5)
- **Hidden dimensions**: 2048 (LLaMA-3.2-1B hidden size)

**Structure**:
```python
{
  "hidden_states": torch.Tensor,  # (76800, 2048) - continuous thought vectors
  "target_token_ids": torch.Tensor,  # Target token IDs
  "layers": torch.Tensor,  # Layer indices (0-15)
  "positions": torch.Tensor,  # Position indices (0-5)
  "problem_ids": torch.Tensor,  # GSM8K problem identifiers
  "cot_sequences": List[List[str]],  # CoT calculation steps (e.g., ["16-3-4=9"])
  "cot_token_ids": List[List[int]]  # Tokenized CoT sequences
}
```

**Extraction Details**:
- **Base data**: `tuned_lens/data/train_data_llama_post_mlp.pt` (76,800 samples)
- **CoT match rate**: 100% (1,000/1,000 problems matched to GSM8K)
- **CoT format**: GSM8K `<<calculation>>` blocks (e.g., `<<16-3-4=9>>`)
- **Average CoT steps**: ~2.6 calculation steps per problem

**Generation Command**:
```bash
cd src/experiments/sae_cot_decoder/scripts
python extract_cot_alignments.py
```

**Used By**:
- SAE training for all 6 positions
- Feature-CoT correlation analysis
- Position-specific interpretability analysis

**Created**: 2025-10-26

---

### 15.2 Enriched CoT-Aligned Test Data
**File**: [`src/experiments/sae_cot_decoder/data/enriched_test_data_with_cot.pt`](../src/experiments/sae_cot_decoder/data/enriched_test_data_with_cot.pt)

**Purpose**: Test set for SAE validation and feature analysis

**Size**: 19,200 samples (150.7 MB)

**Composition**: Same structure as training data (0.2 test split)

**Used By**:
- SAE quality validation (explained variance, feature death rate, L0 norm)
- Feature extraction for interpretability analysis
- Layer selectivity analysis

**Created**: 2025-10-26

---

### 15.3 Trained SAE Models (Position-Specific)
**Location**: [`src/experiments/sae_cot_decoder/models/`](../src/experiments/sae_cot_decoder/models/)

**Purpose**: 6 position-specific sparse autoencoders trained to decompose continuous thoughts into monosemantic features

**Files**:
- `sae_position_0.pt` - Position 0 SAE (first CoT token)
- `sae_position_1.pt` - Position 1 SAE
- `sae_position_2.pt` - Position 2 SAE
- `sae_position_3.pt` - Position 3 SAE
- `sae_position_4.pt` - Position 4 SAE
- `sae_position_5.pt` - Position 5 SAE (last CoT token)

**Architecture**:
```python
class SparseAutoencoder:
    encoder: Linear(2048 → 2048)  # Sparse features
    decoder: Linear(2048 → 2048)  # Reconstruction
    L1 coefficient: 0.0005
    Total parameters per SAE: 8,388,608 (33.6 MB)
```

**Training Configuration**:
- **Epochs**: 50
- **Batch size**: 4096
- **Learning rate**: 1e-3 with CosineAnnealingLR
- **Optimizer**: Adam
- **L1 penalty**: 0.0005
- **Training time**: ~90 minutes total (all 6 SAEs)

**Quality Metrics**:
| Position | Explained Variance | Feature Death Rate | L0 Norm | Status |
|----------|-------------------|-------------------|---------|--------|
| 0 | 37.4% ❌ | 69.6% ❌ | 19.0 | ⚠️ WARNING |
| 1 | 70.9% ✅ | 68.4% ❌ | 51.8 | ⚠️ WARNING |
| 2 | 71.0% ✅ | 80.7% ❌ | 55.4 | ⚠️ WARNING |
| 3 | 72.6% ✅ | 55.7% ❌ | 50.7 | ⚠️ WARNING |
| 4 | 66.2% ❌ | 49.5% ❌ | 30.3 | ⚠️ WARNING |
| 5 | 74.3% ✅ | 73.4% ❌ | 55.7 | ⚠️ WARNING |

**Targets**:
- Explained Variance: ≥70% (4/6 positions pass)
- Feature Death Rate: ≤15% (0/6 positions pass)
- L0 Norm: 50-100 active features

**Key Insight**: High feature death (50-81%) suggests sparse but interpretable features. Position 0 shows markedly lower EV (37.4%), suggesting different encoding.

**Generation Command**:
```bash
cd src/experiments/sae_cot_decoder/scripts
python train_saes.py --no-wandb --epochs 50
```

**Used By**:
- Feature extraction for interpretability analysis
- CoT token correlation analysis
- Layer selectivity analysis

**Created**: 2025-10-26

**Size**: ~200 MB total (6 models × 33.6 MB each)

---

### 15.4 Feature Catalog (Interpretable Features)
**File**: [`src/experiments/sae_cot_decoder/analysis/feature_catalog.json`](../src/experiments/sae_cot_decoder/analysis/feature_catalog.json)

**Purpose**: Comprehensive catalog of interpretable SAE features with CoT token correlations

**Size**: 1,455 interpretable features (out of 12,288 total)

**Distribution by Position**:
- Position 0: 224 interpretable features
- Position 1: 258 interpretable features
- Position 2: 225 interpretable features
- Position 3: 225 interpretable features
- Position 4: 269 interpretable features
- Position 5: 254 interpretable features

**Structure**:
```json
{
  "summary": {
    "total_features": 12288,
    "interpretable_features_per_position": {...}
  },
  "positions": {
    "0": {
      "total_features": 2048,
      "interpretable_features": 224,
      "top_100_features": [
        {
          "feature_id": 1155,
          "position": 0,
          "activation_threshold": 0.47,
          "num_active_samples": 145,
          "enriched_tokens": [
            {
              "token_id": 931,
              "token_str": "000",
              "active_count": 56,
              "inactive_count": 104,
              "enrichment": 0.533,
              "p_value": 4.04e-63
            }
          ],
          "interpretability_score": 14,
          "selectivity": {
            "selectivity_index": 0.458,
            "most_selective_layer": 15,
            "layer_means": {...}
          }
        }
      ]
    }
  }
}
```

**Feature Types Discovered**:
- **Number features**: Correlate with digits (0-9) and multi-digit numbers (100, 200, 810)
- **Operation features**: Correlate with arithmetic operators (*, =, -)
- **Calculation features**: Mixed number-operation patterns

**Example Feature**:
- **Feature 1155 (Position 0)**: "000" detector
  - Enrichment: 53.3% when active
  - p-value: 4.04×10⁻⁶³ (highly significant)
  - Also correlates with "0", "00", "300", "120", "200"

**Statistical Criteria**:
- **Chi-squared test**: p-value < 0.01 for token-feature association
- **Activation threshold**: 75th percentile of feature activations
- **Enrichment**: Fraction of active samples containing token

**Generation Command**:
```bash
cd src/experiments/sae_cot_decoder/scripts
python analyze_features.py
```

**Used By**:
- Identifying monosemantic features
- Understanding CoT token encoding
- CODI Figure 6-style analysis

**Created**: 2025-10-26

---

### 15.5 Feature-CoT Correlation Analysis
**File**: [`src/experiments/sae_cot_decoder/analysis/feature_cot_correlations.json`](../src/experiments/sae_cot_decoder/analysis/feature_cot_correlations.json)

**Purpose**: Statistical correlation between SAE features and explicit CoT tokens

**Size**: 1,455 feature-token correlation mappings

**Analysis Method**:
1. Extract features for all test samples
2. Identify active features (above activation threshold)
3. Build contingency tables (feature active/inactive × token present/absent)
4. Chi-squared test for independence (p < 0.01)
5. Calculate enrichment scores

**Key Findings**:
- **Position 0**: 224 features with significant CoT correlations
- **Position 4**: 269 features (highest interpretability)
- **Number tokens**: Strong correlations with digits 0-9, multi-digit numbers
- **Operation tokens**: Correlations with *, =, +, -

**Created**: 2025-10-26

---

### 15.6 Layer Selectivity Analysis
**File**: [`src/experiments/sae_cot_decoder/analysis/layer_selectivity.json`](../src/experiments/sae_cot_decoder/analysis/layer_selectivity.json)

**Purpose**: Measure which layers each feature is most active in (layer specialization)

**Metrics**:
- **Selectivity index**: Normalized entropy of layer activations (0=uniform, 1=single layer)
- **Most selective layer**: Layer with highest mean activation
- **Layer means**: Average activation per layer (L0-L15)

**Key Findings**:
- Features show layer specialization (selectivity index ~0.4-0.5)
- Late layers (L12-L15) tend to have higher feature activations
- Position 0 features activate more uniformly across layers

**Used By**:
- Understanding layer-wise feature specialization
- Identifying which layers encode which aspects of reasoning

**Created**: 2025-10-26

---

### 15.7 Extracted Features (Test Set)
**File**: [`src/experiments/sae_cot_decoder/analysis/extracted_features.pt`](../src/experiments/sae_cot_decoder/analysis/extracted_features.pt)

**Purpose**: SAE-encoded feature activations for all test samples

**Size**: 19,200 samples × 6 positions × 2048 features

**Structure**:
```python
{
  "position_0": torch.Tensor,  # (3200, 2048) - features for position 0
  "position_1": torch.Tensor,  # (3200, 2048)
  ...
  "position_5": torch.Tensor,  # (3200, 2048)
  "metadata": {...}
}
```

**Used By**:
- Feature-CoT correlation analysis
- Layer selectivity analysis
- Feature catalog generation

**Created**: 2025-10-26

---

### 15.8 Validation Results
**File**: [`src/experiments/sae_cot_decoder/analysis/validation_results.json`](../src/experiments/sae_cot_decoder/analysis/validation_results.json)

**Purpose**: Quality metrics for all 6 trained SAEs

**Structure**: See section 15.3 for metrics table

**Created**: 2025-10-26

---

### 15.9 Visualizations

**Files**:
- [`src/experiments/sae_cot_decoder/analysis/position_comparison.png`](../src/experiments/sae_cot_decoder/analysis/position_comparison.png)
  - 3-panel visualization comparing explained variance, feature death rate, and L0 norm across positions

- [`src/experiments/sae_cot_decoder/analysis/training_curves.png`](../src/experiments/sae_cot_decoder/analysis/training_curves.png)
  - 6-panel grid showing training curves for each position (explained variance and feature death rate vs epochs)

**Created**: 2025-10-26

---

### 15.10 Training Summary
**File**: [`src/experiments/sae_cot_decoder/analysis/training_summary.md`](../src/experiments/sae_cot_decoder/analysis/training_summary.md)

**Purpose**: Human-readable markdown summary of SAE training results

**Content**:
- Quality validation targets
- Results by position (explained variance, feature death rate, L0 norm)
- Summary statistics (positions passing targets)

**Created**: 2025-10-26

---

### 15.11 Dataset Relationships - SAE CoT Decoder Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Tuned Lens Activation Data (Pre-existing)              │
│  tuned_lens/data/train_data_llama_post_mlp.pt          │
│  76,800 samples (602 MB)                                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ↓ CoT Alignment (extract_cot_alignments.py)
                   │
┌──────────────────────────────────────────────────────────┐
│  Enriched Training Data with CoT                         │
│  enriched_train_data_with_cot.pt (603 MB)               │
│  76,800 samples with GSM8K CoT sequences                │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ↓ SAE Training (train_saes.py)
                   │
┌──────────────────────────────────────────────────────────┐
│  6 Position-Specific SAE Models (~200 MB total)          │
│  models/sae_position_{0-5}.pt                           │
│  2048 features each, L1=0.0005                          │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ├─► Validation (validate_results.py)
                   │   └─► validation_results.json
                   │       training_summary.md
                   │       training_curves.png
                   │       position_comparison.png
                   │
                   └─► Feature Analysis (analyze_features.py)
                       ├─► extracted_features.pt
                       │   (19,200 test samples × 6 positions)
                       │
                       ├─► feature_cot_correlations.json
                       │   (1,455 interpretable features)
                       │
                       ├─► layer_selectivity.json
                       │   (Layer specialization analysis)
                       │
                       └─► feature_catalog.json
                           (Top 100 features per position)
```

---

### 15.12 Key Statistics Summary

**Dataset Sizes**:
| Dataset | Samples | Size | Purpose |
|---------|---------|------|---------|
| Train (enriched) | 76,800 | 603.0 MB | SAE training |
| Test (enriched) | 19,200 | 150.7 MB | Validation & analysis |
| SAE models (6) | - | ~200 MB | Feature extraction |
| Extracted features | 19,200×6 | - | Correlation analysis |

**SAE Quality**:
| Metric | Target | Positions Passing |
|--------|--------|------------------|
| Explained Variance | ≥70% | 4/6 (67%) |
| Feature Death Rate | ≤15% | 0/6 (0%) |
| L0 Norm | 50-100 | 4/6 (67%) |

**Feature Interpretability**:
| Position | Total Features | Interpretable | Percentage |
|----------|---------------|---------------|------------|
| 0 | 2048 | 224 | 10.9% |
| 1 | 2048 | 258 | 12.6% |
| 2 | 2048 | 225 | 11.0% |
| 3 | 2048 | 225 | 11.0% |
| 4 | 2048 | 269 | 13.1% |
| 5 | 2048 | 254 | 12.4% |
| **Total** | **12,288** | **1,455** | **11.8%** |

---

### 15.13 Experiment Documentation

**Research Journal**: [`docs/research_journal.md`](research_journal.md) (2025-10-26 entry)

**Detailed Report**: [`docs/experiments/10-26_llama_gsm8k_sae_cot_decoder.md`](experiments/10-26_llama_gsm8k_sae_cot_decoder.md)

**Code Location**: [`src/experiments/sae_cot_decoder/`](../src/experiments/sae_cot_decoder/)

**Base Experiment**: [`src/experiments/tuned_lens/`](../src/experiments/tuned_lens/) (activation data source)

**Reference**: CODI paper Figure 6 (CoT token correlation methodology)

**Created**: 2025-10-26



---

## 16. Mechanistic Interpretability Datasets

### 16.1 Stratified Test Problems for Step Importance Analysis
**File**: [`src/experiments/mechanistic_interp/data/stratified_test_problems.json`](../src/experiments/mechanistic_interp/data/stratified_test_problems.json)

**Purpose**: Stratified GSM8K dataset for measuring causal importance of continuous thought positions via ablation experiments

**Size**: 1,000 problems

**Stratification** (by reasoning steps):
- 1-step: 27 problems (2.7%)
- 2-step: 223 problems (22.3%)
- 3-step: 250 problems (25.0%)
- 4-step: 250 problems (25.0%)
- 5-step: 175 problems (17.5%)
- 6-step: 56 problems (5.6%)
- 7-step: 16 problems (1.6%)
- 8-step: 3 problems (0.3%)

**Structure**:
```json
{
  "gsm8k_id": "test_123",
  "question": "Problem text...",
  "answer": "42",
  "reasoning_steps": 3
}
```

**Source**: GSM8K test set, stratified by counting calculation blocks in solutions

**Generation Command**:
```bash
cd src/experiments/mechanistic_interp/scripts
export PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH
python 01_validate_data.py
```

**How to Recreate**:
1. Load GSM8K test set
2. Extract reasoning difficulty by counting calculation blocks
3. Stratify by difficulty buckets (1-8 steps)
4. Sample proportionally to target distribution
5. Save as JSON with gsm8k_id, question, answer, reasoning_steps

**Used By**:
- MECH-02: Step Importance Analysis (position-wise ablation)
- Future: MECH-03 (SAE feature extraction), MECH-04 (correlation analysis), MECH-06 (interventions)

**Experiment**: MECH-02 - Step Importance Analysis

**Created**: 2025-10-26

**Documentation**: [`docs/experiments/10-26_llama_gsm8k_step_importance.md`](experiments/10-26_llama_gsm8k_step_importance.md)

---

### 16.2 Step Importance Analysis Results

**Files**:
- [`step_importance_scores.json`](../src/experiments/mechanistic_interp/data/step_importance_scores.json) (1.6 MB) - Full results for 1,000 problems
- [`step_importance_summary_stats.json`](../src/experiments/mechanistic_interp/data/step_importance_summary_stats.json) (1.8 KB) - Aggregate statistics
- [`step_importance_validation.json`](../src/experiments/mechanistic_interp/data/step_importance_validation.json) (152 KB) - Validation on 87 problems

**Key Results** (from summary_stats.json):
| Position | Importance | Interpretation |
|----------|-----------|----------------|
| 0 | 0.000 | Baseline (no ablation) |
| 1 | 0.145 | Exploration (recoverable) |
| 2 | 0.468 | Solution space narrowing |
| 3 | 0.528 | Refinement begins |
| 4 | 0.556 | Solution converging |
| 5 | **0.868** | **Final commitment (critical\!)** |

**Key Finding**: **Progressive refinement strategy** - Late positions (4,5) most critical, contrary to "planning first" hypothesis

**Statistical Validation**:
- Monotonic trend: Spearman ρ=0.99, p<0.001
- Effect size: Cohen's d=2.1 (very large)
- Universal pattern: Late > Early for ALL 8 difficulty levels (3.1× ratio)

**Created**: 2025-10-26

---

## 17. Matryoshka SAE Pilot Datasets

### 17.1 Position 3 Activations
**File**: [`src/experiments/matryoshka_sae_pilot/data/position_3_activations.pt`](../src/experiments/matryoshka_sae_pilot/data/position_3_activations.pt)

**Purpose**: Position 3 continuous thought activations for training hierarchical SAE architectures

**Size**: 95,648 vectors (748 MB)

**Structure**:
- **Activations**: [95,648, 2048] tensor
- **Metadata**: Problem IDs for each activation

**Breakdown**:
- Source: 5,978 unique GSM8K problems
- Layers: 16 layers per problem
- 5,978 problems × 16 layers = 95,648 total activations

**Stratification**: None (all available Position 3 data from sae_cot_decoder)

**Source**: Extracted from [`sae_cot_decoder/data/`](../src/experiments/sae_cot_decoder/data/) experiment

**Generation Command**:
```bash
cd src/experiments/matryoshka_sae_pilot
python extract_position3_data.py
```

**How to Recreate**:
1. Load all activation files from sae_cot_decoder/data/ (16 layers)
2. Filter to Position 3 only (from 6 positions)
3. Concatenate all layers
4. Save as position_3_activations.pt with metadata

**Used By**:
- Vanilla Matryoshka SAE training
- Matryoshka-TopK SAE training
- Feature extraction and classification

**Train/Val Split**: 80/20 split (76,518 train / 19,130 val)

**Experiment**: Matryoshka SAE Pilot

**Created**: 2025-10-27

---

### 17.2 Vanilla Matryoshka SAE Model
**File**: [`src/experiments/matryoshka_sae_pilot/models/pos3_hierarchical.pt`](../src/experiments/matryoshka_sae_pilot/models/pos3_hierarchical.pt)

**Purpose**: Trained Matryoshka SAE with 3 hierarchical levels for continuous thought interpretation

**Architecture**:
- **Levels**: [512, 1024, 2048] features (nested hierarchy)
- **Activation**: ReLU + L1 penalty (λ=0.0005)
- **Level weights**: [0.3, 0.3, 0.4] (prioritize fine level)
- **Parameters**: 3.7M

**Training Config**:
- **Data**: 76,518 Position 3 activations
- **Epochs**: 50
- **Batch size**: 4096
- **Optimizer**: AdamW with CosineAnnealingLR
- **Training time**: 1.6 minutes

**Performance** (Level 3):
- **Explained Variance**: 72.1%
- **L0 Norm**: 27.5 active features/sample
- **Feature Death**: 62.5% (769/2048 active)
- **Utilization**: 37.5%

**Improvements over ReLU SAE**:
- Feature death: 97.0% → 62.5% (-34.5 pts)
- Active features: 248 → 769 (+3.1×)
- Classification: 78.9% → 80.0% (+1.1 pts)

**Experiment**: Matryoshka SAE Pilot

**Created**: 2025-10-27

---

### 17.3 Matryoshka-TopK Hybrid SAE Model
**File**: [`src/experiments/matryoshka_sae_pilot/models/pos3_hierarchical_topk.pt`](../src/experiments/matryoshka_sae_pilot/models/pos3_hierarchical_topk.pt)

**Purpose**: Hybrid SAE combining hierarchical structure (Matryoshka) with TopK activation for better efficiency

**Architecture**:
- **Levels**: [128, 256, 512] features (total: 896)
- **K values**: [25, 35, 40] TopK per level (total: 100 active/sample)
- **Activation**: TopK (no L1 penalty)
- **Level weights**: [0.3, 0.3, 0.4]
- **Parameters**: 3.7M

**Training Config**:
- **Data**: 76,518 Position 3 activations
- **Epochs**: 50
- **Batch size**: 4096
- **Optimizer**: AdamW with CosineAnnealingLR
- **Training time**: 1.1 minutes

**Performance** (Level 3):
- **Explained Variance**: 77.9%
- **L0 Norm**: 40.0 active features/sample (enforced by TopK)
- **Feature Death**: 42.0% (297/512 active)
- **Utilization**: 58.0%

**Comparison to Baselines**:
- vs Vanilla Matryoshka: +5.8 pts EV, -20.5 pts feature death
- vs TopK SAE: -9.9 pts EV, +42 pts feature death
- vs ReLU SAE: -0.7 pts EV, -55 pts feature death

**Experiment**: Matryoshka SAE Pilot

**Created**: 2025-10-27

---

### 17.4 Fair Comparison Results
**File**: [`src/experiments/matryoshka_sae_pilot/results/fair_comparison.json`](../src/experiments/matryoshka_sae_pilot/results/fair_comparison.json)

**Purpose**: Fair classification comparison between ReLU and Vanilla Matryoshka SAEs using identical test sets

**Test Protocol**:
- **Test set**: 19,130 samples (same for both models)
- **Train/test split**: 80/20 with random_state=42
- **Classifier**: Logistic Regression (max_iter=1000)
- **Task**: Operation detection (multiplication/addition/division)

**Results**:
| Model | Features | Accuracy | vs ReLU |
|-------|----------|----------|---------|
| ReLU SAE | 8,192 | 78.9% | baseline |
| Matryoshka L1 | 512 | 78.4% | -0.5 pts |
| Matryoshka L2 | 1,024 | 78.7% | -0.1 pts |
| Matryoshka L3 | 2,048 | 78.9% | 0.0 pts |
| **Matryoshka Concat** | **3,584** | **80.0%** | **+1.1 pts** |

**Key Finding**: Concatenating hierarchical features outperforms ReLU baseline

**Experiment**: Matryoshka SAE Pilot

**Created**: 2025-10-27

---

### 17.5 Comprehensive Comparison
**File**: [`src/experiments/matryoshka_sae_pilot/results/comprehensive_comparison.json`](../src/experiments/matryoshka_sae_pilot/results/comprehensive_comparison.json)

**Purpose**: Complete comparison of all SAE architectures across reconstruction and classification metrics

**Models Compared**:
1. ReLU SAE (8,192 features)
2. TopK SAE (512 features, K=100)
3. Vanilla Matryoshka SAE (3,584 concat features)
4. Matryoshka-TopK Hybrid (896 features)

**Reconstruction Metrics** (Explained Variance):
| Model | EV | Utilization | Feature Death |
|-------|-----|-------------|---------------|
| TopK SAE | **87.8%** | **100%** | **0%** |
| ReLU SAE | 78.6% | 3.0% | 97% |
| Matryoshka-TopK | 77.9% | 58.0% | 42% |
| Vanilla Matryoshka | 72.1% | 37.5% | 62.5% |

**Classification Metrics** (Operation Detection):
| Model | Accuracy |
|-------|----------|
| Vanilla Matryoshka (concat) | **80.0%** |
| ReLU SAE | 78.9% |
| Matryoshka-TopK | Pending |
| TopK SAE | No model available |

**Key Findings**:
- **Reconstruction winner**: TopK SAE (87.8% EV, perfect utilization)
- **Classification winner**: Vanilla Matryoshka (80.0% accuracy)
- **Efficiency winner**: TopK SAE (0.171 EV per feature)
- **Feature death loser**: ReLU SAE (97% waste)

**Experiment**: Matryoshka SAE Pilot

**Created**: 2025-10-27

**Documentation**: Pending in research_journal.md

---
