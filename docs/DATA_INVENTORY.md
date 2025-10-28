# Data Inventory - CoT Exploration Project

**Last Updated**: 2025-10-28 (Added Section 14.7: Proper Question-Level Held-Out Splits for Sprint 1 & 4)

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
| **CODI450 Dataset** | SAE intervention pilot | 450 problems (400 train, 50 test) | GPT-2 or LLaMA | [`data/step_by_step/gsm8k_codi450.manifest.json`](../data/step_by_step/) ⚠️ |
| **Step-by-Step SAE Models** | Per-step feature dictionaries | 6 models (~96 MB) | GPT-2 or LLaMA | [`models/step_by_step/sae_step{k}/sae.pt`](../models/step_by_step/) ⚠️ |
| **Intervention Results** | Smoke test feature interventions | ~450 rows | GPT-2 or LLaMA | [`results/step_by_step/smoke_test_results.csv`](../results/step_by_step/smoke_test_results.csv) ⚠️ |

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

### 14.7 Proper Question-Level Held-Out Splits (Sprint 1 & 4 Corrected Methodology)
**Location**: [`src/experiments/liars_bench_codi/data/processed/`](../src/experiments/liars_bench_codi/data/processed/)

**Purpose**: Corrected dataset splits with proper question-level held-out methodology (fixing data leakage from original Sprint 1)

**Critical Change**: Original methodology had 100% question overlap between CODI training and probe evaluation. New methodology enforces zero overlap at question level.

**Files**:
- `probe_train_proper.json` - Probe training set (288 samples, 144 honest + 144 deceptive)
- `probe_test_proper.json` - Probe test set (288 samples, 144 honest + 144 deceptive)
- `probe_activations_gpt2_proper.json` - GPT-2 continuous thought activations (264 MB)
- `probe_activations_llama3b_proper.json` - LLaMA-3.2-3B continuous thought activations (797 MB)
- `response_activations_gpt2_proper.json` - GPT-2 response token activations (13 MB)
- `splits_metadata_proper.json` - Split validation metadata

**Dataset Split Strategy**:
```
Total Questions: 960 unique questions from Liars-Bench

CODI Training:    672 questions (6,405 samples, 100% honest)
CODI Validation:  672 questions (690 samples, 100% honest)
Probe Training:   144 questions (288 samples, 50/50 balance)
Probe Test:       144 questions (288 samples, 50/50 balance)

✅ ZERO OVERLAP: CODI questions ∩ Probe questions = ∅
✅ ZERO OVERLAP: Probe train questions ∩ Probe test questions = ∅
```

**Why This Matters**:
- **Original (INVALID)**: Tested deception detection on questions the model was trained to answer
  - Result: 98% accuracy (FALSE - measured memorization)
- **Corrected (VALID)**: Test deception detection on completely unseen questions
  - Result: 50% accuracy (TRUE - measures generalization)

**Activation Extraction Details**:

**GPT-2** (`probe_activations_gpt2_proper.json`):
- **Model**: GPT-2 (124M), 12 layers, 768-dim
- **Layers extracted**: L4, L8, L11 (early/middle/late)
- **Tokens per layer**: 6 continuous thought tokens
- **Train samples**: 288 (144 honest, 144 deceptive)
- **Test samples**: 288 (144 honest, 144 deceptive)
- **Total dimensions**: 13,824 (3 layers × 6 tokens × 768)

**LLaMA-3.2-3B** (`probe_activations_llama3b_proper.json`):
- **Model**: LLaMA-3.2-3B-Instruct, 28 layers, 3072-dim
- **Layers extracted**: L9, L18, L27 (early/middle/late)
- **Tokens per layer**: 6 continuous thought tokens
- **Train samples**: 288 (144 honest, 144 deceptive)
- **Test samples**: 288 (144 honest, 144 deceptive)
- **Total dimensions**: 55,296 (3 layers × 6 tokens × 3072)

**Response Tokens** (`response_activations_gpt2_proper.json`):
- **Model**: GPT-2 (124M)
- **Layer**: Final layer (L11)
- **Tokens**: Mean-pooled response tokens (after answer generation)
- **Dimensions**: 768
- **Purpose**: Baseline comparison (response vs continuous thoughts)

**Generation Commands**:
```bash
cd src/experiments/liars_bench_codi/scripts

# Create proper splits (question-level held-out)
python create_proper_splits.py

# Extract GPT-2 continuous thought activations
python extract_activations_gpt2_proper.py

# Extract LLaMA-3B continuous thought activations
python extract_activations_llama3b_proper.py

# Extract GPT-2 response token activations
python extract_response_activations_proper.py

# Train probes on proper data
python train_probes_proper_v2.py          # Continuous thoughts
python train_response_probe_proper.py      # Response tokens
```

**Key Results** (Proper Methodology):

**Continuous Thoughts**:
- GPT-2: **50.00% accuracy** (random chance, 0.500 AUROC)
- LLaMA-3.2-3B: **50.00% accuracy** (random chance, 0.500 AUROC)
- Finding: **Zero generalizable deception signal** in continuous space (scale-invariant)

**Response Tokens**:
- GPT-2: **70.49% accuracy** (0.760 AUROC)
- Finding: Response tokens encode generalizable deception patterns

**Gap**: +20.49 percentage points in favor of response tokens

**Used By**:
- Sprint 1 (Corrected): GPT-2 deception detection with proper methodology
- Sprint 4: LLaMA-3.2-3B scale test (124M → 3B parameters)

**Experiment Documentation**:
- [`docs/experiments/10-28_gpt2_liars_bench_sprint1_CORRECTED_FINAL.md`](experiments/10-28_gpt2_liars_bench_sprint1_CORRECTED_FINAL.md)
- [`docs/experiments/10-28_llama3b_liars_bench_sprint4_FINAL.md`](experiments/10-28_llama3b_liars_bench_sprint4_FINAL.md)

**Created**: 2025-10-28 (Sprint 1 correction + Sprint 4)

**Data Quality Audit**: [`src/experiments/liars_bench_codi/data/processed/SPRINT4_DATA_AUDIT_REPORT.md`](../src/experiments/liars_bench_codi/data/processed/SPRINT4_DATA_AUDIT_REPORT.md)

---

### 14.8 Experiment Documentation

**Research Journal**: [`docs/research_journal.md`](research_journal.md) (2025-10-25 entry, updated 2025-10-28)

**Detailed Reports**:
- [`docs/experiments/10-25_gpt2_liars_bench_deception_detection.md`](experiments/10-25_gpt2_liars_bench_deception_detection.md) - Original (invalidated)
- [`docs/experiments/10-28_gpt2_liars_bench_sprint1_CORRECTED_FINAL.md`](experiments/10-28_gpt2_liars_bench_sprint1_CORRECTED_FINAL.md) - Corrected methodology
- [`docs/experiments/10-28_llama3b_liars_bench_sprint4_FINAL.md`](experiments/10-28_llama3b_liars_bench_sprint4_FINAL.md) - Scale test

**Reference Paper**: [Measuring Deceptive Alignment in Language Models](https://arxiv.org/pdf/2502.03407) (Apollo Research)

**Code Location**: [`src/experiments/liars_bench_codi/`](../src/experiments/liars_bench_codi/)

**Last Updated**: 2025-10-28

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

## 18. GPT-2 TopK SAE Parameter Sweep Datasets

### 18.1 GPT-2 Activation Data (Train/Val)
**Files**:
- [`src/experiments/gpt2_sae_training/data/gpt2_full_train_activations.pt`](../src/experiments/gpt2_sae_training/data/gpt2_full_train_activations.pt) (177 MB)
- [`src/experiments/gpt2_sae_training/data/gpt2_full_val_activations.pt`](../src/experiments/gpt2_sae_training/data/gpt2_full_val_activations.pt) (44 MB)

**Purpose**: Continuous thought activations from GPT-2 CODI for TopK SAE training

**Size**:
- Train: 57,600 samples (800 problems × 12 layers × 6 positions)
- Val: 14,400 samples (200 problems × 12 layers × 6 positions)

**Source**: 1,000 GPT-2 predictions from [`gpt2_shared_data/gpt2_predictions_1000.json`](../src/experiments/gpt2_shared_data/gpt2_predictions_1000_checkpoint_1000.json)

**Structure**:
```python
{
  'activations': torch.Tensor,  # (N, 768) - GPT-2 hidden dims
  'metadata': {
    'problem_ids': List[int],  # Problem identifiers (0-999)
    'layers': List[int],       # Layer indices (0-11)
    'positions': List[int],    # Position indices (0-5)
  },
  'config': {
    'model': 'gpt2',
    'num_problems': int,       # 800 train / 200 val
    'num_layers': 12,          # GPT-2 has 12 layers
    'num_ct_tokens': 6,        # CODI uses 6 continuous thought tokens
    'hidden_size': 768         # GPT-2 hidden dimension
  }
}
```

**How to Recreate**:
```bash
# Convert from existing GPT-2 predictions JSON
python src/experiments/gpt2_sae_training/scripts/convert_gpt2_data.py
```

**Used By**:
- Parameter sweep: 8 configs tested (d × K combinations)
- Sweet spot training: 72 SAEs (12 layers × 6 positions)

**Created**: 2025-10-27

---

### 18.2 Parameter Sweep Results (8 Configs)
**Files**: [`src/experiments/gpt2_sae_training/results/gpt2_pos3_layer8_d{192,256,384,512}_k{20,30,40,50,75,100,150}.pt`](../src/experiments/gpt2_sae_training/results/)

**Purpose**: Trained TopK SAE checkpoints for parameter sweep

**Size**: 8 checkpoints (1.2-3.1 MB each)

**Configs Tested**:
| Latent Dim (d) | K | Sparsity | EV | Death Rate |
|----------------|---|----------|-----|------------|
| 512 | 150 | 29.3% | 94.8% | 4.1% |
| 512 | 100 | 19.5% | 94.1% | 26.0% |
| 384 | 75 | 19.5% | 93.3% | 36.5% |
| 256 | 75 | 29.3% | 92.7% | 26.6% |
| 256 | 50 | 19.5% | 91.4% | 43.8% |
| 192 | 40 | 20.8% | 90.6% | 39.6% |
| 256 | 30 | 11.7% | 88.5% | 59.0% |
| 192 | 20 | 10.4% | 83.3% | 76.6% |

**Training Details**:
- Position: 3 (middle token)
- Layer: 8 (middle layer)
- Epochs: 25
- Batch size: 256
- Training time: ~2-3 seconds per config (parallel execution)

**Sweet Spot**: d=512, K=150 (94.8% EV, 4.1% death rate)

**Created**: 2025-10-27

---

### 18.3 Sweet Spot Model (All Layers×Positions)
**Files**: [`src/experiments/gpt2_sae_training/results/sweet_spot_all/gpt2_sweet_spot_pos{0-5}_layer{0-11}.pt`](../src/experiments/gpt2_sae_training/results/sweet_spot_all/)

**Purpose**: Sweet spot config (d=512, K=150) trained on all 72 layer-position combinations for heatmap visualization

**Size**: 72 checkpoints

**Metrics File**: [`sweet_spot_metrics_all.json`](../src/experiments/gpt2_sae_training/results/sweet_spot_all/sweet_spot_metrics_all.json)

**Key Patterns**:
- **Layer progression**:
  - Early layers (L0-L3): High EV (>96%), low death (<20%)
  - Middle layers (L4-L7): Medium EV (~93-97%), very low death (<10%)
  - Late layers (L8-L11): Lower EV (~75-95%), near-zero death (<2%)

- **Position specialization**:
  - Odd positions (1,3,5): Consistently higher EV, lower reconstruction loss
  - Even positions (0,2,4): Lower EV in late layers, suggesting complex/abstract encoding

**Created**: 2025-10-27

---

### 18.4 Analysis Results
**File**: [`src/experiments/gpt2_sae_training/results/analysis_summary.json`](../src/experiments/gpt2_sae_training/results/analysis_summary.json)

**Purpose**: Comprehensive analysis of all 8 configs with sweet spot selection rationale

**Contents**:
- `all_configs`: Metrics for all 8 parameter sweep configs
- `sweet_spot`: Selected configuration (d=512, K=150)
- `rationale`: Selection criteria and justification
- `comparison_table`: Sorted comparison table

**Selection Criteria**:
1. Explained Variance ≥ 70% (reconstruction quality)
2. Lowest Feature Death Rate (feature utilization)
3. Balanced sparsity (not too dense)

**Created**: 2025-10-27

---

### 18.5 Visualizations
**Files**:
- [`src/experiments/gpt2_sae_training/results/gpt2_sweet_spot_reconstruction_loss.png`](../src/experiments/gpt2_sae_training/results/gpt2_sweet_spot_reconstruction_loss.png)
- [`src/experiments/gpt2_sae_training/results/gpt2_sweet_spot_feature_death_rate.png`](../src/experiments/gpt2_sae_training/results/gpt2_sweet_spot_feature_death_rate.png)

**Purpose**: Layer × Position heatmaps for sweet spot config (d=512, K=150)

**Format**: 12 layers × 6 positions heatmaps showing:
1. Reconstruction Loss (MSE) - lower is better
2. Feature Death Rate (0-1) - lower is better

**Key Insights**:
- Position 0 consistently shows higher reconstruction loss in early layers
- Late layers (L8-L11) show near-zero feature death across all positions
- Odd positions (1,3,5) reconstruct better than even positions (0,2,4)

**Created**: 2025-10-27

---

### 18.6 Experiment Summary

**Dataset Pipeline**:
```
gpt2_predictions_1000.json (1.7 GB)
  ↓ convert_gpt2_data.py
gpt2_full_train_activations.pt (177 MB) + gpt2_full_val_activations.pt (44 MB)
  ↓ train_gpt2_grid.py (8 configs)
8 parameter sweep checkpoints
  ↓ analyze_results.py
Sweet spot identified: d=512, K=150
  ↓ train_sweet_spot_all_layers_positions.py
72 sweet spot checkpoints (all layers×positions)
  ↓ visualize_sweet_spot.py
2 heatmap visualizations
```

**Key Statistics**:
| Metric | Value |
|--------|-------|
| Total problems | 1,000 |
| Train problems | 800 (80%) |
| Val problems | 200 (20%) |
| Layers | 12 (GPT-2) |
| Positions | 6 (CODI) |
| Train samples | 57,600 |
| Val samples | 14,400 |
| Configs tested | 8 |
| Sweet spot SAEs | 72 (all layers×positions) |
| Total training time | ~5 minutes |

**Experiment**: GPT-2 TopK SAE Parameter Sweep

**Created**: 2025-10-27

**Documentation**: [`docs/experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md`](experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md)

----

## 19. GPT-2 Feature Interpretability Datasets

### 19.1 GPT-2 Extracted Features
**File**: [`src/experiments/gpt2_feature_interpretability/data/gpt2_extracted_features.pt`](../src/experiments/gpt2_feature_interpretability/data/gpt2_extracted_features.pt) ✅

**Purpose**: Feature activations extracted from all 72 GPT-2 TopK SAE checkpoints for interpretability analysis

**Size**: 142.4 MB

**Structure**:
```python
{
    'features': {
        (layer, position): torch.Tensor(N, 512),  # N=1000 problems
        # 72 keys: (0-11, 0-5)
    },
    'metadata': {
        'problem_ids': List[int],  # 1000 problem IDs
        'layers': List[int],       # Layer for each sample
        'positions': List[int],    # Position for each sample
        'num_samples': 1000,
        'num_saes': 72,
        'features_per_sae': 512
    }
}
```

**Key Stats**:
- Problems: 1,000 GSM8K problems
- SAEs: 72 (12 layers × 6 positions)
- Features per SAE: 512
- Total features: 36,864
- SAE config: d=512, K=150 (sweet spot)

**Used By**:
- Feature-token correlation analysis
- Monosemantic feature labeling
- Interactive dashboard

**Recreation**:
```bash
python src/experiments/gpt2_feature_interpretability/scripts/1_extract_features.py
# Runtime: ~5 minutes
```

---

### 19.2 GPT-2 CoT Tokens
**File**: [`src/experiments/gpt2_feature_interpretability/data/gpt2_cot_tokens.json`](../src/experiments/gpt2_feature_interpretability/data/gpt2_cot_tokens.json) ✅

**Purpose**: Parsed tokens from GSM8K ground truth calculation blocks for correlation analysis

**Size**: 154.6 KB

**Structure**:
```json
{
  "token_to_problems": {
    "50": [12, 45, 89, ...],
    "*": [3, 7, 15, ...],
    "=": [1, 2, 3, ...]
  },
  "problem_to_tokens": {
    "12": ["16", "7", "112", "*", "="],
    "45": ["50", "2", "100", "*", "="]
  },
  "metadata": {
    "unique_tokens": 590,
    "total_problems": 1000
  }
}
```

**Key Stats**:
- Unique tokens: 590
- Total problems: 1,000
- Token types: Numbers (0-50000), operators (+,-,*,/,=), parentheses

**Token Extraction**:
- Source: Ground truth `<<calculation>>` blocks in GSM8K solutions
- Example: `<<16*7=112>>` → tokens: ["16", "7", "112", "*", "="]

**Used By**:
- Chi-squared correlation analysis
- Feature labeling

**Recreation**:
```bash
python src/experiments/gpt2_feature_interpretability/scripts/2_parse_cot_tokens.py
# Runtime: <1 minute
```

---

### 19.3 GPT-2 Feature-Token Correlations
**File**: [`src/experiments/gpt2_feature_interpretability/data/gpt2_feature_token_correlations.json`](../src/experiments/gpt2_feature_interpretability/data/gpt2_feature_token_correlations.json) ✅

**Purpose**: Statistical correlations between SAE features and CoT tokens

**Size**: 19.5 MB

**Structure**:
```json
{
  "metadata": {
    "total_features": 36864,
    "features_analyzed": 26744,
    "interpretable_features": 15399,
    "interpretability_rate": 0.418,
    "total_correlations": 49748,
    "criteria": {
      "min_activations": 20,
      "p_value_threshold": 0.01,
      "enrichment_threshold": 2.0
    }
  },
  "correlations": {
    "0": {  // layer
      "3": {  // position
        "5": {  // feature_id
          "num_activations": 485,
          "activation_rate": 0.485,
          "num_correlations": 3,
          "correlations": [
            {
              "token": "50",
              "p_value": 0.002809,
              "enrichment": 4.042,
              "chi2": 8.957,
              "active_with_token": 101,
              "active_without_token": 384
            }
          ]
        }
      }
    }
  }
}
```

**Key Stats**:
- Features analyzed: 26,744 (≥20 activations)
- Interpretable features: 15,399 (41.8%)
- Total correlations: 49,748
- Chi-squared tests performed: 21.7 million (26,744 features × 590 tokens)

**Statistical Method**:
- Chi-squared test (p < 0.01)
- Enrichment score: P(token | feature active) / P(token | feature inactive)
- Minimum enrichment: 2.0 (token 2× more likely when feature active)

**Used By**:
- Monosemantic feature labeling
- Dashboard generation
- Model comparison

**Recreation**:
```bash
python src/experiments/gpt2_feature_interpretability/scripts/3_compute_correlations.py
# Runtime: ~19 minutes
```

---

### 19.4 GPT-2 Labeled Features
**File**: [`src/experiments/gpt2_feature_interpretability/data/gpt2_labeled_features.json`](../src/experiments/gpt2_feature_interpretability/data/gpt2_labeled_features.json) ✅

**Purpose**: Interpretable features with human-readable labels and monosemanticity classifications

**Size**: 17.5 MB

**Structure**:
```json
{
  "metadata": {
    "total_features": 15399,
    "monosemantic_features": 11187,
    "monosemantic_rate": 0.726,
    "category_counts": {
      "number": 10229,
      "polysemantic": 4212,
      "numbers": 874,
      "operator": 52
    }
  },
  "features": {
    "L4_P3_F241": {
      "layer": 4,
      "position": 3,
      "feature_id": 241,
      "label": "number_50000",
      "is_monosemantic": true,
      "explanation": "Strongly correlates with number 50000 (enrichment=169.9)",
      "num_activations": 201,
      "activation_rate": 0.201,
      "num_correlations": 1,
      "top_correlations": [
        {
          "token": "50000",
          "enrichment": 169.91,
          "p_value": 1.23e-50
        }
      ]
    }
  }
}
```

**Key Stats**:
- Total labeled features: 15,399
- Monosemantic features: 11,187 (72.6%)
- Number features: 10,229 (66.4%)
- Polysemantic features: 4,212 (27.4%)
- High enrichment (≥10.0): 6,596 features

**Labeling Criteria**:
1. Strong single correlation (enrichment ≥ 5.0) → monosemantic
2. Top 3 correlations same category → monosemantic
3. Composite patterns (operator + number) → monosemantic
4. Otherwise → polysemantic

**Used By**:
- Interactive dashboard
- Model comparison
- Feature analysis

**Recreation**:
```bash
python src/experiments/gpt2_feature_interpretability/scripts/4_label_features.py
# Runtime: ~1 minute
```

---

### 19.5 Model Comparison Summary
**File**: [`src/experiments/gpt2_feature_interpretability/data/model_comparison.json`](../src/experiments/gpt2_feature_interpretability/data/model_comparison.json) ✅

**Purpose**: Comparison of GPT-2 vs LLaMA feature interpretability with capacity hypothesis

**Size**: 5.2 KB

**Structure**:
```json
{
  "gpt2_analysis": {
    "model": "GPT-2",
    "model_size": "124M parameters",
    "monosemantic_rate": 0.726,
    "interpretability_rate": 0.418,
    "feature_type_distribution": {...},
    "top_enrichment_examples": [...]
  },
  "llama_framework": {
    "model": "LLaMA",
    "model_size": "1B parameters",
    "expected_monosemantic_rate": 0.50,
    "status": "Not yet analyzed",
    "next_steps": [...]
  },
  "insights": {
    "model_capacity_hypothesis": {
      "claim": "Smaller models require more monosemantic features",
      "interpretation": "GPT-2 uses denser, more specialized features; LLaMA distributes computation"
    }
  }
}
```

**Key Insights**:
- GPT-2 (124M): 72.6% monosemantic, 29.3% sparsity → specialized features
- LLaMA (1B): ~50% monosemantic (estimated), 19.5% sparsity → distributed redundancy
- Model capacity determines encoding strategy

**Used By**:
- Dashboard model comparison section
- Research documentation

**Recreation**:
```bash
python src/experiments/gpt2_feature_interpretability/scripts/5_compare_models.py
# Runtime: <1 minute
```

---

### 19.6 Interactive Dashboard
**File**: [`src/experiments/gpt2_feature_interpretability/dashboard.html`](../src/experiments/gpt2_feature_interpretability/dashboard.html) ✅

**Purpose**: Interactive HTML dashboard for exploring all 15,399 interpretable features

**Size**: 14.3 MB

**Features**:
- Browse all interpretable features in sortable table
- Filter by layer, position, type, monosemanticity
- Search for specific labels
- Click "View" to see detailed correlations in modal
- Model comparison summary section
- Statistics cards with key metrics

**Technology**: Standalone HTML/CSS/JavaScript (no dependencies)

**Access**: Open in browser at `file:///home/paperspace/dev/CoT_Exploration/src/experiments/gpt2_feature_interpretability/dashboard.html`

**Recreation**:
```bash
python src/experiments/gpt2_feature_interpretability/scripts/6_create_dashboard.py
# Runtime: ~2 minutes
```

---

### Summary

| Dataset | Purpose | Size | Features |
|---------|---------|------|----------|
| Extracted Features | Raw activations | 142.4 MB | 36,864 features |
| CoT Tokens | Token vocabulary | 154.6 KB | 590 unique tokens |
| Correlations | Statistical analysis | 19.5 MB | 49,748 correlations |
| Labeled Features | Interpretability catalog | 17.5 MB | 15,399 interpretable |
| Model Comparison | GPT-2 vs LLaMA | 5.2 KB | Hypothesis framework |
| Dashboard | Interactive visualization | 14.3 MB | 15,399 browsable |

**Total Pipeline Runtime**: ~30 minutes

**Experiment**: GPT-2 Feature Interpretability Catalog

**Created**: 2025-10-27

**Documentation**: [`docs/experiments/10-27_gpt2_gsm8k_feature_interpretability.md`](experiments/10-27_gpt2_gsm8k_feature_interpretability.md)

----

## 20. LLaMA TopK SAE Grid Experiment Datasets

### 20.1 LLaMA Activation Data (Train/Val)
**Files**:
- Training data extracted from existing CODI datasets (see Section 15)
- Position 3, Layer 14 activations (2048-dim)

**Purpose**: Continuous thought activations from LLaMA-3.2-1B CODI for TopK SAE grid search

**Size**:
- Train: 5,978 samples (from GSM8K training set)
- Val: 1,495 samples (from GSM8K test set)
- Dimensions: 2048 (hidden size of LLaMA)
- Position: 3 (middle of continuous thought)
- Layer: 14 (late layer)

**Source**: LLaMA-3.2-1B CODI model predictions on GSM8K

**Used By**:
- Initial grid search (12 SAEs)
- Multi-layer analysis (1,152 SAEs across all layers×positions)
- TopK vs Matryoshka comparison
- Feature semantics analysis

**Train/Val Split**: 80/20 split from original GSM8K dataset

**Recreation**:
```bash
# Activations extracted from existing CODI checkpoint
# See Section 15 for base activation extraction
# Grid experiment used Position 3, Layer 14 subset
```

**Stratification**: None - uses continuous activations from reasoning problems

**Experiment**: TopK SAE Grid Pilot (`src/experiments/topk_grid_pilot/`)

**Created**: 2025-10-26

**Documentation**:
- [`docs/experiments/10-26_llama_gsm8k_topk_sae_grid.md`](experiments/10-26_llama_gsm8k_topk_sae_grid.md)
- [`docs/experiments/10-26_llama_gsm8k_topk_sae_multilayer.md`](experiments/10-26_llama_gsm8k_topk_sae_multilayer.md)

----

### 19.2 Initial Grid Search SAE Checkpoints (12 Configs)
**Files**: [`src/experiments/topk_grid_pilot/results/checkpoints/pos3_d{512,1024,2048}_k{5,10,20,100}.pt`](../src/experiments/topk_grid_pilot/results/checkpoints/)

**Purpose**: Trained TopK SAE checkpoints for initial grid search (Position 3, Layer 14 only)

**Size**: 12 checkpoint files
- d=512 models: ~2.1 MB each (4 files)
- d=1024 models: ~4.2 MB each (4 files)
- d=2048 models: ~8.4 MB each (4 files)
- Total: ~58 MB

**Grid Configuration**:
- **K values**: {5, 10, 20, 100}
- **Latent dimensions**: {512, 1024, 2048}
- **Position**: 3 (continuous thought)
- **Layer**: 14 (late layer)

**Architecture**:
- Encoder: Linear(2048 → latent_dim)
- Decoder: Linear(latent_dim → 2048) with unit-norm columns
- Activation: TopK (exact K-sparsity, no L1 penalty)

**Training Details**:
- Epochs: 25
- Batch size: 256
- Optimizer: Adam (lr=1e-3)
- Loss: MSE reconstruction only
- Training time: 1.9-5.2s per SAE

**Performance Summary**:
| Config | EV | Death Rate | Active Features |
|--------|-----|------------|-----------------|
| d=512, K=5 | 70.2% | 85.5% | 74/512 |
| d=512, K=100 | 87.8% | 0.0% | 512/512 |
| d=1024, K=100 | 88.0% | 1.7% | 1,006/1,024 |
| d=2048, K=100 | 87.9% | 7.8% | 1,888/2,048 |

**Key Finding**: All 12 configurations are Pareto-optimal - K dominates quality, latent_dim provides marginal gains

**Experiment**: TopK SAE Grid Pilot

**Created**: 2025-10-26

**Documentation**: [`docs/experiments/10-26_llama_gsm8k_topk_sae_grid.md`](experiments/10-26_llama_gsm8k_topk_sae_grid.md)

----

### 19.3 Multi-Layer SAE Checkpoints (1,152 Configs)
**Files**: [`src/experiments/topk_grid_pilot/results/checkpoints/pos{0-5}_layer0_d{512,1024,2048}_k{5,10,20,100}.pt`](../src/experiments/topk_grid_pilot/results/checkpoints/)

**Purpose**: Comprehensive TopK SAE checkpoints across all layers and positions

**Size**: 1,152 checkpoint files
- 16 layers × 6 positions × 12 configs = 1,152 SAEs
- File sizes: 2.1-8.4 MB each (depending on latent_dim)
- Total: ~5-6 GB

**Grid Configuration**:
- **Layers**: 0-15 (all 16 LLaMA layers)
- **Positions**: 0-5 (all 6 CODI continuous thought positions)
- **K values**: {5, 10, 20, 100}
- **Latent dimensions**: {512, 1024, 2048}

**Training Details**:
- Same architecture as initial grid (Section 19.2)
- Parallel training on A100 80GB
- Total training time: ~30-40 minutes

**Key Findings**:
1. **Position pattern**: Early positions (0-2) reconstruct better than late positions (3-5)
2. **Layer pattern**: Early layers (0-5) reconstruct better than late layers (10-15)
3. **Universal sweet spot**: K=100, d=512 optimal across all layers/positions
4. **Explained variance range**: 75-92% across all 96 layer-position pairs

**Performance by Layer Group** (K=100, d=512):
| Layer Group | Avg EV | Avg Death Rate |
|-------------|--------|----------------|
| Early (0-5) | 90.5% | 2.1% |
| Mid (6-10) | 86.3% | 3.8% |
| Late (11-15) | 82.7% | 5.2% |

**Experiment**: TopK SAE Multi-Layer Analysis

**Created**: 2025-10-26

**Documentation**: [`docs/experiments/10-26_llama_gsm8k_topk_sae_multilayer.md`](experiments/10-26_llama_gsm8k_topk_sae_multilayer.md)

----

### 19.4 Grid Metrics Data
**Files**:
- [`src/experiments/topk_grid_pilot/results/data/grid_metrics_latent{512,1024,2048}.json`](../src/experiments/topk_grid_pilot/results/data/)
- [`src/experiments/topk_grid_pilot/results/data/grid_metrics_pos{0-5}_layer0_latent{512,1024,2048}.json`](../src/experiments/topk_grid_pilot/results/data/)

**Purpose**: Evaluation metrics for all trained TopK SAEs

**Size**: ~50 JSON files total
- Initial grid: 3 files (one per latent_dim)
- Multi-layer: 48 files (16 layers × 6 positions ÷ 2, due to partial coverage)
- Each file: ~5-20 KB

**Metrics Included**:
```json
{
  "config": {"latent_dim": 512, "k": 100, "position": 3, "layer": 14},
  "reconstruction": {
    "explained_variance": 0.878,
    "reconstruction_loss": 0.0527,
    "mean_squared_error": 0.0527
  },
  "sparsity": {
    "feature_death_rate": 0.0,
    "active_features": 512,
    "l0_norm": 100.0,
    "effective_sparsity": 0.195
  },
  "activations": {
    "mean_activation": 1.84,
    "max_activation": 10.87,
    "activation_std": 2.13
  },
  "training": {
    "epochs": 25,
    "batch_size": 256,
    "training_time_seconds": 2.1
  }
}
```

**Experiment**: TopK SAE Grid Pilot

**Created**: 2025-10-26

**Documentation**: [`docs/experiments/10-26_llama_gsm8k_topk_sae_grid.md`](experiments/10-26_llama_gsm8k_topk_sae_grid.md)

----

### 19.5 Visualization Data
**Files**: [`src/experiments/topk_grid_pilot/results/viz/`](../src/experiments/topk_grid_pilot/results/viz/)

**Purpose**: Heatmaps and plots for TopK SAE analysis

**Visualizations**:
1. `heatmap_explained_variance.png` - EV across K × latent_dim grid
2. `heatmap_feature_death_rate.png` - Feature death across grid
3. `heatmap_mean_activation.png` - Average feature magnitude
4. `heatmap_max_activation.png` - Peak feature magnitude
5. `heatmap_reconstruction_loss.png` - MSE loss across grid

**Size**: 5 PNG files, ~50-100 KB each

**Content**:
- X-axis: K values {5, 10, 20, 100}
- Y-axis: Latent dimensions {512, 1024, 2048}
- Color: Metric value (explained variance, death rate, etc.)

**Insights from Visualizations**:
- Clear K dominance: Vertical gradients (same latent_dim, different K)
- Weak latent_dim effect: Similar colors horizontally (same K, different latent_dim)
- Feature death inversely correlates with K

**Experiment**: TopK SAE Grid Pilot

**Created**: 2025-10-26

**Documentation**: [`docs/experiments/10-26_llama_gsm8k_topk_sae_grid.md`](experiments/10-26_llama_gsm8k_topk_sae_grid.md)

----

### 19.6 Summary Statistics

**Total Data Created**:
| Category | Count | Size |
|----------|-------|------|
| SAE checkpoints (initial) | 12 | ~58 MB |
| SAE checkpoints (multi-layer) | 1,152 | ~5-6 GB |
| Metrics JSON files | ~50 | ~500 KB |
| Visualization PNG files | 5 | ~300 KB |
| **Total** | **1,219 files** | **~6 GB** |

**Experiments Using This Data**:
1. Initial grid search (10-26f)
2. Multi-layer analysis (10-26g)
3. TopK vs Matryoshka comparison (10-27)
4. Feature semantics analysis (10-27)

**Key Scientific Contributions**:
- ✅ Characterized quality-sparsity tradeoff for TopK SAEs
- ✅ Identified universal sweet spot: K=100, d=512
- ✅ Proved K dominates quality over latent_dim
- ✅ Demonstrated 1.8-2.3× efficiency improvement over Matryoshka SAEs

----


## 16. LLaMA SAE Feature Hierarchy Datasets

### 16.1 Feature Taxonomy (Top 20 Features)
**File**: [`src/experiments/llama_sae_hierarchy/feature_labels_layer14_pos3.json`](../src/experiments/llama_sae_hierarchy/feature_labels_layer14_pos3.json)

**Purpose**: Ground truth feature interpretations for top-20 most frequent features in Layer 14, Position 3

**Size**: 20 features with complete metadata

**Structure**:
```json
{
  "metadata": {
    "layer": 14,
    "position": 3,
    "sae_config": "K=100, d=512",
    "num_features": 20
  },
  "features": [
    {
      "rank": 1,
      "feature_id": 449,
      "activation_freq": 0.9987,
      "interpretation": {
        "type": "mixed",
        "detected_patterns": ["addition", "multiplication", ...],
        "description": "General arithmetic feature"
      },
      "top_samples": [...]
    }
  ]
}
```

**Key Stats**:
- All 20 features classified as "mixed" (general-purpose, >87% activation)
- Activation frequency range: 87.8% - 99.9%
- All features activate on multiple operations and numbers

**Generation**:
```bash
python src/experiments/llama_sae_hierarchy/feature_taxonomy.py --layer 14 --position 3 --top_n 20
```

**Used By**: Feature validation experiments (Story 4), baseline for activation analysis (Story 2)

**Documentation**: [`docs/experiments/10-27_llama_gsm8k_feature_taxonomy.md`](experiments/10-27_llama_gsm8k_feature_taxonomy.md)

---

### 16.2 Activation Analysis - Mid-Frequency Features
**File**: [`src/experiments/llama_sae_hierarchy/activation_analysis_layer14_pos3_rank50-200.json`](../src/experiments/llama_sae_hierarchy/activation_analysis_layer14_pos3_rank50-200.json)

**Purpose**: Specialization analysis for mid-frequency features (rank 50-200, activation 11.6%-57.0%)

**Size**: 151 features with specialization scores

**Key Results**:
- 150 general features (99.3%)
- 1 operation-specialized feature (multiplication, 0.7%)
- 0 value-specialized features

**Generation**:
```bash
python src/experiments/llama_sae_hierarchy/analyze_activations.py --layer 14 --position 3 --start_rank 50 --end_rank 200
```

---

### 16.3 Activation Analysis - Rare Features (Most Important)
**File**: [`src/experiments/llama_sae_hierarchy/activation_analysis_layer14_pos3_rank400-512.json`](../src/experiments/llama_sae_hierarchy/activation_analysis_layer14_pos3_rank400-512.json)

**Purpose**: Specialization analysis for rare features where specialized features are most likely found

**Size**: 109 features analyzed, **5 specialized features identified**

**Key Results - Specialized Features**:
1. Feature 332 (rank 496): Addition specialist (0.3% activation)
2. Feature 194 (rank 505): Subtraction specialist (0.1% activation)
3. Feature 392 (rank 506): Addition + number 100 (0.1% activation) - **highly-specialized**
4. Feature 350 (rank 507): Addition + number 50 (0.1% activation) - **highly-specialized**
5. Feature 487 (rank 508): Addition + number 30 (0.1% activation) - **highly-specialized**

**Swap Pairs Generated**: 1 operation pair (addition ↔ subtraction)

**Major Finding**: Specialized features only appear in rare feature range (<3% activation)

**Generation**:
```bash
python src/experiments/llama_sae_hierarchy/analyze_activations.py --layer 14 --position 3 --start_rank 400 --end_rank 512
```

**Used By**: Validation experiments (Story 4+5), swap pair selection

**Documentation**: [`docs/experiments/10-27_llama_gsm8k_activation_patterns.md`](experiments/10-27_llama_gsm8k_activation_patterns.md)

---

### 16.4 Activation Analysis - Early Layer
**File**: [`src/experiments/llama_sae_hierarchy/activation_analysis_layer3_pos3_rank20-100.json`](../src/experiments/llama_sae_hierarchy/activation_analysis_layer3_pos3_rank20-100.json)

**Purpose**: Test hypothesis that early layers have more specialized features

**Size**: 81 features analyzed

**Key Results**:
- 81 general features (100%)
- 0 specialized features
- **Falsified hypothesis**: Early layers are MORE general, not more specialized

**Generation**:
```bash
python src/experiments/llama_sae_hierarchy/analyze_activations.py --layer 3 --position 3 --start_rank 20 --end_rank 100
```

---

### 16.5 Feature Validation Results
**File**: [`src/experiments/llama_sae_hierarchy/validation_results_layer14_pos3.json`](../src/experiments/llama_sae_hierarchy/validation_results_layer14_pos3.json)

**Purpose**: Causal validation of feature interpretations via ablation experiments

**Size**: Validation results for 10 general + 5 specialized features

**Structure**:
```json
{
  "metadata": {
    "layer": 14,
    "position": 3,
    "num_samples": 1495,
    "sanity_checks": {...}
  },
  "general_features": [
    {
      "rank": 1,
      "feature_id": 449,
      "mean_impact": 0.097888,
      "max_impact": 1.052734,
      "classification": "MEDIUM"
    }
  ],
  "specialized_features": [
    {
      "feature_id": 332,
      "specialization_type": "operation-specialized",
      "mean_impact": 0.000036,
      "classification": "MINIMAL"
    }
  ]
}
```

**Key Results**:
- General features: 3 HIGH impact, 7 MEDIUM impact, 0 LOW
- Specialized features: All MINIMAL impact (expected due to rarity)
- Validation: ✅ General features validated, ⊘ Specialized features inconclusive

**Generation**:
```bash
python src/experiments/llama_sae_hierarchy/validate_features.py --layer 14 --position 3 --top_n 10
```

**Used By**: Final results analysis, architecture validation

**Documentation**: [`docs/experiments/10-27_llama_gsm8k_feature_hierarchy.md`](experiments/10-27_llama_gsm8k_feature_hierarchy.md)

---

### 16.6 Summary Statistics

**Total Analysis Coverage**:
- Features analyzed: 361 (70.5% of 512 total features)
- Specialized features found: 6 (1.8%)
- Activation frequency spectrum: 0.1% - 99.9%

**Dataset Sizes**:
- Feature labels: ~50 KB
- Activation analyses: ~200 KB total (3 files)
- Validation results: ~15 KB
- **Total**: ~265 KB (all JSON)

**Experiments Using This Data**:
1. Feature taxonomy (10-27c Story 1)
2. Activation pattern analysis (10-27c Story 2)
3. Causal validation experiments (10-27c Stories 4+5)
4. Feature hierarchy investigation (10-27c Complete)

**Key Scientific Contributions**:
- ✅ Characterized feature hierarchy in TopK SAEs (1.8% specialized)
- ✅ Discovered specialization inverse correlation with activation frequency
- ✅ Validated general feature importance via ablation (impact: 0.075-0.118)
- ✅ Falsified "early layer specialization" hypothesis
- ❌ Refuted feasibility of swap experiments with rare specialized features
- ✅ Demonstrated no pure value-specific features (values contextualized in operations)

**Replication**:
```bash
# Run complete analysis pipeline
python src/experiments/llama_sae_hierarchy/feature_taxonomy.py --layer 14 --position 3
python src/experiments/llama_sae_hierarchy/analyze_activations.py --layer 14 --position 3 --start_rank 50 --end_rank 200
python src/experiments/llama_sae_hierarchy/analyze_activations.py --layer 14 --position 3 --start_rank 400 --end_rank 512
python src/experiments/llama_sae_hierarchy/analyze_activations.py --layer 3 --position 3 --start_rank 20 --end_rank 100
python src/experiments/llama_sae_hierarchy/validate_features.py --layer 14 --position 3
```

**Time to Generate**: ~3 hours (automated)

---

---

## 21. Attention Flow Analysis Datasets

### 21.1 Attention Flow Training Sample (100 problems)
**File**: [`src/experiments/codi_attention_flow/data/attention_dataset_100_train.json`](../src/experiments/codi_attention_flow/data/attention_dataset_100_train.json)

**Purpose**: Random sample of 100 GSM8K training problems for attention pattern extraction and critical head identification

**Size**: 100 problems (63 KB)

**Status**: ✅ Generated

**Structure**:
```json
[
  {
    "gsm8k_id": "train_53",
    "question": "Problem text from GSM8K...",
    "answer": "42",
    "full_solution": "Step-by-step solution with #### 42"
  }
]
```

**Sampling Method**:
- Source: GSM8K training set (7,473 problems total)
- Method: Random sampling with seed=42
- No stratification (pure random sample)
- Reproducible: same seed produces same sample

**Usage**:
- Story 1.2: Extract 6×6 attention matrices between continuous thought positions
- Story 1.3-1.5: Identify hub positions, flow patterns, skip connections
- Story 2.1-2.6: Rank critical attention heads, compare GPT-2 vs LLaMA

**Recreation**:
```bash
cd /home/paperspace/dev/CoT_Exploration
python src/experiments/codi_attention_flow/scripts/1_sample_dataset.py --seed 42 --n_samples 100
```

**Related Experiments**:
- Used in: Attention flow analysis (Phase 1-2)
- Models: LLaMA-3.2-1B, GPT-2-124M
- Documented in: `docs/experiments/10-27_llama_gsm8k_attention_flow_analysis.md`

**Validation**:
- ✅ No duplicates: 100 unique IDs
- ✅ All questions non-empty (min length: 85 chars)
- ✅ All answers present
- ✅ All IDs from training set (train_53 to train_7308)

---

## 22. Step-by-Step SAE Intervention Datasets

### 22.1 CODI450 Dataset (Frozen GSM8K Train/Test Splits)
**Files**:
- [`data/step_by_step/gsm8k_codi400_train.csv`](../data/step_by_step/gsm8k_codi400_train.csv)
- [`data/step_by_step/gsm8k_codi50_test.csv`](../data/step_by_step/gsm8k_codi50_test.csv)

**Purpose**: Deterministic GSM8K subsets for high-quality SAE training (400 items) and stable intervention testing (50 items)

**Size**: 450 problems total (400 train, 50 test)

**Status**: ⚠️ To be generated (Story 1)

**Structure** (same for both train and test):
```csv
id,question,answer,split,difficulty,baseline_rank
gsm8k_test_042,"Question text...",42,test,easy,1
gsm8k_test_157,"Question text...",157,test,decompose,5
...
```

**Columns**:
- `id`: Unique GSM8K test item ID
- `question`: Problem text
- `answer`: Numerical answer
- `split`: Always "test" (from GSM8K test set)
- `difficulty`: "easy" | "decompose" (tagged via CODI baseline rank)
- `baseline_rank`: Rank of correct answer in CODI baseline logits (1=top)

**Sampling Method**:
- Source: GSM8K test set (1,319 problems total)
- Method: Deterministic sampling with seed=42
- Stratification:
  - **Train (400)**: 200 easy (baseline_rank=1) + 200 decompose (baseline_rank>1)
  - **Test (50)**: 25 easy + 25 decompose
- Tagging heuristic: CODI baseline rank (primary) or arithmetic operation count (fallback)

**Recreation**:
```bash
cd /home/paperspace/dev/CoT_Exploration
python src/experiments/step_by_step/scripts/create_codi450_dataset.py --seed 42
```

**Used By**:
- SAE training (Story 2): 400-item train set for high-quality per-step SAEs
- Intervention smoke test (Story 4): 50-item test set for feature interventions
- Feature analysis (Story 5): Visualization and attribution

**Validation**:
- 450 unique items total (no duplicates)
- Train: Exactly 200 easy + 200 decompose
- Test: Exactly 25 easy + 25 decompose
- All items from GSM8K test split (no train leakage)
- Manifest SHA256 hashes match both CSVs

---

### 22.2 CODI450 Manifest
**File**: [`data/step_by_step/gsm8k_codi450.manifest.json`](../data/step_by_step/gsm8k_codi450.manifest.json)

**Purpose**: Metadata for reproducibility and validation

**Status**: ⚠️ To be generated (Story 1)

**Structure**:
```json
{
  "created": "2025-10-28",
  "seed": 42,
  "total_items": 450,
  "train_items": 400,
  "test_items": 50,
  "train_splits": {
    "easy": 200,
    "decompose": 200
  },
  "test_splits": {
    "easy": 25,
    "decompose": 25
  },
  "heuristic": "codi_baseline_rank",
  "baseline_checkpoint": "/home/paperspace/codi_ckpt/gpt2_gsm8k/pytorch_model.bin",
  "sha256_train": "abc123...",
  "sha256_test": "def456...",
  "gsm8k_version": "1.0",
  "source_split": "test"
}
```

**Validation Fields**:
- `sha256_train`, `sha256_test`: Hash of CSV content (for integrity checks)
- `seed`: Random seed (42)
- `baseline_checkpoint`: Path to CODI model used for tagging

**Usage**: Dataset versioning, reproducibility validation, citation in papers

---

### 22.3 CODI450 Baseline Cache
**File**: [`data/step_by_step/gsm8k_codi450_baseline_cache.pt`](../data/step_by_step/gsm8k_codi450_baseline_cache.pt)

**Purpose**: Cached CODI model outputs (logits, ranks) for 450 items to avoid recomputation

**Size**: ~22 MB (450 items × vocab_size logits)

**Status**: ⚠️ To be generated (Story 1)

**Structure** (PyTorch dict):
```python
{
  "gsm8k_test_042": {
    "logits": torch.Tensor([vocab_size]),  # Raw logits
    "rank": 1,  # Rank of correct answer
    "top5_tokens": ["42", "43", "41", ...],
    "probs": torch.Tensor([vocab_size])  # Softmax probabilities
  },
  # ... 450 items total
}
```

**Usage**:
- Story 1: Difficulty tagging (easy vs decompose) for all 450 items
- Story 4: Pre-intervention baseline comparison
- Story 5: Visualization (rank improvement calculations)

**Recreation**:
```bash
python src/experiments/step_by_step/scripts/create_baseline_cache.py \
  --train data/step_by_step/gsm8k_codi400_train.csv \
  --test data/step_by_step/gsm8k_codi50_test.csv \
  --checkpoint /home/paperspace/codi_ckpt/gpt2_gsm8k/pytorch_model.bin
```

---

### 22.4 Per-Step SAE Models (6 models)
**Files**:
- [`models/step_by_step/sae_step0/sae.pt`](../models/step_by_step/sae_step0/sae.pt)
- [`models/step_by_step/sae_step1/sae.pt`](../models/step_by_step/sae_step1/sae.pt)
- [`models/step_by_step/sae_step2/sae.pt`](../models/step_by_step/sae_step2/sae.pt)
- [`models/step_by_step/sae_step3/sae.pt`](../models/step_by_step/sae_step3/sae.pt)
- [`models/step_by_step/sae_step4/sae.pt`](../models/step_by_step/sae_step4/sae.pt)
- [`models/step_by_step/sae_step5/sae.pt`](../models/step_by_step/sae_step5/sae.pt)

**Purpose**: Sparse autoencoder dictionaries trained on CODI latent representations at each reasoning step

**Size**: ~16 MB per model (512 input × 8192 dictionary × 2 matrices)

**Status**: ⚠️ To be generated (Story 2)

**Architecture**:
```python
SparseAutoencoder(
  input_dim=512,      # CODI latent dimension
  dict_size=8192,     # 16× expansion
  tied_weights=False
)
```

**Training Configuration**:
- Dataset: CODI400 train (400 items)
- L1 coefficient: 0.005 (validated from Section 18 experiments)
- Optimizer: Adam (lr=1e-4)
- Epochs: 100 (early stopping on validation MSE)
- Seed: 42
- Batch size: 64

**Quality Thresholds**:
- Reconstruction MSE: <0.2 (normalized activations)
- Sparsity: 10–80 active features/sample (L0 norm)
- Dead features: <20% of dictionary (improved with 400 training samples)

**Used By**:
- Story 3: Intervention hook (loads SAE for feature editing)
- Story 4: Smoke test sweep (applies interventions)
- Story 5: Feature attribution (identifies top features)

**Recreation**:
```bash
python src/experiments/step_by_step/scripts/train_step_saes.py \
  --dataset data/step_by_step/gsm8k_codi400_train.csv \
  --config configs/step_by_step/sae_training.yaml
```

---

### 22.5 SAE Training Statistics
**Files**:
- `models/step_by_step/sae_step{k}/train_stats.json` (k=0 to 5)
- `models/step_by_step/sae_summary.csv`

**Purpose**: Training metrics and quality validation for each SAE

**Status**: ⚠️ To be generated (Story 2)

**Structure** (train_stats.json):
```json
{
  "step": 3,
  "final_recon_mse": 0.152,
  "final_sparsity": 0.92,
  "dead_features": 1200,
  "dead_feature_pct": 14.6,
  "training_time_sec": 1234,
  "epochs_trained": 87,
  "early_stopped": true,
  "config": {
    "l1_coeff": 0.001,
    "dict_size": 8192,
    "learning_rate": 0.0001
  }
}
```

**Summary CSV**:
```csv
step,recon_mse,sparsity,dead_features,dead_pct,training_time
0,0.145,0.91,1150,14.0,1205
1,0.158,0.89,1300,15.9,1187
...
```

**Usage**: Quality validation, model selection, paper reporting

---

### 22.6 SAE Feature Exemplars
**Files**: `models/step_by_step/sae_step{k}/feature_exemplars.parquet` (k=0 to 5)

**Purpose**: Max-activating examples for each SAE feature (for interpretability)

**Size**: ~2 MB per step (8192 features × 5 exemplars)

**Status**: ⚠️ To be generated (Story 2)

**Structure**:
```
feature_id | item_id         | activation | question
------------------------------------------------------
0          | gsm8k_test_042  | 3.45       | "Question..."
0          | gsm8k_test_157  | 2.87       | "Question..."
...
8191       | gsm8k_test_999  | 0.12       | "Question..."
```

**Columns**:
- `feature_id`: SAE dictionary feature (0 to 8191)
- `item_id`: GSM8K problem ID
- `activation`: Feature activation magnitude
- `question`: Problem text (for human inspection)

**Usage**:
- Story 2: Feature analysis (what does feature X respond to?)
- Story 5: Visualization (show exemplars for top causal features)
- Future: Manual feature labeling

---

### 22.7 Top Features by Step
**Files**: `models/step_by_step/sae_step{k}/top_features.json` (k=0 to 5)

**Purpose**: Ranked list of most important features per step (by max activation)

**Status**: ⚠️ To be generated (Story 2)

**Structure**:
```json
{
  "step": 3,
  "top_features": [
    {"feature_id": 1547, "max_activation": 4.23, "mean_activation": 1.12},
    {"feature_id": 892, "max_activation": 3.87, "mean_activation": 0.98},
    ...
  ]
}
```

**Usage**:
- Story 4: Feature selection for smoke test (pick top-K features)
- Story 5: Feature importance visualization

---

### 22.8 Intervention Results (Smoke Test)
**File**: [`results/step_by_step/smoke_test_results.csv`](../results/step_by_step/smoke_test_results.csv)

**Purpose**: Raw intervention experiment results (feature × Δ × item grid)

**Size**: ~450 rows (3 features × 3 steps × 5 Δ values × 10 items)

**Status**: ⚠️ To be generated (Story 4)

**Structure**:
```csv
item_id,step,feature_id,delta,pre_rank,post_rank,rank_delta,answer_changed,answer_correct
gsm8k_test_042,2,1547,0.5,5,1,-4,True,True
gsm8k_test_042,2,1547,0.3,5,3,-2,True,False
gsm8k_test_042,2,1547,0.0,5,5,0,False,False
...
```

**Columns**:
- `item_id`: GSM8K problem ID
- `step`: CODI latent step (0-5)
- `feature_id`: SAE feature intervened on
- `delta`: Intervention magnitude (-0.6 to +0.6)
- `pre_rank`: Answer rank before intervention
- `post_rank`: Answer rank after intervention
- `rank_delta`: post_rank - pre_rank (negative = improvement)
- `answer_changed`: Boolean (rank changed)
- `answer_correct`: Boolean (post_rank == 1)

**Usage**:
- Story 4: Identify promising features
- Story 5: Heatmap visualizations, statistical analysis

**Recreation**:
```bash
python src/experiments/step_by_step/scripts/run_smoke_test.py \
  --dataset data/step_by_step/gsm8k_codi50.csv \
  --config configs/step_by_step/intervention.yaml
```

---

### 22.9 Intervention Summary Statistics
**File**: [`results/step_by_step/smoke_test_summary.csv`](../results/step_by_step/smoke_test_summary.csv)

**Purpose**: Aggregated intervention effects per feature

**Status**: ⚠️ To be generated (Story 4)

**Structure**:
```csv
feature_id,step,mean_rank_improvement,flip_to_correct_rate,flip_to_wrong_rate,n_interventions
1547,2,2.4,0.35,0.05,50
892,3,1.8,0.22,0.08,50
...
```

**Columns**:
- `mean_rank_improvement`: Average rank improvement (negative delta)
- `flip_to_correct_rate`: % interventions that moved answer to rank 1
- `flip_to_wrong_rate`: % interventions that moved correct answer down
- `n_interventions`: Number of tests run

**Usage**: Identify top-3 most impactful features for future experiments

---

### 22.10 Intervention Logs (JSONL)
**File**: [`logs/step_by_step/interventions.jsonl`](../logs/step_by_step/interventions.jsonl)

**Purpose**: Structured log of all interventions (richer than CSV)

**Size**: ~500 KB (450 interventions with metadata)

**Status**: ⚠️ To be generated (Story 4)

**Structure** (one JSON object per line):
```json
{"item_id": "gsm8k_test_042", "step": 2, "feature_id": 1547, "delta": 0.5, "mode": "direction_space", "edit_norm": 0.23, "pre_rank": 5, "post_rank": 1, "rank_delta": -4, "answer_changed": true, "answer_correct": true, "timestamp": "2025-10-28T14:32:01"}
```

**Additional Fields**:
- `mode`: "feature_space" | "direction_space"
- `edit_norm`: L2 norm of applied intervention
- `timestamp`: ISO 8601 timestamp

**Usage**: Debugging, reproducibility, audit trail

---

### 22.11 Visualizations
**Files**:
- `results/step_by_step/visualizations/heatmap_feature_{id}.png`
- `results/step_by_step/visualizations/step_comparison.png`
- `results/step_by_step/visualizations/feature_importance.png`
- `results/step_by_step/visualizations/control_validation.png`
- `results/step_by_step/visualizations/sae_quality.png`
- `results/step_by_step/visualizations/dashboard.html`

**Purpose**: Publication-ready plots and interactive dashboard

**Status**: ⚠️ To be generated (Story 5)

**Plots**:
1. **Heatmap**: Feature × Δ → rank improvement (color-coded)
2. **Step Comparison**: Box plots of rank deltas across steps
3. **Feature Importance**: Bar chart of top-10 features by mean |rank_delta|
4. **Control Validation**: Wrong-step vs target-step effect sizes
5. **SAE Quality**: Reconstruction MSE vs sparsity per step
6. **Dashboard**: Lightweight HTML with all plots embedded

**Usage**: Papers, presentations, exploratory analysis

---

## Dataset Relationships (Step-by-Step)

```
GSM8K Test (1,319)
       ↓ sample(seed=42, n=450)
CODI450 Dataset (450)
       ├─ CODI400 Train (400)
       │    ↓ tag(baseline_rank)
       │  200 Easy + 200 Decompose
       │    ↓ extract_activations(K=6 steps)
       │  Activation Tensors × 6 × 400
       │    ↓ train_sae(dict_size=8192)
       │  SAE Models × 6
       │    ↓ select_top_features(k=3/step)
       │  9 Features (3 per step)
       │
       └─ CODI50 Test (50)
            ↓ tag(baseline_rank)
          25 Easy + 25 Decompose
            ↓ intervene(features=9, deltas=5, items=50)
          450 Intervention Results (9×5×10 target items)
            ↓ aggregate()
          Summary Statistics + Visualizations
```

---

## Critical Configuration Files

### Dataset Config
**File**: [`configs/step_by_step/dataset.yaml`](../configs/step_by_step/dataset.yaml)
**Status**: ⚠️ To be created (Story 0)

### SAE Training Config
**File**: [`configs/step_by_step/sae_training.yaml`](../configs/step_by_step/sae_training.yaml)
**Status**: ⚠️ To be created (Story 0)

### Intervention Config
**File**: [`configs/step_by_step/intervention.yaml`](../configs/step_by_step/intervention.yaml)
**Status**: ⚠️ To be created (Story 0)

### Main Pipeline Config
**File**: [`configs/step_by_step/main.yaml`](../configs/step_by_step/main.yaml)
**Status**: ⚠️ To be created (Story 0)

---

## Experiments Using Step-by-Step Datasets

| Experiment | Datasets Used | Models | Documented In |
|------------|---------------|--------|---------------|
| SAE Intervention Pilot | CODI50, SAE models, Intervention results | GPT-2 or LLaMA | `docs/experiments/MM-DD_model_gsm8k_sae_intervention.md` (future) |

---

## Storage Estimates

| Dataset | Size | Git Tracked |
|---------|------|-------------|
| CODI400 Train CSV | <200 KB | ✅ Yes |
| CODI50 Test CSV | <25 KB | ✅ Yes |
| CODI450 Manifest | <5 KB | ✅ Yes |
| Baseline Cache | ~22 MB | ❌ No (.gitignore) |
| SAE Models (6×) | ~96 MB | ❌ No (.gitignore) |
| Feature Exemplars (6×) | ~12 MB | ❌ No (.gitignore) |
| Intervention Results | <1 MB | ✅ Yes |
| Visualizations | <5 MB | ✅ Yes (PNGs) |
| **Total** | **~136 MB** | **~6 MB tracked** |

---

## Validation Checklist (Step-by-Step)

### Dataset Integrity
- [ ] CODI400 train has exactly 400 items (200 easy, 200 decompose)
- [ ] CODI50 test has exactly 50 items (25 easy, 25 decompose)
- [ ] Manifest SHA256 hashes match both CSVs
- [ ] All items from GSM8K test split (no train leakage)
- [ ] Baseline cache has entries for all 450 items
- [ ] No duplicate questions across train and test
- [ ] Difficulty split validated (easy=rank1, decompose>rank1)

### SAE Quality
- [ ] All 6 SAE models trained successfully on 400 samples
- [ ] Reconstruction MSE <0.2 for all steps
- [ ] Sparsity 0.80–0.95 for all steps
- [ ] Dead features <20% for all steps (improved threshold)
- [ ] Feature exemplars extracted for all 8192 features/step

### Intervention Validity
- [ ] 450+ interventions logged
- [ ] Zero-delta controls show rank_delta ≈ 0
- [ ] Wrong-step controls show |rank_delta| < target-step
- [ ] At least 1 feature shows mean rank improvement >2
- [ ] Flip-to-correct rate >20% for at least 1 feature

### Reproducibility
- [ ] All configs version-controlled (git)
- [ ] Seeds documented in manifest
- [ ] WandB run IDs logged
- [ ] Recreation scripts tested

---


---

## 23. LLaMA Feature Interpretability Datasets

**Purpose**: Comprehensive feature interpretability analysis on LLaMA-3.2-1B to compare with GPT-2 and test capacity hypothesis

**Experiment**: Feature-token correlation analysis using chi-squared tests

**Status**: ✅ Complete (2025-10-28)

### 23.1 LLaMA Extracted Features
**File**: [`src/experiments/llama_feature_interpretability/data/llama_extracted_features.pt`](../src/experiments/llama_feature_interpretability/data/llama_extracted_features.pt)

**Purpose**: Features extracted from all 96 LLaMA SAEs (16 layers × 6 positions)

**Size**: 195.9 MB

**Samples**: 96,000 (1,000 problems × 96 SAEs)

**Config**:
- SAE: K=100, d=512 (sweet spot)
- Sparsity: 19.5%
- Source: `src/experiments/topk_grid_pilot/results/checkpoints/`

**Generation**:
```bash
python src/experiments/llama_feature_interpretability/scripts/1_extract_features.py
```

---

### 23.2 LLaMA CoT Tokens
**File**: [`src/experiments/llama_feature_interpretability/data/llama_cot_tokens.json`](../src/experiments/llama_feature_interpretability/data/llama_cot_tokens.json)

**Purpose**: Parsed calculation tokens from LLaMA's CoT sequences

**Size**: 436.7 KB

**Tokens**: 628 unique tokens

**Avg per problem**: 16.9 tokens

**Generation**:
```bash
python src/experiments/llama_feature_interpretability/scripts/2_parse_cot_tokens.py
```

---

### 23.3 LLaMA Feature-Token Correlations
**File**: [`src/experiments/llama_feature_interpretability/data/llama_feature_token_correlations.json`](../src/experiments/llama_feature_interpretability/data/llama_feature_token_correlations.json)

**Purpose**: Statistical correlations between features and CoT tokens

**Size**: 23.7 MB

**Features analyzed**: 31,057 (63.2% of 49,152 total)

**Interpretable features**: 18,551 (37.7%)

**Total correlations**: 60,296

**Criteria**: p < 0.01, enrichment ≥ 2.0, min 20 activations

**Runtime**: ~24.5 minutes on A100

**Generation**:
```bash
python src/experiments/llama_feature_interpretability/scripts/3_compute_correlations.py
```

---

### 23.4 LLaMA Labeled Features
**File**: [`src/experiments/llama_feature_interpretability/data/llama_labeled_features.json`](../src/experiments/llama_feature_interpretability/data/llama_labeled_features.json)

**Purpose**: Human-readable labels for monosemantic features

**Size**: 21.4 MB

**Labeled features**: 18,551

**Monosemantic**: 13,890 (74.9%)

**Polysemantic**: 4,661 (25.1%)

**Labeling criteria**: Enrichment ≥ 5.0 OR top 3 correlations same category

**Generation**:
```bash
python src/experiments/llama_feature_interpretability/scripts/4_label_features.py
```

---

### 23.5 Model Comparison
**File**: [`src/experiments/llama_feature_interpretability/data/model_comparison.json`](../src/experiments/llama_feature_interpretability/data/model_comparison.json)

**Purpose**: Direct comparison of LLaMA vs GPT-2 feature interpretability

**Size**: 7.9 KB

**Key findings**:
- LLaMA monosemantic rate: 74.9% (vs GPT-2: 72.6%)
- Number features: 98.9% (vs GPT-2: 98.5%)
- Max enrichment: 195.0× (vs GPT-2: 169.9×)
- **Capacity hypothesis**: REJECTED

**Generation**:
```bash
python src/experiments/llama_feature_interpretability/scripts/5_compare_models.py
```

---

### 23.6 Interactive Dashboard
**File**: [`src/experiments/llama_feature_interpretability/visualizations/dashboard.html`](../src/experiments/llama_feature_interpretability/visualizations/dashboard.html)

**Purpose**: Interactive exploration of top features

**Size**: 89.9 KB

**Features shown**: Top 100 by enrichment + category breakdowns

**Generation**:
```bash
python src/experiments/llama_feature_interpretability/scripts/6_create_dashboard.py
```

---

### Dataset Summary

| File | Size | Samples/Features | Purpose |
|------|------|-----------------|---------|
| llama_extracted_features.pt | 195.9 MB | 96,000 samples | Feature extraction |
| llama_cot_tokens.json | 436.7 KB | 628 tokens | Token vocabulary |
| llama_feature_token_correlations.json | 23.7 MB | 18,551 features | Correlations |
| llama_labeled_features.json | 21.4 MB | 13,890 monosemantic | Labels |
| model_comparison.json | 7.9 KB | 2 models | Comparison |
| dashboard.html | 89.9 KB | Top 100 features | Visualization |

**Total**: ~241 MB

---

## 24. Liars-Bench Deception Detection Datasets

###Overview
**Experiment**: Sprint 1 & 4 - Cross-scale deception detection analysis (Oct 25-28, 2025)
**Models**: GPT-2 (124M) and LLaMA-3.2-3B (3B)
**Purpose**: Test if continuous thoughts can detect deception; compare to response tokens; test scale effects
**Key Finding**: Continuous thoughts CANNOT detect deception (50% = random) at any scale; response tokens achieve 70.49%

**Location**: `src/experiments/liars_bench_codi/`

---

### 24.1 Proper Question-Level Splits (ZERO Overlap)

**Base Dataset**: Liars-Bench Instructed Deception
- **Source**: 960 unique questions, 20,798 total examples (honest + deceptive pairs)
- **Split methodology**: 70% CODI train / 15% probe train / 15% probe test (question-level, zero overlap)

#### CODI Training Set
**File**: [`src/experiments/liars_bench_codi/data/processed/liars_bench_instructed_deception/train_proper.json`](../src/experiments/liars_bench_codi/data/processed/liars_bench_instructed_deception/train_proper.json) ✅

**Purpose**: Train CODI models (GPT-2 and LLaMA-3.2-3B)

**Size**:
- 672 unique questions
- 13,440 examples (6,720 honest + 6,720 deceptive)
- Balanced 50/50 distribution

**Generation**:
```bash
cd src/experiments/liars_bench_codi
python scripts/create_proper_splits.py
```

---

#### Probe Training Set
**File**: [`src/experiments/liars_bench_codi/data/processed/liars_bench_instructed_deception/probe_train_proper.json`](../src/experiments/liars_bench_codi/data/processed/liars_bench_instructed_deception/probe_train_proper.json) ✅

**Purpose**: Train linear probes to classify honest vs deceptive from activations

**Size**:
- 144 unique questions (DISJOINT from CODI train and probe test)
- 288 examples (144 honest + 144 deceptive)

---

#### Probe Test Set
**File**: [`src/experiments/liars_bench_codi/data/processed/liars_bench_instructed_deception/probe_test_proper.json`](../src/experiments/liars_bench_codi/data/processed/liars_bench_instructed_deception/probe_test_proper.json) ✅

**Purpose**: Evaluate probes on completely unseen questions (true generalization test)

**Size**:
- 144 unique questions (DISJOINT from all training data)
- 288 examples (144 honest + 144 deceptive)

---

### 24.2 Activation & Results Files

| File | Size | Purpose | Key Result |
|------|------|---------|------------|
| probe_activations_gpt2_proper.json | ~15 MB | GPT-2 continuous thoughts | 50% accuracy |
| probe_activations_response_proper.json | ~3 MB | GPT-2 response tokens | 70.49% accuracy |
| probe_activations_llama3b_proper.json | ~45 MB | LLaMA continuous thoughts | 50% accuracy |
| probe_results_gpt2_proper_v2.json | ~20 KB | GPT-2 probe results | Random chance |
| probe_results_response_proper.json | ~5 KB | Response probe results | Superior |
| probe_results_llama3b_proper.json | ~30 KB | LLaMA probe results | Scale doesn't help |

**Total**: ~147 MB

**Key Takeaway**: Continuous thoughts (50% accuracy) cannot detect deception at any scale. Response tokens (70.49%) are superior. This is a fundamental encoding limitation, not a capacity issue.

---
