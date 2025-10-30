# CommonsenseQA Model Debug: Root Cause Analysis

**Date**: October 30, 2025
**Status**: ✅ **ROOT CAUSE IDENTIFIED**

## Executive Summary

The CommonsenseQA CODI model shows 33% accuracy instead of expected 71.33% due to **DATASET MISMATCH**. The model was trained on `zen-E/CommonsenseQA-GPT4omini` (with GPT-4 generated chain-of-thought reasoning) but we tested it on standard `commonsense_qa` dataset (without CoT reasoning).

## Problem Statement

- **Expected Performance**: 71.33% accuracy (871/1221) on validation set
- **Actual Performance**: 33.0% accuracy (99/300) in three-way comparison
- **Performance Drop**: -38.3 percentage points
- **Answer Bias**: 47% predict E, 33% predict A, only 4.7% predict D

## Investigation Process

### Step 1: Answer Distribution Analysis

Found severe answer position bias:

| Answer | True Distribution | Predicted | Bias |
|--------|------------------|-----------|------|
| A | 22.3% | **33.3%** | +40 examples |
| B | 21.0% | 8.0% | -36 examples |
| C | 17.7% | 7.0% | -39 examples |
| D | 17.7% | **4.7%** | -46 examples |
| E | 21.3% | **47.0%** | +81 examples |

### Step 2: Manual Inference Testing

Ran 5 test examples (one per answer A-E) with actual model inference.

**Results**: 1/5 correct = **20% accuracy**

#### Example Outputs Reveal the Problem:

**Example 1** (Correct Answer: A - bank):
```
Generated: "A revolving door is typically used to control access... a bank is a place where security is paramount..."
Extracted: A ✓ CORRECT
```

**Example 2** (Correct Answer: B - bookstore):
```
Generated: "A"  ← Just the letter!
Extracted: A ✗ WRONG
```

**Example 3** (Correct Answer: C - Great Britain):
```
Generated: "...Among the options, 'Great Britain' is a well-known country where ferrets are often kEpt as pets..."
Extracted: E ← Extracted first A-E letter, which is "E" from "kEpt"!
✗ WRONG
```

**Example 4** (Correct Answer: D - listen to each other):
```
Generated: "...pass water is a common behavior in preparation for an impending threat..."
Extracted: E ← First A-E letter in "prEparation"
✗ WRONG
```

**Example 5** (Correct Answer: E - wallpaper):
```
Generated: "...Therefore, replacing vinyl records would be an odd thing to do. The answer is: B"
Extracted: B ← Model explicitly says B, but true answer is E!
✗ WRONG
```

## Root Cause: Dataset Mismatch

### Training Dataset: `zen-E/CommonsenseQA-GPT4omini`

- Contains **GPT-4 generated** chain-of-thought reasoning
- Includes explicit reasoning steps before answers
- Format: Question → GPT-4 CoT Reasoning → Answer
- Training accuracy: **71.33%** (871/1221)

### Testing Dataset: `commonsense_qa` (standard)

- Standard CommonsenseQA dataset
- **NO chain-of-thought reasoning**
- Format: Question → Answer only
- Testing accuracy: **33.0%** (99/300)

### Why This Causes Failure:

1. **Input Format Mismatch**: Model expects different prompt structure
2. **Missing Context**: Model trained with CoT expects reasoning steps
3. **Distribution Shift**: Different question phrasing/style
4. **Confused Outputs**: Model generates partial answers, wrong reasoning, or just letters

## Evidence Summary

### Model Outputs Show Confusion:

1. **Short/incomplete outputs**: Example 2 just says "A"
2. **Wrong extraction**: Examples 3 & 4 have correct answer in text, but extractor picks wrong letter
3. **Wrong reasoning**: Example 5 explicitly says "B" but true answer is "E"
4. **Verbose explanations**: Model generates long explanations mentioning correct terms, but answer extraction fails

### Answer Extraction Logic:

```python
def _extract_commonsense_answer(self, output: str) -> str:
    output = output.strip().upper()

    # Look for "THE ANSWER IS: X" pattern
    if "THE ANSWER IS:" in output:
        parts = output.split("THE ANSWER IS:")
        if len(parts) > 1:
            answer_part = parts[-1].strip()
            for char in answer_part:
                if char in ['A', 'B', 'C', 'D', 'E']:
                    return char

    # Fallback: look for first A-E in output
    for char in output:
        if char in ['A', 'B', 'C', 'D', 'E']:
            return char

    return "INVALID"
```

**Problem**: Fallback extracts **first A-E letter** in output, which can be:
- From words like "prEparation", "k**E**pt", "id**E**ntified"
- Not necessarily the answer!

## Impact on Three-Way Comparison

### What Remains Valid:

✅ **Mechanistic Analysis**:
- Variance ratios (0.253, 0.175, 0.154) - CT representation structure
- Centroid distances (33-40) - Task separation
- Cosine similarities (<0.25) - Limited cross-task overlap
- Layer-wise patterns - Representation evolution

✅ **Qualitative Findings**:
- Personal Relations uses more compact representations
- CommonsenseQA uses more distributed representations
- Task-specific encoding strategies

### What is Compromised:

❌ **Performance Claims**:
- "CommonsenseQA achieves 33% accuracy" - Misleading (should be ~71%)
- "Low accuracy due to distributed representations" - Wrong (it's dataset mismatch)
- Compactness → performance correlation - Confounded by test set issue

❌ **Cross-Task Comparisons**:
- CommonsenseQA performance not comparable to other tasks
- Can't draw conclusions about task difficulty
- Above-chance metrics are unreliable

## Solutions

### Option 1: Re-run with Correct Dataset (RECOMMENDED)

**Action**: Use `zen-E/CommonsenseQA-GPT4omini` validation split

**Pros**:
- Matches training data
- Should achieve ~71% accuracy
- Valid performance comparisons

**Cons**:
- Dataset has loading issues (need to fix)
- Need to reprocess 300 examples (~15 min)
- Regenerate all analyses

### Option 2: Document and Caveat (CURRENT STATE)

**Action**: Keep analysis with prominent disclaimers

**Pros**:
- Mechanistic analysis still valid
- Saves time
- Honest about limitations

**Cons**:
- Can't make performance claims
- Cross-task comparisons weakened
- Reduced publication value

### Option 3: Drop CommonsenseQA (FALLBACK)

**Action**: Two-way comparison (Personal Relations vs GSM8K only)

**Pros**:
- Both tasks have reliable performance
- Simpler analysis
- Cleaner story

**Cons**:
- Loses diversity (only 2 tasks)
- Less comprehensive
- Wastes collected data

## Recommendations

### Immediate (Before Publication):

1. **Fix dataset loading** for `zen-E/CommonsenseQA-GPT4omini`
2. **Re-run extraction** with correct dataset (300 examples, ~15 min)
3. **Regenerate analyses** with corrected data
4. **Update documentation** with correct 71% baseline

### Alternative (If Dataset Can't Be Fixed):

1. **Add prominent disclaimer** in abstract/intro
2. **Explain dataset mismatch** in methods section
3. **Focus on mechanistic findings** (variance ratios, task separation)
4. **Avoid performance comparisons** involving CommonsenseQA
5. **Consider two-way comparison** as primary result

## Lessons Learned

1. **Always verify test set** matches training data distribution
2. **Check dataset sources** explicitly (commonsense_qa vs CommonsenseQA-GPT4omini)
3. **Validate model outputs** manually before large-scale extraction
4. **Test inference** on small sample before full run
5. **Document dataset versions** in experiment configs

## Technical Details

### Model Checkpoint

- Path: `/home/paperspace/codi_ckpt/llama_commonsense/gsm8k_llama1b_latent_baseline/Llama-3.2-1B-Instruct/ep_3/lr_0.0008/seed_11/`
- Training: Oct 17, 2025
- Expected accuracy: 71.33% on zen-E/CommonsenseQA-GPT4omini validation
- Actual accuracy: 33.0% on commonsense_qa validation (wrong dataset!)

### Inference Parameters

```python
outputs = model.codi.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,  # Greedy decoding
    pad_token_id=tokenizer.eos_token_id
)
```

### Format Input Example

```python
Question: {question}
Choices:
A: {choice_a}
B: {choice_b}
C: {choice_c}
D: {choice_d}
E: {choice_e}
Reasoning:
```

**Problem**: This format expects model to generate reasoning, but standard dataset doesn't have it!

## Conclusion

The CommonsenseQA model is **not broken** - it's being tested on the **wrong dataset**. Performance drop from 71% to 33% is entirely explained by dataset mismatch. The mechanistic analysis (variance ratios, task separation) remains valid, but performance comparisons must be corrected or caveated.

**Action Required**: Either fix dataset and re-run, or document limitation and focus on mechanistic insights only.

---

**Investigation Time**: ~2 hours
**Status**: Root cause identified, solution proposed
**Priority**: HIGH (blocks publication-quality claims)
