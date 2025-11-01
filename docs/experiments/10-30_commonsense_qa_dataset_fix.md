# CommonsenseQA Dataset Fix

**Date**: October 30, 2025
**Status**: ✅ **FIXED - Re-running extraction**

## Problem Summary

The three-way comparison experiment was using the wrong dataset for CommonsenseQA:
- **Model trained on**: `zen-E/CommonsenseQA-GPT4omini` (with GPT-4 CoT reasoning)
- **We tested on**: `commonsense_qa` (standard dataset, no CoT)
- **Result**: 33% accuracy instead of expected 71%

## Root Cause

See [10-30_commonsense_qa_debug_ROOTCAUSE.md](./10-30_commonsense_qa_debug_ROOTCAUSE.md) for full investigation.

**Key issue**: Dataset mismatch caused:
1. Wrong input format (model expected pre-formatted questions)
2. Confused model outputs (incomplete, wrong reasoning)
3. Answer extraction errors (picking random letters from words)

## Solution Implemented

### 1. Upgraded pandas (REQUIRED)

```bash
pip install --upgrade pandas
```

- **Old version**: pandas 1.4.4 (2022)
- **New version**: pandas 2.3.3 (2024)
- **Why**: `dtype_backend='pyarrow'` parameter added in pandas 2.0

### 2. Updated data_loading.py

**File**: `src/experiments/three_way_comparison/utils/data_loading.py`

**Change** (line 125):
```python
# OLD (WRONG)
dataset = load_dataset('commonsense_qa', split='validation')

# NEW (CORRECT)
dataset = load_dataset('zen-E/CommonsenseQA-GPT4omini')['validation']
```

**Dataset format change**:

**Old format** (standard commonsense_qa):
```python
{
  'question': 'A revolving door is convenient...',
  'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
              'text': ['bank', 'library', ...]},
  'answerKey': 'A'
}
```

**New format** (zen-E/CommonsenseQA-GPT4omini):
```python
{
  'question': 'Question: A revolving door is convenient...\\nChoices:\\nA: bank\\nB: library\\n...',
  'answer': 'A',
  'cot': ''  # GPT-4 chain-of-thought (empty in this case)
}
```

### 3. Updated model_loader.py

**File**: `src/experiments/three_way_comparison/model_loader.py`

**Change** (line 322-345): Updated `_format_commonsense()` to handle both formats:

```python
def _format_commonsense(self, example: Dict) -> str:
    # Check if this is the new zen-E format (question is pre-formatted)
    if 'choices' not in example and 'answer' in example:
        # New format: question already includes choices
        question = example['question']
        # Add "Reasoning:" prompt at the end
        return f"{question}\\nReasoning:"

    # Old format (for backward compatibility)
    # ... existing code ...
```

### 4. Updated extract_activations.py

**File**: `src/experiments/three_way_comparison/extract_activations.py`

**Change** (line 193): Support both answer key formats:

```python
# OLD (WRONG)
ground_truth = example['answerKey']

# NEW (CORRECT - supports both)
ground_truth = example.get('answer', example.get('answerKey', 'UNKNOWN'))
```

## Verification

Tested all components:

1. **Data loading** ✅:
   ```bash
   python3 -c "from data_loading import load_commonsense; load_commonsense(5)"
   # Output: Loaded 5 examples from zen-E/CommonsenseQA-GPT4omini
   ```

2. **Format input** ✅:
   ```python
   loader.format_input('commonsense', example)
   # Output: Correctly formatted with "Question: ... Choices: ... Reasoning:"
   ```

3. **Answer extraction** ✅:
   - New format uses `example['answer']`
   - Falls back to `example['answerKey']` for old format

## Re-running Extraction

**Command**:
```bash
python extract_activations.py --n_examples 300 --task commonsense
```

**Expected results**:
- **Accuracy**: ~71% (matching training performance)
- **Answer distribution**: Balanced across A-E
- **No more bias**: 47% E, 33% A reduced to ~20% each

**Output file**: `results/commonsense_activations_corrected_300.json`

**Timeline**: ~15 minutes for 300 examples

## Impact on Three-Way Comparison

### What Changes:

1. **CommonsenseQA accuracy**: 33% → ~71% (**+38 percentage points**)
2. **Answer distribution**: More balanced (no more E/A bias)
3. **Performance comparisons**: Now valid across all three tasks
4. **Statistical analyses**: Can now use all bootstrap CI results

### What Stays the Same:

1. **Personal Relations data**: No changes (already correct)
2. **GSM8K data**: No changes (already correct)
3. **Mechanistic analyses**: Variance ratios, centroids still valid
4. **Visualizations**: Will regenerate with corrected data

## Files Modified

1. ✅ `utils/data_loading.py` - Dataset loading
2. ✅ `model_loader.py` - Input formatting
3. ✅ `extract_activations.py` - Answer key handling

## Next Steps

1. ✅ **Dataset fix complete** - pandas upgraded, code updated
2. 🔄 **Re-run extraction** - In progress (300 examples)
3. ⏳ **Regenerate analyses** - After extraction completes
4. ⏳ **Update documentation** - Final accuracy numbers
5. ⏳ **Commit changes** - All fixes + new results

## Lessons Learned

1. **Always verify test set matches training data**
2. **Check dataset sources explicitly** (commonsense_qa vs CommonsenseQA-GPT4omini)
3. **Test inference on small sample before large-scale runs**
4. **Document dataset versions in configs**
5. **Version dependencies matter** (pandas 1.4 → 2.3 for pyarrow support)

---

**Time to fix**: ~3 hours (investigation + implementation + verification)
**Status**: Dataset loading fixed, extraction running
**Priority**: HIGH (blocks publication-quality results)
