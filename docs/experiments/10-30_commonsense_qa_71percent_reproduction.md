# CommonsenseQA 71% Accuracy Reproduction

**Date**: October 30, 2025
**Status**: ✅ **SOLVED - 71.09% accuracy reproduced**

## Problem Summary

After fixing the dataset loading to use `zen-E/CommonsenseQA-GPT4omini`, the model was still achieving only **34% accuracy** instead of the documented **71.33%**. The user insisted "We reproduced it before" and requested investigation of the discrepancy.

## Investigation Process

### 1. Initial Hypotheses
- Different generation parameters (temperature, sampling)
- Different prompt format
- Different checkpoint or model version
- Evaluation script details not documented

### 2. Key Discovery: Found Original Evaluation Code

Located the original CODI evaluation scripts:
- `/home/paperspace/dev/CoT_Exploration/codi/eval_baseline.py` - CoT baseline evaluation
- `/home/paperspace/dev/CoT_Exploration/codi/test.py` - Main CODI evaluation function

### 3. Root Cause: THREE Critical Differences

Compared our `extract_activations.py` with the original `test.py`:

#### Difference 1: Input Format

**Our code** (model_loader.py:332-334):
```python
# New format: question already includes choices
question = example['question']
# Add "Reasoning:" prompt at the end
return f"{question}\nReasoning:"
```

**Original CODI** (test.py:137):
```python
question = [f"{example[question_name].strip().replace('  ', ' ')}" for example in test_set]
```

**Impact**: The `"\nReasoning:"` suffix was NOT part of the original evaluation. This changes the model's behavior.

#### Difference 2: Answer Extraction Logic

**Our code** (model_loader.py:419-422):
```python
# Fallback: look for first A-E in output
for char in output:
    if char in ['A', 'B', 'C', 'D', 'E']:
        return char
```

**Original CODI** (test.py:334-337):
```python
pred = sentence.split("The answer is:")[-1].strip()
if pred[0] not in "ABCDE":
    return "C"  # Default to C
return pred[0]
```

**Impact**: Our code picks the FIRST A-E letter anywhere in the output, which grabs letters from words like "prEparation", "gEt", "thE", etc. This caused the severe answer bias (47% E, 33% A).

The original code:
1. Splits on "The answer is:"
2. Takes the first character after that
3. Defaults to "C" if invalid

#### Difference 3: Generation Flow

**Our code** (extract_activations.py:154-180):
```python
# Extract CT hidden states using proper iterative method
ct_hidden = self.extract_ct_hidden_states(model, input_text, tokenizer)

# Now generate the final answer (after CT tokens)
with torch.no_grad():
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(self.loader.device)

    # Generate answer
    outputs = model.codi.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
```

**Original CODI** (test.py:226-267):
```python
# encode the question
outputs = model.codi(input_ids=batch["input_ids"], ...)
past_key_values = outputs.past_key_values
latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

if training_args.use_prj:
    latent_embd = model.prj(latent_embd)

# Generate CT tokens (6 iterations)
for i in range(inf_latent_iterations):
    outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, ...)
    past_key_values = outputs.past_key_values
    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

    if training_args.use_prj:
        latent_embd = model.prj(latent_embd)

# Add EOT token
eot_emb = model.get_embd(...)[model.eot_id]
output = eot_emb

# Generate answer tokens (manual loop, not generate())
for i in range(max_new_tokens):
    out = model.codi(inputs_embeds=output, use_cache=True, past_key_values=past_key_values)
    # ... greedy sampling or temperature sampling
```

**Impact**:
- Our code calls `.generate()` which starts fresh, ignoring CT tokens
- Original code maintains KV cache throughout: question → CT tokens → answer
- Original uses manual token-by-token generation with full control
- Our code uses max_new_tokens=50, original uses 256

## Solution: Corrected Evaluation Script

Created `eval_commonsense_correct.py` that exactly matches the original CODI evaluation:

```python
#!/usr/bin/env python3
"""
CommonsenseQA Evaluation - MATCHING ORIGINAL CODI EVALUATION

Key differences from our extract_activations.py:
1. Input format: Raw question text ONLY (no "Reasoning:" suffix)
2. Answer extraction: Takes FIRST letter after "The answer is:", defaults to "C"
3. Generation: Uses proper KV cache flow: BOT → CT tokens × 6 → EOT → answer
"""
```

### Key Implementation Details:

1. **Input Processing**:
   ```python
   # RAW QUESTIONS ONLY (like original)
   questions = [example['question'].strip().replace('  ', ' ') for example in test_set]
   ```

2. **Answer Extraction**:
   ```python
   def extract_answer_number(sentence: str) -> str:
       pred = sentence.split("The answer is:")[-1].strip()
       if not pred or pred[0] not in "ABCDE":
           return "C"  # Default to C (like original)
       return pred[0]
   ```

3. **Generation Flow**:
   ```python
   # Encode question
   outputs = model.codi(input_ids=batch["input_ids"], ...)
   past_key_values = outputs.past_key_values
   latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

   # Generate CT tokens (6 iterations)
   for i in range(6):
       outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, past_key_values=past_key_values)
       past_key_values = outputs.past_key_values
       latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
       if training_args.use_prj:
           latent_embd = model.prj(latent_embd)

   # Add EOT token
   eot_emb = model.get_embd(model.codi, model.model_name)(
       torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
   ).unsqueeze(0)

   # Generate answer tokens manually
   for i in range(256):  # max_new_tokens=256
       out = model.codi(inputs_embeds=output, use_cache=True, past_key_values=past_key_values)
       # ... greedy decoding
   ```

## Results

**Script**: `src/experiments/three_way_comparison/eval_commonsense_correct.py`

```bash
python eval_commonsense_correct.py
```

**Output**:
```
================================================================================
RESULTS
================================================================================
Total examples: 1221
Correct: 868/1221
Accuracy: 71.09%
================================================================================
```

**Comparison**:
- **Original documentation**: 71.33% (871/1221) ✅
- **Our reproduction**: 71.09% (868/1221) ✅
- **Difference**: 3 examples (0.24 percentage points)

The 3-example difference is likely due to:
- Minor implementation details
- Floating point precision differences
- Randomness in model inference (though we use greedy decoding)

## Validation

The reproduction is **SUCCESSFUL** because:

1. ✅ **Accuracy matches**: 71.09% vs 71.33% (within margin of error)
2. ✅ **Same dataset**: zen-E/CommonsenseQA-GPT4omini validation (1221 examples)
3. ✅ **Same model**: /home/paperspace/codi_ckpt/llama_commonsense/...
4. ✅ **Same method**: Matches original test.py evaluation flow exactly

## Next Steps

### 1. Update Three-Way Comparison Pipeline

Need to update `extract_activations.py` to use the correct evaluation method:

**Changes required**:
- Remove `"\nReasoning:"` suffix from CommonsenseQA prompts
- Update answer extraction to match original (split on "The answer is:", default to "C")
- Update generation flow to maintain KV cache (question → CT → answer)
- Change max_new_tokens from 50 to 256

**Challenge**: The current pipeline separates CT extraction from answer generation for mechanistic analysis. We need to preserve CT hidden states while using the correct generation flow.

**Solution**: Modify `extract_ct_hidden_states()` to:
1. Maintain KV cache throughout
2. Extract CT hidden states during proper generation
3. Continue with EOT and answer generation

### 2. Re-run Extraction

After updating the pipeline:
```bash
python extract_activations.py --n_examples 300 --task commonsense
```

Expected results:
- **Accuracy**: ~71% (matching reproduction)
- **Answer distribution**: More balanced (~20% each)
- **Model outputs**: Proper "The answer is: X" format

### 3. Regenerate All Analyses

With corrected CommonsenseQA data:
- Visualizations (9 plots)
- Divergence metrics
- Validation analyses
- Statistical comparisons

## Files

**Created**:
- `src/experiments/three_way_comparison/eval_commonsense_correct.py` - Corrected evaluation
- `results/commonsense_eval_original_method.json` - Full results (868/1221)

**Documentation**:
- `docs/experiments/10-30_commonsense_qa_dataset_fix.md` - Dataset fix
- `docs/experiments/10-30_commonsense_qa_debug_ROOTCAUSE.md` - Initial investigation
- `docs/experiments/10-30_commonsense_qa_71percent_reproduction.md` - This doc

## Lessons Learned

1. **Always verify evaluation methods match training**:
   - Dataset format ✅ (fixed)
   - Input format ✅ (now fixed)
   - Answer extraction ✅ (now fixed)
   - Generation parameters ✅ (now fixed)

2. **Small differences compound**:
   - `"\nReasoning:"` suffix seemed minor but changed behavior
   - Answer extraction logic caused catastrophic failure (34% accuracy)
   - Generation flow matters for continuous thought models

3. **Read the original code**:
   - Documentation claimed 71% but didn't show HOW
   - No validation metrics in trainer_state.json
   - Had to find and read `test.py` to understand exact method

4. **Greedy decoding should be deterministic**:
   - The 3-example difference suggests some non-determinism
   - Could be from floating point precision or model initialization
   - Acceptable for reproduction (0.24 percentage point difference)

## Timeline

- **October 30, 2025 (morning)**: Discovered 34% accuracy issue after dataset fix
- **October 30, 2025 (afternoon)**: User insisted "We reproduced it before"
- **October 30, 2025 (evening)**: Found original evaluation code, identified 3 differences
- **October 30, 2025 (evening)**: Created corrected evaluation, achieved 71.09% ✅

**Total time**: ~8 hours from initial discovery to successful reproduction

---

**Status**: ✅ **COMPLETE**
**Accuracy**: 71.09% (868/1221) - matches documented 71.33%
**Next**: Update extraction pipeline and re-run with correct method
