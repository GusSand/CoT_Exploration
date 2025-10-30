# Bug Report: CT Swapping Implementation

**Date:** 2025-10-30
**Severity:** CRITICAL
**Status:** IDENTIFIED, FIX READY

---

## Summary

The CT token swapping implementation in `2_implement_swapping.py` had a critical bug where swapped hidden states were set AFTER the forward pass instead of being used AS INPUT to the forward pass. This caused the KV cache to be updated with non-swapped states, making the swap ineffective.

---

## Evidence of Bug

### 1. Suspicious Results
- **CT5:** 0% resampling impact (despite 26% ablation impact)
- **CT3:** -1% impact (performance improved!)
- **Negative correlation:** r = -0.751 (backwards from expected)

### 2. Diagnostic Test Results
**CT5 Per-Problem Analysis:**
```
Problem 0: Baseline=False → ALL 5 swaps=False (100% same)
Problem 2: Baseline=True → ALL 5 swaps=True (100% same)
```

**Interpretation:** Swapping CT5 has ZERO effect - answers are identical regardless of which problem we swap from.

### 3. Code Analysis

**Original Buggy Code (lines 85-106):**
```python
for step in range(6):
    # Forward pass with CURRENT latent_embd
    outputs = model.codi(inputs_embeds=latent_embd, ...)
    past_key_values = outputs.past_key_values

    # Set latent_embd for NEXT iteration
    if step == swap_position:
        latent_embd = problem_B_cache['ct_hidden_states'][step]...  # BUG!
    else:
        latent_embd = outputs.hidden_states[-1][:, -1, :]...

    if model.use_prj:
        latent_embd = model.prj(latent_embd)
```

**Problem:**
- At `step=5`: Uses `latent_embd` from step 4, runs forward pass, THEN sets `latent_embd` to B[5]
- But loop ends! The swapped B[5] is never used
- KV cache contains generated CT5, not swapped CT5

---

## Root Cause

The loop structure sets embeddings for the NEXT iteration after the forward pass. For the last iteration (CT5), there is no "next" iteration, so the swapped state is set but never consumed.

This also affects earlier positions partially:
- Swap at position `i` sets the embedding but uses it at position `i+1`'s forward pass
- This creates an off-by-one error where we're swapping the wrong position

---

## Impact Assessment

### What Actually Happened
- **CT0-CT4:** Partial swap (off-by-one error)
  - Swapping CT0 actually contaminated CT1
  - Swapping CT4 actually contaminated CT5
  - This explains the weird pattern!
- **CT5:** No swap at all (set but never used)

### Why CT4 Showed High Impact (25%)
- When we "swapped CT4", we actually swapped what goes into CT5's forward pass
- CT5 is the critical computation token (26% ablation)
- So contaminating CT5's input had high impact
- This accidentally measured CT5's sensitivity to input!

### Why CT5 Showed Zero Impact (0%)
- We never actually swapped CT5
- We set the value after the last forward pass
- So it had no effect on answer generation

---

## Fix Strategy

### Approach 1: Prepare Embedding Before Forward Pass
```python
for step in range(6):
    # DECIDE which embedding to use BEFORE forward pass
    if step == swap_position:
        latent_embd = problem_B_cache['ct_hidden_states'][step]...
        if model.use_prj:
            latent_embd = model.prj(latent_embd)
    # else: use latent_embd from previous iteration

    # NOW do forward pass with chosen embedding
    outputs = model.codi(inputs_embeds=latent_embd, ...)
    past_key_values = outputs.past_key_values

    # Prepare for NEXT iteration (if not swapping)
    if step + 1 < 6 and step + 1 != swap_position:
        latent_embd = outputs.hidden_states[-1][:, -1, :]...
        if model.use_prj:
            latent_embd = model.prj(latent_embd)
```

### Approach 2: Simpler - Check Swap BEFORE Forward
```python
for step in range(6):
    # Check if we should swap THIS step
    if step == swap_position:
        latent_embd = problem_B_cache['ct_hidden_states'][step]...
        if model.use_prj:
            latent_embd = model.prj(latent_embd)

    # Forward pass
    outputs = model.codi(inputs_embeds=latent_embd, ...)
    past_key_values = outputs.past_key_values

    # Prepare for next (only if not swapping next)
    latent_embd = outputs.hidden_states[-1][:, -1, :]...
    if model.use_prj:
        latent_embd = model.prj(latent_embd)
```

---

## Verification Plan

After fixing:

1. **Re-run diagnostic tests**
   - CT5 swap should now show changes
   - All swaps should affect output

2. **Re-run pilot experiment**
   - Expect positive correlation with ablation
   - CT0 ~15-20% impact
   - CT5 ~20-25% impact

3. **Check interpretation**
   - If results converge with ablation → Strong success
   - If still dissociate → Investigate further

---

## Lessons Learned

1. **Always validate with extreme tests**
   - "Swap all positions" test would have caught this
   - "No-op swap" test was good but not enough

2. **Loop structure matters**
   - When setting state for "next iteration", make sure there IS a next iteration

3. **Suspicious results = investigate**
   - 0% impact on critical token was the smoking gun
   - Negative correlation was the red flag

---

## Time Impact

- **Time lost:** ~4 hours (pilot experiment with buggy code)
- **Time to fix:** ~1 hour (reimplement + test)
- **Time to re-run:** ~30 min (pilot with fixed code)

**Total delay:** ~5.5 hours

---

## Next Steps

1. ✅ Implement fix (Approach 2 - cleaner)
2. ⏳ Test fixed version on 5 problems
3. ⏳ Re-run full pilot (20 problems × 5 samples)
4. ⏳ Analyze corrected results
5. ⏳ Make new Go/No-Go decision

---

**Credit:** User correctly identified suspicious results and requested diagnostic tests, which revealed the bug. Thank you for the scrutiny!
