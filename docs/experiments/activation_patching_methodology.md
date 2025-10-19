# Activation Patching Methodology - Detailed Explanation

**What exactly are we patching?** This document explains the technical details.

---

## Visual Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-2 TRANSFORMER                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: "Janet's ducks lay 16 eggs... [THINK] [THINK]..."  │
│         └─ Tokenized into sequence of token IDs             │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Embedding Layer (wte + wpe)                        │    │
│  │   → Converts token IDs to vectors (768-dim)        │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Transformer Block 0 (h[0])                         │    │
│  │   ├─ LayerNorm → Attention → Residual              │    │
│  │   └─ LayerNorm → MLP → Residual                    │    │
│  │                                                     │    │
│  │   Output: hidden_states [batch, seq_len, 768]      │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│                       ...                                    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Transformer Block 3 (h[3])  ← 🎯 EARLY LAYER       │    │
│  │   ├─ LayerNorm → Attention → Residual              │    │
│  │   └─ LayerNorm → MLP → Residual                    │    │
│  │                                                     │    │
│  │   Output: hidden_states [1, seq_len, 768]          │    │
│  │            └─ We patch position [-1] (last token)  │    │
│  │            └─ This is the [THINK] token!           │ ◄──┼── HOOK HERE
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│                       ...                                    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Transformer Block 6 (h[6])  ← 🎯 MIDDLE LAYER      │    │
│  │   (same structure, we hook here too)               │ ◄──┼── HOOK HERE
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│                       ...                                    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Transformer Block 11 (h[11]) ← 🎯 LATE LAYER       │    │
│  │   (same structure, we hook here too)               │ ◄──┼── HOOK HERE
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Final LayerNorm (ln_f)                             │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Language Model Head (lm_head)                      │    │
│  │   → Projects 768-dim to vocab_size (50257)         │    │
│  │   → Softmax to get token probabilities             │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  Output: "The answer is 18"                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## What We Patch

### Location: **Residual Stream**

We patch the **output of the entire transformer block**, which is the **residual stream** after all computations in that block are complete.

**Why this location?**
- The residual stream is the "information highway" in transformers
- All information flows through it from layer to layer
- Interventions here test: "Does the information at this point causally matter?"
- This is a standard approach in mechanistic interpretability research

---

## Detailed Data Flow in One Block

```
┌─────────────────────────────────────────────────────────────┐
│              INSIDE TRANSFORMER BLOCK h[i]                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: x (from previous block)                             │
│  Shape: [batch_size, seq_len, hidden_dim]                   │
│  For us: [1, ~20 tokens, 768]                               │
│          └─ Last token is [THINK]                           │
│                                                              │
│  ┌──────────────────────────────────────┐                   │
│  │ ATTENTION SUB-BLOCK                  │                   │
│  ├──────────────────────────────────────┤                   │
│  │                                      │                   │
│  │  1. norm1 = LayerNorm(x)            │                   │
│  │     └─ Normalize across hidden_dim   │                   │
│  │                                      │                   │
│  │  2. attn_out = Attention(norm1)     │                   │
│  │     └─ Multi-head self-attention     │                   │
│  │     └─ Tokens attend to each other   │                   │
│  │                                      │                   │
│  │  3. x = x + attn_out                │                   │
│  │     └─ RESIDUAL CONNECTION           │                   │
│  │                                      │                   │
│  └──────────────────────────────────────┘                   │
│                   ↓                                          │
│  ┌──────────────────────────────────────┐                   │
│  │ MLP SUB-BLOCK                        │                   │
│  ├──────────────────────────────────────┤                   │
│  │                                      │                   │
│  │  4. norm2 = LayerNorm(x)            │                   │
│  │     └─ Normalize again               │                   │
│  │                                      │                   │
│  │  5. mlp_out = MLP(norm2)            │                   │
│  │     └─ 2-layer feedforward           │                   │
│  │     └─ Expand to 4*768, then back    │                   │
│  │                                      │                   │
│  │  6. x = x + mlp_out                 │                   │
│  │     └─ RESIDUAL CONNECTION           │                   │
│  │                                      │                   │
│  └──────────────────────────────────────┘                   │
│                   ↓                                          │
│  Output: x (to next block)                                  │
│          └─ This is what we intercept! 🎯                   │
│                                                              │
│  🪝 HOOK LOCATION: Right here, after step 6                 │
│     We replace: x[:, -1, :] (last token, all features)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## What Token Position?

### We Patch: `hidden_states[:, -1, :]`

**Breakdown:**
- `[:]` - All batch items (we only use batch_size=1)
- `[-1]` - **Last token position** in the sequence
- `[:]` - All 768 hidden dimensions

**Why the last token?**
During generation, the model processes tokens autoregressively:

```
Step 0: "Janet's ducks lay 16 eggs... [THINK]"
        └─ Model generates first [THINK] token
        └─ This becomes position -1 (last token)
        └─ 🎯 WE PATCH THIS POSITION

Step 1: "Janet's ducks lay 16 eggs... [THINK] [THINK]"
        └─ Model generates second [THINK] token
        └─ Now this is position -1
        └─ (We only patch at step 0 - the first [THINK])

Step 2+: Continue generating until answer is produced
```

**For CODI:**
- CODI uses 6 latent `[THINK]` tokens for reasoning
- We currently only patch the **first [THINK]** token
- This is a limitation (should test all 6 positions)

---

## The Patching Process

### Step 1: Cache Clean Activation

```python
# Run clean problem: "Janet's ducks lay 16 eggs, eats 3, bakes 4..."
# Expected answer: $18

# At generation step 0 (first [THINK] token):
with torch.no_grad():
    output = model(input_ids)
    # output.hidden_states[layer_idx] shape: [1, seq_len, 768]

    # Extract last token (the [THINK] we just generated)
    clean_activation = output.hidden_states[layer_idx][:, -1, :]
    # Shape: [1, 768]

    # Save this!
    cache[layer_name] = clean_activation
```

### Step 2: Patch Corrupted Run

```python
# Run corrupted problem: "Janet's ducks lay 16 eggs, eats 10, bakes 4..."
# Expected answer: $4 (but model gets wrong without patch)

# Register hook at target layer
def patch_hook(module, input, output):
    if at_first_THINK_token:
        # Clone output (it's a tuple: (hidden_states, ...))
        hidden_states = output[0].clone()

        # REPLACE the last token with cached clean activation
        hidden_states[:, -1, :] = clean_activation
        #             ^^^^  ^^^
        #             last  all features
        #             token

        return (hidden_states,) + output[1:]
    return output

model.transformer.h[layer_idx].register_forward_hook(patch_hook)

# Now generate with the hook active
output = model.generate(corrupted_input)
# The model will use the CLEAN activation instead of computing
# its own activation for the corrupted problem
```

---

## What We're NOT Patching

**We are NOT patching:**
- ❌ Specific attention heads
- ❌ Specific MLP neurons
- ❌ Query/Key/Value matrices
- ❌ Individual attention patterns
- ❌ Gradient flows

**We ARE patching:**
- ✓ The entire residual stream at a specific layer
- ✓ Only for one token position ([THINK])
- ✓ All 768 features simultaneously

This is a **coarse-grained** intervention, not fine-grained.

---

## Why Residual Stream and Not Components?

### Option 1: Patch Residual Stream (What We Do)
```
✓ Tests: "Does information at this layer matter?"
✓ Neutral about mechanism (attention vs MLP)
✓ Standard in mechanistic interpretability literature
✓ Easier to implement and interpret
```

### Option 2: Patch Attention Output Only
```
✓ Tests: "Does attention matter?"
✗ Ignores MLP contributions
✗ More complex (which head?)
✗ Harder to interpret interactions
```

### Option 3: Patch MLP Output Only
```
✓ Tests: "Does MLP matter?"
✗ Ignores attention contributions
✗ Misses attention-MLP interactions
```

### Option 4: Patch Individual Attention Heads
```
✓ Most fine-grained
✗ 12 heads × 12 layers = 144 interventions!
✗ Multiple testing problem
✗ Harder to find effects (diluted signal)
```

**We chose Option 1** because:
1. It's the standard approach (Meng et al. 2022, Wang et al. 2022)
2. It tests the high-level question: "Does this layer's output matter?"
3. It's interpretable: if patching helps, information at this layer is causally relevant

---

## Code Implementation

### Key Files

**`cache_activations.py`** - Extracts and saves activations
```python
# Register hook to capture activations
def cache_hook(module, input, output):
    # Save output of the entire block
    activation = output.hidden_states[layer_idx][:, -1, :]
    cache[layer_name] = activation
```

**`patch_and_eval.py`** - Injects cached activations
```python
# Register hook to replace activations
def patch_hook(module, input, output):
    hidden_states = output[0].clone()
    hidden_states[:, -1, :] = cached_activation  # REPLACE
    return (hidden_states,) + output[1:]
```

### Hook Mechanics (PyTorch)

```python
# Get the module we want to hook
layer_module = model.transformer.h[layer_idx]
#                                 ^^^^^^^^^^^
#                                 This is the entire block

# Register forward hook
hook_handle = layer_module.register_forward_hook(patch_hook)
#                                                 ^^^^^^^^^^
#                                                 Our custom function

# Run model with hook active
output = model.generate(input_ids)
# Every time this layer is called during generation,
# our patch_hook function intercepts the output

# Remove hook when done
hook_handle.remove()
```

---

## Comparison to Other Methods

| Method | Our Approach | Alternative |
|--------|--------------|-------------|
| **Component** | Residual stream | Attention heads, MLP neurons |
| **Granularity** | Coarse (entire layer) | Fine (individual components) |
| **Token** | Last position ([THINK]) | All tokens, specific positions |
| **Operation** | Replace (patch) | Add, scale, ablate |
| **Comparison** | Clean → Corrupted | Random, zero, mean |

---

## Limitations

### What We're Missing

1. **Only first [THINK] token**
   - CODI uses 6 [THINK] tokens
   - We only test the first one
   - Should test all 6 positions

2. **Only 3 layers**
   - GPT-2 has 12 layers
   - We only test L3, L6, L11
   - Should scan all layers

3. **Residual stream only**
   - Doesn't isolate attention vs MLP
   - Can't identify which component matters
   - Needs follow-up with component-level patching

4. **Single token replacement**
   - Could patch multiple tokens simultaneously
   - Could patch over multiple generation steps
   - More complex but potentially more powerful

---

## References

**Mechanistic Interpretability:**
- Meng et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS.
  - Uses residual stream patching to locate factual knowledge
  - Standard methodology we follow

- Wang et al. (2022). "Interpretability in the Wild." arXiv.
  - Activation patching for circuit discovery
  - Shows residual stream interventions work

- Geiger et al. (2021). "Causal Abstractions of Neural Networks." arXiv.
  - Theoretical foundation for causal interventions
  - Defines what "causal" means in this context

**Transformer Architecture:**
- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
- Radford et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)

---

## Summary

**What we patch:**
- **Component**: Residual stream (output of entire transformer block)
- **Layer**: h[3], h[6], h[11] (early, middle, late)
- **Token**: Position -1 (last token = first [THINK])
- **Dimensions**: All 768 hidden features

**How we patch:**
1. Run clean problem, cache activation at target layer
2. Run corrupted problem with hook registered
3. Hook intercepts output, replaces last token's activation
4. Model continues with patched activation
5. Measure if performance recovers

**Why this approach:**
- Standard in mechanistic interpretability literature
- Tests high-level question: "Does this layer's info matter?"
- Doesn't assume specific mechanism (attention vs MLP)
- Easier to implement and interpret than fine-grained alternatives

**Current limitations:**
- Small sample size (n=9, need 634) ← **Most critical**
- Only first [THINK] token (should test all 6)
- Only 3 layers (should test all 12)
- Coarse-grained (could do component-level patching)
