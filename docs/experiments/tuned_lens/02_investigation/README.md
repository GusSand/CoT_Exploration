# Investigation: LoRA Loading Issue

## Problem Discovered

When attempting to scale the successful single L10→L11 probe to all layer pairs (L0-L10 → L11), we discovered that the LoRA weights were not being fully loaded.

## Root Cause

The original LoRA configuration only targeted  modules:

```python
lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=['c_attn'],  # INCOMPLETE!
    ...
)
```

However, the CODI-GPT2 checkpoint includes LoRA weights for three module types:
-  (attention projection)
-  (attention output projection)
-  (feed-forward layer)

## Impact

- Multi-layer probes trained with incomplete LoRA weights showed degraded performance
- Layer 11 predictions had unusually low confidence (~3-4%) instead of expected >50%
- Visualization HTML files show the difference between incorrect and corrected loading

## Fix

Updated LoRA configuration to include all modules:

```python
lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=['c_attn', 'c_proj', 'c_fc'],  # COMPLETE
    ...
)
```

## Status

The multi-layer probe weights trained with incomplete LoRA are **not archived** in this repository. Only the successful single L10→L11 baseline probe (trained before this issue was discovered) is preserved.

## Visualizations

See  for HTML visualizations showing the difference between incomplete and complete LoRA loading.
