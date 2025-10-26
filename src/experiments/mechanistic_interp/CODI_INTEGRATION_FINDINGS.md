# CODI Integration Findings - Study Report

**Date:** 2025-10-26
**Study Duration:** 1 hour
**Purpose:** Understand existing CODI integration patterns for MECH-02 implementation

---

## Summary

✅ **EXCELLENT NEWS:** Existing experiments have comprehensive CODI integration infrastructure that we can reuse directly!

**Key Finding:** The `activation_patching` experiment has production-ready code for:
1. Loading CODI LLaMA model with LoRA
2. Extracting continuous thought activations
3. Using forward hooks to intervene/patch activations
4. Generating answers with modified thoughts

**Reusability:** ~90% - Can adapt existing `ActivationCacherLLaMA` and `NTokenPatcher` classes

---

## Key Files Discovered

###  1. Core Infrastructure: `activation_patching/core/cache_activations_llama.py`
**Purpose:** Load CODI and extract continuous thoughts

**Key Class:**
```python
class ActivationCacherLLaMA:
    """Caches activations from CODI LLaMA model at specified layers."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        # Loads CODI with:
        # - Model: meta-llama/Llama-3.2-1B-Instruct
        # - 6 latent tokens
        # - LoRA config
        # - Projection layer (2048 dims)
        # - Checkpoint loading (safetensors or pytorch_model.bin)
```

**Key Method:**
```python
def cache_problem_activations(
    self,
    problem_text: str,
    problem_id: int,
    layer_indices: Dict[str, int] = None
) -> Dict[str, torch.Tensor]:
    """
    Extract continuous thought activations at specified layers.

    Process:
    1. Tokenize problem text
    2. Get input embeddings
    3. Forward through model
    4. Process 6 latent tokens (BOT → 6 THINK tokens → EOT)
    5. Extract hidden states at specified layers
    6. Return activations dict
    """
```

###  2. Intervention Infrastructure: `scripts/experiments/run_ablation_N_tokens_llama.py`
**Purpose:** Patch/intervene on continuous thoughts using forward hooks

**Key Class:**
```python
class NTokenPatcher:
    """Patches first N [THINK] tokens using forward hooks."""

    def run_with_N_tokens_patched(
        self,
        problem_text: str,
        patch_activations: List[torch.Tensor],
        layer_name: str,
        max_new_tokens: int = 200
    ) -> str:
        """
        Generate answer with patched continuous thoughts.

        Process:
        1. Set patch_activations (e.g., zeros for ablation)
        2. Get target layer module
        3. Register forward hook
        4. Generate with hook active (replaces activations)
        5. Remove hook
        6. Return generated answer
        """
```

**Forward Hook Pattern:**
```python
def _create_patch_hook(self):
    def patch_hook(module, input, output):
        if self.patch_activations is not None:
            activation_to_patch = self.patch_activations[self.current_step]

            if isinstance(output, tuple):
                hidden_states = output[0].clone()
                hidden_states[:, -1, :] = activation_to_patch.to(self.device)
                return (hidden_states,) + output[1:]
            else:
                hidden_states = output.clone()
                hidden_states[:, -1, :] = activation_to_patch.to(self.device)
                return hidden_states
        return output

    return patch_hook
```

###  3. Usage Examples

#### Example 1: CoT Necessity Test (Zero Ablation)
**File:** `activation_patching/manual_cot_necessity_test.py`

```python
# Load model
cacher = ActivationCacherLLaMA(model_path)
patcher = NTokenPatcher(cacher, num_tokens=6)

# Create ZERO activations (ablate all reasoning)
sample_act = patcher.cache_N_token_activations(question, 'middle')[0]
zero_activations = [torch.zeros_like(sample_act) for _ in range(6)]

# Run with zeros
ablated_output = patcher.run_with_N_tokens_patched(
    problem_text=question,
    patch_activations=zero_activations,
    layer_name='middle',
    max_new_tokens=200
)
```

#### Example 2: Operation Intervention
**File:** `operation_intervention/run_intervention.py`

```python
class OperationIntervener:
    def run_with_intervention(
        self,
        problem_text: str,
        intervention_vector: Optional[torch.Tensor] = None,
        intervention_token: int = 1,
        intervention_layer: str = 'middle'
    ) -> str:
        # Register hook at specific layer
        # Inject intervention_vector at intervention_token
        # Generate answer
```

---

## CODI Loading Pattern (Standard)

### Step 1: Setup Arguments
```python
from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType

parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(
    args=[
        '--model_name_or_path', 'meta-llama/Llama-3.2-1B-Instruct',
        '--output_dir', './tmp',
        '--num_latent', '6',
        '--use_lora', 'True',
        '--ckpt_dir', model_path,  # ~/codi_ckpt/llama_gsm8k/
        '--use_prj', 'True',
        '--prj_dim', '2048',
        '--lora_r', '128',
        '--lora_alpha', '32',
        '--lora_init', 'True',
    ]
)

model_args.train = False
training_args.greedy = True
```

### Step 2: Create LoRA Config
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=128,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    init_lora_weights=True,
)
```

### Step 3: Load Model
```python
model = CODI(model_args, training_args, lora_config)

# Load checkpoint
from safetensors.torch import load_file
try:
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
except Exception:
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))

model.load_state_dict(state_dict, strict=False)
model.codi.tie_weights()
model.float()  # Convert to float32
model.to(device)
model.eval()
```

---

## Continuous Thought Extraction Pattern

### Full Process (6 Latent Tokens)
```python
with torch.no_grad():
    # 1. Tokenize
    inputs = tokenizer(problem_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # 2. Get initial embeddings
    input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

    # 3. Forward for context
    outputs = model.codi(
        inputs_embeds=input_embd,
        use_cache=True,
        output_hidden_states=True
    )
    past_key_values = outputs.past_key_values

    # 4. Get BOT embedding
    bot_emb = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.bot_id], dtype=torch.long, device=device)
    ).unsqueeze(0)

    # 5. Process 6 latent tokens
    latent_embd = bot_emb
    continuous_thoughts = []

    for latent_step in range(6):
        outputs = model.codi(
            inputs_embeds=latent_embd,
            use_cache=True,
            output_hidden_states=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values

        # Extract activation at layer
        activation = outputs.hidden_states[layer_idx][:, -1, :]
        continuous_thoughts.append(activation)

        # Update for next step
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Apply projection
        if model.use_prj:
            latent_embd = model.prj(latent_embd)

    # continuous_thoughts is List[Tensor(1, 2048)] - one per position
```

---

## Forward Hook Pattern for Interventions

### Hook Registration
```python
# Get layer module
target_layer = model.codi.base_model.model.model.layers[layer_idx]

# Create hook function
def intervention_hook(module, input, output):
    if intervention_active:
        if isinstance(output, tuple):
            hidden_states = output[0].clone()
            hidden_states[:, -1, :] = modified_activation
            return (hidden_states,) + output[1:]
        else:
            hidden_states = output.clone()
            hidden_states[:, -1, :] = modified_activation
            return hidden_states
    return output

# Register hook
hook_handle = target_layer.register_forward_hook(intervention_hook)

# Run inference
answer = generate_with_model(...)

# Remove hook
hook_handle.remove()
```

---

## Layer Configuration (LLaMA-3.2-1B)

```python
LAYER_CONFIG = {
    'early': 4,    # 25% through 16 layers
    'middle': 8,   # 50% through model
    'late': 14     # 87.5% through model
}
```

**Total Layers:** 16 (0-indexed: 0-15)
**Hidden Dim:** 2048
**Latent Tokens:** 6

---

## Key Insights for MECH-02

### ✅ What Works (Proven in Production)
1. **Forward hooks** - Clean, non-invasive intervention method
2. **Zero ablation** - Setting activations to zeros tests importance
3. **Layer-specific** - Can intervene at early/middle/late layers
4. **Position-specific** - Can target specific latent tokens (0-5)
5. **Batching** - Can process multiple problems (though existing code doesn't batch well)

### 🔨 What We Need to Implement
1. **KL divergence measurement** - Not in existing code
   - Baseline: Full continuous thoughts → answer distribution
   - Ablated: Zero positions [0...i-1] → answer distribution
   - Importance: KL(baseline || ablated)

2. **Position-wise zeroing** - Existing code zeros ALL tokens
   - We need: Zero [0...i-1], keep [i...5]
   - Requires tracking which step we're on

3. **Efficient batching** - Existing code is single-problem
   - Need: Batch 32 problems at once for GPU efficiency
   - Checkpoint every 500 problems

### ⚠️ Gotchas Discovered
1. **Model loading is slow** (~10-20 seconds)
   - Load once, reuse for all problems

2. **Mixed precision issues** - Checkpoint has bfloat16/float32 mix
   - Use `.float()` to convert everything to float32

3. **Layer module access** - Different for base_model vs model
   - Try: `model.codi.base_model.model.model.layers[i]`
   - Fallback: `model.codi.model.layers[i]`

4. **Hook timing** - Must register/remove carefully
   - Use try/finally to ensure cleanup

5. **current_step tracking** - Need global counter for multi-token hooks
   - Reset to 0 before each forward pass

---

## Recommended Implementation for MECH-02

### Option 1: Adapt Existing Classes (RECOMMENDED)
**Time:** 4-6 hours
**Approach:** Copy `ActivationCacherLLaMA` and `NTokenPatcher`, modify for our needs

**Pros:**
- Proven infrastructure
- 90% of code already works
- Just need to add KL divergence + position-wise zeroing

**Cons:**
- Need to understand existing code patterns

### Option 2: Write from Scratch
**Time:** 8-10 hours
**Approach:** Implement based on patterns learned

**Pros:**
- Clean, purpose-built code
- No dependencies on other experiments

**Cons:**
- Higher risk of bugs
- Reinventing wheel

---

## Next Steps

### Immediate (Next 2 hours)
1. ✅ Create `utils/codi_interface.py` module
   - Copy `ActivationCacherLLaMA` class (model loading)
   - Create `StepImportanceMeasurer` class
   - Implement position-wise zeroing
   - Implement KL divergence measurement

2. ✅ Test on 1 sample problem
   - Load model
   - Extract continuous thoughts
   - Zero positions [0...2], keep [3...5]
   - Measure KL divergence

### After Testing (Next 4 hours)
3. ✅ Implement batching (32 problems at once)
4. ✅ Add checkpointing (every 500 problems)
5. ✅ Validate on 100 problems

### Final Phase (Next 2-3 hours + compute)
6. ✅ Run full sweep (7,473 problems)
7. ✅ Generate summary statistics

**Total Estimated Time:** 8-9 hours development + 2-3 hours compute = **10-12 hours**

✅ **Within original 12-hour estimate!**

---

## Code Reuse Plan

### Files to Copy/Adapt
1. `activation_patching/core/cache_activations_llama.py`
   - Copy `ActivationCacherLLaMA` class → `utils/codi_interface.py`
   - Rename to `CODIInterface`
   - Keep model loading logic intact

2. `scripts/experiments/run_ablation_N_tokens_llama.py`
   - Copy `NTokenPatcher` class → `utils/codi_interface.py`
   - Modify for position-wise zeroing (not all-or-nothing)
   - Add KL divergence measurement

3. New Code Needed
   - `StepImportanceMeasurer` class
   - KL divergence calculation
   - Batching infrastructure
   - Checkpointing logic

### Dependencies to Add
```python
# Already in environment
import torch
import torch.nn.functional as F
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

# From codi/
from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
```

---

## Validation Criteria

### ✅ Model Loading Test
```python
# Should complete in <30 seconds
interface = CODIInterface(model_path)
print("✅ Model loaded")
```

### ✅ Extraction Test
```python
# Should extract (6, 2048) activations
thoughts = interface.extract_continuous_thoughts("Problem text")
assert len(thoughts) == 6
assert thoughts[0].shape == (1, 2048)
print("✅ Extraction works")
```

### ✅ Intervention Test
```python
# Should generate different answers
baseline = interface.generate("Problem text")
ablated = interface.generate_with_zeroed_positions("Problem text", zero_until=3)
assert baseline != ablated  # (usually)
print("✅ Intervention works")
```

### ✅ KL Divergence Test
```python
# Should compute KL > 0
kl = interface.measure_step_importance("Problem", position=0)
assert kl > 0
print(f"✅ KL divergence: {kl:.3f}")
```

---

## Risk Assessment

### 🟢 LOW RISK
- Model loading (proven code)
- Continuous thought extraction (proven code)
- Forward hooks (proven code)

### 🟡 MEDIUM RISK
- Position-wise zeroing (new logic, but straightforward)
- KL divergence (standard PyTorch function)
- Batching (may need GPU memory tuning)

### 🔴 NO HIGH RISKS IDENTIFIED
- All patterns proven in production
- Infrastructure exists and works

---

## Conclusion

✅ **READY TO PROCEED** with high confidence

**Key Takeaways:**
1. Existing code provides 90% of what we need
2. Patterns are well-established and proven
3. Main work is adapting, not inventing
4. Forward hooks approach (AD-002) is validated
5. 10-12 hour estimate is realistic

**Blocker Status:** 🟢 **RESOLVED**
- CODI integration patterns understood
- Code reuse path identified
- Implementation plan validated

**Next Action:** Create `utils/codi_interface.py` based on discovered patterns

---

**Study Complete:** 1 hour
**Findings:** Excellent - exceeded expectations
**Confidence:** HIGH (90%+ reusable code found)
**Recommendation:** Proceed immediately to implementation
