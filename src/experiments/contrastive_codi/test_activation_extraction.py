#!/usr/bin/env python3
"""Test activation extraction to ensure CT tokens ≠ regular hidden states."""

import torch
import json
import numpy as np
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add CODI path for imports
sys.path.append('/home/paperspace/dev/CoT_Exploration/codi')

from src.model import CODI, ModelArguments, TrainingArguments

def find_checkpoint_bin(ckpt_root):
    """Find pytorch_model.bin in checkpoint directory."""
    candidates = list(ckpt_root.rglob("pytorch_model.bin"))
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[-1]

def load_codi_from_checkpoint(ckpt_dir: str, base_model: str, device: torch.device) -> CODI:
    """Load CODI model from checkpoint."""
    ckpt_root = Path(ckpt_dir)
    weights_path = find_checkpoint_bin(ckpt_root)

    model_args = ModelArguments(model_name_or_path=base_model, full_precision=False, train=False)
    training_args = TrainingArguments(output_dir=str(ckpt_root), bf16=True, num_latent=6, use_lora=False)
    model = CODI(model_args=model_args, training_args=training_args, lora_config=None).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model

def extract_sample_activations():
    """Extract activations from a small sample to test feature differences."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = "/home/paperspace/codi_ckpt/contrastive_liars_llama1b_smoke_test"
    base_model = "meta-llama/Llama-3.2-1B-Instruct"
    test_data_path = "/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi/data/contrastive_liars_test.json"

    print("🔍 Testing activation extraction differences...")
    print(f"📱 Device: {device}")

    # Load test data (just first few samples)
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)

    sample_data = test_data[:3]  # Just test 3 samples
    print(f"📊 Testing with {len(sample_data)} samples")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load CODI model and extract CT token activations
    print("🔄 Loading CODI model...")
    codi_model = load_codi_from_checkpoint(ckpt_dir, base_model, device)
    codi_model.eval()

    print("🔄 Extracting CT token activations...")
    ct_activations = []

    for i, sample in enumerate(sample_data):
        question = sample.get('question', '')
        print(f"  Sample {i+1}: {question[:50]}...")

        # Prepare input (same format as in extract_activations.py)
        user_only_messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Extract CT tokens using CODI protocol (same as extract_activations.py)

            # 1. Encode question with CODI
            question_outputs = codi_model.codi(
                input_ids=inputs["input_ids"],
                use_cache=True,
                output_hidden_states=True
            )
            past_key_values = question_outputs.past_key_values

            # 2. BOT token embedding
            bot_emb = codi_model.get_embd(codi_model.codi, codi_model.model_name)(
                torch.tensor([codi_model.bot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # 3. Generate 6 CT iterations and collect hidden states
            ct_vectors = []
            for ct_step in range(6):
                outputs = codi_model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Get middle layer (layer 8 for 16-layer model)
                hidden_states = outputs.hidden_states
                layer_idx = len(hidden_states) // 2  # Middle layer
                ct_vec = hidden_states[layer_idx][:, -1, :].squeeze(0).to(torch.float32)
                ct_vectors.append(ct_vec)

                # Next latent embedding is last-layer last-token
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if getattr(codi_model, "use_prj", False):
                    latent_embd = codi_model.prj(latent_embd)

            # 4. Average across 6 CT steps
            mean_ct_vector = torch.stack(ct_vectors, dim=0).mean(dim=0)
            ct_activations.append(mean_ct_vector.cpu().numpy())
            print(f"    ✅ Extracted CT activations: {mean_ct_vector.shape}")

    ct_activations = np.array(ct_activations)  # Shape: (num_samples, hidden_dim)
    print(f"✅ CT activations shape: {ct_activations.shape}")

    # Clear GPU memory
    del codi_model
    torch.cuda.empty_cache()

    # 2. Load base model and extract regular hidden states
    print("🔄 Loading base model...")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_model_obj.eval()

    print("🔄 Extracting regular hidden state activations...")
    regular_activations = []

    for i, sample in enumerate(sample_data):
        question = sample.get('question', '')

        user_only_messages = [{"role": "user", "content": question}]
        text = tokenizer.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = base_model_obj(**inputs, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[-1]  # Last layer
            # Convert to float32 to avoid dtype issues
            reg_vec = hidden_states[0].mean(dim=0).to(torch.float32).cpu().numpy()
            regular_activations.append(reg_vec)  # Mean across seq

    regular_activations = np.array(regular_activations)  # Shape: (num_samples, hidden_dim)
    print(f"✅ Regular activations shape: {regular_activations.shape}")

    # 3. Compare features
    print("🔍 Comparing features...")

    # Check if they're identical (which would be bad)
    are_identical = np.allclose(ct_activations, regular_activations, rtol=1e-5)
    print(f"❓ Are features identical? {are_identical}")

    if not are_identical:
        # Compute differences
        diff_norms = np.linalg.norm(ct_activations - regular_activations, axis=1)
        mean_diff = np.mean(diff_norms)
        std_diff = np.std(diff_norms)

        print(f"✅ Features are different!")
        print(f"📊 Mean L2 difference: {mean_diff:.6f}")
        print(f"📊 Std L2 difference: {std_diff:.6f}")

        # Compute correlation
        ct_flat = ct_activations.flatten()
        reg_flat = regular_activations.flatten()
        correlation = np.corrcoef(ct_flat, reg_flat)[0, 1]
        print(f"📊 Feature correlation: {correlation:.6f}")

        if correlation > 0.95:
            print("⚠️  High correlation - features may be too similar")
        else:
            print("✅ Good feature diversity")

    else:
        print("❌ Features are identical - this indicates a problem!")

    # 4. Check feature statistics
    print("📊 Feature statistics:")
    print(f"CT tokens - Mean: {ct_activations.mean():.6f}, Std: {ct_activations.std():.6f}")
    print(f"Regular  - Mean: {regular_activations.mean():.6f}, Std: {regular_activations.std():.6f}")

    return ct_activations, regular_activations, are_identical

if __name__ == "__main__":
    ct_acts, reg_acts, identical = extract_sample_activations()

    if identical:
        print("\n💥 PROBLEM: Features are identical!")
        print("   This explains why probe performance is the same for both conditions.")
        print("   Need to debug CODI model output extraction.")
    else:
        print("\n🎉 SUCCESS: Features are different!")
        print("   CODI model is producing distinct continuous thought representations.")
        print("   The probe evaluation issues are likely elsewhere.")