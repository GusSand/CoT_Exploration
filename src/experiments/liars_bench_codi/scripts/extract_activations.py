"""
Story 5.1: Extract Activations for Probe Training

Extracts continuous thought activations from GPT-2 CODI model for:
- Honest examples (from test set)
- Deceptive examples (from deceptive_for_probes dataset)

Creates balanced dataset for training deception detection probes.
"""

import sys
import json
import os
import math
import logging
import torch
import transformers
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file

# Add CODI directory to path
codi_path = Path(__file__).parent.parent.parent.parent.parent / "codi"
sys.path.insert(0, str(codi_path))

from src.model import CODI, ModelArguments, DataArguments, TrainingArguments
from peft import LoraConfig, TaskType

logging.basicConfig(level=logging.INFO)


def extract_activations(
    model_name: str = "gpt2",
    n_honest: int = 500,
    n_deceptive: int = 500
):
    """
    Extract continuous thought activations for probe training.

    Args:
        model_name: "gpt2" or "llama"
        n_honest: Number of honest examples to extract
        n_deceptive: Number of deceptive examples to extract
    """

    print("=" * 80)
    print(f"STORY 5.1: Extracting Activations for {model_name.upper()} Probe Training")
    print("=" * 80)

    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "processed"

    if model_name == "gpt2":
        ckpt_dir = Path("~/codi_ckpt/gpt2_liars_bench/liars_bench_gpt2_codi/gpt2/ep_20/lr_0.003/seed_42").expanduser()
        model_path = "gpt2"
        hidden_size = 768
        prj_dim = 768
        target_modules = ["c_attn", "c_proj", 'c_fc']
    elif model_name == "llama":
        ckpt_dir = Path("~/codi_ckpt/llama_liars_bench/liars_bench_llama_codi/llama/ep_10/lr_0.0008/seed_42").expanduser()
        model_path = "meta-llama/Llama-3.2-1B-Instruct"
        hidden_size = 2048
        prj_dim = 2048
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n[1/8] Loading checkpoint from: {ckpt_dir}")

    # Model arguments
    model_args = ModelArguments(
        model_name_or_path=model_path,
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir=str(ckpt_dir),
        token=None
    )

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir),
        model_max_length=512,
        num_latent=6,
        use_prj=True,
        prj_dim=prj_dim,
        prj_dropout=0.0,
        remove_eos=True,
        greedy=True
    )

    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    print(f"[2/8] Loading CODI model...")
    model = CODI(model_args, training_args, lora_config)

    # Load trained weights
    state_dict = torch.load(ckpt_dir / "pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    # Setup tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    device = "cuda"
    model = model.to('cuda')
    model.to(torch.bfloat16)
    model.eval()

    print(f"[3/8] Loading datasets...")

    # Load honest examples
    with open(data_dir / "test_honest.json") as f:
        honest_data = json.load(f)

    # Load deceptive examples
    with open(data_dir / "deceptive_for_probes.json") as f:
        deceptive_data = json.load(f)

    print(f"  Available: {len(honest_data)} honest, {len(deceptive_data)} deceptive")

    # Sample balanced dataset
    import random
    random.seed(42)

    honest_sample = random.sample(honest_data, min(n_honest, len(honest_data)))
    deceptive_sample = random.sample(deceptive_data, min(n_deceptive, len(deceptive_data)))

    print(f"  Sampled: {len(honest_sample)} honest, {len(deceptive_sample)} deceptive")

    # Define layers to extract from
    if model_name == "gpt2":
        # GPT-2 has 12 layers (0-11)
        layers = {
            "layer_4": 4,   # Early
            "layer_8": 8,   # Middle
            "layer_11": 11  # Late
        }
    else:
        # LLaMA has 16 layers (0-15)
        layers = {
            "layer_4": 4,   # Early
            "layer_10": 10, # Middle
            "layer_15": 15  # Late
        }

    print(f"\n[4/8] Extracting activations from layers: {list(layers.keys())}")

    def extract_from_examples(examples, is_honest):
        """Extract activations from a list of examples."""
        results = []

        questions = [ex['question'] for ex in examples]
        batch_size = 16
        num_batches = math.ceil(len(questions) / batch_size)

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc=f"{'Honest' if is_honest else 'Deceptive'} examples"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(questions))
                batch_questions = questions[start_idx:end_idx]

                # Tokenize
                batch = tokenizer(
                    batch_questions,
                    return_tensors="pt",
                    padding="longest",
                )

                # Add BOT token
                bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
                batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
                batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
                batch = {k: v.to(device) for k, v in batch.items()}

                batch_size_curr = batch["input_ids"].size(0)

                # Encode question
                outputs = model.codi(
                    input_ids=batch["input_ids"],
                    use_cache=True,
                    output_hidden_states=True,
                    attention_mask=batch["attention_mask"]
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

                # Store activations for each latent token
                token_activations = {layer_name: [[] for _ in range(batch_size_curr)]
                                   for layer_name in layers.keys()}

                # Iterate through latent tokens
                for token_idx in range(training_args.num_latent):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values
                    hidden_states = outputs.hidden_states
                    latent_embd = hidden_states[-1][:, -1, :].unsqueeze(1)

                    # Extract from specified layers
                    for layer_name, layer_idx in layers.items():
                        layer_activation = hidden_states[layer_idx][:, -1, :]  # [batch_size, hidden_size]
                        for b in range(batch_size_curr):
                            token_activations[layer_name][b].append(
                                layer_activation[b].cpu().float().numpy().tolist()
                            )

                    if training_args.use_prj:
                        latent_embd = model.prj(latent_embd)

                # Store results
                for b in range(batch_size_curr):
                    example_idx = start_idx + b
                    results.append({
                        "question": examples[example_idx]['question'],
                        "answer": examples[example_idx]['answer'],
                        "is_honest": is_honest,
                        "thoughts": {layer_name: token_activations[layer_name][b] for layer_name in layers.keys()}  # Only this sample's activations
                    })

        return results

    print(f"[5/8] Extracting from honest examples...")
    honest_results = extract_from_examples(honest_sample, is_honest=True)

    print(f"[6/8] Extracting from deceptive examples...")
    deceptive_results = extract_from_examples(deceptive_sample, is_honest=False)

    print(f"[7/8] Saving results...")

    # Combine and save
    all_results = {
        "model": model_name,
        "n_honest": len(honest_results),
        "n_deceptive": len(deceptive_results),
        "layers": list(layers.keys()),
        "num_tokens": training_args.num_latent,
        "hidden_size": hidden_size,
        "samples": honest_results + deceptive_results
    }

    output_dir = script_dir.parent / "data" / "processed"
    output_file = output_dir / f"probe_dataset_{model_name}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"  ✅ Saved to: {output_file}")

    # Print statistics
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"\n[8/8] Summary:")
    print(f"  Total examples: {len(all_results['samples'])}")
    print(f"  Honest: {all_results['n_honest']}")
    print(f"  Deceptive: {all_results['n_deceptive']}")
    print(f"  Layers: {all_results['layers']}")
    print(f"  Tokens per layer: {all_results['num_tokens']}")
    print(f"  File size: {file_size_mb:.2f} MB")

    print("\n" + "=" * 80)
    print("✅ STORY 5.1 COMPLETE: Activations extracted successfully!")
    print("=" * 80)
    print(f"\n🎯 Next: Train deception detection probes (Story 5.2)")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract activations for probe training")
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "llama"],
                       help="Model to extract from")
    parser.add_argument("--n-honest", type=int, default=500,
                       help="Number of honest examples")
    parser.add_argument("--n-deceptive", type=int, default=500,
                       help="Number of deceptive examples")

    args = parser.parse_args()

    extract_activations(
        model_name=args.model,
        n_honest=args.n_honest,
        n_deceptive=args.n_deceptive
    )
