"""
Extract response token activations from GPT-2 CODI model.

This extracts activations from the GENERATED RESPONSE tokens (final layer),
NOT from continuous thought tokens. This allows comparison to Apollo Research
baseline which used response token activations.
"""

import sys
import json
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


def extract_response_activations(num_honest=500, num_deceptive=500):
    """
    Extract final layer activations from generated response tokens.

    This replicates Apollo Research methodology:
    - Generate response to each question
    - Extract final layer hidden states from response tokens
    - Use for deception detection probes
    """

    print("=" * 80)
    print("Extracting Response Token Activations from GPT-2 CODI")
    print("=" * 80)

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data" / "processed"

    # Load checkpoint
    ckpt_dir = Path("~/codi_ckpt/gpt2_liars_bench/liars_bench_gpt2_codi/gpt2/ep_20/lr_0.003/seed_42").expanduser()

    print(f"\n[1/6] Loading checkpoint from: {ckpt_dir}")

    model_args = ModelArguments(
        model_name_or_path="gpt2",
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
        prj_dim=768,
        prj_dropout=0.0,
        remove_eos=True,
        greedy=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", 'c_fc'],
        init_lora_weights=True,
    )

    print(f"[2/6] Loading CODI model...")
    model = CODI(model_args, training_args, lora_config)

    state_dict = torch.load(ckpt_dir / "pytorch_model.bin")
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "gpt2",
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

    print(f"[3/6] Loading datasets...")

    # Load honest examples
    with open(data_dir / "test_honest.json") as f:
        honest_data = json.load(f)

    # Load deceptive examples
    with open(data_dir / "deceptive_for_probes.json") as f:
        deceptive_data = json.load(f)

    # Sample and deduplicate
    honest_unique = {}
    for ex in honest_data:
        if ex['question'] not in honest_unique:
            honest_unique[ex['question']] = ex

    deceptive_unique = {}
    for ex in deceptive_data:
        if ex['question'] not in deceptive_unique:
            deceptive_unique[ex['question']] = ex

    honest_samples = list(honest_unique.values())[:num_honest]
    deceptive_samples = list(deceptive_unique.values())[:num_deceptive]

    print(f"  Honest examples: {len(honest_samples)}")
    print(f"  Deceptive examples: {len(deceptive_samples)}")

    # Combine for processing
    all_samples = []
    for ex in honest_samples:
        all_samples.append({
            'question': ex['question'],
            'answer': ex['answer'],
            'is_honest': True
        })
    for ex in deceptive_samples:
        all_samples.append({
            'question': ex['question'],
            'answer': ex['answer'],
            'is_honest': False
        })

    print(f"\n[4/6] Extracting response token activations...")
    print(f"  Total examples: {len(all_samples)}")
    print(f"  Batch size: 16")
    print(f"  Max response tokens: 50 (will pool across all tokens)")

    results = []
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(range(0, len(all_samples), batch_size), desc="Processing batches"):
            batch_samples = all_samples[i:i+batch_size]
            questions = [s['question'] for s in batch_samples]

            # Tokenize questions
            batch = tokenizer(
                questions,
                return_tensors="pt",
                padding="longest",
            )

            # Add BOT token
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
            batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
            batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
            batch = batch.to(device)

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

            # Process continuous thought tokens
            for token_idx in range(training_args.num_latent):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Add EOT token
            eot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
            ).unsqueeze(0).expand(batch_size_curr, -1, -1).to(device)

            # Generate response and collect activations
            output = eot_emb
            response_activations = [[] for _ in range(batch_size_curr)]  # Store all response token activations

            max_tokens = 50  # Generate up to 50 response tokens

            for gen_step in range(max_tokens):
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=True,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values

                # Extract final layer hidden state for this token
                final_hidden = out.hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)

                for b in range(batch_size_curr):
                    response_activations[b].append(final_hidden[b].cpu().float().numpy().tolist())

                # Generate next token
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
                next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)

                # Check for EOS
                all_finished = True
                for b in range(batch_size_curr):
                    if next_token_ids[b] != tokenizer.eos_token_id:
                        all_finished = False
                        break

                if all_finished:
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            # Mean pool across all response tokens for each example
            for b, sample in enumerate(batch_samples):
                if len(response_activations[b]) > 0:
                    # Mean pool across all tokens in the response
                    import numpy as np
                    pooled_activation = np.mean(response_activations[b], axis=0).tolist()

                    results.append({
                        'question': sample['question'],
                        'answer': sample['answer'],
                        'is_honest': sample['is_honest'],
                        'response_activation': pooled_activation,  # Mean-pooled final layer activation
                        'num_response_tokens': len(response_activations[b])
                    })

    print(f"\n[5/6] Extracted activations for {len(results)} examples")
    print(f"  Honest: {sum(1 for r in results if r['is_honest'])}")
    print(f"  Deceptive: {sum(1 for r in results if not r['is_honest'])}")

    # Save results
    output_dir = script_dir.parent / "data" / "processed"
    output_file = output_dir / "response_activations_gpt2.json"

    print(f"\n[6/6] Saving to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump({
            'model': 'gpt2',
            'extraction_type': 'response_tokens',
            'layer': 'final_layer',
            'hidden_dim': 768,
            'pooling': 'mean_across_tokens',
            'n_honest': sum(1 for r in results if r['is_honest']),
            'n_deceptive': sum(1 for r in results if not r['is_honest']),
            'samples': results
        }, f)

    print(f"  ✅ Saved {len(results)} samples")

    print("\n" + "=" * 80)
    print("✅ Complete!")
    print("=" * 80)
    print("\nNext step: Train probes on response activations")
    print("  python train_probes_response.py")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_honest', type=int, default=500)
    parser.add_argument('--num_deceptive', type=int, default=500)
    args = parser.parse_args()

    extract_response_activations(args.num_honest, args.num_deceptive)
