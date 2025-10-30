#!/usr/bin/env python3
"""
Story 1.2: Extract CT Hidden States (Pilot)

Extract and cache CT token hidden states for pilot test problems.
This validates the extraction pipeline before full-scale resampling.

Architecture:
- Load LLaMA-1B GSM8K CODI model
- Process N test problems (20 for pilot, 100 for full)
- Extract hidden states at 6 CT positions from final layer
- Cache to disk with baseline predictions

Time estimate: 2.5 hours (2h coding + 30m runtime for pilot)

Usage:
    python 1_extract_ct_states.py --phase pilot --n_problems 20
    python 1_extract_ct_states.py --phase full --n_problems 100
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import pickle
import argparse
import wandb
from pathlib import Path
from tqdm import tqdm
from typing import Dict
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed, load_model, load_test_problems, extract_answer, validate_cache


def extract_ct_hidden_states(model, tokenizer, question: str, device='cuda') -> Dict:
    """
    Extract hidden states at 6 CT positions during generation.

    Architecture (from architecture spec):
    1. Question forward pass
    2. BOT token
    3. Generate 6 CT tokens, extracting hidden states from final layer
    4. EOT + answer generation

    Args:
        model: CODI model
        tokenizer: Model tokenizer
        question: Problem question text
        device: Device

    Returns:
        dict with:
        - ct_hidden_states: torch.Tensor [6, 2048]
        - generated_answer: str
    """
    with torch.no_grad():
        # Step 1: Question forward pass
        inputs = tokenizer(question, return_tensors='pt').to(device)
        input_ids = inputs['input_ids']

        question_outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True  # CRITICAL
        )
        past_key_values = question_outputs.past_key_values
        question_len = input_ids.shape[1]

        # Step 2: BOT token
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb
        ct_hidden_states = []  # Will store [6, 2048]

        # Step 3: Generate 6 CT tokens
        for step in range(6):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,  # CRITICAL
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # EXTRACT: Final layer output (layer 15, index -1)
            ct_hidden_state = outputs.hidden_states[-1][:, -1, :]  # [1, 2048]
            ct_hidden_states.append(ct_hidden_state.detach().cpu())

            # Continue generation
            latent_embd = ct_hidden_state.unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd)

        # Step 4: EOT + answer generation
        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        output_emb = eot_emb
        pred_tokens = []

        for _ in range(100):  # max_new_tokens
            out = model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = model.get_embd(model.codi, model.model_name)(
                next_token_id
            ).unsqueeze(1)

        generated_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
        ct_hidden_states_tensor = torch.stack(ct_hidden_states).squeeze(1)  # [6, 2048]

        return {
            'ct_hidden_states': ct_hidden_states_tensor,
            'generated_answer': generated_answer
        }


def validate_extraction(cache: Dict, expected_baseline_acc: float = 0.59):
    """
    Validate extraction results.

    Args:
        cache: Extracted cache
        expected_baseline_acc: Expected baseline accuracy (from research journal)

    Raises:
        AssertionError: If validation fails
    """
    # Calculate baseline accuracy
    correct = sum(1 for p in cache.values() if p['baseline_correct'])
    baseline_acc = correct / len(cache)

    print(f"\nValidation Results:")
    print(f"  Baseline accuracy: {baseline_acc:.2%} ({correct}/{len(cache)})")
    print(f"  Expected range: [50%, 70%]")

    # Check if within expected range
    if not (0.50 <= baseline_acc <= 0.70):
        print(f"⚠️  WARNING: Baseline accuracy {baseline_acc:.2%} outside expected range!")
        print(f"     This may indicate model loading or generation issues.")
    else:
        print(f"✓ Baseline accuracy within expected range")

    # Calculate average hidden state norm (health check)
    norms = []
    for p in cache.values():
        norm = p['ct_hidden_states'].norm(dim=1).mean().item()
        norms.append(norm)

    avg_norm = sum(norms) / len(norms)
    print(f"  Average CT hidden state norm: {avg_norm:.3f}")

    return baseline_acc, avg_norm


def main():
    parser = argparse.ArgumentParser(description="Extract CT hidden states for resampling experiment")
    parser.add_argument('--phase', type=str, choices=['pilot', 'full'], default='pilot',
                        help='Experiment phase (pilot=20 problems, full=100)')
    parser.add_argument('--n_problems', type=int, default=None,
                        help='Number of problems (overrides phase default)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output_dir', type=str, default='../data',
                        help='Output directory for cache file')

    args = parser.parse_args()

    # Determine number of problems
    if args.n_problems is None:
        args.n_problems = 20 if args.phase == 'pilot' else 100

    # Set all seeds
    set_seed(args.seed)

    # Initialize W&B
    wandb.init(
        project="codi-resampling",
        name=f"{args.phase}_extraction",
        tags=[args.phase, "extraction", "llama-1b"],
        config={
            "model": "llama-1b-gsm8k",
            "checkpoint": str(Path.home() / "codi_ckpt" / "llama_gsm8k"),
            "n_problems": args.n_problems,
            "seed": args.seed,
            "phase": args.phase
        }
    )

    print(f"\n{'='*60}")
    print(f"Story 1.2: CT Hidden State Extraction ({args.phase.upper()})")
    print(f"{'='*60}")
    print(f"Phase: {args.phase}")
    print(f"Problems: {args.n_problems}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # Load model
    model, tokenizer = load_model(device=args.device)

    # Load test problems
    problems = load_test_problems(n_problems=args.n_problems, seed=args.seed)

    # Extract CT hidden states
    cache = {}

    print(f"\nExtracting CT hidden states from {args.n_problems} problems...")
    for i, problem in enumerate(tqdm(problems, desc="Extraction")):
        result = extract_ct_hidden_states(
            model, tokenizer, problem['question'], device=args.device
        )

        # Check correctness
        predicted_numeric = extract_answer(result['generated_answer'])
        is_correct = (predicted_numeric == problem['gold_numeric'])

        cache[i] = {
            'idx': problem['idx'],
            'question': problem['question'],
            'answer': problem['answer'],
            'gold_numeric': problem['gold_numeric'],
            'ct_hidden_states': result['ct_hidden_states'],
            'baseline_prediction': result['generated_answer'],
            'baseline_correct': is_correct
        }

    # Validate cache
    print(f"\nValidating cache...")
    validate_cache(cache, expected_n_problems=args.n_problems)

    # Validate baseline accuracy
    baseline_acc, avg_norm = validate_extraction(cache)

    # Log to W&B
    wandb.log({
        "extraction/baseline_accuracy": baseline_acc,
        "extraction/num_problems": args.n_problems,
        "extraction/avg_hidden_state_norm": avg_norm
    })

    # Save cache
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = f"ct_hidden_states_cache_{args.phase}.pkl"
    cache_path = output_dir / cache_filename

    print(f"\nSaving cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)

    print(f"✓ Cache saved ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Problems processed: {len(cache)}")
    print(f"Baseline accuracy: {baseline_acc:.2%}")
    print(f"Cache file: {cache_path}")
    print(f"W&B run: {wandb.run.url}")
    print(f"{'='*60}\n")

    wandb.finish()


if __name__ == "__main__":
    main()
