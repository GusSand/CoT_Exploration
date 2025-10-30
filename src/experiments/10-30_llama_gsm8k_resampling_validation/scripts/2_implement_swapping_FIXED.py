#!/usr/bin/env python3
"""
Story 1.3: FIXED CT Token Swapping Function

BUGFIX: The original implementation did forward pass BEFORE swapping,
which meant the KV cache was updated with non-swapped states.

Fixed version: Prepare the embedding FIRST, then do forward pass.
"""

# CRITICAL: Set PYTHONHASHSEED before imports
import os
os.environ['PYTHONHASHSEED'] = '42'

import torch
import pickle
import argparse
from pathlib import Path
from typing import Dict
import sys

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils import set_seed, load_model, extract_answer


def generate_with_swapped_ct_FIXED(
    model,
    tokenizer,
    problem_A_cache: dict,
    problem_B_cache: dict,
    swap_position: int,  # 0-5
    device: str = "cuda",
    max_new_tokens: int = 100
) -> str:
    """
    FIXED VERSION: Generate answer for problem A with CT position swapped from problem B.

    Key fix: Use swapped hidden state AS INPUT to forward pass, not after.

    Args:
        model: CODI model
        tokenizer: Model tokenizer
        problem_A_cache: Cached data for problem A
        problem_B_cache: Cached data for problem B
        swap_position: Which CT position to swap (0-5)
        device: 'cuda' or 'cpu'
        max_new_tokens: Max answer tokens

    Returns:
        Generated answer text (str)
    """

    with torch.no_grad():
        # Step 1: Question forward pass (problem A)
        inputs = tokenizer(problem_A_cache['question'], return_tensors='pt').to(device)
        input_ids = inputs['input_ids']

        question_outputs = model.codi(
            input_ids=input_ids,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = question_outputs.past_key_values

        # Step 2: BOT token
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Step 3: Generate CT tokens with swap
        for step in range(6):
            # DECISION POINT: Determine which embedding to use BEFORE forward pass
            if step == swap_position:
                # SWAP: Use problem B's hidden state from cache
                # Cache stores [6, 2048], need [1, 1, 2048] for inputs_embeds
                latent_embd = problem_B_cache['ct_hidden_states'][step].unsqueeze(0).unsqueeze(0).to(device)

                # Apply projection to match what extraction did
                if model.use_prj:
                    latent_embd = model.prj(latent_embd)
            # else: latent_embd is already set from previous iteration

            # NOW do forward pass with the chosen embedding
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Prepare embedding for NEXT iteration (if not swapping next time)
            if step + 1 < 6 and step + 1 != swap_position:
                # NORMAL: Use generated hidden state for next iteration
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

        # Step 4: EOT + answer generation
        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        output_emb = eot_emb
        pred_tokens = []

        for _ in range(max_new_tokens):
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

        generated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        return generated_text


# For testing
if __name__ == "__main__":
    print("This is the FIXED version of CT swapping")
    print("Run 6_test_fixed_swapping.py to verify")
