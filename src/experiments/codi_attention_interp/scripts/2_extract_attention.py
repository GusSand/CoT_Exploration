#!/usr/bin/env python3
"""
Attention Weight Extraction

Extracts attention weights from answer tokens to continuous thought tokens.

Focus: Attention from the FIRST ANSWER TOKEN to the 6 continuous thought tokens.
This tells us which continuous thoughts the model relies on most when generating the answer.

Usage:
    python 2_extract_attention.py [--test_mode]
"""
import json
import sys
import torch
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))

from cache_activations_llama import ActivationCacherLLaMA, LAYER_CONFIG


def extract_attention_weights(test_mode=False):
    """
    Extract attention from answer tokens to continuous thought tokens.

    Args:
        test_mode: If True, use 10-problem test set.
    """
    # Paths
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    if test_mode:
        dataset_file = Path(__file__).parent.parent / 'results' / 'test_dataset_10.json'
        output_file = Path(__file__).parent.parent / 'results' / 'attention_weights_test.json'
        print("=" * 80)
        print("RUNNING IN TEST MODE (10 problems)")
        print("=" * 80)
    else:
        dataset_file = Path(__file__).parent.parent / 'results' / 'full_dataset_100.json'
        output_file = Path(__file__).parent.parent / 'results' / 'attention_weights_100.json'
        print("=" * 80)
        print("RUNNING IN FULL MODE (100 problems)")
        print("=" * 80)

    print(f"\nLoading LLaMA CODI model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)
    model = cacher.model
    tokenizer = cacher.tokenizer
    device = cacher.device

    # Extract attention at multiple layers
    layers_to_extract = [4, 8, 14]  # early, middle, late
    print(f"Extracting attention at layers: {layers_to_extract}")

    print(f"\nLoading dataset from {dataset_file}...")
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} problems\n")

    results = []

    for problem in tqdm(dataset, desc="Extracting attention"):
        problem_id = problem['gsm8k_id']
        question = problem['question']
        expected_answer = problem['answer']
        difficulty = problem['difficulty']

        try:
            with torch.no_grad():
                # Tokenize input
                inputs = tokenizer(question, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]
                input_len = input_ids.size(1)

                # Get initial embeddings
                input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

                # Forward through question
                outputs = model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True  # KEY: Enable attention extraction
                )

                past_key_values = outputs.past_key_values
                question_length = input_len

                # BOT token
                bot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.bot_id], dtype=torch.long, device=device)
                ).unsqueeze(0)

                latent_embd = bot_emb

                # Generate 6 continuous thoughts
                for latent_step in range(6):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        output_attentions=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values

                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if model.use_prj:
                        latent_embd = model.prj(latent_embd)

                # EOT token
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device=device)
                ).unsqueeze(0)

                # Generate first answer token WITH ATTENTION
                outputs = model.codi(
                    inputs_embeds=eot_emb,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    past_key_values=past_key_values
                )

                # Extract attention weights
                # outputs.attentions: tuple of (num_layers, [batch, num_heads, seq_len, seq_len])
                # We care about the LAST position (answer token) attending to positions

                attention_data = {
                    'problem_id': problem_id,
                    'difficulty': difficulty,
                    'expected_answer': expected_answer,
                    'attention_by_layer': {}
                }

                # Current sequence positions:
                # [0:question_length] = question tokens
                # [question_length] = BOT
                # [question_length+1:question_length+7] = 6 continuous thoughts
                # [question_length+7] = EOT
                # [question_length+8] = first answer token (current position)

                continuous_token_start = question_length + 1
                continuous_token_end = question_length + 7
                answer_token_pos = -1  # Last position in sequence

                for layer_idx in layers_to_extract:
                    # Get attention from this layer
                    # Shape: [batch=1, num_heads, seq_len, seq_len]
                    attn = outputs.attentions[layer_idx]

                    # Average over heads: [seq_len, seq_len]
                    attn_avg = attn[0].mean(dim=0)

                    # Extract attention from answer token (last position) to all tokens
                    answer_attn = attn_avg[answer_token_pos, :]  # Shape: [seq_len]

                    # Extract attention to continuous thoughts specifically
                    continuous_attn = answer_attn[continuous_token_start:continuous_token_end]

                    # Convert to list
                    continuous_attn_list = continuous_attn.cpu().numpy().tolist()

                    attention_data['attention_by_layer'][f'layer_{layer_idx}'] = {
                        'continuous_token_attention': continuous_attn_list,  # List of 6 values
                        'total_attention_to_continuous': float(sum(continuous_attn_list)),
                        'mean_attention_to_continuous': float(np.mean(continuous_attn_list))
                    }

                results.append(attention_data)

        except Exception as e:
            print(f"\nError on problem {problem_id}: {e}")
            import traceback
            traceback.print_exc()

            results.append({
                'problem_id': problem_id,
                'difficulty': difficulty,
                'error': str(e)
            })

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("ATTENTION EXTRACTION COMPLETE")
    print("=" * 80)

    successful = [r for r in results if 'error' not in r]
    print(f"Problems processed: {len(successful)}/{len(dataset)}")

    # Compute average attention per token across all problems
    for layer_idx in layers_to_extract:
        layer_key = f'layer_{layer_idx}'
        token_attentions = [[] for _ in range(6)]

        for result in successful:
            attn_list = result['attention_by_layer'][layer_key]['continuous_token_attention']
            for token_pos in range(6):
                token_attentions[token_pos].append(attn_list[token_pos])

        print(f"\nLayer {layer_idx} - Average attention to each continuous token:")
        for token_pos in range(6):
            avg_attn = np.mean(token_attentions[token_pos])
            std_attn = np.std(token_attentions[token_pos])
            print(f"  Token {token_pos}: {avg_attn:.4f} ± {std_attn:.4f}")

    print(f"\n✓ Results saved to {output_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true', help='Run on 10-problem test set')
    args = parser.parse_args()

    extract_attention_weights(test_mode=args.test_mode)
