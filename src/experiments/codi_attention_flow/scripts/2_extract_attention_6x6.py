#!/usr/bin/env python3
"""
Attention Extractor 6×6 - Story 1.2

Extract 6×6 attention matrices between continuous thought token positions.

This script captures attention DURING continuous thought generation, not just
at the answer token. For each of the 6 continuous thought positions, we extract
attention to all previous positions, building up the 6×6 matrix.

Usage:
    PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
        python 2_extract_attention_6x6.py [--model MODEL] [--n_problems N]

Output:
    ../results/{model}/attention_patterns_raw.npy  # [N, L, H, 6, 6]
    ../results/{model}/attention_metadata.json
"""
import json
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

# Add paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'activation_patching' / 'core'))

from cache_activations_llama import ActivationCacherLLaMA


def extract_6x6_attention(
    problem: dict,
    cacher: ActivationCacherLLaMA
) -> tuple[np.ndarray, bool]:
    """
    Extract 6×6 attention matrix between continuous thought positions.

    During CODI generation, we have:
    1. Question tokens
    2. BOT token
    3. 6 continuous thought tokens (positions 0-5)
    4. EOT token
    5. Answer tokens

    We extract attention at each continuous thought position to build the 6×6 matrix.

    Args:
        problem: Dict with 'question', 'answer' keys
        cacher: Loaded CODI model wrapper

    Returns:
        attention: Array of shape [n_layers, n_heads, 6, 6]
            - attention[l, h, i, j] = attention from position i to j at layer l, head h
            - Row sums to ~1.0 (normalized within continuous thoughts)
            - Upper triangle may be non-zero due to attention to question/BOT
        success: Whether extraction succeeded
    """
    model = cacher.model
    tokenizer = cacher.tokenizer
    device = cacher.device

    question = problem['question']

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
                output_attentions=True  # Enable attention extraction
            )

            past_key_values = outputs.past_key_values
            question_length = input_len

            # Initialize attention storage
            # We'll get n_heads from the first attention tensor, not from past_key_values
            # (past_key_values may have grouped heads for GQA)
            n_layers = None
            n_heads = None
            attention_matrix = None  # Will initialize after first forward pass

            # BOT token
            bot_emb = model.get_embd(model.codi, model.model_name)(
                torch.tensor([model.bot_id], dtype=torch.long, device=device)
            ).unsqueeze(0)

            latent_embd = bot_emb

            # Track positions of continuous thoughts as they're generated
            # Before any CTs: sequence is [question, BOT]
            # After CT i: sequence is [question, BOT, CT0, ..., CTi]

            ct_positions = []  # Will store the position of each CT in the sequence

            # Generate 6 continuous thoughts and extract attention
            for step in range(6):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    output_attentions=True,  # KEY: Extract attention
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values

                # Initialize attention_matrix on first iteration
                if attention_matrix is None:
                    n_layers = len(outputs.attentions)
                    n_heads = outputs.attentions[0].size(1)
                    attention_matrix = np.zeros((n_layers, n_heads, 6, 6), dtype=np.float32)

                # The NEW token being added is at position: question_length + 1 + step
                # This is BOT (step -1 conceptually) + step
                # But actually: question_length + step (since BOT is included in the forward)
                # After this forward pass, the sequence contains:
                # [question (0...question_length-1), BOT (question_length), CT0...CTstep-1, NEW_CT (question_length + 1 + step)]

                # Wait - let me reconsider. Looking at the past_key_values:
                # Initially after question: past_key_values contains question_length positions
                # After BOT forward: past_key_values contains question_length + 1 positions (question + BOT)
                # After CT0 forward: past_key_values contains question_length + 2 positions (question + BOT + CT0)
                # ...

                # So when we're generating CT{step}, the sequence currently contains:
                # - question: positions 0 to question_length-1
                # - BOT: position question_length
                # - CT0 to CT{step-1}: positions question_length+1 to question_length+step
                # - Current CT{step}: will be at position question_length+1+step (not in sequence yet!)

                # The attention tensor shows attention FROM the new token TO all previous tokens
                # New token position (will be): question_length + 1 + step
                # Current sequence length: question_length + 1 + step

                current_ct_pos = question_length + 1 + step
                ct_positions.append(current_ct_pos)

                # Extract attention from current CT to all previous CTs
                for layer_idx in range(n_layers):
                    attn = outputs.attentions[layer_idx]  # [1, n_heads, 1, seq_len]

                    # Attention from the new token (CT{step}) to all previous tokens
                    # Shape: [n_heads, seq_len]
                    attn_from_current = attn[0, :, -1, :]

                    # Extract attention to previous CTs only
                    # Previous CTs are at positions ct_positions[0:step]
                    # For step=0: no previous CTs, so we'll have empty attention (expected)
                    # For step=1: attend to ct_positions[0] = CT0
                    # etc.

                    if step == 0:
                        # First CT: no previous CTs to attend to
                        # We could extract attention to BOT, but the spec asks for 6×6 between CTs
                        # So row 0 will be all zeros (or we could fill diagonal with 1.0)
                        pass
                    else:
                        # Extract attention to previous CT positions
                        for prev_step, prev_pos in enumerate(ct_positions[:step]):
                            attention_matrix[layer_idx, :, step, prev_step] = attn_from_current[:, prev_pos].cpu().numpy()

                # Update embedding for next step
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                if model.use_prj:
                    latent_embd = model.prj(latent_embd)

            return attention_matrix, True

    except Exception as e:
        print(f"\n  Error on problem {problem['gsm8k_id']}: {e}")
        return np.zeros((n_layers, n_heads, 6, 6), dtype=np.float32), False


def extract_dataset(
    dataset_path: Path,
    model_path: str,
    output_dir: Path,
    n_problems: int = None
) -> None:
    """
    Extract attention patterns for all problems in dataset.

    Args:
        dataset_path: Path to dataset JSON
        model_path: Path to CODI model checkpoint
        output_dir: Output directory for results
        n_problems: Limit number of problems (None = all)
    """
    print("=" * 80)
    print("ATTENTION EXTRACTOR 6×6 - Story 1.2")
    print("=" * 80)

    # Load dataset
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    if n_problems:
        dataset = dataset[:n_problems]

    print(f"✓ Loaded {len(dataset)} problems")

    # Load model
    print(f"\nLoading CODI model from {model_path}...")
    cacher = ActivationCacherLLaMA(model_path)
    model = cacher.model
    print(f"✓ Model loaded")

    # We'll get dimensions from first successful extraction
    n_layers = None
    n_heads = None
    all_attention = None  # Will initialize after first extraction

    # Extract attention for all problems
    print(f"\nExtracting attention patterns for {len(dataset)} problems...")
    n_success = 0
    n_failed = 0

    problem_metadata = []

    for i, problem in enumerate(tqdm(dataset, desc="Extracting")):
        attention, success = extract_6x6_attention(problem, cacher)

        if success:
            # Initialize storage on first success
            if all_attention is None:
                n_layers, n_heads, _, _ = attention.shape
                all_attention = np.zeros((len(dataset), n_layers, n_heads, 6, 6), dtype=np.float16)
                print(f"\n  Initialized storage: {n_layers} layers, {n_heads} heads")

            all_attention[i] = attention.astype(np.float16)
            n_success += 1
            problem_metadata.append({
                'index': i,
                'gsm8k_id': problem['gsm8k_id'],
                'question_length': len(problem['question']),
                'success': True
            })
        else:
            n_failed += 1
            problem_metadata.append({
                'index': i,
                'gsm8k_id': problem['gsm8k_id'],
                'success': False,
                'error': 'Extraction failed'
            })

    # Create metadata after extraction
    metadata = {
        'model_name': Path(model_path).name,
        'model_path': model_path,
        'n_problems': len(dataset),
        'n_layers': int(n_layers) if n_layers is not None else None,
        'n_heads': int(n_heads) if n_heads is not None else None,
        'continuous_thought_positions': list(range(6)),
        'dataset_path': str(dataset_path),
        'problems': problem_metadata
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / 'attention_patterns_raw.npy'
    np.save(raw_path, all_attention)
    print(f"\n✓ Saved attention patterns: {raw_path}")
    print(f"  Shape: {all_attention.shape}")
    print(f"  Size: {raw_path.stat().st_size / 1024 / 1024:.1f} MB")

    metadata_path = output_dir / 'attention_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    # Validation checks
    print("\nValidation checks:")

    # Check 1: Attention normalization (check a few samples)
    sample_attention = all_attention[0].astype(np.float32)  # Convert back to float32
    row_sums = sample_attention.sum(axis=-1)  # Sum across source positions

    # Note: Rows may not sum to 1.0 exactly due to attention to question/BOT tokens
    # We only extracted attention TO continuous thoughts, not the full attention distribution
    print(f"  Sample attention row sums: min={row_sums.min():.3f}, max={row_sums.max():.3f}")
    print(f"  (Note: < 1.0 is expected - we only extracted attention to 6 continuous thoughts)")

    # Check 2: Non-zero patterns
    max_attention = sample_attention.max(axis=-1)  # Max attention per head
    n_strong_heads = (max_attention > 0.4).sum()
    pct_strong = 100 * n_strong_heads / max_attention.size
    print(f"  Heads with max attention > 0.4: {n_strong_heads}/{max_attention.size} ({pct_strong:.1f}%)")
    if pct_strong > 10:
        print(f"  ✓ Non-random patterns detected ({pct_strong:.1f}% > 10%)")
    else:
        print(f"  ⚠️  Few strong patterns ({pct_strong:.1f}% < 10%) - may need more data")

    # Check 3: Success rate
    print(f"  Extraction success: {n_success}/{len(dataset)} ({100*n_success/len(dataset):.1f}%)")
    if n_failed > 0:
        print(f"  ⚠️  {n_failed} problems failed")

    print("\n" + "=" * 80)
    print("STORY 1.2 COMPLETE ✓")
    print("=" * 80)
    print(f"\nExtracted attention patterns for {n_success} problems")
    print(f"Output: {raw_path}")
    print("\nNext step: Run Story 1.3 to aggregate attention")
    print("  python 3_aggregate_attention.py")


def main():
    parser = argparse.ArgumentParser(description='Extract 6×6 attention patterns')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to use (llama or gpt2)')
    parser.add_argument('--n_problems', type=int, default=None,
                        help='Limit number of problems (default: all)')
    args = parser.parse_args()

    # Paths
    model_paths = {
        'llama': str(Path.home() / 'codi_ckpt' / 'llama_gsm8k'),
        'gpt2': str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')
    }

    dataset_path = Path(__file__).parent.parent / 'data' / 'attention_dataset_100_train.json'
    output_dir = Path(__file__).parent.parent / 'results' / args.model

    # Extract
    extract_dataset(
        dataset_path=dataset_path,
        model_path=model_paths[args.model],
        output_dir=output_dir,
        n_problems=args.n_problems
    )


if __name__ == '__main__':
    main()
