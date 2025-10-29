#!/usr/bin/env python3
"""
Sanity Check & Baseline Establishment - Story 0

Validate that inference pipeline works and establish baseline accuracy
before running ablation experiments.

Usage:
    python 0_sanity_check.py [--model MODEL] [--n_problems N]

Output:
    ../results/{model}_baseline.json
"""
import json
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from utils import load_model, extract_answer, validate_model_architecture


def run_sanity_check(model_name='llama', n_problems=100):
    """
    Validate baseline inference pipeline.

    Args:
        model_name: 'llama' or 'gpt2'
        n_problems: Number of test problems (default 100)

    Returns:
        baseline_accuracy: float (0-1)
    """
    print("=" * 80)
    print(f"SANITY CHECK - Story 0")
    print("=" * 80)
    print(f"\nModel: {model_name.upper()}")
    print(f"Test problems: {n_problems}")

    # Load model
    print(f"\nLoading CODI model...")
    model, tokenizer = load_model(model_name)
    print(f"✓ Model loaded")

    # Validate architecture
    print(f"\nValidating architecture...")
    validate_model_architecture(model, model_name, tokenizer)

    # Load test set
    print(f"\nLoading GSM8K test set...")
    dataset = load_dataset('gsm8k', 'main', split='test')
    test_problems = dataset.select(range(n_problems))
    print(f"✓ Loaded {len(test_problems)} problems")

    # Run inference
    print(f"\nRunning baseline inference...")
    n_correct = 0
    n_failed = 0
    results_detail = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    with torch.no_grad():
        for i, problem in enumerate(tqdm(test_problems, desc="Evaluating")):
            question = problem['question']
            gold_answer = extract_answer(problem['answer'])

            if gold_answer is None:
                n_failed += 1
                continue

            try:
                # Generate answer using CODI
                # Use the same pattern as the extraction script
                inputs = tokenizer(question, return_tensors="pt").to(device)
                input_ids = inputs["input_ids"]
                input_len = input_ids.size(1)

                # Get initial embeddings
                input_embd = model.get_embd(model.codi, model.model_name)(input_ids).to(device)

                # Forward through question
                outputs = model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True
                )

                past_key_values = outputs.past_key_values

                # BOT token
                bot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.bot_id], dtype=torch.long, device=device)
                ).unsqueeze(0)

                latent_embd = bot_emb

                # Generate 6 continuous thoughts
                for step in range(6):
                    outputs = model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )
                    past_key_values = outputs.past_key_values

                    # Update embedding for next step
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    if model.use_prj:
                        latent_embd = model.prj(latent_embd)

                # EOT token
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device=device)
                ).unsqueeze(0)

                output_emb = eot_emb

                # Generate answer tokens (manual token-by-token generation)
                pred_tokens = []
                max_new_tokens = 256
                for _ in range(max_new_tokens):
                    out = model.codi(
                        inputs_embeds=output_emb,
                        use_cache=True,
                        past_key_values=past_key_values
                    )

                    past_key_values = out.past_key_values
                    logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                    # Greedy decoding
                    next_token_id = torch.argmax(logits, dim=-1)

                    if next_token_id.item() == tokenizer.eos_token_id:
                        break

                    pred_tokens.append(next_token_id.item())
                    output_emb = model.get_embd(model.codi, model.model_name)(
                        next_token_id
                    ).unsqueeze(1)

                # Decode answer
                generated_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
                pred_answer = extract_numeric_answer(generated_text, question)

                is_correct = (pred_answer == gold_answer)
                if is_correct:
                    n_correct += 1

                results_detail.append({
                    'problem_idx': i,
                    'question': question[:100] + '...',  # Truncate for storage
                    'gold_answer': gold_answer,
                    'pred_answer': pred_answer,
                    'correct': is_correct
                })

            except Exception as e:
                print(f"\nError on problem {i}: {e}")
                n_failed += 1
                results_detail.append({
                    'problem_idx': i,
                    'error': str(e)
                })

    accuracy = n_correct / (n_problems - n_failed) if (n_problems - n_failed) > 0 else 0.0

    # Save baseline
    results = {
        'model': model_name,
        'n_problems': n_problems,
        'n_correct': n_correct,
        'n_failed': n_failed,
        'accuracy': accuracy,
        'results_detail': results_detail  # Save ALL predictions for qualitative analysis
    }

    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f'{model_name}_baseline.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("BASELINE RESULTS")
    print("=" * 80)
    print(f"\n{model_name.upper()} Baseline Accuracy: {accuracy:.2%} ({n_correct}/{n_problems - n_failed})")
    print(f"Failed problems: {n_failed}")

    # Validate results
    if accuracy < 0.50:
        print("\n⚠️  WARNING: Accuracy is low (<50%). Check model or inference pipeline.")
    elif accuracy > 0.95:
        print("\n⚠️  WARNING: Accuracy is very high (>95%). Verify test set is correct.")
    else:
        print("\n✓ Baseline accuracy is in expected range (50-95%)")

    print(f"\n✓ Saved: {output_path}")

    return accuracy


def extract_numeric_answer(generated_text, question):
    """
    Extract numeric answer from generated text.

    CODI generates only the answer text (not the question).
    We need to extract just the numeric answer.

    Args:
        generated_text: Generated answer text
        question: Original question (unused, kept for compatibility)

    Returns:
        int or None: Extracted answer
    """
    # Use the full generated text as answer
    answer_part = generated_text.strip()

    # Try to find #### marker (if model learned GSM8K format)
    if '####' in answer_part:
        try:
            answer_str = answer_part.split('####')[1].strip()
            answer_str = answer_str.split()[0]  # Take first token after ####
            answer_str = answer_str.replace(',', '')
            return int(answer_str)
        except (IndexError, ValueError):
            pass

    # Otherwise, try to extract last number in the text
    import re
    numbers = re.findall(r'-?\d+(?:,\d{3})*', answer_part)
    if numbers:
        last_number = numbers[-1].replace(',', '')
        try:
            return int(last_number)
        except ValueError:
            pass

    return None


def main():
    parser = argparse.ArgumentParser(description='Run sanity check and establish baseline')
    parser.add_argument('--model', type=str, default='llama',
                        choices=['llama', 'gpt2'],
                        help='Model to test (llama or gpt2)')
    parser.add_argument('--n_problems', type=int, default=100,
                        help='Number of test problems (default: 100)')
    args = parser.parse_args()

    run_sanity_check(model_name=args.model, n_problems=args.n_problems)


if __name__ == '__main__':
    main()
