"""
Vanilla Model Control Analysis (GPT-2 and Llama)
=================================================

Baseline comparison: Analyze vanilla models (no CODI) on same problems.
For each generated token, compute similarity between:
- Hidden state before token generation
- Actual token embedding that was generated

This serves as a control to compare against CODI's continuous thought similarities.
"""

import torch
import transformers
from torch.nn import functional as F
import os
import json
import re
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
import numpy as np
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def extract_answer_number(sentence: str) -> float:
    """Extract numerical answer from generated text"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def compute_similarity_metrics(
    hidden_state: torch.Tensor,  # [hidden_dim]
    token_embedding: torch.Tensor,  # [hidden_dim]
) -> Tuple[float, float]:
    """
    Compute cosine similarity and normalized L2 distance.

    Returns:
        cosine_similarity: float
        norm_l2_distance: float
    """
    # Normalize to unit length
    hidden_norm = F.normalize(hidden_state.unsqueeze(0), p=2, dim=1)  # [1, hidden_dim]
    token_norm = F.normalize(token_embedding.unsqueeze(0), p=2, dim=1)  # [1, hidden_dim]

    # Cosine similarity
    cosine_sim = torch.mm(hidden_norm, token_norm.t()).squeeze().item()

    # Normalized L2 distance
    diff = hidden_norm - token_norm
    norm_l2_dist = torch.norm(diff, p=2).item()

    return cosine_sim, norm_l2_dist


def vanilla_analysis(
    model_name: str,
    num_examples: int = 100,
    max_new_tokens: int = 100,
    output_dir: str = None
):
    """
    Run vanilla model analysis on GSM8K problems.

    Args:
        model_name: "gpt2" or "llama"
        num_examples: Number of examples to process
        max_new_tokens: Maximum tokens to generate per example
        output_dir: Output directory path
    """

    if output_dir is None:
        output_dir = Path(f"outputs/vanilla_control/{model_name}")
    else:
        output_dir = Path(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"vanilla_{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {run_dir}")

    # Load model based on type
    print(f"Loading vanilla {model_name} model...")
    if model_name == "gpt2":
        model = transformers.GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        embedding_layer = model.transformer.wte
    elif model_name == "llama":
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
        embedding_layer = model.model.embed_tokens
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    model.eval()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("zen-E/GSM8k-Aug")
    test_set = dataset['test'].select(range(num_examples))

    questions = []
    answers = []

    for example in test_set:
        questions.append(example['question'].strip())
        ans_text = example['answer']
        try:
            answers.append(float(ans_text.replace(',', '')))
        except:
            answers.append(float('inf'))

    print(f"Processing {len(questions)} examples...")

    all_results = []

    for q_idx, question in enumerate(questions):
        if (q_idx + 1) % 10 == 0:
            print(f"Processing question {q_idx+1}/{len(questions)}")

        # Tokenize question
        inputs = tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Storage for token-level analysis
        generated_tokens = []
        token_similarities = []  # (cosine_sim, norm_l2_dist) for each generated token

        with torch.no_grad():
            # Generate answer token by token
            for step in range(max_new_tokens):
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False
                )

                # Get hidden state at last position (before generating next token)
                hidden_states = outputs.hidden_states[-1]  # Last layer
                last_hidden = hidden_states[0, -1, :]  # [hidden_dim]

                # Get next token (greedy)
                logits = outputs.logits[0, -1, :]
                next_token_id = torch.argmax(logits).item()

                # Stop if EOS
                if next_token_id == tokenizer.eos_token_id:
                    break

                # Get embedding of the generated token
                next_token_embedding = embedding_layer(torch.tensor([next_token_id]).to(device)).squeeze(0)

                # Compute similarity between hidden state and generated token embedding
                cosine_sim, norm_l2_dist = compute_similarity_metrics(
                    last_hidden,
                    next_token_embedding
                )

                generated_tokens.append({
                    'token_id': next_token_id,
                    'token_text': tokenizer.decode([next_token_id]),
                    'cosine_similarity': cosine_sim,
                    'norm_l2_distance': norm_l2_dist
                })
                token_similarities.append((cosine_sim, norm_l2_dist))

                # Append token to input for next iteration
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]]).to(device)], dim=1)

        # Decode generated text
        generated_ids = [t['token_id'] for t in generated_tokens]
        generated_text = tokenizer.decode(generated_ids)
        predicted_answer = extract_answer_number(generated_text)
        ground_truth = answers[q_idx]
        is_correct = abs(predicted_answer - ground_truth) < 0.01 if ground_truth != float('inf') else False

        # Calculate aggregate statistics
        if token_similarities:
            avg_cosine_sim = np.mean([s[0] for s in token_similarities])
            avg_norm_l2_dist = np.mean([s[1] for s in token_similarities])
            std_cosine_sim = np.std([s[0] for s in token_similarities])
            std_norm_l2_dist = np.std([s[1] for s in token_similarities])
        else:
            avg_cosine_sim = avg_norm_l2_dist = std_cosine_sim = std_norm_l2_dist = 0.0

        result = {
            'question_id': q_idx,
            'question': question,
            'ground_truth': ground_truth,
            'predicted_answer': predicted_answer,
            'generated_text': generated_text,
            'is_correct': is_correct,
            'num_generated_tokens': len(generated_tokens),
            'generated_tokens': generated_tokens,
            'avg_cosine_similarity': avg_cosine_sim,
            'avg_norm_l2_distance': avg_norm_l2_dist,
            'std_cosine_similarity': std_cosine_sim,
            'std_norm_l2_distance': std_norm_l2_dist
        }

        all_results.append(result)

    # Save results
    with open(run_dir / "results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Calculate overall statistics
    num_correct = sum(1 for r in all_results if r['is_correct'])
    accuracy = num_correct / len(all_results) if all_results else 0.0

    # Aggregate similarity metrics across all tokens
    all_cosine_sims = []
    all_norm_l2_dists = []
    for result in all_results:
        for token in result['generated_tokens']:
            all_cosine_sims.append(token['cosine_similarity'])
            all_norm_l2_dists.append(token['norm_l2_distance'])

    summary = {
        'model': f'vanilla {model_name}',
        'model_name': model_name,
        'num_examples': len(all_results),
        'accuracy': accuracy,
        'correct_predictions': num_correct,
        'similarity_metrics': {
            'all_tokens': {
                'avg_cosine_similarity': float(np.mean(all_cosine_sims)) if all_cosine_sims else 0.0,
                'std_cosine_similarity': float(np.std(all_cosine_sims)) if all_cosine_sims else 0.0,
                'avg_norm_l2_distance': float(np.mean(all_norm_l2_dists)) if all_norm_l2_dists else 0.0,
                'std_norm_l2_distance': float(np.std(all_norm_l2_dists)) if all_norm_l2_dists else 0.0,
                'total_tokens_analyzed': len(all_cosine_sims)
            },
            'per_example_averages': {
                'avg_cosine_similarity': float(np.mean([r['avg_cosine_similarity'] for r in all_results])),
                'avg_norm_l2_distance': float(np.mean([r['avg_norm_l2_distance'] for r in all_results]))
            }
        },
        'timestamp': timestamp
    }

    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print(f"VANILLA {model_name.upper()} CONTROL ANALYSIS - SUMMARY")
    print("="*80)
    print(f"Model: vanilla {model_name} (no CODI)")
    print(f"Examples: {len(all_results)}")
    print(f"Accuracy: {accuracy*100:.1f}% ({num_correct}/{len(all_results)})")
    print(f"\nSimilarity Metrics (all {len(all_cosine_sims)} generated tokens):")
    print(f"  Avg Cosine Similarity: {summary['similarity_metrics']['all_tokens']['avg_cosine_similarity']:.4f} "
          f"(±{summary['similarity_metrics']['all_tokens']['std_cosine_similarity']:.4f})")
    print(f"  Avg Norm L2 Distance:  {summary['similarity_metrics']['all_tokens']['avg_norm_l2_distance']:.4f} "
          f"(±{summary['similarity_metrics']['all_tokens']['std_norm_l2_distance']:.4f})")
    print(f"\nOutputs saved to: {run_dir}")
    print("="*80)

    return all_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=["gpt2", "llama"],
                       help="Model to analyze")
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Number of examples to process")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory")
    args = parser.parse_args()

    num_examples = args.num_examples or int(os.getenv("NUM_EXAMPLES", "100"))

    results, summary = vanilla_analysis(
        model_name=args.model,
        num_examples=num_examples,
        output_dir=args.output_dir
    )
