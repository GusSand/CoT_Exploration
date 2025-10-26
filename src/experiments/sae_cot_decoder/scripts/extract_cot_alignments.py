"""
Extract CoT Token Alignments for SAE Training.

This script enriches the tuned_lens activation data with CoT token sequences
from GSM8K, enabling analysis of feature-token correlations.

Usage:
    python extract_cot_alignments.py
"""

import torch
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoTokenizer


def extract_cot_steps(answer_text: str) -> List[str]:
    """Extract CoT calculation steps from GSM8K answer.

    GSM8K format: "16 - 3 - 4 = <<16-3-4=9>>9"
    Extracts: ["16-3-4=9"]
    """
    calculations = re.findall(r'<<([^>]+)>>', answer_text)
    return calculations


def extract_final_answer(answer_text: str) -> str:
    """Extract final answer after ####."""
    match = re.search(r'#### (.+)$', answer_text, re.MULTILINE)
    return match.group(1).strip() if match else None


def tokenize_cot_steps(cot_steps: List[str], tokenizer) -> List[int]:
    """Tokenize CoT steps into token IDs."""
    if not cot_steps:
        return []

    # Concatenate all CoT steps
    cot_text = ' '.join(cot_steps)

    # Tokenize
    token_ids = tokenizer.encode(cot_text, add_special_tokens=False)

    return token_ids


def match_problem_to_gsm8k(problem: Dict, gsm8k_data: List[Dict]) -> Dict:
    """Match stratified problem to GSM8K full dataset by question text."""
    for gsm8k_problem in gsm8k_data:
        if gsm8k_problem['question'] == problem['question']:
            return gsm8k_problem
    return None


def enrich_dataset_with_cot(
    tuned_lens_data: Dict,
    stratified_data: List[Dict],
    gsm8k_full: List[Dict],
    tokenizer,
    split_name: str
) -> Dict:
    """Enrich tuned_lens data with CoT token alignments.

    Args:
        tuned_lens_data: Pre-extracted activation data
        stratified_data: Stratified problem dataset
        gsm8k_full: Full GSM8K dataset with CoT sequences
        tokenizer: LLaMA tokenizer
        split_name: "train" or "test"

    Returns:
        Enriched dataset with CoT token metadata
    """
    print(f"\n{'='*70}")
    print(f"Enriching {split_name} dataset with CoT alignments")
    print(f"{'='*70}")

    # Create question->problem mapping for stratified data
    question_to_problem = {p['question']: p for p in stratified_data}

    # Create problem_id->cot mapping
    problem_cot_map = {}
    matched = 0
    unmatched = []

    print(f"\nMatching {len(stratified_data)} problems to GSM8K...")
    for problem in tqdm(stratified_data, desc="Matching"):
        gsm8k_match = match_problem_to_gsm8k(problem, gsm8k_full)

        if gsm8k_match:
            cot_steps = extract_cot_steps(gsm8k_match['answer'])
            cot_token_ids = tokenize_cot_steps(cot_steps, tokenizer)
            final_answer = extract_final_answer(gsm8k_match['answer'])

            problem_cot_map[problem['gsm8k_id']] = {
                'cot_steps': cot_steps,
                'cot_token_ids': cot_token_ids,
                'cot_text': ' '.join(cot_steps),
                'final_answer_gsm8k': final_answer,
                'final_answer_stratified': problem['answer']
            }
            matched += 1
        else:
            unmatched.append(problem['gsm8k_id'])

    print(f"\n✓ Matched: {matched}/{len(stratified_data)} ({matched/len(stratified_data)*100:.1f}%)")
    if unmatched:
        print(f"✗ Unmatched: {len(unmatched)} problems")
        print(f"  Sample IDs: {unmatched[:5]}")

    # Enrich metadata
    print(f"\nEnriching {len(tuned_lens_data['metadata']['problem_ids'])} samples...")

    enriched_metadata = tuned_lens_data['metadata'].copy()
    enriched_metadata['cot_steps'] = []
    enriched_metadata['cot_token_ids'] = []
    enriched_metadata['cot_text'] = []
    enriched_metadata['num_cot_steps'] = []
    enriched_metadata['has_cot'] = []

    for problem_id in tqdm(tuned_lens_data['metadata']['problem_ids'], desc="Enriching"):
        if problem_id in problem_cot_map:
            cot_data = problem_cot_map[problem_id]
            enriched_metadata['cot_steps'].append(cot_data['cot_steps'])
            enriched_metadata['cot_token_ids'].append(cot_data['cot_token_ids'])
            enriched_metadata['cot_text'].append(cot_data['cot_text'])
            enriched_metadata['num_cot_steps'].append(len(cot_data['cot_steps']))
            enriched_metadata['has_cot'].append(True)
        else:
            # No CoT available
            enriched_metadata['cot_steps'].append([])
            enriched_metadata['cot_token_ids'].append([])
            enriched_metadata['cot_text'].append('')
            enriched_metadata['num_cot_steps'].append(0)
            enriched_metadata['has_cot'].append(False)

    # Create enriched dataset
    enriched_data = {
        'hidden_states': tuned_lens_data['hidden_states'],
        'target_token_ids': tuned_lens_data['target_token_ids'],
        'metadata': enriched_metadata,
        'config': tuned_lens_data['config'],
        'cot_stats': {
            'total_problems': len(set(tuned_lens_data['metadata']['problem_ids'])),
            'matched_problems': matched,
            'match_rate': matched / len(stratified_data),
            'samples_with_cot': sum(enriched_metadata['has_cot']),
            'samples_without_cot': len(enriched_metadata['has_cot']) - sum(enriched_metadata['has_cot'])
        }
    }

    # Print statistics
    print(f"\n{'='*70}")
    print(f"Enrichment Statistics")
    print(f"{'='*70}")
    print(f"Total samples: {len(enriched_metadata['has_cot'])}")
    print(f"Samples with CoT: {sum(enriched_metadata['has_cot'])} ({sum(enriched_metadata['has_cot'])/len(enriched_metadata['has_cot'])*100:.1f}%)")
    print(f"Samples without CoT: {len(enriched_metadata['has_cot']) - sum(enriched_metadata['has_cot'])}")

    # CoT step distribution
    from collections import Counter
    step_counts = [n for n in enriched_metadata['num_cot_steps'] if n > 0]
    if step_counts:
        step_dist = Counter(step_counts)
        print(f"\nCoT Step Distribution:")
        for steps, count in sorted(step_dist.items()):
            print(f"  {steps} steps: {count} samples")

    return enriched_data


def main():
    print("="*70)
    print("CoT TOKEN ALIGNMENT EXTRACTION")
    print("="*70)

    # Paths
    base_dir = Path("/home/paperspace/dev/CoT_Exploration")
    tuned_lens_dir = base_dir / "src/experiments/tuned_lens/data"
    output_dir = base_dir / "src/experiments/sae_cot_decoder/results"

    train_data_path = tuned_lens_dir / "train_data_llama_post_mlp.pt"
    test_data_path = tuned_lens_dir / "test_data_llama_post_mlp.pt"
    stratified_path = base_dir / "src/experiments/activation_patching/data/llama_cot_original_stratified_1000.json"
    gsm8k_full_path = base_dir / "src/experiments/operation_circuits/gsm8k_full.json"

    # Load data
    print("\nLoading datasets...")
    train_data = torch.load(train_data_path, weights_only=False)
    test_data = torch.load(test_data_path, weights_only=False)

    with open(stratified_path, 'r') as f:
        stratified = json.load(f)

    with open(gsm8k_full_path, 'r') as f:
        gsm8k_full = json.load(f)

    print(f"✓ Loaded tuned_lens train: {len(train_data['hidden_states'])} samples")
    print(f"✓ Loaded tuned_lens test: {len(test_data['hidden_states'])} samples")
    print(f"✓ Loaded stratified: {len(stratified)} problems")
    print(f"✓ Loaded GSM8K full: {len(gsm8k_full)} problems")

    # Load tokenizer
    print("\nLoading LLaMA tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    print(f"✓ Tokenizer vocab size: {tokenizer.vocab_size}")

    # Enrich train dataset
    enriched_train = enrich_dataset_with_cot(
        train_data, stratified, gsm8k_full, tokenizer, "train"
    )

    # Enrich test dataset
    enriched_test = enrich_dataset_with_cot(
        test_data, stratified, gsm8k_full, tokenizer, "test"
    )

    # Save enriched datasets
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / "enriched_train_data_with_cot.pt"
    test_output = output_dir / "enriched_test_data_with_cot.pt"

    print(f"\nSaving enriched datasets...")
    torch.save(enriched_train, train_output)
    torch.save(enriched_test, test_output)

    train_size_mb = train_output.stat().st_size / (1024 * 1024)
    test_size_mb = test_output.stat().st_size / (1024 * 1024)

    print(f"✓ Train data saved: {train_output}")
    print(f"  Size: {train_size_mb:.1f} MB")
    print(f"✓ Test data saved: {test_output}")
    print(f"  Size: {test_size_mb:.1f} MB")

    # Save summary statistics
    summary = {
        'train_stats': enriched_train['cot_stats'],
        'test_stats': enriched_test['cot_stats'],
        'total_samples': len(enriched_train['hidden_states']) + len(enriched_test['hidden_states']),
        'config': enriched_train['config']
    }

    summary_path = output_dir / "cot_alignment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved: {summary_path}")

    print(f"\n{'='*70}")
    print("CoT ALIGNMENT EXTRACTION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nNext step: Train SAEs using enriched data")


if __name__ == "__main__":
    main()
