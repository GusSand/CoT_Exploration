#!/usr/bin/env python3
"""
Create matched pairs file from validation results.

This script creates the data/problem_pairs_matched.json file needed for
CoT necessity testing by filtering to pairs where both LLaMA and GPT-2
achieve both-correct baseline.
"""

import json
from pathlib import Path


def create_matched_pairs():
    """Create matched pairs file from validation results."""
    
    # Load validation results
    llama_validation_file = 'validation_results_llama_gpt4_532.json'
    gpt2_validation_file = 'validation_results_gpt2_gpt4_532.json'
    pairs_file = 'problem_pairs_gpt4_answers.json'
    
    print("Loading validation results...")
    with open(llama_validation_file, 'r') as f:
        llama_validation = json.load(f)
    
    with open(gpt2_validation_file, 'r') as f:
        gpt2_validation = json.load(f)
    
    with open(pairs_file, 'r') as f:
        all_pairs = json.load(f)
    
    # Index by pair_id
    llama_by_id = {r['pair_id']: r for r in llama_validation['results']}
    gpt2_by_id = {r['pair_id']: r for r in gpt2_validation['results']}
    pairs_by_id = {p['pair_id']: p for p in all_pairs}
    
    # Find matched pairs (both models both-correct)
    matched_pairs = []
    
    for pair_id, pair in pairs_by_id.items():
        # Check if both models have this pair
        if pair_id not in llama_by_id or pair_id not in gpt2_by_id:
            continue
        
        llama_result = llama_by_id[pair_id]
        gpt2_result = gpt2_by_id[pair_id]
        
        # Both models must have both-correct baseline
        if llama_result.get('both_correct', False) and gpt2_result.get('both_correct', False):
            # Create matched pair with validation metadata
            matched_pair = pair.copy()
            matched_pair['matched_validation'] = {
                'llama': {
                    'clean_correct': llama_result['clean']['correct'],
                    'corrupted_correct': llama_result['corrupted']['correct'],
                    'both_correct': True
                },
                'gpt2': {
                    'clean_correct': gpt2_result['clean']['correct'],
                    'corrupted_correct': gpt2_result['corrupted']['correct'],
                    'both_correct': True
                }
            }
            matched_pairs.append(matched_pair)
    
    # Save matched pairs
    output_file = 'data/problem_pairs_matched.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(matched_pairs, f, indent=2)
    
    print("\n" + "=" * 80)
    print("MATCHED PAIRS CREATION")
    print("=" * 80)
    print(f"Total pairs in dataset:           {len(all_pairs)}")
    print(f"LLaMA both-correct:               {llama_validation['statistics']['both_correct']}")
    print(f"GPT-2 both-correct:               {gpt2_validation['statistics']['both_correct']}")
    print(f"MATCHED (both models):            {len(matched_pairs)}")
    print("=" * 80)
    print(f"\n✓ Matched pairs saved to {output_file}")
    print(f"✓ Ready for CoT necessity testing")


if __name__ == "__main__":
    create_matched_pairs()


