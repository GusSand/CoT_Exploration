#!/usr/bin/env python3
"""
Create Matched Pairs Dataset

Filters problem pairs to only include those where BOTH LLaMA AND GPT-2
achieve both-correct baseline performance (Tier 1 pairs).

This ensures fair cross-model comparison - both models are tested on
identical problem sets where both succeed at baseline.

Usage:
    python create_matched_pairs.py \
        --pairs problem_pairs_gpt4_answers.json \
        --llama_results validation_results_llama_gpt4_532.json \
        --gpt2_results validation_results_gpt2_gpt4_532.json \
        --output data/problem_pairs_matched.json \
        --report results/matched_pairs_report.txt
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_json(filepath: str) -> any:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def find_matched_pairs(pairs: List[Dict], llama_results: Dict, gpt2_results: Dict) -> tuple:
    """Find pairs where both models achieve both-correct.

    Returns:
        (matched_pairs, stats)
    """
    # Index results by pair_id
    llama_by_id = {r['pair_id']: r for r in llama_results['results']}
    gpt2_by_id = {r['pair_id']: r for r in gpt2_results['results']}

    matched = []
    stats = {
        'total_pairs': len(pairs),
        'llama_both_correct': 0,
        'gpt2_both_correct': 0,
        'both_models_both_correct': 0,
        'excluded_llama_failed': 0,
        'excluded_gpt2_failed': 0,
        'excluded_both_failed': 0
    }

    for pair in pairs:
        pair_id = pair['pair_id']

        llama_val = llama_by_id.get(pair_id, {})
        gpt2_val = gpt2_by_id.get(pair_id, {})

        llama_both = llama_val.get('both_correct', False)
        gpt2_both = gpt2_val.get('both_correct', False)

        # Track overall stats
        if llama_both:
            stats['llama_both_correct'] += 1
        if gpt2_both:
            stats['gpt2_both_correct'] += 1

        # Only include if BOTH models achieve both-correct
        if llama_both and gpt2_both:
            # Add validation metadata
            pair['matched_validation'] = {
                'llama': {
                    'clean_correct': llama_val.get('clean', {}).get('correct', False),
                    'corrupted_correct': llama_val.get('corrupted', {}).get('correct', False),
                    'clean_predicted': llama_val.get('clean', {}).get('predicted'),
                    'corrupted_predicted': llama_val.get('corrupted', {}).get('predicted')
                },
                'gpt2': {
                    'clean_correct': gpt2_val.get('clean', {}).get('correct', False),
                    'corrupted_correct': gpt2_val.get('corrupted', {}).get('correct', False),
                    'clean_predicted': gpt2_val.get('clean', {}).get('predicted'),
                    'corrupted_predicted': gpt2_val.get('corrupted', {}).get('predicted')
                }
            }
            matched.append(pair)
            stats['both_models_both_correct'] += 1
        else:
            # Track exclusion reasons
            if not llama_both and not gpt2_both:
                stats['excluded_both_failed'] += 1
            elif not llama_both:
                stats['excluded_llama_failed'] += 1
            else:
                stats['excluded_gpt2_failed'] += 1

    return matched, stats


def generate_sanity_check_report(matched_pairs: List[Dict], stats: Dict) -> str:
    """Generate detailed sanity check report."""
    report = []
    report.append("=" * 80)
    report.append("MATCHED PAIRS SANITY CHECK REPORT")
    report.append("=" * 80)
    report.append("")

    report.append("FILTERING STATISTICS:")
    report.append(f"  Total candidate pairs:           {stats['total_pairs']}")
    report.append(f"  LLaMA both-correct:              {stats['llama_both_correct']} ({100*stats['llama_both_correct']/stats['total_pairs']:.1f}%)")
    report.append(f"  GPT-2 both-correct:              {stats['gpt2_both_correct']} ({100*stats['gpt2_both_correct']/stats['total_pairs']:.1f}%)")
    report.append(f"  MATCHED (both models both-correct): {stats['both_models_both_correct']} ({100*stats['both_models_both_correct']/stats['total_pairs']:.1f}%)")
    report.append("")

    report.append("EXCLUSION BREAKDOWN:")
    report.append(f"  Excluded (LLaMA failed):         {stats['excluded_llama_failed']}")
    report.append(f"  Excluded (GPT-2 failed):         {stats['excluded_gpt2_failed']}")
    report.append(f"  Excluded (both failed):          {stats['excluded_both_failed']}")
    report.append("")

    report.append("MATCHED PAIRS LIST:")
    pair_ids = [p['pair_id'] for p in matched_pairs]
    report.append(f"  Total: {len(pair_ids)}")
    report.append(f"  IDs: {pair_ids[:10]}..." if len(pair_ids) > 10 else f"  IDs: {pair_ids}")
    report.append("")

    # Sample verification
    report.append("SAMPLE VERIFICATION (first 3 pairs):")
    for i, pair in enumerate(matched_pairs[:3]):
        report.append(f"\nPair {i+1} (ID: {pair['pair_id']}):")
        report.append(f"  Clean Question: {pair['clean']['question'][:80]}...")
        report.append(f"  Clean Answer: {pair['clean']['answer']}")
        report.append(f"  Corrupted Question: {pair['corrupted']['question'][:80]}...")
        report.append(f"  Corrupted Answer: {pair['corrupted']['answer']}")

        llama = pair['matched_validation']['llama']
        gpt2 = pair['matched_validation']['gpt2']

        report.append(f"  LLaMA: clean={llama['clean_predicted']} (✓), corrupted={llama['corrupted_predicted']} (✓)")
        report.append(f"  GPT-2: clean={gpt2['clean_predicted']} (✓), corrupted={gpt2['corrupted_predicted']} (✓)")

    report.append("")
    report.append("=" * 80)
    report.append("VERIFICATION STATUS: ✓ ALL PAIRS VALIDATED")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Create matched pairs dataset")
    parser.add_argument('--pairs', type=str, required=True, help='All pairs with GPT-4 answers')
    parser.add_argument('--llama_results', type=str, required=True, help='LLaMA validation results')
    parser.add_argument('--gpt2_results', type=str, required=True, help='GPT-2 validation results')
    parser.add_argument('--output', type=str, required=True, help='Output matched pairs JSON')
    parser.add_argument('--report', type=str, default=None, help='Output sanity check report (optional)')

    args = parser.parse_args()

    print("Loading data...")
    pairs = load_json(args.pairs)
    llama_results = load_json(args.llama_results)
    gpt2_results = load_json(args.gpt2_results)

    print(f"Finding matched pairs (both models both-correct)...")
    matched_pairs, stats = find_matched_pairs(pairs, llama_results, gpt2_results)

    # Save matched pairs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(matched_pairs, f, indent=2)

    print(f"\n✓ Saved {len(matched_pairs)} matched pairs to {args.output}")

    # Generate and save report
    report = generate_sanity_check_report(matched_pairs, stats)
    print("\n" + report)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, 'w') as f:
            f.write(report)
        print(f"\n✓ Saved sanity check report to {args.report}")


if __name__ == "__main__":
    main()
