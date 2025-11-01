"""
Parse CoT tokens from Personal Relations CODI model predictions.

Analyzes whether explicit CoT reasoning is interpretable for relation reasoning tasks.
Extracts tokens from generated CoT sequences to understand reasoning patterns.

Output: personal_relations_cot_analysis.json
"""

import json
import re
from pathlib import Path
from collections import defaultdict, Counter


def extract_cot_reasoning(generated_text):
    """
    Extract CoT reasoning steps from generated text.

    Example:
      " = Paul\nThe answer is: Paul" → ["Paul"]
      " = Amber Chloe's friend = Paul Chloe's friend's parent = Uma\nThe answer is: Uma"
        → ["Amber", "Chloe's friend", "Paul", "Chloe's friend's parent", "Uma"]

    Returns:
        List of reasoning steps/tokens
    """
    # Remove "The answer is:" part
    if "The answer is:" in generated_text:
        reasoning_part = generated_text.split("The answer is:")[0].strip()
    else:
        reasoning_part = generated_text.strip()

    # Split by "=" to get reasoning steps
    steps = []
    if "=" in reasoning_part:
        # Pattern: "entity = value" or "entity's relation = value"
        parts = reasoning_part.split("=")
        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                # Add the entity/relation being computed
                if i < len(parts) - 1:
                    # Get the entity part before "="
                    entity_match = re.findall(r"([A-Z][a-z]+(?:'s (?:child|parent|friend|enemy))+)", part)
                    if entity_match:
                        steps.extend(entity_match)

                # Get the value after "="
                # Look for names at the start of next part
                words = part.split()
                for word in words:
                    if word and word[0].isupper() and word.isalpha():
                        steps.append(word)

    return steps


def extract_relations_and_entities(generated_text):
    """
    Extract relation types and entity names from CoT.

    Returns:
        dict with 'entities' and 'relation_types'
    """
    # Entity names (capitalized words)
    entities = re.findall(r'\b([A-Z][a-z]+)\b', generated_text)

    # Relation types
    relations = re.findall(r"'s (child|parent|friend|enemy)", generated_text)

    return {
        'entities': entities,
        'relations': relations
    }


def categorize_complexity(generated_text):
    """
    Categorize the complexity of reasoning based on chain length.

    Returns:
        'direct' - single hop (e.g., "X's friend")
        'two_hop' - two relations (e.g., "X's friend's parent")
        'multi_hop' - three or more (e.g., "X's friend's parent's child")
    """
    # Count the number of 's
    apostrophe_s_count = generated_text.count("'s")

    if apostrophe_s_count <= 1:
        return 'direct'
    elif apostrophe_s_count == 2:
        return 'two_hop'
    else:
        return 'multi_hop'


def analyze_personal_relations_cot():
    """Analyze CoT token interpretability for Personal Relations dataset."""
    print("="*80)
    print("PERSONAL RELATIONS COT TOKEN ANALYSIS")
    print("="*80)

    # Load predictions
    data_path = Path('models/personal_relations_1b_codi_v2/evaluation_results/personal_relations_1b_eval_FINAL_CORRECT.json')
    print(f"\nLoading predictions from: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    results = data['results']
    print(f"Loaded {len(results)} predictions")
    print(f"Accuracy: {data['accuracy']:.1f}%")

    # Analysis structures
    all_entities = []
    all_relations = []
    complexity_distribution = Counter()
    reasoning_patterns = []

    # Separate correct and incorrect
    correct_examples = []
    incorrect_examples = []

    print(f"\nAnalyzing CoT tokens...")
    for result in results:
        generated = result['generated_full']
        is_correct = result['correct']

        # Extract components
        components = extract_relations_and_entities(generated)
        all_entities.extend(components['entities'])
        all_relations.extend(components['relations'])

        # Categorize complexity
        complexity = categorize_complexity(generated)
        complexity_distribution[complexity] += 1

        # Store reasoning pattern
        pattern = {
            'example_id': result['example_id'],
            'generated': generated,
            'entities': components['entities'],
            'relations': components['relations'],
            'complexity': complexity,
            'correct': is_correct,
            'predicted': result['predicted'],
            'correct_answer': result['correct_answer']
        }
        reasoning_patterns.append(pattern)

        if is_correct:
            correct_examples.append(pattern)
        else:
            incorrect_examples.append(pattern)

    # Compute statistics
    entity_counts = Counter(all_entities)
    relation_counts = Counter(all_relations)

    print(f"\n✓ Analysis complete")
    print(f"\n{'='*80}")
    print("INTERPRETABILITY ANALYSIS")
    print(f"{'='*80}")

    print(f"\n1. Entity Distribution:")
    print(f"   Total entities mentioned: {len(all_entities)}")
    print(f"   Unique entities: {len(entity_counts)}")
    print(f"\n   Most common entities:")
    for entity, count in entity_counts.most_common(10):
        pct = (count / len(all_entities)) * 100
        print(f"     {entity}: {count} ({pct:.1f}%)")

    print(f"\n2. Relation Types:")
    print(f"   Total relations used: {len(all_relations)}")
    print(f"\n   Relation distribution:")
    for relation, count in relation_counts.most_common():
        pct = (count / len(all_relations)) * 100 if all_relations else 0
        print(f"     {relation}: {count} ({pct:.1f}%)")

    print(f"\n3. Reasoning Complexity:")
    total = sum(complexity_distribution.values())
    for complexity, count in complexity_distribution.most_common():
        pct = (count / total) * 100
        print(f"   {complexity}: {count} ({pct:.1f}%)")

    print(f"\n4. Interpretability Assessment:")

    # Check if CoT shows reasoning steps
    has_reasoning_steps = sum(1 for p in reasoning_patterns if len(p['relations']) > 0)
    pct_with_reasoning = (has_reasoning_steps / len(reasoning_patterns)) * 100
    print(f"   Examples with explicit reasoning: {has_reasoning_steps}/{len(reasoning_patterns)} ({pct_with_reasoning:.1f}%)")

    # Average chain length
    avg_relations = len(all_relations) / len(reasoning_patterns)
    print(f"   Average relations per example: {avg_relations:.2f}")

    # Interpretability score based on presence of relation reasoning
    interpretability_score = pct_with_reasoning
    print(f"\n   Interpretability Score: {interpretability_score:.1f}%")
    if interpretability_score > 80:
        print(f"   Assessment: HIGHLY INTERPRETABLE - CoT shows clear relation reasoning")
    elif interpretability_score > 50:
        print(f"   Assessment: MODERATELY INTERPRETABLE - Some reasoning visible")
    else:
        print(f"   Assessment: LOW INTERPRETABILITY - Limited reasoning shown")

    print(f"\n5. Sample Reasoning Patterns:")

    # Show examples of different complexities
    print(f"\n   Direct reasoning (single hop):")
    direct_examples = [p for p in reasoning_patterns if p['complexity'] == 'direct'][:3]
    for ex in direct_examples:
        status = "✓" if ex['correct'] else "✗"
        print(f"     {status} Generated: {ex['generated'][:80]}...")

    print(f"\n   Two-hop reasoning:")
    two_hop_examples = [p for p in reasoning_patterns if p['complexity'] == 'two_hop'][:3]
    for ex in two_hop_examples:
        status = "✓" if ex['correct'] else "✗"
        print(f"     {status} Generated: {ex['generated'][:80]}...")

    print(f"\n   Multi-hop reasoning:")
    multi_hop_examples = [p for p in reasoning_patterns if p['complexity'] == 'multi_hop'][:3]
    for ex in multi_hop_examples:
        status = "✓" if ex['correct'] else "✗"
        print(f"     {status} Generated: {ex['generated'][:80]}...")

    # Save detailed results
    output_dir = Path('src/experiments/10-31_cot_token_interpretability')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'personal_relations_cot_analysis.json'

    print(f"\n{'='*80}")
    print(f"Saving detailed results to: {output_path}")

    output_data = {
        'summary': {
            'total_examples': len(results),
            'accuracy': data['accuracy'],
            'total_entities': len(all_entities),
            'unique_entities': len(entity_counts),
            'total_relations': len(all_relations),
            'avg_relations_per_example': avg_relations,
            'interpretability_score': interpretability_score,
            'complexity_distribution': dict(complexity_distribution),
        },
        'entity_counts': dict(entity_counts),
        'relation_counts': dict(relation_counts),
        'reasoning_patterns': reasoning_patterns[:100],  # First 100 for space
        'correct_examples': correct_examples[:20],
        'incorrect_examples': incorrect_examples[:20],
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Saved ({size_kb:.1f} KB)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

    return output_data


if __name__ == '__main__':
    analyze_personal_relations_cot()
