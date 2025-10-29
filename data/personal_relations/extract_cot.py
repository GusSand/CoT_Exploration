#!/usr/bin/env python3
"""
Extract and generate Chain-of-Thought reasoning for Personal Relations Task.

This script:
1. Loads the universe_questions_grouped.csv dataset
2. Filters for Extensional English examples
3. Parses the universe relationships
4. Generates step-by-step CoT for each question
5. Validates CoT leads to correct answer
"""

import pandas as pd
import re
from typing import Dict, List, Tuple


def parse_universe(universe_str: str) -> Dict[Tuple[str, str], str]:
    """
    Parse universe string into a dictionary of relationships.

    Args:
        universe_str: String like "Autumn's enemy = Daniel  Autumn's friend = Chad ..."

    Returns:
        Dict mapping (person, relation) -> target_person
        e.g., ("Autumn", "enemy") -> "Daniel"
    """
    relationships = {}

    # Split on ;; for major separations, then parse each segment
    segments = universe_str.replace(';;', '  ').split('  ')

    for segment in segments:
        segment = segment.strip()
        if not segment or '=' not in segment:
            continue

        # Parse patterns like "Autumn's enemy = Daniel"
        match = re.match(r"(\w+)'s\s+(\w+)\s*=\s*(\w+)", segment)
        if match:
            person, relation, target = match.groups()
            relationships[(person, relation)] = target

    return relationships


def parse_question(question: str) -> Tuple[str, List[str]]:
    """
    Parse question into starting person and chain of relations.

    Args:
        question: String like "Brooke's friend's parent's enemy"

    Returns:
        (starting_person, [relation1, relation2, ...])
        e.g., ("Brooke", ["friend", "parent", "enemy"])
    """
    # Remove "Who is " if present
    question = question.replace("Who is ", "").replace("?", "").strip()

    # Split by possessive 's
    parts = question.split("'s ")

    if len(parts) == 1:
        # No possessive, might be "the friend of X" format
        return None, []

    # First part is the starting person
    starting_person = parts[0].strip()

    # Rest are relations
    relations = [p.strip() for p in parts[1:]]

    return starting_person, relations


def generate_cot(starting_person: str, relations: List[str],
                 universe: Dict[Tuple[str, str], str]) -> Tuple[List[str], str]:
    """
    Generate step-by-step Chain-of-Thought reasoning.

    Args:
        starting_person: e.g., "Brooke"
        relations: e.g., ["friend", "parent", "enemy"]
        universe: Relationship mapping

    Returns:
        (steps, final_answer)
        steps: ["1. Brooke's friend = Erin", "2. Erin's parent = Autumn", ...]
        final_answer: The person at the end of the chain
    """
    steps = []
    current_person = starting_person

    for i, relation in enumerate(relations, 1):
        # Look up the relationship
        key = (current_person, relation)
        if key not in universe:
            # Relationship not found - return error
            return [], f"ERROR: {current_person}'s {relation} not found in universe"

        next_person = universe[key]

        # Create step description
        if i == 1:
            step = f"{i}. {starting_person}'s {relation} = {next_person}"
        else:
            # Build accumulated path
            path = "'s ".join(relations[:i])
            step = f"{i}. {starting_person}'s {path} = {next_person}"

        steps.append(step)
        current_person = next_person

    return steps, current_person


def main():
    print("=" * 80)
    print("PERSONAL RELATIONS TASK - CoT EXTRACTION")
    print("=" * 80)

    # Load dataset
    df = pd.read_csv('universe_questions_grouped.csv')
    print(f"\nLoaded {len(df)} total examples")

    # Filter for Extensional English
    ext_eng = df[
        (df['Approach'] == 'SemanticApproach.EXTENSIONAL') &
        (df['Representation'] == 'Representation.ENGLISH')
    ]
    print(f"Filtered to {len(ext_eng)} Extensional English examples")

    # Process each example
    results = []
    errors = []

    for idx, (i, row) in enumerate(ext_eng.iterrows()):
        question = row['Question']
        correct_answer = row['Answer']
        universe_str = row['question_universe']
        complexity = row['Relation_Count']

        # Parse universe
        universe = parse_universe(universe_str)

        # Parse question
        starting_person, relations = parse_question(question)

        if not starting_person:
            errors.append({
                'index': idx,
                'question': question,
                'error': 'Could not parse question format'
            })
            continue

        # Generate CoT
        steps, generated_answer = generate_cot(starting_person, relations, universe)

        # Validate
        if generated_answer != correct_answer:
            errors.append({
                'index': idx,
                'question': question,
                'expected': correct_answer,
                'generated': generated_answer,
                'error': 'Answer mismatch'
            })

        # Store result
        results.append({
            'questionId': row['questionId'],
            'question': question,
            'answer': correct_answer,
            'complexity': complexity,
            'group': row['group'],
            'cot_steps': steps,
            'cot_text': "\n".join(steps),
            'validated': generated_answer == correct_answer
        })

    # Summary
    print(f"\n{'=' * 80}")
    print("GENERATION RESULTS")
    print(f"{'=' * 80}")
    print(f"Successfully generated CoT: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Validation rate: {sum(r['validated'] for r in results) / len(results) * 100:.1f}%")

    if errors:
        print(f"\nFirst 5 errors:")
        for err in errors[:5]:
            print(f"  - Q: {err['question']}")
            print(f"    Error: {err['error']}")
            if 'expected' in err:
                print(f"    Expected: {err['expected']}, Got: {err['generated']}")

    # Show examples
    print(f"\n{'=' * 80}")
    print("SAMPLE GENERATED CoT")
    print(f"{'=' * 80}")
    for complexity in [2, 4, 5]:
        samples = [r for r in results if r['complexity'] == complexity and r['validated']]
        if samples:
            sample = samples[0]
            print(f"\nComplexity {complexity}:")
            print(f"Q: {sample['question']}")
            print(f"CoT:")
            for step in sample['cot_steps']:
                print(f"  {step}")
            print(f"A: {sample['answer']}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_json('personal_relations_with_cot.json', orient='records', indent=2)
    print(f"\n{'=' * 80}")
    print(f"Saved {len(results)} examples to: personal_relations_with_cot.json")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
