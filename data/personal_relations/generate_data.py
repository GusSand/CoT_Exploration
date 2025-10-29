#!/usr/bin/env python3
"""
Generate additional Personal Relations Task examples.

Target: 1,000-5,000 examples for proper CODI training.
"""

import json
import random
from typing import List, Dict, Tuple, Set
from itertools import permutations, combinations
import re


# Common names to use
NAMES = [
    "Alice", "Bob", "Charlie", "David", "Emma", "Frank",
    "Grace", "Henry", "Ivy", "Jack", "Kate", "Liam",
    "Mia", "Noah", "Olivia", "Paul", "Quinn", "Ruby",
    "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zach", "Amber", "Blake", "Chloe", "Dylan"
]

RELATIONS = ["parent", "child", "friend", "enemy"]

# Inverse relations
INVERSE = {
    "parent": "child",
    "child": "parent",
    "friend": "friend",  # symmetric
    "enemy": "enemy"     # symmetric
}


def generate_universe(num_people: int = 6) -> Dict[Tuple[str, str], str]:
    """
    Generate a valid universe with relationship constraints.

    Constraints:
    - Each person has exactly 1 parent
    - Each person has 0-2 children
    - Each person has 1-2 friends
    - Each person has 1-2 enemies
    """
    people = random.sample(NAMES, num_people)
    universe = {}

    # Assign parents (each person has 1 parent)
    # Make sure not everyone is a parent (to avoid cycles)
    available_parents = people.copy()
    for person in people:
        possible_parents = [p for p in available_parents if p != person]
        if possible_parents:
            parent = random.choice(possible_parents)
            universe[(person, "parent")] = parent
            universe[(parent, "child")] = person  # Add inverse

    # Assign friends (1-2 per person, symmetric)
    assigned_friendships = set()
    for person in people:
        current_friends = sum(1 for (p, rel), _ in universe.items()
                             if p == person and rel == "friend")
        num_friends = random.randint(1, 2) - current_friends

        possible_friends = [p for p in people
                           if p != person
                           and (person, p) not in assigned_friendships
                           and (p, person) not in assigned_friendships]

        for _ in range(min(num_friends, len(possible_friends))):
            if possible_friends:
                friend = random.choice(possible_friends)
                universe[(person, "friend")] = friend
                universe[(friend, "friend")] = person  # symmetric
                assigned_friendships.add((person, friend))
                assigned_friendships.add((friend, person))
                possible_friends.remove(friend)

    # Assign enemies (1-2 per person, symmetric)
    assigned_enemies = set()
    for person in people:
        current_enemies = sum(1 for (p, rel), _ in universe.items()
                             if p == person and rel == "enemy")
        num_enemies = random.randint(1, 2) - current_enemies

        possible_enemies = [p for p in people
                           if p != person
                           and (person, p) not in assigned_enemies
                           and (p, person) not in assigned_enemies
                           # Don't make friends enemies
                           and universe.get((person, "friend")) != p]

        for _ in range(min(num_enemies, len(possible_enemies))):
            if possible_enemies:
                enemy = random.choice(possible_enemies)
                universe[(person, "enemy")] = enemy
                universe[(enemy, "enemy")] = person  # symmetric
                assigned_enemies.add((person, enemy))
                assigned_enemies.add((enemy, person))
                possible_enemies.remove(enemy)

    return universe


def universe_to_string(universe: Dict[Tuple[str, str], str]) -> str:
    """Convert universe dict to string format."""
    statements = []
    for (person, relation), target in sorted(universe.items()):
        statements.append(f"{person}'s {relation} = {target}")
    return ";;  ".join(statements)


def generate_cot(starting_person: str,
                 relations: List[str],
                 universe: Dict[Tuple[str, str], str]) -> Tuple[List[str], str]:
    """
    Generate step-by-step Chain-of-Thought reasoning.

    Returns:
        (cot_steps, final_answer)
    """
    steps = []
    current_person = starting_person

    for i, relation in enumerate(relations, 1):
        key = (current_person, relation)
        if key not in universe:
            # Invalid path - this question can't be answered
            return None, None

        next_person = universe[key]
        path = "'s ".join(relations[:i])
        step = f"{starting_person}'s {path} = {next_person}"
        steps.append(step)
        current_person = next_person

    return steps, current_person


def generate_questions_for_universe(
    universe: Dict[Tuple[str, str], str],
    complexity_range: Tuple[int, int] = (2, 6),
    questions_per_complexity: int = 2
) -> List[Dict]:
    """
    Generate questions of varying complexity for a universe.

    Complexity = number of relationship hops + 1
    (e.g., "Alice's parent" = complexity 2, "Alice's parent's friend" = complexity 3)
    """
    people = list(set(p for (p, r), _ in universe.items()))
    questions = []

    for complexity in range(complexity_range[0], complexity_range[1] + 1):
        num_relations = complexity - 1
        generated_count = 0
        attempts = 0
        max_attempts = 100  # Prevent infinite loops

        while generated_count < questions_per_complexity and attempts < max_attempts:
            attempts += 1

            # Pick random starting person
            starting_person = random.choice(people)

            # Generate random relation chain
            relations = [random.choice(RELATIONS) for _ in range(num_relations)]

            # Try to generate CoT
            cot_steps, answer = generate_cot(starting_person, relations, universe)

            if cot_steps is None or answer is None:
                continue  # Invalid path, try again

            # Create question text
            relation_chain = "'s ".join(relations)
            question_text = f"{starting_person}'s {relation_chain}"

            # Check for duplicates
            if any(q['question'] == question_text for q in questions):
                continue

            questions.append({
                'question': question_text,
                'answer': answer,
                'complexity': complexity,
                'cot_steps': cot_steps,
                'universe': universe_to_string(universe),
                'starting_person': starting_person,
                'relations': relations
            })

            generated_count += 1

    return questions


def generate_dataset(
    num_universes: int = 200,
    questions_per_complexity: int = 5,
    complexity_range: Tuple[int, int] = (2, 6)
) -> List[Dict]:
    """
    Generate full dataset with multiple universes.

    Args:
        num_universes: Number of distinct universes to generate
        questions_per_complexity: Questions per complexity level per universe
        complexity_range: (min, max) complexity levels

    Returns:
        List of all questions across all universes
    """
    all_questions = []

    for universe_id in range(num_universes):
        if universe_id % 20 == 0:
            print(f"Generating universe {universe_id}/{num_universes}...")

        # Generate universe
        universe = generate_universe(num_people=6)

        # Generate questions for this universe
        questions = generate_questions_for_universe(
            universe,
            complexity_range=complexity_range,
            questions_per_complexity=questions_per_complexity
        )

        # Add universe ID (group) to each question
        for q in questions:
            q['group'] = f"universe_{universe_id}"

        all_questions.extend(questions)

    return all_questions


def validate_dataset(dataset: List[Dict]) -> Dict:
    """
    Validate that all generated CoT leads to correct answers.

    Returns:
        Validation statistics
    """
    valid_count = 0
    invalid_questions = []

    for item in dataset:
        # Parse universe
        universe_str = item['universe']
        universe = {}
        segments = universe_str.replace(';;', '  ').split('  ')
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            match = re.match(r"(\w+)'s\s+(\w+)\s*=\s*(\w+)", segment)
            if match:
                person, relation, target = match.groups()
                universe[(person, relation)] = target

        # Regenerate answer using CoT
        starting_person = item['starting_person']
        relations = item['relations']

        _, computed_answer = generate_cot(starting_person, relations, universe)

        if computed_answer == item['answer']:
            valid_count += 1
        else:
            invalid_questions.append({
                'question': item['question'],
                'expected': item['answer'],
                'computed': computed_answer
            })

    return {
        'total': len(dataset),
        'valid': valid_count,
        'invalid': len(dataset) - valid_count,
        'validation_rate': valid_count / len(dataset) * 100 if dataset else 0,
        'invalid_questions': invalid_questions
    }


def create_splits(
    dataset: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split dataset by universe (not questions) to prevent leakage.
    """
    # Group by universe
    universes = {}
    for item in dataset:
        group = item['group']
        if group not in universes:
            universes[group] = []
        universes[group].append(item)

    # Shuffle universe keys
    universe_keys = list(universes.keys())
    random.shuffle(universe_keys)

    # Split universes
    num_universes = len(universe_keys)
    train_end = int(num_universes * train_ratio)
    val_end = train_end + int(num_universes * val_ratio)

    train_universes = universe_keys[:train_end]
    val_universes = universe_keys[train_end:val_end]
    test_universes = universe_keys[val_end:]

    # Collect questions
    train_data = [q for u in train_universes for q in universes[u]]
    val_data = [q for u in val_universes for q in universes[u]]
    test_data = [q for u in test_universes for q in universes[u]]

    return train_data, val_data, test_data


def main():
    print("=" * 80)
    print("PERSONAL RELATIONS - DATA GENERATION")
    print("=" * 80)

    # Configuration
    NUM_UNIVERSES = 200  # Will generate ~5,000 questions
    QUESTIONS_PER_COMPLEXITY = 5
    COMPLEXITY_RANGE = (2, 6)

    print(f"\nConfiguration:")
    print(f"  Universes: {NUM_UNIVERSES}")
    print(f"  Questions per complexity: {QUESTIONS_PER_COMPLEXITY}")
    print(f"  Complexity range: {COMPLEXITY_RANGE}")
    print(f"  Expected total: ~{NUM_UNIVERSES * (COMPLEXITY_RANGE[1] - COMPLEXITY_RANGE[0] + 1) * QUESTIONS_PER_COMPLEXITY}")

    # Generate dataset
    print(f"\nGenerating dataset...")
    dataset = generate_dataset(
        num_universes=NUM_UNIVERSES,
        questions_per_complexity=QUESTIONS_PER_COMPLEXITY,
        complexity_range=COMPLEXITY_RANGE
    )

    print(f"\nGenerated {len(dataset)} questions")

    # Validate
    print(f"\nValidating dataset...")
    validation_stats = validate_dataset(dataset)
    print(f"  Total: {validation_stats['total']}")
    print(f"  Valid: {validation_stats['valid']}")
    print(f"  Invalid: {validation_stats['invalid']}")
    print(f"  Validation rate: {validation_stats['validation_rate']:.1f}%")

    if validation_stats['invalid'] > 0:
        print(f"\n⚠️  WARNING: {validation_stats['invalid']} invalid questions found!")
        print(f"  Removing invalid questions...")
        # Filter out invalid questions
        valid_dataset = [q for q in dataset
                        if q['answer'] in [compute_answer_for_validation(q) for _ in [None]]]
        dataset = valid_dataset
        print(f"  Remaining: {len(dataset)} questions")

    # Create splits
    print(f"\nCreating train/val/test splits...")
    train_data, val_data, test_data = create_splits(dataset)

    print(f"  Train: {len(train_data)} questions")
    print(f"  Val: {len(val_data)} questions")
    print(f"  Test: {len(test_data)} questions")

    # Breakdown by complexity
    print(f"\nComplexity distribution (train):")
    for complexity in range(2, 7):
        count = sum(1 for q in train_data if q['complexity'] == complexity)
        print(f"  Complexity {complexity}: {count}")

    # Save datasets
    print(f"\nSaving datasets...")

    with open('generated_full.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"  ✓ generated_full.json ({len(dataset)} examples)")

    with open('generated_train.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"  ✓ generated_train.json ({len(train_data)} examples)")

    with open('generated_val.json', 'w') as f:
        json.dump(val_data, f, indent=2)
    print(f"  ✓ generated_val.json ({len(val_data)} examples)")

    with open('generated_test.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"  ✓ generated_test.json ({len(test_data)} examples)")

    print(f"\n{'=' * 80}")
    print("✅ Data generation complete!")
    print(f"{'=' * 80}")


def compute_answer_for_validation(question: Dict) -> str:
    """Helper to compute answer for validation."""
    # Parse universe
    universe_str = question['universe']
    universe = {}
    segments = universe_str.replace(';;', '  ').split('  ')
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        match = re.match(r"(\w+)'s\s+(\w+)\s*=\s*(\w+)", segment)
        if match:
            person, relation, target = match.groups()
            universe[(person, relation)] = target

    # Regenerate answer
    starting_person = question['starting_person']
    relations = question['relations']
    _, computed_answer = generate_cot(starting_person, relations, universe)
    return computed_answer


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
