"""
Analyze layer-wise contributions for error detection.

Focus on late layers (10-16) since error localization showed late layer (L14)
contains 56% of error-predictive features.

Goals:
1. Extract continuous thoughts from layers 10-16
2. Test each layer individually
3. Test combinations of layers
4. Identify which layers contribute most to error detection
"""

import sys
import json
from pathlib import Path
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Reuse the extraction infrastructure
sys.path.insert(0, str(Path(__file__).parent.parent / "operation_circuits"))
from extract_continuous_thoughts import ContinuousThoughtExtractor


def extract_late_layers(solutions, extractor, layers=[10, 11, 12, 13, 14, 15]):
    """
    Extract continuous thoughts for specified late layers.

    Args:
        solutions: List of dicts with 'question' field
        extractor: ContinuousThoughtExtractor instance
        layers: List of layer indices to extract

    Returns:
        extracted_data: List of dicts with layer activations
    """
    print(f"\nExtracting layers: {layers}")

    extracted_data = []

    for sol in tqdm(solutions, desc="Extracting"):
        question = sol['question']

        # Build layer config for extraction
        layer_config = {f'L{i}': i for i in layers}

        # Use the proper extraction method
        with torch.no_grad():
            thoughts = extractor.extract_continuous_thoughts(question, layer_indices=layer_config)

        # Convert thoughts to numpy arrays
        # thoughts is a dict: {layer_name: [tensor_0, ..., tensor_5]}
        # Each tensor is shape [1, 2048]
        layer_activations = {}
        for layer_name in layer_config.keys():
            thought_list = thoughts[layer_name]  # List of 6 tensors
            # Stack and convert to numpy: [6, 2048]
            thought_array = torch.stack([t.squeeze(0) for t in thought_list]).cpu().numpy()
            layer_activations[layer_name] = thought_array

        extracted_data.append({
            'pair_id': sol['pair_id'],
            'variant': sol['variant'],
            'is_correct': sol['is_correct'],
            'layer_activations': layer_activations
        })

    return extracted_data


def test_single_layer(layer_name, layer_data, y, test_size=0.2, random_state=42):
    """Test classification performance using a single layer."""
    # Concatenate 6 token vectors: [n_samples, 6*2048]
    X = np.array([np.concatenate(d) for d in layer_data])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=random_state, verbose=0)
    clf.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    return {
        'layer': layer_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc,
        'clf': clf
    }


def test_layer_combinations(layer_data_dict, y, combinations, test_size=0.2, random_state=42):
    """Test classification using combinations of layers."""
    results = []

    for combo in combinations:
        # Concatenate specified layers
        combo_features = []
        for sample_idx in range(len(y)):
            sample_features = []
            for layer_name in combo:
                sample_features.append(layer_data_dict[layer_name][sample_idx])
            combo_features.append(np.concatenate(sample_features))

        X = np.array(combo_features)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=random_state, verbose=0)
        clf.fit(X_train, y_train)

        # Evaluate
        test_acc = accuracy_score(y_test, clf.predict(X_test))

        results.append({
            'layers': combo,
            'test_acc': test_acc,
            'n_layers': len(combo)
        })

    return results


def visualize_layer_contributions(single_layer_results, combo_results, output_dir):
    """Visualize which layers contribute most to error detection."""
    print("\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Individual layer performance
    ax = axes[0, 0]
    layers = [r['layer'] for r in single_layer_results]
    test_accs = [r['test_acc'] * 100 for r in single_layer_results]

    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    bars = ax.bar(layers, test_accs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlabel('Layer')
    ax.set_title('Individual Layer Performance')
    ax.set_ylim([0, 100])
    ax.axhline(y=73.22, color='red', linestyle='--', alpha=0.5,
               label='Best (3 layers: L4,L8,L14)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)

    # 2. Overfitting gap per layer
    ax = axes[0, 1]
    gaps = [r['gap'] * 100 for r in single_layer_results]
    bars = ax.bar(layers, gaps, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Train-Test Gap (%)')
    ax.set_xlabel('Layer')
    ax.set_title('Overfitting by Layer')
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5,
               label='Acceptable (<10%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()

    # 3. Layer combinations performance
    ax = axes[1, 0]
    combo_names = ['+'.join(r['layers']) for r in combo_results]
    combo_accs = [r['test_acc'] * 100 for r in combo_results]

    # Color by number of layers
    n_layers = [r['n_layers'] for r in combo_results]
    colors_combo = plt.cm.coolwarm(np.array(n_layers) / max(n_layers))

    bars = ax.barh(range(len(combo_names)), combo_accs, color=colors_combo,
                   alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(combo_names)))
    ax.set_yticklabels(combo_names, fontsize=8)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Layer Combination Performance')
    ax.set_xlim([0, 100])
    ax.axvline(x=73.22, color='red', linestyle='--', alpha=0.5,
               label='Current best')
    ax.grid(alpha=0.3, axis='x')
    ax.legend()

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' {width:.1f}%',
                ha='left', va='center', fontsize=8)

    # 4. Best layer vs number of layers
    ax = axes[1, 1]
    # Group by number of layers, show best accuracy for each
    n_layers_unique = sorted(set(n_layers))
    best_by_n = []
    for n in n_layers_unique:
        best_acc = max([r['test_acc'] for r in combo_results if r['n_layers'] == n])
        best_by_n.append(best_acc * 100)

    ax.plot(n_layers_unique, best_by_n, 'o-', linewidth=2, markersize=8,
            color='blue', label='Best accuracy')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Performance vs Number of Layers')
    ax.set_xticks(n_layers_unique)
    ax.grid(alpha=0.3)
    ax.axhline(y=73.22, color='red', linestyle='--', alpha=0.5,
               label='Current best (3 layers)')
    ax.legend()

    # Add value labels
    for n, acc in zip(n_layers_unique, best_by_n):
        ax.text(n, acc, f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # Save
    for ext in ['png', 'pdf']:
        save_path = output_dir / f'layer_contribution_analysis.{ext}'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='src/experiments/sae_error_analysis/data/error_analysis_dataset.json',
                        help='Path to error analysis dataset')
    parser.add_argument('--model_path', type=str,
                        default='/home/paperspace/codi_ckpt/llama_gsm8k',
                        help='Path to CODI LLaMA model')
    parser.add_argument('--output_dir', type=str,
                        default='src/experiments/sae_error_analysis/results',
                        help='Output directory')
    parser.add_argument('--layers', type=int, nargs='+',
                        default=[10, 11, 12, 13, 14, 15],
                        help='Layers to analyze')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("LAYER CONTRIBUTION ANALYSIS (LAYERS 10-16)")
    print("="*80)
    print(f"Layers to test: {args.layers}")
    print(f"Goal: Understand which late layers contribute most to error detection")

    # Load dataset
    print("\n" + "="*80)
    print("LOADING DATASET")
    print("="*80)
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    print(f"Total solutions: {dataset['metadata']['total']}")
    print(f"Incorrect: {dataset['metadata']['n_incorrect']}")
    print(f"Correct: {dataset['metadata']['n_correct']}")

    # Initialize extractor
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    extractor = ContinuousThoughtExtractor(args.model_path, device='cuda')
    print("Model loaded successfully")

    # Extract layers for incorrect solutions
    print("\n" + "="*80)
    print("EXTRACTING INCORRECT SOLUTIONS")
    print("="*80)
    incorrect_extracted = extract_late_layers(
        dataset['incorrect_solutions'],
        extractor,
        args.layers
    )

    # Extract layers for correct solutions
    print("\n" + "="*80)
    print("EXTRACTING CORRECT SOLUTIONS")
    print("="*80)
    correct_extracted = extract_late_layers(
        dataset['correct_solutions'],
        extractor,
        args.layers
    )

    # Combine and create labels
    all_extracted = incorrect_extracted + correct_extracted
    y = np.array([1 if d['is_correct'] else 0 for d in all_extracted])

    print(f"\n✅ Extracted {len(all_extracted)} solutions")
    print(f"  Incorrect: {np.sum(y==0)}")
    print(f"  Correct: {np.sum(y==1)}")

    # Organize data by layer
    layer_data_dict = {}
    for layer_idx in args.layers:
        layer_name = f'L{layer_idx}'
        layer_data_dict[layer_name] = [
            d['layer_activations'][layer_name] for d in all_extracted
        ]

    # Test individual layers
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL LAYERS")
    print("="*80)

    single_layer_results = []
    for layer_name in sorted(layer_data_dict.keys(), key=lambda x: int(x[1:])):
        print(f"\nTesting {layer_name}...")
        result = test_single_layer(layer_name, layer_data_dict[layer_name], y,
                                   args.test_size, args.random_seed)
        print(f"  Train: {result['train_acc']*100:.2f}%")
        print(f"  Test:  {result['test_acc']*100:.2f}%")
        print(f"  Gap:   {result['gap']*100:.2f} pts")
        single_layer_results.append(result)

    # Find best single layer
    best_single = max(single_layer_results, key=lambda x: x['test_acc'])
    print(f"\n🏆 Best single layer: {best_single['layer']}")
    print(f"  Test accuracy: {best_single['test_acc']*100:.2f}%")

    # Test layer combinations
    print("\n" + "="*80)
    print("TESTING LAYER COMBINATIONS")
    print("="*80)

    layer_names = sorted(layer_data_dict.keys(), key=lambda x: int(x[1:]))

    # Test some strategic combinations
    combinations = []

    # All individual layers (already tested above, but for consistency)
    for layer in layer_names:
        combinations.append([layer])

    # All pairs
    for i in range(len(layer_names)):
        for j in range(i+1, len(layer_names)):
            combinations.append([layer_names[i], layer_names[j]])

    # All triplets with L14 (since L14 was best in previous analysis)
    if 'L14' in layer_names:
        for i in range(len(layer_names)):
            for j in range(i+1, len(layer_names)):
                if layer_names[i] != 'L14' and layer_names[j] != 'L14':
                    combinations.append(sorted([layer_names[i], layer_names[j], 'L14']))

    # All layers
    combinations.append(layer_names)

    # Remove duplicates
    combinations = [list(c) for c in set(tuple(c) for c in combinations)]

    print(f"Testing {len(combinations)} combinations...")
    combo_results = test_layer_combinations(
        layer_data_dict, y, combinations, args.test_size, args.random_seed
    )

    # Sort by accuracy
    combo_results = sorted(combo_results, key=lambda x: x['test_acc'], reverse=True)

    # Print top 10
    print("\nTop 10 combinations:")
    for i, result in enumerate(combo_results[:10], 1):
        layers_str = '+'.join(result['layers'])
        print(f"  {i}. {layers_str:20s}: {result['test_acc']*100:.2f}%")

    # Visualize
    visualize_layer_contributions(single_layer_results, combo_results[:20], output_dir)

    # Save results
    summary = {
        'layers_tested': args.layers,
        'single_layer_results': [
            {
                'layer': r['layer'],
                'test_accuracy': float(r['test_acc']),
                'train_accuracy': float(r['train_acc']),
                'overfitting_gap': float(r['gap'])
            }
            for r in single_layer_results
        ],
        'best_single_layer': {
            'layer': best_single['layer'],
            'test_accuracy': float(best_single['test_acc'])
        },
        'top_10_combinations': [
            {
                'layers': r['layers'],
                'n_layers': r['n_layers'],
                'test_accuracy': float(r['test_acc'])
            }
            for r in combo_results[:10]
        ],
        'best_combination': {
            'layers': combo_results[0]['layers'],
            'test_accuracy': float(combo_results[0]['test_acc']),
            'improvement_vs_baseline': float(combo_results[0]['test_acc'] - 0.6557),
            'vs_current_best': float(combo_results[0]['test_acc'] - 0.7322)
        }
    }

    results_path = output_dir / 'layer_contribution_results.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved to {results_path}")

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Best single layer: {best_single['layer']} ({best_single['test_acc']*100:.2f}%)")
    print(f"Best combination: {'+'.join(combo_results[0]['layers'])} ({combo_results[0]['test_acc']*100:.2f}%)")
    print(f"Current baseline (L4+L8+L14 with XGBoost): 73.22%")
    print(f"Improvement: {(combo_results[0]['test_acc'] - 0.7322)*100:+.2f} pts")
    print("="*80)


if __name__ == "__main__":
    main()
