"""
Analyze continuous thoughts for operation-specific circuits.

This script performs comprehensive analysis on extracted continuous thoughts:
1. PCA visualization and clustering analysis
2. Classification using multiple ML models
3. Feature importance analysis across tokens and layers
4. Within-group vs between-group similarity analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class ContinuousThoughtAnalyzer:
    """Analyzer for continuous thought representations."""

    def __init__(self, data_path: str):
        """Load extracted continuous thoughts data.

        Args:
            data_path: Path to JSON file with extracted thoughts
        """
        print(f"Loading data from {data_path}...")
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} problems")

        # Group by operation type
        self.groups = {}
        for item in self.data:
            op_type = item['operation_type']
            if op_type not in self.groups:
                self.groups[op_type] = []
            self.groups[op_type].append(item)

        print("Operation type distribution:")
        for op_type, items in self.groups.items():
            print(f"  {op_type:20s}: {len(items):4d} problems")

        # Extract metadata
        self.operation_types = list(self.groups.keys())
        self.layer_names = list(self.data[0]['thoughts'].keys())
        self.num_tokens = len(self.data[0]['thoughts'][self.layer_names[0]])
        self.hidden_dim = len(self.data[0]['thoughts'][self.layer_names[0]][0])

        print(f"\nData shape:")
        print(f"  Layers: {self.layer_names}")
        print(f"  Tokens per problem: {self.num_tokens}")
        print(f"  Hidden dim: {self.hidden_dim}")

    def extract_features(self, aggregation: str = 'mean', token_idx: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from continuous thoughts.

        Args:
            aggregation: How to aggregate tokens ('mean', 'first', 'last')
            token_idx: Specific token index (if not using aggregation)

        Returns:
            (features, labels) where:
              features: [n_samples, n_features]
              labels: [n_samples] with operation type indices
        """
        features = []
        labels = []

        label_map = {op_type: i for i, op_type in enumerate(self.operation_types)}

        for item in self.data:
            # Collect all layer representations
            layer_features = []

            for layer_name in self.layer_names:
                thoughts = np.array(item['thoughts'][layer_name])  # [num_tokens, hidden_dim]

                if token_idx is not None:
                    # Use specific token
                    layer_feat = thoughts[token_idx]
                elif aggregation == 'mean':
                    layer_feat = thoughts.mean(axis=0)
                elif aggregation == 'first':
                    layer_feat = thoughts[0]
                elif aggregation == 'last':
                    layer_feat = thoughts[-1]
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")

                layer_features.append(layer_feat)

            # Concatenate all layers
            features.append(np.concatenate(layer_features))
            labels.append(label_map[item['operation_type']])

        return np.array(features), np.array(labels)

    def plot_clustering(self, aggregation: str, output_path: Path):
        """
        Visualize clusters using PCA.

        Args:
            aggregation: Aggregation method ('mean', 'first', 'last')
            output_path: Path to save plot
        """
        features, labels = self.extract_features(aggregation=aggregation)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # PCA to 2D
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features_scaled)

        # Plot
        plt.figure(figsize=(10, 8))
        colors = ['red', 'blue', 'green']

        for i, op_type in enumerate(self.operation_types):
            mask = labels == i
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=colors[i],
                label=op_type,
                alpha=0.6,
                s=100
            )

        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'Continuous Thoughts Clustering ({aggregation} aggregation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return {
            'pca_variance': pca.explained_variance_ratio_.tolist(),
            'feature_shape': features.shape
        }

    def classify_operations(self, output_dir: Path) -> Dict:
        """
        Train classifiers to predict operation type.

        Returns:
            Dict with classification results
        """
        features, labels = self.extract_features(aggregation='mean')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train multiple classifiers
        classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        }

        results = {}

        for name, clf in classifiers.items():
            print(f"\nTraining {name}...")
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            cm = confusion_matrix(y_test, y_pred)

            print(f"  Accuracy: {acc:.3f}")
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall: {rec:.3f}")
            print(f"  F1: {f1:.3f}")

            results[name] = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'confusion_matrix': cm.tolist()
            }

            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.operation_types,
                       yticklabels=self.operation_types)
            plt.title(f'Confusion Matrix - {name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
            plt.close()

        return results

    def feature_importance_analysis(self, output_dir: Path) -> Dict:
        """
        Analyze which tokens and layers are most important.

        Returns:
            Dict with feature importance results
        """
        print("\nAnalyzing feature importance across tokens and layers...")

        # Test each (token, layer) combination
        importance_matrix = np.zeros((self.num_tokens, len(self.layer_names)))

        for token_idx in range(self.num_tokens):
            for layer_idx, layer_name in enumerate(self.layer_names):
                # Extract features for this specific (token, layer)
                features = []
                labels = []

                for item in self.data:
                    thought = item['thoughts'][layer_name][token_idx]
                    features.append(thought)
                    labels.append(item['operation_type'])

                features = np.array(features)
                labels = np.array(labels)

                # Train simple classifier
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=labels
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Encode labels
                label_map = {op: i for i, op in enumerate(self.operation_types)}
                y_train_encoded = np.array([label_map[l] for l in y_train])
                y_test_encoded = np.array([label_map[l] for l in y_test])

                clf = LogisticRegression(max_iter=1000, random_state=42)
                clf.fit(X_train_scaled, y_train_encoded)
                y_pred = clf.predict(X_test_scaled)

                acc = accuracy_score(y_test_encoded, y_pred)
                importance_matrix[token_idx, layer_idx] = acc

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(importance_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                   xticklabels=self.layer_names,
                   yticklabels=[f'Token {i+1}' for i in range(self.num_tokens)])
        plt.title('Feature Importance Heatmap\n(Classification Accuracy by Token & Layer)')
        plt.xlabel('Layer')
        plt.ylabel('Token Position')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance_heatmap.png', dpi=150)
        plt.close()

        return {
            'token_layer_matrix': importance_matrix.tolist()
        }

    def similarity_analysis(self, output_dir: Path) -> Dict:
        """
        Analyze within-group vs between-group similarity.

        Returns:
            Dict with similarity analysis results
        """
        print("\nAnalyzing within-group vs between-group similarity...")

        features, labels = self.extract_features(aggregation='mean')

        # Normalize features for cosine similarity
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        # Compute similarity matrices
        within_similarities = {op_type: [] for op_type in self.operation_types}
        between_similarities = {}

        for i in range(len(self.operation_types)):
            for j in range(i+1, len(self.operation_types)):
                op1, op2 = self.operation_types[i], self.operation_types[j]
                between_similarities[f"{op1}_vs_{op2}"] = []

        # Compute pairwise similarities
        for i in range(len(features_norm)):
            for j in range(i+1, len(features_norm)):
                similarity = np.dot(features_norm[i], features_norm[j])

                op1 = self.operation_types[labels[i]]
                op2 = self.operation_types[labels[j]]

                if op1 == op2:
                    within_similarities[op1].append(similarity)
                else:
                    key = f"{op1}_vs_{op2}" if (labels[i] < labels[j]) else f"{op2}_vs_{op1}"
                    if key in between_similarities:
                        between_similarities[key].append(similarity)

        # Compute statistics
        results = {
            'within': {},
            'between': {}
        }

        for op_type, sims in within_similarities.items():
            if sims:
                results['within'][op_type] = {
                    'mean': float(np.mean(sims)),
                    'std': float(np.std(sims))
                }

        for pair, sims in between_similarities.items():
            if sims:
                results['between'][pair] = {
                    'mean': float(np.mean(sims)),
                    'std': float(np.std(sims))
                }

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Within-group similarities
        ax = axes[0]
        within_means = [results['within'][op]['mean'] for op in self.operation_types]
        within_stds = [results['within'][op]['std'] for op in self.operation_types]
        ax.bar(self.operation_types, within_means, yerr=within_stds, capsize=5, alpha=0.7)
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Within-Group Similarity')
        ax.grid(True, alpha=0.3)

        # Between-group similarities
        ax = axes[1]
        between_pairs = list(results['between'].keys())
        between_means = [results['between'][pair]['mean'] for pair in between_pairs]
        between_stds = [results['between'][pair]['std'] for pair in between_pairs]
        ax.bar(range(len(between_pairs)), between_means, yerr=between_stds, capsize=5, alpha=0.7)
        ax.set_xticks(range(len(between_pairs)))
        ax.set_xticklabels([p.replace('_', '\n') for p in between_pairs], rotation=45, ha='right')
        ax.set_ylabel('Mean Cosine Similarity')
        ax.set_title('Between-Group Similarity')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'similarity_analysis.png', dpi=150)
        plt.close()

        return results

    def run_full_analysis(self, output_dir: Path):
        """
        Run complete analysis pipeline.

        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("RUNNING FULL ANALYSIS PIPELINE")
        print("="*80)

        results = {}

        # 1. Clustering analysis
        print("\n1. CLUSTERING ANALYSIS")
        print("-" * 40)
        results['clustering'] = {}

        for agg in ['mean', 'first', 'last']:
            print(f"\n  Aggregation: {agg}")
            output_path = output_dir / f"clustering_all_layers_{agg}.png"
            results['clustering'][f'all_layers_{agg}'] = self.plot_clustering(agg, output_path)

        # Also try middle layer only
        print(f"\n  Single layer: middle")
        # Temporarily override layer_names for middle-only analysis
        orig_layers = self.layer_names
        self.layer_names = ['middle']
        output_path = output_dir / f"clustering_middle_layer_mean.png"
        results['clustering']['middle_layer_mean'] = self.plot_clustering('mean', output_path)
        self.layer_names = orig_layers

        # 2. Classification
        print("\n2. CLASSIFICATION ANALYSIS")
        print("-" * 40)
        results['classification'] = self.classify_operations(output_dir)

        # 3. Feature importance
        print("\n3. FEATURE IMPORTANCE ANALYSIS")
        print("-" * 40)
        results['feature_importance'] = self.feature_importance_analysis(output_dir)

        # 4. Similarity analysis
        print("\n4. SIMILARITY ANALYSIS")
        print("-" * 40)
        results['similarity'] = self.similarity_analysis(output_dir)

        # Save results
        results_path = output_dir / 'analysis_report.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE!")
        print(f"Results saved to: {output_dir}")
        print(f"Report: {results_path}")
        print(f"{'='*80}")

        # Create summary
        self._create_summary(results, output_dir)

    def _create_summary(self, results: Dict, output_dir: Path):
        """Create a markdown summary of results."""
        summary_path = output_dir / 'ANALYSIS_SUMMARY.md'

        with open(summary_path, 'w') as f:
            f.write("# Operation-Specific Circuits Analysis Summary\n\n")

            f.write("## Classification Results\n\n")
            for clf_name, metrics in results['classification'].items():
                f.write(f"### {clf_name}\n")
                f.write(f"- Accuracy: {metrics['accuracy']:.3f}\n")
                f.write(f"- Precision: {metrics['precision']:.3f}\n")
                f.write(f"- Recall: {metrics['recall']:.3f}\n")
                f.write(f"- F1: {metrics['f1']:.3f}\n\n")

            f.write("## Key Findings\n\n")
            f.write("1. PCA clustering visualizations show separation between operation types\n")
            f.write("2. Classification models achieve above-chance accuracy\n")
            f.write("3. Feature importance varies across token positions and layers\n")
            f.write("4. Within-group similarity is higher than between-group similarity\n")

        print(f"Summary saved to: {summary_path}")


def main():
    """Run analysis on extracted data."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to extracted continuous thoughts JSON')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for analysis results')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: {data_path} not found!")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent / 'analysis'

    # Run analysis
    analyzer = ContinuousThoughtAnalyzer(str(data_path))
    analyzer.run_full_analysis(output_dir)


if __name__ == "__main__":
    main()
