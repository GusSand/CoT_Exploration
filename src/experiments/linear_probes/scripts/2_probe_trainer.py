"""
Story 1.2: Build Linear Probe Training Infrastructure

Reusable infrastructure for training linear probes on continuous thoughts.

Usage:
    from probe_trainer import ProbeTrainer

    trainer = ProbeTrainer('probe_dataset_100.json')
    results = trainer.train_probe(layer=14, token=0)
    print(f"Accuracy: {results['accuracy']:.2%}")
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from dataclasses import dataclass
import wandb


@dataclass
class ProbeResults:
    """Results from training a single probe."""
    layer: int
    token: int
    accuracy: float
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    weights: np.ndarray
    intercept: float
    confusion_matrix: np.ndarray
    n_samples: int
    n_features: int
    cv_scores: List[float]
    best_C: float


class ProbeTrainer:
    """Trains logistic regression probes on continuous thoughts."""

    def __init__(
        self,
        dataset_path: str,
        use_wandb: bool = True,
        project_name: str = "linear-probes",
        seed: int = 42
    ):
        """
        Initialize the probe trainer.

        Args:
            dataset_path: Path to probe dataset JSON
            use_wandb: Whether to log to wandb
            project_name: WandB project name
            seed: Random seed for reproducibility
        """
        self.dataset_path = dataset_path
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.seed = seed

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load dataset
        self.data = self._load_dataset()
        self.layer_map = {
            8: 'layer_8',
            14: 'layer_14',
            15: 'layer_15'
        }

        print(f"ProbeTrainer initialized")
        print(f"  Dataset: {dataset_path}")
        print(f"  Samples: {len(self.data['samples'])}")
        print(f"  Layers: {self.data['metadata']['layers']}")
        print(f"  Tokens: {self.data['metadata']['n_tokens']}")
        print(f"  Features: {self.data['metadata']['hidden_dim']}")

    def _load_dataset(self) -> Dict:
        """Load the probe dataset."""
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        return data

    def _extract_features(self, layer: int, token: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels for a specific layer and token.

        Args:
            layer: Layer index (8, 14, or 15)
            token: Token index (0-5)

        Returns:
            Tuple of (features, labels)
            features: shape [n_samples, hidden_dim]
            labels: shape [n_samples] (1 = correct, 0 = incorrect)
        """
        if layer not in self.layer_map:
            raise ValueError(f"Layer {layer} not in dataset. Available: {list(self.layer_map.keys())}")

        if token < 0 or token >= self.data['metadata']['n_tokens']:
            raise ValueError(f"Token {token} out of range [0, {self.data['metadata']['n_tokens']-1}]")

        layer_name = self.layer_map[layer]

        features = []
        labels = []

        for sample in self.data['samples']:
            # Extract activation for this layer and token
            activation = sample['thoughts'][layer_name][token]  # List of 2048 floats
            features.append(activation)

            # Label: 1 = correct, 0 = incorrect
            labels.append(1 if sample['is_correct'] else 0)

        features = np.array(features)
        labels = np.array(labels)

        return features, labels

    def train_probe(
        self,
        layer: int,
        token: int,
        Cs: Optional[List[float]] = None,
        cv_folds: int = 5,
        run_name: Optional[str] = None
    ) -> ProbeResults:
        """
        Train a logistic regression probe.

        Args:
            layer: Layer index (8, 14, or 15)
            token: Token index (0-5)
            Cs: List of regularization strengths to try (default: [0.001, 0.01, 0.1, 1.0, 10.0])
            cv_folds: Number of cross-validation folds
            run_name: Optional name for wandb run

        Returns:
            ProbeResults object with accuracy, weights, etc.
        """
        if Cs is None:
            Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

        # Initialize wandb
        if self.use_wandb:
            if run_name is None:
                run_name = f"probe_L{layer}_T{token}"

            wandb.init(
                project=self.project_name,
                name=run_name,
                config={
                    'layer': layer,
                    'token': token,
                    'cv_folds': cv_folds,
                    'Cs': Cs,
                    'seed': self.seed
                },
                reinit=True
            )

        # Extract features and labels
        X, y = self._extract_features(layer, token)

        print(f"\nTraining probe: Layer {layer}, Token {token}")
        print(f"  Features: {X.shape}")
        print(f"  Labels: {y.shape} (Correct: {y.sum()}, Incorrect: {len(y) - y.sum()})")

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train logistic regression with cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        clf = LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            scoring='accuracy',
            random_state=self.seed,
            max_iter=1000,
            n_jobs=-1
        )

        clf.fit(X_scaled, y)

        # Get predictions
        y_pred = clf.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)

        # Calculate confidence intervals via bootstrap
        accuracy_ci = self._bootstrap_ci(clf, X_scaled, y, n_bootstrap=1000)

        # Get cross-validation scores
        cv_scores = []
        for train_idx, val_idx in cv.split(X_scaled, y):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf_temp = LogisticRegressionCV(
                Cs=Cs,
                cv=3,
                scoring='accuracy',
                random_state=self.seed,
                max_iter=1000
            )
            clf_temp.fit(X_train, y_train)
            score = clf_temp.score(X_val, y_val)
            cv_scores.append(score)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        # Results
        results = ProbeResults(
            layer=layer,
            token=token,
            accuracy=accuracy,
            accuracy_ci_lower=accuracy_ci[0],
            accuracy_ci_upper=accuracy_ci[1],
            weights=clf.coef_[0],
            intercept=clf.intercept_[0],
            confusion_matrix=cm,
            n_samples=len(y),
            n_features=X.shape[1],
            cv_scores=cv_scores,
            best_C=clf.C_[0]
        )

        # Print results
        print(f"  Accuracy: {accuracy:.4f} [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]")
        print(f"  CV Scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"  Best C: {clf.C_[0]:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                'layer': layer,
                'token': token,
                'accuracy': accuracy,
                'accuracy_ci_lower': accuracy_ci[0],
                'accuracy_ci_upper': accuracy_ci[1],
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'best_C': clf.C_[0],
                'confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y,
                    preds=y_pred,
                    class_names=['Incorrect', 'Correct']
                )
            })
            wandb.finish()

        return results

    def _bootstrap_ci(
        self,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals for accuracy.

        Args:
            clf: Trained classifier
            X: Features
            y: Labels
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (0.95 = 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        accuracies = []
        n_samples = len(y)

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Predict and calculate accuracy
            y_pred = clf.predict(X_boot)
            acc = accuracy_score(y_boot, y_pred)
            accuracies.append(acc)

        # Calculate percentiles
        alpha = 1 - confidence
        lower = np.percentile(accuracies, alpha/2 * 100)
        upper = np.percentile(accuracies, (1 - alpha/2) * 100)

        return (lower, upper)

    def train_sweep(
        self,
        layers: List[int],
        tokens: List[int],
        sweep_name: Optional[str] = None
    ) -> List[ProbeResults]:
        """
        Train probes for multiple layer-token combinations.

        Args:
            layers: List of layer indices
            tokens: List of token indices
            sweep_name: Name for this sweep

        Returns:
            List of ProbeResults
        """
        results = []
        total = len(layers) * len(tokens)
        count = 0

        print(f"\n{'='*60}")
        print(f"PROBE TRAINING SWEEP: {sweep_name or 'Unnamed'}")
        print(f"{'='*60}")
        print(f"Layers: {layers}")
        print(f"Tokens: {tokens}")
        print(f"Total probes: {total}")
        print(f"{'='*60}\n")

        for layer in layers:
            for token in tokens:
                count += 1
                print(f"\n[{count}/{total}] Layer {layer}, Token {token}")

                run_name = f"{sweep_name}_L{layer}_T{token}" if sweep_name else f"L{layer}_T{token}"
                result = self.train_probe(layer, token, run_name=run_name)
                results.append(result)

        print(f"\n{'='*60}")
        print(f"SWEEP COMPLETE: {len(results)} probes trained")
        print(f"{'='*60}")

        return results

    def save_results(self, results: List[ProbeResults], output_path: str):
        """Save probe results to JSON."""
        results_dict = {
            'metadata': {
                'n_probes': len(results),
                'dataset': self.dataset_path,
                'seed': self.seed
            },
            'results': []
        }

        for r in results:
            results_dict['results'].append({
                'layer': r.layer,
                'token': r.token,
                'accuracy': float(r.accuracy),
                'accuracy_ci_lower': float(r.accuracy_ci_lower),
                'accuracy_ci_upper': float(r.accuracy_ci_upper),
                'cv_mean': float(np.mean(r.cv_scores)),
                'cv_std': float(np.std(r.cv_scores)),
                'best_C': float(r.best_C),
                'n_samples': r.n_samples,
                'n_features': r.n_features,
                'confusion_matrix': r.confusion_matrix.tolist()
            })

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

        print(f"\nResults saved to {output_path}")


def test_single_probe():
    """Test training a single probe."""
    project_root = Path(__file__).parent.parent.parent.parent.parent
    dataset_path = project_root / "src/experiments/linear_probes/data/probe_dataset_100.json"

    trainer = ProbeTrainer(str(dataset_path), use_wandb=False)
    result = trainer.train_probe(layer=14, token=0)

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Accuracy: {result.accuracy:.4f} [{result.accuracy_ci_lower:.4f}, {result.accuracy_ci_upper:.4f}]")
    print("="*60)


if __name__ == "__main__":
    test_single_probe()
