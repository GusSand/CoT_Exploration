#!/usr/bin/env python3
"""Debug probe training data to identify why accuracy is 9%."""

import pickle
import numpy as np
import json
from pathlib import Path

def analyze_activations():
    """Analyze the saved activation data to identify issues."""

    results_dir = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi/results")

    print("🔍 Debugging probe training data...")
    print(f"📁 Results directory: {results_dir}")

    # 1. Load saved activations
    ct_file = results_dir / "ct_token_activations.pkl"
    regular_file = results_dir / "regular_hidden_activations.pkl"

    if not ct_file.exists():
        print(f"❌ CT activations file missing: {ct_file}")
        return

    if not regular_file.exists():
        print(f"❌ Regular activations file missing: {regular_file}")
        return

    print("📂 Loading saved activations...")

    with open(ct_file, 'rb') as f:
        ct_data = pickle.load(f)

    with open(regular_file, 'rb') as f:
        regular_data = pickle.load(f)

    print(f"✅ Loaded CT data: {ct_data.keys()}")
    print(f"✅ Loaded regular data: {regular_data.keys()}")

    # 2. Analyze CT data
    print("\n🔍 CT Token Data Analysis:")
    print(f"  Activation type: {ct_data.get('activation_type', 'unknown')}")
    print(f"  Total samples: {len(ct_data['labels'])}")
    print(f"  Layers: {list(ct_data['activations'].keys())}")

    labels = np.array(ct_data['labels'])
    honest_count = np.sum(labels)
    deceptive_count = len(labels) - honest_count
    print(f"  Label distribution: {honest_count} honest, {deceptive_count} deceptive")
    print(f"  Label balance: {honest_count/len(labels):.2%} honest")

    # Check a specific layer
    layer = list(ct_data['activations'].keys())[0]
    X_ct = ct_data['activations'][layer]
    print(f"  Layer {layer} shape: {X_ct.shape}")
    print(f"  Layer {layer} stats: mean={X_ct.mean():.6f}, std={X_ct.std():.6f}")
    print(f"  Layer {layer} range: [{X_ct.min():.6f}, {X_ct.max():.6f}]")

    # Check for NaN or infinite values
    has_nan = np.isnan(X_ct).any()
    has_inf = np.isinf(X_ct).any()
    print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")

    # 3. Analyze Regular data
    print("\n🔍 Regular Hidden State Data Analysis:")
    print(f"  Activation type: {regular_data.get('activation_type', 'unknown')}")
    print(f"  Total samples: {len(regular_data['labels'])}")
    print(f"  Layers: {list(regular_data['activations'].keys())}")

    reg_labels = np.array(regular_data['labels'])
    reg_honest_count = np.sum(reg_labels)
    reg_deceptive_count = len(reg_labels) - reg_honest_count
    print(f"  Label distribution: {reg_honest_count} honest, {reg_deceptive_count} deceptive")
    print(f"  Label balance: {reg_honest_count/len(reg_labels):.2%} honest")

    # Check same layer
    X_reg = regular_data['activations'][layer]
    print(f"  Layer {layer} shape: {X_reg.shape}")
    print(f"  Layer {layer} stats: mean={X_reg.mean():.6f}, std={X_reg.std():.6f}")
    print(f"  Layer {layer} range: [{X_reg.min():.6f}, {X_reg.max():.6f}]")

    # Check for NaN or infinite values
    has_nan_reg = np.isnan(X_reg).any()
    has_inf_reg = np.isinf(X_reg).any()
    print(f"  Has NaN: {has_nan_reg}, Has Inf: {has_inf_reg}")

    # 4. Check label consistency
    print("\n🔍 Label Consistency Check:")
    labels_match = np.array_equal(ct_data['labels'], regular_data['labels'])
    print(f"  Labels match between CT and regular: {labels_match}")

    hashes_match = ct_data['question_hashes'] == regular_data['question_hashes']
    print(f"  Question hashes match: {hashes_match}")

    # 5. Quick sanity check: test simple probe on one layer
    print(f"\n🔍 Quick Probe Test on Layer {layer}:")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import accuracy_score

    # Test with CT data
    print("  Testing CT token probe...")
    y = labels.astype(int)  # Convert to 0/1

    scaler = StandardScaler()
    X_ct_scaled = scaler.fit_transform(X_ct)

    probe_ct = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')

    # Cross-validation
    cv_scores = cross_val_score(
        probe_ct, X_ct_scaled, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )

    print(f"    CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Fit and check if predictions are reasonable
    probe_ct.fit(X_ct_scaled, y)
    y_pred = probe_ct.predict(X_ct_scaled)
    train_accuracy = accuracy_score(y, y_pred)

    print(f"    Training accuracy: {train_accuracy:.3f}")
    print(f"    Prediction distribution: {np.bincount(y_pred)}")
    print(f"    True label distribution: {np.bincount(y)}")

    # Check if probe is just predicting one class
    unique_preds = len(np.unique(y_pred))
    print(f"    Unique predictions: {unique_preds} (should be 2)")

    if unique_preds == 1:
        print("    ⚠️  Probe is predicting only one class!")

    # Test probability predictions
    y_proba = probe_ct.predict_proba(X_ct_scaled)
    print(f"    Probability range: [{y_proba.min():.6f}, {y_proba.max():.6f}]")
    print(f"    Probability mean: {y_proba.mean():.6f}")

    # 6. Compare with random baseline
    print("\n🔍 Random Baseline Comparison:")
    from sklearn.dummy import DummyClassifier

    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy_scores = cross_val_score(
        dummy, X_ct_scaled, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )

    print(f"  Random baseline: {dummy_scores.mean():.3f} ± {dummy_scores.std():.3f}")
    print(f"  Expected random: 0.500")

    if cv_scores.mean() < dummy_scores.mean():
        print("  ❌ Probe performs worse than random!")
    else:
        print("  ✅ Probe beats random baseline")

if __name__ == "__main__":
    analyze_activations()