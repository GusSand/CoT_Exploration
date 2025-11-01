#!/usr/bin/env python3
"""Debug the probe methodology to identify why all predictions are the same."""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

def debug_probe_methodology():
    """Debug the probe training to identify the issue."""

    results_dir = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi/results")

    # Load CT data for testing
    with open(results_dir / "ct_token_activations.pkl", 'rb') as f:
        ct_data = pickle.load(f)

    labels = np.array(ct_data['labels']).astype(int)
    X = ct_data['activations']['12']  # Use middle layer

    print("🔍 Deep debugging of probe methodology...")
    print(f"📊 Feature shape: {X.shape}")
    print(f"📊 Label shape: {labels.shape}")
    print(f"📊 Label distribution: {np.bincount(labels)}")
    print(f"📊 Label unique values: {np.unique(labels)}")

    # 1. Check data before scaling
    print("\\n1️⃣ Raw feature analysis:")
    print(f"   Mean: {X.mean():.6f}")
    print(f"   Std: {X.std():.6f}")
    print(f"   Range: [{X.min():.6f}, {X.max():.6f}]")

    # Check if features have zero variance
    feature_stds = X.std(axis=0)
    zero_var_features = np.sum(feature_stds == 0)
    print(f"   Zero variance features: {zero_var_features}/{X.shape[1]}")

    if zero_var_features > 0:
        print("   ⚠️  Some features have zero variance!")

    # 2. Test simple train/test split without CV
    print("\\n2️⃣ Simple train/test split test:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.3, stratify=labels, random_state=42
    )

    print(f"   Train set: {X_train.shape}, labels: {np.bincount(y_train)}")
    print(f"   Test set: {X_test.shape}, labels: {np.bincount(y_test)}")

    # Test without scaling first
    print("\\n   🔸 Without scaling:")
    probe_raw = LogisticRegression(max_iter=1000, random_state=42)
    probe_raw.fit(X_train, y_train)
    y_pred_raw = probe_raw.predict(X_test)
    accuracy_raw = accuracy_score(y_test, y_pred_raw)

    print(f"     Test accuracy: {accuracy_raw:.3f}")
    print(f"     Predictions: {np.bincount(y_pred_raw)}")
    print(f"     Unique predictions: {len(np.unique(y_pred_raw))}")

    # Test with scaling
    print("\\n   🔸 With StandardScaler:")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"     Scaled train mean: {X_train_scaled.mean():.6f}")
    print(f"     Scaled train std: {X_train_scaled.std():.6f}")
    print(f"     Scaled test mean: {X_test_scaled.mean():.6f}")
    print(f"     Scaled test std: {X_test_scaled.std():.6f}")

    probe_scaled = LogisticRegression(max_iter=1000, random_state=42)
    probe_scaled.fit(X_train_scaled, y_train)
    y_pred_scaled = probe_scaled.predict(X_test_scaled)
    accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

    print(f"     Test accuracy: {accuracy_scaled:.3f}")
    print(f"     Predictions: {np.bincount(y_pred_scaled)}")
    print(f"     Unique predictions: {len(np.unique(y_pred_scaled))}")

    # 3. Check probability outputs
    print("\\n3️⃣ Probability analysis:")
    y_proba_raw = probe_raw.predict_proba(X_test)
    y_proba_scaled = probe_scaled.predict_proba(X_test_scaled)

    print(f"   Raw probabilities:")
    print(f"     Shape: {y_proba_raw.shape}")
    print(f"     Range: [{y_proba_raw.min():.6f}, {y_proba_raw.max():.6f}]")
    print(f"     Class 0 prob range: [{y_proba_raw[:, 0].min():.6f}, {y_proba_raw[:, 0].max():.6f}]")
    print(f"     Class 1 prob range: [{y_proba_raw[:, 1].min():.6f}, {y_proba_raw[:, 1].max():.6f}]")

    print(f"   Scaled probabilities:")
    print(f"     Shape: {y_proba_scaled.shape}")
    print(f"     Range: [{y_proba_scaled.min():.6f}, {y_proba_scaled.max():.6f}]")
    print(f"     Class 0 prob range: [{y_proba_scaled[:, 0].min():.6f}, {y_proba_scaled[:, 0].max():.6f}]")
    print(f"     Class 1 prob range: [{y_proba_scaled[:, 1].min():.6f}, {y_proba_scaled[:, 1].max():.6f}]")

    # 4. Test with different algorithms
    print("\\n4️⃣ Testing different algorithms:")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    # Random Forest (doesn't need scaling)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    print(f"   Random Forest accuracy: {accuracy_rf:.3f}")
    print(f"   RF predictions: {np.bincount(y_pred_rf)}")
    print(f"   RF unique predictions: {len(np.unique(y_pred_rf))}")

    # SVM
    svm = SVC(random_state=42)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    print(f"   SVM accuracy: {accuracy_svm:.3f}")
    print(f"   SVM predictions: {np.bincount(y_pred_svm)}")
    print(f"   SVM unique predictions: {len(np.unique(y_pred_svm))}")

    # 5. Test with synthetic data to verify pipeline
    print("\\n5️⃣ Sanity check with synthetic data:")

    # Create linearly separable synthetic data
    np.random.seed(42)
    n_samples = len(X)
    n_features = 10

    X_synth = np.random.randn(n_samples, n_features)
    # Make class 1 have higher values in first feature
    X_synth[labels == 1, 0] += 2.0

    X_synth_train, X_synth_test, y_synth_train, y_synth_test = train_test_split(
        X_synth, labels, test_size=0.3, stratify=labels, random_state=42
    )

    scaler_synth = StandardScaler()
    X_synth_train_scaled = scaler_synth.fit_transform(X_synth_train)
    X_synth_test_scaled = scaler_synth.transform(X_synth_test)

    probe_synth = LogisticRegression(max_iter=1000, random_state=42)
    probe_synth.fit(X_synth_train_scaled, y_synth_train)
    y_pred_synth = probe_synth.predict(X_synth_test_scaled)
    accuracy_synth = accuracy_score(y_synth_test, y_pred_synth)

    print(f"   Synthetic data accuracy: {accuracy_synth:.3f}")
    print(f"   Synthetic predictions: {np.bincount(y_pred_synth)}")
    print(f"   Synthetic unique predictions: {len(np.unique(y_pred_synth))}")

    if accuracy_synth > 0.8:
        print("   ✅ Pipeline works correctly with synthetic data")
    else:
        print("   ❌ Pipeline fails even with synthetic data!")

    # 6. Summary
    print("\\n📋 Summary:")
    print(f"   Raw LogReg accuracy: {accuracy_raw:.3f}")
    print(f"   Scaled LogReg accuracy: {accuracy_scaled:.3f}")
    print(f"   Random Forest accuracy: {accuracy_rf:.3f}")
    print(f"   SVM accuracy: {accuracy_svm:.3f}")
    print(f"   Synthetic accuracy: {accuracy_synth:.3f}")

    if all(acc < 0.6 for acc in [accuracy_raw, accuracy_scaled, accuracy_rf, accuracy_svm]):
        print("\\n❌ CONCLUSION: The deception signal is very weak or absent")
        print("   All algorithms fail to detect honest vs deceptive patterns")
        print("   This could mean:")
        print("   1. Deception isn't encoded in these model layers")
        print("   2. The contrastive training didn't work as expected")
        print("   3. The task is genuinely very difficult")
        print("   4. More sophisticated methods are needed")
    else:
        print("\\n✅ Some algorithms show promise - investigate further")

if __name__ == "__main__":
    debug_probe_methodology()