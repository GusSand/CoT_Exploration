#!/usr/bin/env python3
"""Test all layers to see if any can distinguish honest vs deceptive."""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from pathlib import Path

def test_all_layers():
    """Test probe performance across all layers for both CT and regular."""

    results_dir = Path("/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi/results")

    # Load data
    with open(results_dir / "ct_token_activations.pkl", 'rb') as f:
        ct_data = pickle.load(f)

    with open(results_dir / "regular_hidden_activations.pkl", 'rb') as f:
        regular_data = pickle.load(f)

    labels = np.array(ct_data['labels']).astype(int)
    layers = list(ct_data['activations'].keys())

    print("🔍 Testing all layers for deception detection ability...")
    print(f"📊 Layers: {layers}")
    print(f"📊 Total samples: {len(labels)}")
    print(f"📊 Label balance: {np.mean(labels):.1%} honest")

    print("\\n" + "="*80)
    print(f"{'Layer':<8} {'CT Accuracy':<15} {'CT Unique Preds':<15} {'Reg Accuracy':<15} {'Reg Unique Preds':<15}")
    print("="*80)

    best_ct_layer = None
    best_ct_score = 0
    best_reg_layer = None
    best_reg_score = 0

    for layer in layers:
        # Test CT tokens
        X_ct = ct_data['activations'][layer]
        scaler_ct = StandardScaler()
        X_ct_scaled = scaler_ct.fit_transform(X_ct)

        probe_ct = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        cv_scores_ct = cross_val_score(
            probe_ct, X_ct_scaled, labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )

        # Check if it predicts multiple classes
        probe_ct.fit(X_ct_scaled, labels)
        y_pred_ct = probe_ct.predict(X_ct_scaled)
        unique_preds_ct = len(np.unique(y_pred_ct))

        ct_score = cv_scores_ct.mean()
        if ct_score > best_ct_score:
            best_ct_score = ct_score
            best_ct_layer = layer

        # Test regular tokens
        X_reg = regular_data['activations'][layer]
        scaler_reg = StandardScaler()
        X_reg_scaled = scaler_reg.fit_transform(X_reg)

        probe_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        cv_scores_reg = cross_val_score(
            probe_reg, X_reg_scaled, labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )

        probe_reg.fit(X_reg_scaled, labels)
        y_pred_reg = probe_reg.predict(X_reg_scaled)
        unique_preds_reg = len(np.unique(y_pred_reg))

        reg_score = cv_scores_reg.mean()
        if reg_score > best_reg_score:
            best_reg_score = reg_score
            best_reg_layer = layer

        print(f"{layer:<8} {ct_score:.3f}           {unique_preds_ct:<15} {reg_score:.3f}           {unique_preds_reg:<15}")

    print("="*80)
    print("\\n📊 Summary:")
    print(f"Best CT layer: {best_ct_layer} with {best_ct_score:.3f} accuracy")
    print(f"Best Regular layer: {best_reg_layer} with {best_reg_score:.3f} accuracy")
    print(f"Random baseline: ~0.500")

    if best_ct_score < 0.55 and best_reg_score < 0.55:
        print("\\n❌ PROBLEM: Both CT and regular features perform near random!")
        print("   This suggests the deception signal is very weak or the task is very difficult.")
        print("   Possible causes:")
        print("   1. Deception isn't encoded in these representations")
        print("   2. Need more sophisticated probe architecture")
        print("   3. Need different aggregation methods (not just mean)")
        print("   4. Need more training data")

    elif best_ct_score > best_reg_score + 0.02:
        print("\\n✅ CT tokens show some advantage over regular hidden states")

    elif best_reg_score > best_ct_score + 0.02:
        print("\\n⚠️  Regular hidden states outperform CT tokens")

    else:
        print("\\n🤷 No clear difference between CT tokens and regular hidden states")

    # Test with different layer combinations
    print("\\n🔍 Testing layer combinations...")

    # Try averaging features across multiple layers
    layers_to_combine = ['9', '12', '15']  # Later layers
    if all(layer in ct_data['activations'] for layer in layers_to_combine):
        print(f"  Testing combination of layers {layers_to_combine}...")

        # Combine CT features
        ct_combined = []
        for layer in layers_to_combine:
            ct_combined.append(ct_data['activations'][layer])
        X_ct_combined = np.concatenate(ct_combined, axis=1)

        scaler_combined = StandardScaler()
        X_ct_combined_scaled = scaler_combined.fit_transform(X_ct_combined)

        probe_combined = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        cv_scores_combined = cross_val_score(
            probe_combined, X_ct_combined_scaled, labels,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )

        combined_score = cv_scores_combined.mean()
        print(f"  Combined layers accuracy: {combined_score:.3f}")

        if combined_score > best_ct_score:
            print(f"  ✅ Layer combination improves performance!")
        else:
            print(f"  ⚠️  Layer combination doesn't help")

if __name__ == "__main__":
    test_all_layers()