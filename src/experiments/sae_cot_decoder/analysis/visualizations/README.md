# SAE Feature Visualization Directory

This directory contains visualizations from SAE (Sparse Autoencoder) training and analysis on CODI continuous thought tokens.

## Directory Structure

### `baseline_800_samples/`
Visualizations from SAE models trained on **800 unique GSM8K problems** (replicated 16× across layers).

**Contents:**
- `cross_position_token_X_comparison.png` - How tokens ('0', '1', '*', '=') are detected across all 6 positions
- `position_X_feature_token_heatmap.png` - Feature-token correlation heatmaps for positions 0, 1, 3, 5
- `feature_detail_posX_fXXXX.png` - Detailed analysis of specific features
- `token_specific_features.png` - Top features detecting important tokens

**Key Metrics (Baseline):**
- Position 0 EV: 37.4%
- Average EV: 64.7%
- Feature Death Rate: 66.2%

---

### `full_dataset_7473_samples/`
Visualizations from SAE models trained on **full 7,473 unique GSM8K problems**.

**Contents:**
- `cross_position_token_0_comparison_updated.png` - Updated analysis showing how token '0' is detected across all 6 positions using full dataset models

**Key Metrics (Full Dataset):**
- Position 0 EV: 70.5% (+88.6% improvement!)
- Average EV: 78.9% (+20.0% improvement)
- Feature Death Rate: 30.0% (-54.7% reduction)

---

### `comparison/`
Visualizations comparing baseline vs full dataset training results.

**Contents:**
- `training_curves_comparison.png` - EV convergence comparison across all positions
- `death_rate_comparison.png` - Feature death rate improvements
- `cross_position_comparison_updated.png` - 4-panel metrics comparison (EV, Val Loss, Death Rate, L0)
- `improvement_summary.png` - Horizontal bar chart of % improvements per position
- `baseline_vs_full_comparison.png` - Overall comparison summary
- `feature_1893_comparison.png` - Position 3, Feature 1893 activation distributions
- `feature_148_comparison.png` - Position 1, Feature 148 activation distributions

**Key Findings:**
- Full dataset training dramatically improved EV for all positions
- Position 0 showed the largest improvement (+88.6%)
- Feature interpretability maintained/improved
- All positions now meet ≥70% EV threshold

---

## Experiment Details

- **Baseline Training**: 800 problems × 16 layers × 6 positions = 76,800 samples
- **Full Dataset Training**: 7,473 problems × 16 layers × 6 positions = 717,408 samples
- **SAE Architecture**: 2048 → 2048 features, L1=0.0005
- **Training**: 50 epochs, Adam optimizer, CosineAnnealingLR

For complete details, see: `docs/experiments/10-26_codi_gsm8k_sae_full_dataset_retraining.md`
