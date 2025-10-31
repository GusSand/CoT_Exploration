#!/usr/bin/bash
# Full pipeline for contrastive CODI deception detection experiment
# Run this script after training completes

set -e  # Exit on any error

EXPERIMENT_DIR=/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi
CODI_MODEL_PATH=~/codi_ckpt/contrastive_liars_llama1b_smoke_test
TEST_DATA_PATH=$EXPERIMENT_DIR/data/contrastive_liars_test.json
RESULTS_DIR=$EXPERIMENT_DIR/results

echo "============================================================"
echo "CONTRASTIVE CODI DECEPTION DETECTION - FULL PIPELINE"
echo "Research Question: Can CT tokens encode deception when trained contrastively?"
echo "============================================================"
echo "CODI Model: $CODI_MODEL_PATH"
echo "Test Data: $TEST_DATA_PATH"
echo "Results: $RESULTS_DIR"
echo "Layers: [4, 5, 9, 12, 15] (equivalent to Apollo's layer 22/80 = 27.5%)"
echo "============================================================"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Extract activations
echo ""
echo "STEP 1: Extracting activations..."
echo "--------------------------------"
python $EXPERIMENT_DIR/extract_activations.py \\
    --codi_model_path "$CODI_MODEL_PATH" \\
    --base_model_path meta-llama/Llama-3.2-1B-Instruct \\
    --test_data_path "$TEST_DATA_PATH" \\
    --output_dir "$RESULTS_DIR" \\
    --layers 4 5 9 12 15

# Step 2: Train deception detection probes
echo ""
echo "STEP 2: Training deception detection probes..."
echo "----------------------------------------------"
python $EXPERIMENT_DIR/train_deception_probes.py \\
    --ct_activations "$RESULTS_DIR/ct_token_activations.pkl" \\
    --regular_activations "$RESULTS_DIR/regular_hidden_activations.pkl" \\
    --output_dir "$RESULTS_DIR" \\
    --cv_folds 5 \\
    --test_size 0.2 \\
    --seed 42

# Step 3: Generate summary report
echo ""
echo "STEP 3: Generating summary report..."
echo "-----------------------------------"
python $EXPERIMENT_DIR/generate_report.py \\
    --results_dir "$RESULTS_DIR" \\
    --output_path "$RESULTS_DIR/final_report.md"

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo "📋 Final Report: $RESULTS_DIR/final_report.md"
echo "📊 Detailed Results: $RESULTS_DIR/"
echo ""
echo "Key Files:"
echo "  - probe_comparison_results.json: Performance comparison"
echo "  - ct_token_probe_results.json: CT token probe details"
echo "  - regular_hidden_probe_results.json: Regular hidden state probe details"
echo "  - ct_token_activations.pkl: CT token activation data"
echo "  - regular_hidden_activations.pkl: Regular hidden state activation data"
echo "============================================================"