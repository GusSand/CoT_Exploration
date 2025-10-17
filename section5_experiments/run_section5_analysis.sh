#!/bin/bash

# Section 5 Analysis Script for CODI Paper Reproduction
# ======================================================
# This script runs the complete Section 5 interpretability analysis
# including model evaluation, output segregation, and visualization generation

set -e  # Exit on error

# Configuration
MODEL_NAME="gpt2"
DATASET="zen-E/GSM8k-Aug"
CHECKPOINT_DIR="models/CODI-gpt2"  # Update this path
OUTPUT_DIR="outputs/section5_analysis"
BATCH_SIZE=32
SEED=11

echo "======================================================================="
echo "CODI Section 5 Interpretability Analysis"
echo "======================================================================="
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "======================================================================="
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please download the pretrained model first:"
    echo ""
    echo "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='zen-E/CODI-gpt2', local_dir='models/CODI-gpt2')\""
    echo ""
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Step 1: Running Section 5 Analysis..."
echo "This will generate:"
echo "  - Separate JSON files for correct/incorrect predictions"
echo "  - Detailed interpretability data for each continuous thought"
echo "  - Intermediate computation validation results"
echo ""

python section5_analysis.py \
	--data_name "$DATASET" \
	--output_dir "$OUTPUT_DIR" \
	--model_name_or_path "$MODEL_NAME" \
	--seed $SEED \
	--model_max_length 512 \
	--bf16 \
	--lora_r 128 \
	--lora_alpha 32 \
	--lora_init \
	--batch_size $BATCH_SIZE \
	--greedy True \
	--num_latent 6 \
	--use_prj True \
	--prj_dim 768 \
	--prj_no_ln False \
	--prj_dropout 0.0 \
	--inf_latent_iterations 6 \
	--inf_num_iterations 1 \
	--remove_eos True \
	--use_lora True \
	--ckpt_dir "$CHECKPOINT_DIR"

echo ""
echo "Step 2: Generating Visualizations..."

# Find the most recent run directory
LATEST_RUN=$(ls -td $OUTPUT_DIR/section5_run_* | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No run directory found in $OUTPUT_DIR"
    exit 1
fi

echo "Visualizing results from: $LATEST_RUN"

python visualize_interpretability.py \
	--input_dir "$LATEST_RUN" \
	--max_examples 100 \
	--output_name "interpretability_visualization"

echo ""
echo "======================================================================="
echo "Section 5 Analysis Complete!"
echo "======================================================================="
echo ""
echo "Results saved to: $LATEST_RUN"
echo ""
echo "Generated files:"
echo "  - summary_statistics.json: Overall metrics and statistics"
echo "  - correct_predictions/predictions.json: All correct predictions with analysis"
echo "  - incorrect_predictions/predictions.json: All incorrect predictions with analysis"
echo "  - interpretability_analysis.csv: Spreadsheet-friendly summary"
echo "  - interpretability_visualization.html: Interactive HTML visualization"
echo "  - interpretability_visualization.txt: Text-based report"
echo ""
echo "To view the HTML visualization, open in a web browser:"
echo "  file://$PWD/$LATEST_RUN/interpretability_visualization.html"
echo ""
echo "======================================================================="
