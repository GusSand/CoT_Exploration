#!/usr/bin/bash
# Master script to run complete pre-compression signal analysis pipeline
#
# Usage: bash scripts/run_full_analysis.sh [5ep|10ep|15ep]
#
# This script runs:
# 1. Activation extraction (30 min)
# 2. Probe training (2 hours)
# 3. Visualization generation
# 4. Statistical analysis
#
# Author: Claude Code
# Date: 2025-10-30

set -e  # Exit on error

EPOCH=${1:-5ep}  # Default to 5ep if not specified

echo "============================================================"
echo "PRE-COMPRESSION DECEPTION SIGNAL ANALYSIS PIPELINE"
echo "============================================================"
echo "Epoch: $EPOCH"
echo "Expected total time: ~2.5 hours"
echo "============================================================"
echo ""

# Determine checkpoint path based on epoch
case $EPOCH in
  5ep)
    CHECKPOINT=~/codi_ckpt/llama1b_liars_bench_proper/liars_bench_llama1b_codi/Llama-3.2-1B-Instruct/ep_5/lr_0.0008/seed_42/checkpoint-250
    ;;
  10ep)
    CHECKPOINT=~/codi_ckpt/llama1b_liars_bench_proper/liars_bench_llama1b_codi/Llama-3.2-1B-Instruct/ep_10/lr_0.0008/seed_42/checkpoint-500
    ;;
  15ep)
    CHECKPOINT=~/codi_ckpt/llama1b_liars_bench_proper/liars_bench_llama1b_codi/Llama-3.2-1B-Instruct/ep_15/lr_0.0008/seed_42/checkpoint-750
    ;;
  *)
    echo "Error: Invalid epoch specified. Use 5ep, 10ep, or 15ep"
    exit 1
    ;;
esac

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT"
    echo "Make sure training has completed for $EPOCH"
    exit 1
fi

echo "✅ Checkpoint found: $CHECKPOINT"
echo ""

START_TIME=$(date +%s)

# Step 1: Extract activations
echo "============================================================"
echo "STEP 1/4: EXTRACTING MULTI-LAYER ACTIVATIONS"
echo "============================================================"
echo "Expected time: ~30 minutes"
echo ""

python scripts/extract_activations_llama1b_multilayer.py --checkpoint "$CHECKPOINT"

if [ $? -ne 0 ]; then
    echo "❌ Activation extraction failed"
    exit 1
fi

echo ""
echo "✅ Activation extraction complete"
echo ""

# Step 2: Train probes
echo "============================================================"
echo "STEP 2/4: TRAINING PROBES (48 TOTAL)"
echo "============================================================"
echo "Expected time: ~2 hours"
echo ""

python scripts/train_multilayer_probes_llama1b.py --epoch $EPOCH

if [ $? -ne 0 ]; then
    echo "❌ Probe training failed"
    exit 1
fi

echo ""
echo "✅ Probe training complete"
echo ""

# Step 3: Generate visualizations
echo "============================================================"
echo "STEP 3/4: GENERATING VISUALIZATIONS"
echo "============================================================"
echo "Creating heatmap and line plots..."
echo ""

python scripts/visualize_multilayer_results.py --epoch $EPOCH

if [ $? -ne 0 ]; then
    echo "❌ Visualization failed"
    exit 1
fi

echo ""
echo "✅ Visualizations complete"
echo ""

# Step 4: Statistical analysis
echo "============================================================"
echo "STEP 4/4: STATISTICAL ANALYSIS"
echo "============================================================"
echo "Analyzing patterns and generating key findings..."
echo ""

python scripts/analyze_multilayer_patterns.py --epoch $EPOCH

if [ $? -ne 0 ]; then
    echo "❌ Statistical analysis failed"
    exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "============================================================"
echo "✅ ANALYSIS PIPELINE COMPLETE"
echo "============================================================"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to:"
echo "  - Activations: data/processed/multilayer_activations_llama1b_${EPOCH}_*.json"
echo "  - Probe results: results/multilayer_probe_results_llama1b_${EPOCH}.json"
echo "  - Figures: results/figures/multilayer_*_llama1b_${EPOCH}.png"
echo "  - Analysis: results/multilayer_statistical_analysis_llama1b_${EPOCH}.json"
echo ""
echo "Next steps:"
echo "  1. Review visualizations in results/figures/"
echo "  2. Check key findings in statistical analysis JSON"
echo "  3. Decide: Stop here or continue training to next epoch?"
echo "============================================================"
