#!/bin/bash
#
# Master script to run complete layer-wise patching experiment
#

set -e  # Exit on error

echo "========================================================================"
echo "Layer-wise CoT Activation Patching Experiment"
echo "========================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Load and validate data
echo "Step 1/5: Loading and validating dataset..."
python scripts/1_load_data.py
if [ $? -ne 0 ]; then
    echo "❌ Data validation failed"
    exit 1
fi
echo ""

# Step 2: Run patching experiment
echo "Step 2/5: Running activation patching experiment..."
echo "  This will take some time (~2-3 hours for 66 pairs × 22 layers)..."
python scripts/2_run_patching.py
if [ $? -ne 0 ]; then
    echo "❌ Patching experiment failed"
    exit 1
fi
echo ""

# Step 3: Generate individual heatmaps
echo "Step 3/5: Generating individual heatmaps..."
python scripts/3_visualize_individual.py
if [ $? -ne 0 ]; then
    echo "❌ Individual visualization failed"
    exit 1
fi
echo ""

# Step 4: Generate aggregate visualizations
echo "Step 4/5: Generating aggregate visualizations..."
python scripts/4_visualize_aggregate.py
if [ $? -ne 0 ]; then
    echo "❌ Aggregate visualization failed"
    exit 1
fi
echo ""

echo "========================================================================"
echo "✓ EXPERIMENT COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: $SCRIPT_DIR/results/"
echo ""
echo "Generated files:"
echo "  - results/prepared_pairs.json           : Validated dataset"
echo "  - results/patching_results.json         : Raw patching results"
echo "  - results/aggregate_statistics.json     : Aggregate statistics"
echo "  - results/individual_heatmaps/          : Per-example visualizations"
echo "  - results/aggregate_analysis/           : Aggregate visualizations"
echo ""
