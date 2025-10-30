#!/bin/bash
# Quick status check for resampling experiment

echo "============================================================"
echo "Resampling Experiment Status Check"
echo "============================================================"
echo ""

# Check if resampling is running
if pgrep -f "3_run_resampling.py --phase full" > /dev/null; then
    echo "Status: RUNNING ⏳"
    echo ""

    # Try to get progress from W&B or results file
    if [ -f "../results/resampling_full_results.json" ]; then
        POSITIONS=$(python3 -c "import json; data=json.load(open('../results/resampling_full_results.json')); print(len(data))" 2>/dev/null || echo "0")
        echo "Progress: $POSITIONS/6 positions complete"
    else
        echo "Progress: Starting up..."
    fi

    echo ""
    echo "W&B Dashboard: https://wandb.ai/gussand/codi-resampling/runs/55hmkxb2"
    echo ""
    echo "To monitor: tail -f ../results/resampling_full_results.json"
    echo "Or run: bash monitor_and_continue.sh"
else
    echo "Status: COMPLETED ✓ or NOT STARTED"
    echo ""

    if [ -f "../results/resampling_full_results.json" ]; then
        echo "✓ Results file exists"
        POSITIONS=$(python3 -c "import json; data=json.load(open('../results/resampling_full_results.json')); print(len(data))" 2>/dev/null || echo "?")
        echo "  Positions complete: $POSITIONS/6"

        if [ "$POSITIONS" = "6" ]; then
            echo ""
            echo "Next step: Run analysis"
            echo "  python 4_analyze_full.py"
        fi
    else
        echo "✗ Results file not found"
        echo "  Either not started or still initializing"
    fi
fi

echo ""
echo "============================================================"
