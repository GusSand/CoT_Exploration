#!/bin/bash
# Monitor full resampling experiment and auto-continue with remaining stories

echo "Starting experiment monitor..."
echo "Monitoring process: 3_run_resampling.py --phase full"
echo "Expected runtime: ~5 hours"
echo ""

# Wait for resampling to complete
while pgrep -f "3_run_resampling.py --phase full" > /dev/null; do
    # Check progress every 10 minutes
    sleep 600

    # Get current time
    TIME=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if results file exists and get progress
    if [ -f "../results/resampling_full_results.json" ]; then
        # Count how many positions are complete
        PROGRESS=$(python3 -c "import json; data=json.load(open('../results/resampling_full_results.json')); print(f'{len(data)}/6 positions complete')" 2>/dev/null || echo "Reading...")
        echo "[$TIME] Progress: $PROGRESS"
    else
        echo "[$TIME] Experiment running... (results file not yet created)"
    fi
done

echo ""
echo "============================================================"
echo "RESAMPLING EXPERIMENT COMPLETE!"
echo "============================================================"
echo ""
echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Next steps:"
echo "1. Run Story 2.3: Per-problem analysis"
echo "2. Run Story 2.4: Statistical analysis"
echo "3. Create Story 2.5: Visualizations"
echo "4. Write Story 2.6: Final report"
echo ""
echo "To continue automatically, run:"
echo "  python 4_analyze_full.py"
echo ""
