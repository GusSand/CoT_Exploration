#!/bin/bash
# Monitor training completion and trigger analysis

LOG_FILE="src/experiments/topk_grid_pilot/training_log.txt"

echo "Monitoring training progress..."

while true; do
    if grep -q "All training complete!" "$LOG_FILE" 2>/dev/null; then
        echo "Training complete! Starting analysis..."

        # Analyze patterns
        echo "1. Analyzing quality patterns..."
        python src/experiments/topk_grid_pilot/analyze_all_layers.py

        # Generate heatmaps
        echo "2. Generating layer-position heatmaps..."
        python src/experiments/topk_grid_pilot/visualize_all_layers.py

        echo "Analysis complete!"
        break
    fi

    # Show progress every minute
    tail -3 "$LOG_FILE" 2>/dev/null | grep "Progress:"
    sleep 60
done
