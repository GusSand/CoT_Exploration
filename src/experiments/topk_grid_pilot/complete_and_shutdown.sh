#!/bin/bash
# Complete TopK SAE multi-layer experiment and shutdown

set -e  # Exit on error

LOG_FILE="src/experiments/topk_grid_pilot/training_log.txt"

echo "================================================================================================"
echo "TopK SAE Multi-Layer Experiment - Completion Script"
echo "================================================================================================"
echo ""
echo "Waiting for training to complete..."
echo ""

# Wait for training to complete
while true; do
    if grep -q "All training complete!" "$LOG_FILE" 2>/dev/null; then
        echo "✓ Training complete!"
        break
    fi

    # Show progress
    PROGRESS=$(tail -100 "$LOG_FILE" 2>/dev/null | grep "Progress:" | tail -1 || echo "Waiting...")
    echo "  $PROGRESS"
    sleep 60
done

echo ""
echo "================================================================================================"
echo "Step 1: Analysis"
echo "================================================================================================"
echo ""

python src/experiments/topk_grid_pilot/analyze_all_layers.py

echo ""
echo "================================================================================================"
echo "Step 2: Visualization"
echo "================================================================================================"
echo ""

python src/experiments/topk_grid_pilot/visualize_all_layers.py

echo ""
echo "================================================================================================"
echo "Step 3: Documentation"
echo "================================================================================================"
echo ""

# Run documentation script (will create this next)
python src/experiments/topk_grid_pilot/update_documentation.py

echo ""
echo "================================================================================================"
echo "Step 4: Git Commit & Push"
echo "================================================================================================"
echo ""

git add docs/research_journal.md docs/experiments/*.md src/experiments/topk_grid_pilot/

git commit -m "$(cat <<'EOF'
feat: Complete TopK SAE multi-layer analysis (1,152 SAEs across 16 layers × 6 positions)

Trained and analyzed TopK Sparse Autoencoders across all LLaMA layers and
continuous thought positions to identify optimal feature extraction points.

Results:
- 1,152 SAEs trained (16 layers × 6 positions × 12 configs)
- Layer and position quality patterns identified
- Comprehensive layer×position heatmaps generated
- Best configs documented for each layer/position

Analysis:
- Layer effects: [to be filled from results]
- Position effects: [to be filled from results]
- Interaction patterns across 16×6 grid

Training time: ~30-40 minutes on A100 80GB

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

echo "Pushing to GitHub..."
git pull --rebase origin master
git push origin master

echo "✓ Changes committed and pushed to GitHub"

echo ""
echo "================================================================================================"
echo "Step 5: Check for other Claude processes"
echo "================================================================================================"
echo ""

OTHER_CLAUDES=$(ps aux | grep -i claude | grep -v grep | grep -v "complete_and_shutdown" | wc -l)

if [ "$OTHER_CLAUDES" -gt 0 ]; then
    echo "⚠ Found $OTHER_CLAUDES other Claude processes:"
    ps aux | grep -i claude | grep -v grep | grep -v "complete_and_shutdown"
    echo ""
    echo "NOT shutting down - other Claude instances running"
    echo "Please manually review and shutdown when ready"
    exit 0
fi

echo "✓ No other Claude processes found"

echo ""
echo "================================================================================================"
echo "Step 6: Shutdown Machine"
echo "================================================================================================"
echo ""

echo "Shutting down in 10 seconds..."
echo "Press Ctrl+C to cancel"
sleep 10

sudo shutdown -h now
