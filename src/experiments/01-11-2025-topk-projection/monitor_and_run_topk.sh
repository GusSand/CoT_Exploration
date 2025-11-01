#!/bin/bash

# Monitor GPU memory and run top-k experiment when space becomes available
TARGET_DIR="/workspace/CoT_Exploration/src/experiments/01-11-2025-topk-projection"
MIN_FREE_MB=10000  # Need at least 10 GB free

echo "Monitoring GPU memory... will start top-k experiment when >= ${MIN_FREE_MB}MB free"
echo "Current time: $(date)"

while true; do
    # Get free GPU memory in MB
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    
    echo "$(date): Free GPU memory: ${FREE_MB}MB"
    
    if [ "$FREE_MB" -ge "$MIN_FREE_MB" ]; then
        echo "Sufficient memory available! Starting top-k projection experiment..."
        cd "$TARGET_DIR"
        python test_topk_projection_corrected.py 2>&1 | tee topk_test_auto_run.log
        echo "Experiment completed at $(date)"
        exit 0
    fi
    
    # Check every 2 minutes
    sleep 120
done
