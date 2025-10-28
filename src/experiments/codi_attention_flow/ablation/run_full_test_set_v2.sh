#!/bin/bash
# Run attention pattern ablation on full GSM8K test set (1,319 problems)

echo "=========================================="
echo "FULL TEST SET ATTENTION PATTERN ABLATION"
echo "=========================================="
echo "Dataset: GSM8K test set (1,319 problems)"
echo "Patterns: 9"
echo "Estimated time: ~72 minutes"
echo ""

for pattern in hub_to_ct0 skip_connections backward position_0 position_1 position_2 position_3 position_4 position_5; do
    echo ""
    echo "========================================" 
    echo "Running pattern: $pattern"
    echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    
    PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
        python 5_ablate_attention_patterns_v2.py \
        --model llama \
        --pattern $pattern \
        --n_problems 1319
    
    echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
done

echo ""
echo "=========================================="
echo "ALL PATTERNS COMPLETE"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
