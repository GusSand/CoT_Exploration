#!/bin/bash
# Train all 8 GPT-2 TopK SAE configs in parallel

echo "======================================================================"
echo "GPT-2 TOPK SAE PARAMETER SWEEP - PARALLEL TRAINING"
echo "======================================================================"
echo "Training 8 configs simultaneously on A100..."
echo ""

# Launch all 8 configs in background
for i in {0..7}; do
    echo "Launching config $i..."
    python src/experiments/gpt2_sae_training/scripts/train_gpt2_grid.py \
        --config_idx $i \
        --epochs 25 \
        --batch_size 256 \
        > "src/experiments/gpt2_sae_training/results/train_log_config${i}.txt" 2>&1 &
done

echo ""
echo "All 8 training processes launched!"
echo "Monitor progress with:"
echo "  tail -f src/experiments/gpt2_sae_training/results/train_log_config*.txt"
echo ""
echo "Waiting for all processes to complete..."

# Wait for all background jobs
wait

echo ""
echo "======================================================================"
echo "ALL TRAINING COMPLETE!"
echo "======================================================================"
echo "Check results in: src/experiments/gpt2_sae_training/results/"
echo ""

# Show completion status
ls -lh src/experiments/gpt2_sae_training/results/*.pt 2>/dev/null | wc -l | \
    xargs -I {} echo "Trained {} / 8 SAE models"
