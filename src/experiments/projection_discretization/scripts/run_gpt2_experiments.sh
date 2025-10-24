#!/bin/bash
# Run GPT-2 projection discretization experiments

cd /workspace/CoT_Exploration/src/experiments/projection_discretization

echo "=== Running GPT-2 Projection Experiments ==="
echo "Start time: $(date)"

# Vanilla baseline
echo "1/3: Running vanilla baseline..."
python code/run_gpt2_topk_projection.py \
    --batch_size 1 \
    --discretize_positions none \
    --output_dir ./results/gpt2_vanilla_full \
    --device cuda

# k=1 projection
echo "2/3: Running k=1 projection..."
python code/run_gpt2_topk_projection.py \
    --batch_size 1 \
    --discretize_positions thought_only \
    --normalize \
    --k_nearest 1 \
    --output_dir ./results/gpt2_k1_thought_normalized_full \
    --device cuda

# k=5 projection
echo "3/3: Running k=5 projection..."
python code/run_gpt2_topk_projection.py \
    --batch_size 1 \
    --discretize_positions thought_only \
    --normalize \
    --k_nearest 5 \
    --output_dir ./results/gpt2_k5_thought_normalized_full \
    --device cuda

echo "=== All experiments completed ==="
echo "End time: $(date)"
