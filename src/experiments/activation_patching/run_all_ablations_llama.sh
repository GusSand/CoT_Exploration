#!/bin/bash
# Run all N-token ablations for LLaMA

MODEL_PATH=~/codi_ckpt/llama_gsm8k/
PAIRS=data/problem_pairs.json
PROJECT=codi-activation-patching

echo "============================================================"
echo "Running N-Token Ablation Study on LLaMA"
echo "Testing: 1, 2, 3, 4, 5, 6 tokens"
echo "============================================================"

for N in 1 2 3 4 5 6; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Testing $N token(s)..."
    echo "------------------------------------------------------------"
    python3 run_ablation_N_tokens_llama.py \
        --model_path $MODEL_PATH \
        --problem_pairs $PAIRS \
        --num_tokens $N \
        --output_dir results_ablation_${N}_tokens_llama/ \
        --wandb_project $PROJECT

    if [ $? -eq 0 ]; then
        echo "✓ $N tokens: SUCCESS"
    else
        echo "✗ $N tokens: FAILED"
    fi
done

echo ""
echo "============================================================"
echo "All ablation experiments complete!"
echo "============================================================"
