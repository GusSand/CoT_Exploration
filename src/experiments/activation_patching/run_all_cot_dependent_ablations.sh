#!/bin/bash
# Run N-token ablation experiments on CoT-dependent pairs
# Tests 1, 2, and 4 tokens for both LLaMA and GPT-2

set -e  # Exit on error

ACTIVATION_PATCHING_DIR="/home/paperspace/dev/CoT_Exploration/src/experiments/activation_patching"
export PYTHONPATH="${ACTIVATION_PATCHING_DIR}/core:${ACTIVATION_PATCHING_DIR}:/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH"

cd "${ACTIVATION_PATCHING_DIR}/scripts/experiments"

echo "============================================================"
echo "N-TOKEN ABLATION ON COT-DEPENDENT PAIRS (43 pairs)"
echo "============================================================"
echo ""

# LLaMA experiments
echo "==== LLAMA EXPERIMENTS ===="
for N in 1 2 4; do
    echo ""
    echo "------------------------------------------------------------"
    echo "LLaMA: Testing ${N} token(s)..."
    echo "------------------------------------------------------------"
    python run_ablation_N_tokens_llama.py \
        --model_path ~/codi_ckpt/llama_gsm8k \
        --problem_pairs ../../data/problem_pairs_cot_dependent.json \
        --num_tokens $N \
        --output_dir ../../results/cot_dependent_ablation/llama_${N}token \
        --wandb_project codi-activation-patching

    if [ $? -eq 0 ]; then
        echo "✓ LLaMA $N tokens: SUCCESS"
    else
        echo "✗ LLaMA $N tokens: FAILED"
        exit 1
    fi
done

echo ""
echo "==== GPT-2 EXPERIMENTS ===="
for N in 1 2 4; do
    echo ""
    echo "------------------------------------------------------------"
    echo "GPT-2: Testing ${N} token(s)..."
    echo "------------------------------------------------------------"
    python run_ablation_N_tokens.py \
        --model_path ~/codi_ckpt/gpt2_gsm8k \
        --problem_pairs ../../data/problem_pairs_cot_dependent.json \
        --num_tokens $N \
        --output_dir ../../results/cot_dependent_ablation/gpt2_${N}token \
        --wandb_project codi-activation-patching

    if [ $? -eq 0 ]; then
        echo "✓ GPT-2 $N tokens: SUCCESS"
    else
        echo "✗ GPT-2 $N tokens: FAILED"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "ALL ABLATION EXPERIMENTS COMPLETE!"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  - results/cot_dependent_ablation/llama_*token/"
echo "  - results/cot_dependent_ablation/gpt2_*token/"
