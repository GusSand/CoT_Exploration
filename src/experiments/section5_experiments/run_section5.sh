#!/bin/bash

# Quick Section 5 execution script
set -e

cd /workspace/CoT_Exploration
source env/bin/activate

cd codi

echo "=========================================="
echo "Running CODI Section 5 Analysis"
echo "=========================================="
echo ""

python section5_analysis.py \
    --data_name "zen-E/GSM8k-Aug" \
    --output_dir "outputs/section5_analysis" \
    --model_name_or_path gpt2 \
    --seed 11 \
    --model_max_length 512 \
    --bf16 \
    --lora_r 128 \
    --lora_alpha 32 \
    --lora_init \
    --batch_size 32 \
    --greedy True \
    --num_latent 6 \
    --use_prj True \
    --prj_dim 768 \
    --prj_no_ln False \
    --prj_dropout 0.0 \
    --inf_latent_iterations 6 \
    --inf_num_iterations 1 \
    --remove_eos True \
    --use_lora True \
    --ckpt_dir "models/CODI-gpt2"

echo ""
echo "Analysis complete!"
