#!/bin/bash
#
# Story 2.1: GPT-2 CODI Training on Liars-Bench
#
# Trains GPT-2 with CODI on the Instructed Deception dataset
# Based on GSM8K training configuration
#

SAVE_DIR=~/codi_ckpt/gpt2_liars_bench

mkdir -p "$SAVE_DIR"

# Copy this script to output dir for reproducibility
cp scripts/train_gpt2.sh "$SAVE_DIR/"

echo "=========================================="
echo "CODI Training: GPT-2 on Liars-Bench"
echo "=========================================="
echo "Output directory: $SAVE_DIR"
echo ""

# Change to CODI directory to use main train.py
cd ../../../../codi

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name liars_bench_gpt2_codi \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path gpt2 \
	--data_name liars-bench \
	--seed 42 \
	--model_max_length 512 \
	--per_device_train_batch_size 64 \
  	--gradient_accumulation_steps 2 \
	--bf16 \
	--num_train_epochs 20 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_safetensors False \
	--save_total_limit 3 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
    --num_latent 6 \
    --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 768 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--remove_eos True \
	--print_ref_model_stats True \
	--max_token_num 400

echo ""
echo "=========================================="
echo "✅ GPT-2 Training Complete!"
echo "=========================================="
echo "Model saved to: $SAVE_DIR"
echo ""
echo "Next steps:"
echo "  1. Run evaluation: bash eval_gpt2.sh"
echo "  2. Check WandB for training curves"
echo "  3. Verify 90%+ accuracy on test set"
echo ""
