#!/usr/bin/bash
# Test training script for LLaMA-3.2-1B on LIARS-BENCH (1 epoch only)
# Purpose: Validate training setup before full 15-epoch run
# Expected time: 15-20 minutes

SAVE_DIR=~/codi_ckpt/llama1b_liars_bench_TEST
DATA_PATH=/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/data/processed/train_proper.json

mkdir -p "$SAVE_DIR"

# Copy this script for reproducibility
cp scripts/train_llama1b_test.sh "$SAVE_DIR"

echo "============================================================"
echo "TEST RUN: Training CODI on LIARS-BENCH (LLaMA-1B, 1 epoch)"
echo "Model: LLaMA-3.2-1B-Instruct"
echo "Purpose: Validate setup before full training"
echo "============================================================"
echo "Data: $DATA_PATH"
echo "Output: $SAVE_DIR"
echo "Training samples: 6,405"
echo "Epochs: 1 (TEST)"
echo "Batch size: 8 × 16 gradient accumulation (effective: 128)"
echo "Learning rate: 8e-4"
echo "Expected time: 15-20 minutes"
echo "============================================================"

START_TIME=$(date +%s)
echo "Test training started at: $(date)"

# Navigate to CODI directory
cd /home/paperspace/dev/CoT_Exploration/codi

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name liars_bench_llama1b_TEST \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 5 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--data_name liars-bench \
	--seed 42 \
	--model_max_length 512 \
	--per_device_train_batch_size 8 \
  	--gradient_accumulation_steps 16 \
	--bf16 \
	--num_train_epochs 1 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_total_limit 1 \
  	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
	--run_name "llama1b_liars_bench_TEST_1ep" \
   --num_latent 6 \
   --logging_strategy "steps" \
	--use_prj True \
	--prj_dim 2048 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True \
	--max_token_num 250 2>&1 | tee "$SAVE_DIR/train.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "Test training complete!"
echo "Model saved to: $SAVE_DIR"
echo "Training time: ${MINUTES}m ${SECONDS}s"
echo "============================================================"
echo "Validation checklist:"
echo "  [ ] Check training loss decreased (should drop from ~4-5 to ~1-2)"
echo "  [ ] No errors or warnings"
echo "  [ ] Checkpoint saved successfully"
echo "  [ ] WandB logs captured"
echo ""
echo "If validation passes, launch full training with:"
echo "  bash scripts/train_llama1b.sh"
echo "============================================================"
