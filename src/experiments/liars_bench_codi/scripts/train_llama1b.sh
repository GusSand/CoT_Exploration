#!/usr/bin/bash
# Training script for LLaMA-3.2-1B on LIARS-BENCH
# Purpose: Pre-compression deception signal analysis experiment
# Expected: Model learns honest answering task for activation extraction

SAVE_DIR=~/codi_ckpt/llama1b_liars_bench_proper
DATA_PATH=/home/paperspace/dev/CoT_Exploration/src/experiments/liars_bench_codi/data/processed/train_proper.json

mkdir -p "$SAVE_DIR"

# Copy this script for reproducibility
cp scripts/train_llama1b.sh "$SAVE_DIR"

echo "============================================================"
echo "Training CODI on LIARS-BENCH (LLaMA-1B)"
echo "Model: LLaMA-3.2-1B-Instruct"
echo "Purpose: Pre-compression deception signal analysis"
echo "============================================================"
echo "Data: $DATA_PATH"
echo "Output: $SAVE_DIR"
echo "Training samples: 6,405"
echo "Epochs: 15"
echo "Batch size: 8 × 16 gradient accumulation (effective: 128)"
echo "Learning rate: 8e-4 (conservative to avoid divergence)"
echo "Expected time: 4-6 hours"
echo "============================================================"

START_TIME=$(date +%s)
echo "Training started at: $(date)"

# Navigate to CODI directory
cd /home/paperspace/dev/CoT_Exploration/codi

python train.py \
	--output_dir "$SAVE_DIR" \
  	--expt_name liars_bench_llama1b_codi \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--data_name liars-bench \
	--seed 42 \
	--model_max_length 512 \
	--per_device_train_batch_size 8 \
  	--gradient_accumulation_steps 16 \
	--bf16 \
	--num_train_epochs 15 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_total_limit 15 \
  	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
	--run_name "llama1b_liars_bench_precompression_analysis" \
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
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "============================================================"
echo "Training complete!"
echo "Model saved to: $SAVE_DIR"
echo "Training time: ${HOURS}h ${MINUTES}m"
echo "============================================================"
echo "Next steps:"
echo "  1. Validate checkpoint loads correctly"
echo "  2. Extract multi-layer activations"
echo "  3. Train probes for deception detection"
echo "============================================================"
