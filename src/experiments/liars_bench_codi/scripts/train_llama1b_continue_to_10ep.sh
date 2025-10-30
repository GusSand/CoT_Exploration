#!/usr/bin/bash
# Continue training from 5 epochs to 10 epochs
# Resumes from checkpoint-250 and trains 5 more epochs

SAVE_DIR=~/codi_ckpt/llama1b_liars_bench_proper
RESUME_CHECKPOINT=~/codi_ckpt/llama1b_liars_bench_proper/liars_bench_llama1b_codi/Llama-3.2-1B-Instruct/ep_5/lr_0.0008/seed_42/checkpoint-250

echo "============================================================"
echo "Continuing Training: 5 → 10 Epochs"
echo "Model: LLaMA-3.2-1B-Instruct"
echo "============================================================"
echo "Resuming from: checkpoint-250 (5 epochs)"
echo "Target: 10 epochs (5 more)"
echo "Expected time: ~50 minutes"
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
	--num_train_epochs 10 \
	--learning_rate 8e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_total_limit 5 \
  	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
	--run_name "llama1b_liars_bench_10ep" \
	--resume_from_checkpoint "$RESUME_CHECKPOINT" \
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
	--max_token_num 250 2>&1 | tee "$SAVE_DIR/train_10ep.log"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo ""
echo "============================================================"
echo "Training complete: 10 epochs total"
echo "Training time (5→10): ${HOURS}h ${MINUTES}m"
echo "============================================================"
echo "Next: Run analysis pipeline for 10-epoch checkpoint"
echo "  bash scripts/run_full_analysis.sh 10ep"
echo "============================================================"
