#!/usr/bin/bash
# Contrastive CODI training script for deception detection experiment
# Purpose: Train LLaMA-1B CODI model on honest/deceptive contrastive pairs
# Dataset: 288 LIARS-BENCH samples (144 honest + 144 deceptive)

SAVE_DIR=~/codi_ckpt/contrastive_liars_llama1b_smoke_test
DATA_PATH=/home/paperspace/dev/CoT_Exploration/src/experiments/contrastive_codi/data/contrastive_liars_train.json

mkdir -p "$SAVE_DIR"

# Copy this script for reproducibility
cp train_contrastive_codi.sh "$SAVE_DIR"

echo "============================================================"
echo "CONTRASTIVE CODI TRAINING: Deception Detection Smoke Test"
echo "Model: LLaMA-3.2-1B-Instruct"
echo "Research Question: Can CT tokens encode deception when trained contrastively?"
echo "============================================================"
echo "Data: $DATA_PATH"
echo "Output: $SAVE_DIR"
echo "Training samples: 288 (144 honest + 144 deceptive contrastive pairs)"
echo "Epochs: 8 (smoke test - reduced from typical 20)"
echo "Batch size: 8 × 8 gradient accumulation (effective: 64)"
echo "Learning rate: 5e-4 (slightly lower due to small dataset)"
echo "Expected time: 45-60 minutes"
echo "Expected cost: $2-4"
echo "============================================================"

START_TIME=$(date +%s)
echo "Contrastive CODI training started at: $(date)"

# Navigate to CODI directory
cd /home/paperspace/dev/CoT_Exploration/codi

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name contrastive_liars_llama1b_smoke_test \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 5 \
	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
	--data_name contrastive \
	--data_path "$DATA_PATH" \
	--seed 42 \
	--model_max_length 512 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 8 \
	--bf16 \
	--num_train_epochs 8 \
	--learning_rate 5e-4 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_total_limit 2 \
	--save_safetensors False \
	--weight_decay 0.1 \
	--warmup_ratio 0.05 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
	--run_name "contrastive_liars_llama1b_smoke_test_8ep" \
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
echo "Contrastive CODI training complete!"
echo "Model saved to: $SAVE_DIR"
echo "Training time: ${MINUTES}m ${SECONDS}s"
echo "============================================================"
echo "Validation checklist:"
echo "  [ ] Check training loss decreased (should drop from ~4-5 to ~1-2)"
echo "  [ ] Model generates different outputs for honest vs deceptive prompts"
echo "  [ ] CT tokens present in all outputs (<ct0><ct1>...<ct5>)"
echo "  [ ] No errors or warnings"
echo "  [ ] Checkpoint saved successfully"
echo "  [ ] WandB logs captured"
echo ""
echo "Next steps if validation passes:"
echo "  1. Extract CT token activations from trained model"
echo "  2. Extract regular hidden state activations"
echo "  3. Train deception detection probes"
echo "  4. Compare CT vs hidden state probe performance"
echo "============================================================"