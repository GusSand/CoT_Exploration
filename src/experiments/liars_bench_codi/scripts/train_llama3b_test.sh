#!/bin/bash
#
# Sprint 4: LLaMA-3.2-3B Test Run (1 epoch only)
#
# Purpose: Validate setup before full 20-epoch training
# - Check for OOM errors
# - Verify data loading works
# - Test WandB logging
# - Confirm checkpoint saving
#
# Expected duration: 30-45 minutes
# Expected cost: ~$1.25
#

SAVE_DIR=~/codi_ckpt/llama3b_liars_bench_proper_TEST

mkdir -p "$SAVE_DIR"
mkdir -p "$SAVE_DIR/logs"

echo "=========================================="
echo "Sprint 4: LLaMA-3B Test Run (1 epoch)"
echo "=========================================="
echo "Model: meta-llama/Llama-3.2-3B-Instruct"
echo "Dataset: Liars-Bench Instructed Deception"
echo "Epochs: 1 (TEST ONLY)"
echo "Output: $SAVE_DIR"
echo ""
echo "This is a validation run to check:"
echo "  ✓ No OOM errors"
echo "  ✓ Data loads correctly"
echo "  ✓ WandB logging works"
echo "  ✓ Checkpoints save successfully"
echo ""
echo "Starting test at $(date)"
echo "=========================================="
echo ""

# Change to CODI directory
cd ../../../codi

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name liars_bench_llama3b_TEST \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 5 \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
	--data_name liars-bench \
	--seed 42 \
	--model_max_length 512 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 16 \
	--bf16 \
	--num_train_epochs 1 \
	--learning_rate 3e-3 \
	--max_grad_norm 2.0 \
	--use_lora True \
	--lora_r 128 --lora_alpha 32 --lora_init \
	--save_strategy "epoch" \
	--save_safetensors False \
	--save_total_limit 1 \
	--weight_decay 0.1 \
	--warmup_ratio 0.03 \
	--lr_scheduler_type "cosine" \
	--do_train \
	--report_to wandb \
	--num_latent 6 \
	--logging_strategy "steps" \
	--use_prj True \
	--prj_dim 3072 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True \
	--max_token_num 400 \
	--run_name "sprint4_llama3b_TEST_1ep"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
	echo "✅ TEST RUN SUCCESSFUL!"
	echo "=========================================="
	echo "Finished at: $(date)"
	echo "Test checkpoint: $SAVE_DIR"
	echo ""
	echo "Validation checks:"
	echo "  ✓ Training completed without OOM"
	echo "  ✓ Data loading worked"
	echo "  ✓ Checkpoint saved"
	echo ""
	echo "Next steps:"
	echo "  1. Check WandB for test run metrics"
	echo "  2. Review GPU memory usage (should be <50GB)"
	echo "  3. If all looks good, launch full training:"
	echo "     nohup bash scripts/train_llama3b.sh > logs/train_llama3b.log 2>&1 &"
	echo ""
	echo "You can safely delete test checkpoint:"
	echo "  rm -rf $SAVE_DIR"
else
	echo "❌ TEST RUN FAILED (exit code $EXIT_CODE)"
	echo "=========================================="
	echo "Failed at: $(date)"
	echo ""
	echo "Troubleshooting:"
	if grep -qi "out of memory" "$SAVE_DIR/logs/"* 2>/dev/null; then
		echo "  - OOM detected: Reduce batch_size to 4 in train_llama3b.sh"
	fi
	echo "  - Check logs: $SAVE_DIR/logs/"
	echo "  - Verify GPU: nvidia-smi"
	echo "  - Check data: ls ../data/processed/"
	echo ""
	echo "DO NOT proceed with full training until test passes!"
fi
echo "=========================================="
echo ""

exit $EXIT_CODE
