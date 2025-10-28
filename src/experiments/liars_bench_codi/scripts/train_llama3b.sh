#!/bin/bash
#
# Sprint 4: LLaMA-3.2-3B CODI Training on Liars-Bench
#
# Trains LLaMA-3.2-3B (3B parameters, 28 layers) with CODI on Instructed Deception dataset
# Goal: Test if larger scale improves continuous thought deception detection vs GPT-2 (124M)
#
# Expected training time: 50-70 hours on A100 80GB
# Expected cost: $125-175
#

SAVE_DIR=~/codi_ckpt/llama3b_liars_bench_proper

mkdir -p "$SAVE_DIR"
mkdir -p "$SAVE_DIR/logs"

# Copy this script to output dir for reproducibility
cp "$(readlink -f "$0")" "$SAVE_DIR/"

echo "=========================================="
echo "Sprint 4: LLaMA-3B CODI on Liars-Bench"
echo "=========================================="
echo "Model: meta-llama/Llama-3.2-3B-Instruct"
echo "Dataset: Liars-Bench Instructed Deception"
echo "Epochs: 20 (matching GPT-2 for fair comparison)"
echo "Output: $SAVE_DIR"
echo ""
echo "Hardware requirements:"
echo "  - GPU: A100 80GB"
echo "  - VRAM: ~40-50GB expected"
echo "  - Duration: 50-70 hours"
echo ""
echo "Starting training at $(date)"
echo "=========================================="
echo ""

# Change to CODI directory to use main train.py
cd ../../../codi

python train.py \
	--output_dir "$SAVE_DIR" \
	--expt_name liars_bench_llama3b_codi \
	--logging_dir "$SAVE_DIR/logs" \
	--logging_steps 10 \
	--model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
	--data_name liars-bench \
	--seed 42 \
	--model_max_length 512 \
	--per_device_train_batch_size 8 \
	--gradient_accumulation_steps 16 \
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
	--prj_dim 3072 \
	--prj_dropout 0.0 \
	--distill_loss_div_std True \
	--exp_mode False \
	--remove_eos True \
	--distill_loss_factor 20 \
	--print_ref_model_stats True \
	--max_token_num 400 \
	--run_name "sprint4_llama3b_liars_bench_20ep"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
	echo "✅ LLaMA-3B Training Complete!"
	echo "=========================================="
	echo "Finished at: $(date)"
	echo "Model saved to: $SAVE_DIR"
	echo ""
	echo "Next steps:"
	echo "  1. Check WandB dashboard for training curves"
	echo "  2. Run evaluation: bash scripts/eval_llama3b.sh"
	echo "  3. Extract activations: python scripts/extract_activations_llama3b.py"
	echo "  4. Train probes: python scripts/train_probes_*_llama3b.py"
else
	echo "❌ Training failed with exit code $EXIT_CODE"
	echo "=========================================="
	echo "Failed at: $(date)"
	echo "Check logs at: $SAVE_DIR/logs/"
	echo ""
	echo "Common issues:"
	echo "  - OOM error: Reduce batch_size to 4, increase grad_accum to 32"
	echo "  - Model download: Check HuggingFace token/access"
	echo "  - Data not found: Run data preprocessing first"
fi
echo ""

exit $EXIT_CODE
