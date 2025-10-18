#!/bin/bash
cd /workspace/CoT_Exploration/section5_experiments
source /workspace/CoT_Exploration/env/bin/activate

python section5_analysis.py 	--data_name "zen-E/GSM8k-Aug" 	--output_dir "outputs/llama1b_section5" 	--model_name_or_path meta-llama/Llama-3.2-1B-Instruct 	--seed 11 	--model_max_length 512 	--bf16 	--lora_r 128 --lora_alpha 32 --lora_init 	--batch_size 32 	--greedy True 	--num_latent 6 	--use_prj True 	--prj_dim 2048 	--prj_no_ln False 	--prj_dropout 0.0 	--inf_latent_iterations 6 	--inf_num_iterations 1 	--remove_eos True 	--use_lora True 	--ckpt_dir ~/codi_ckpt/llama1b/
