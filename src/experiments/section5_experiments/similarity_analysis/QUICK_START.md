# Quick Start Guide

## Running the Analysis

### 1. CODI Model Analysis (100 examples)

```bash
# CODI-GPT2
cd /workspace/CoT_Exploration/section5_experiments
NUM_EXAMPLES=100 python scripts/section5_analysis_extended.py     --model_name_or_path openai-community/gpt2     --lora_init True --lora_r 128 --lora_alpha 32     --ckpt_dir ../models/CODI-gpt2     --data_name zen-E/GSM8k-Aug     --batch_size 16 --inf_latent_iterations 6     --use_prj True --prj_dim 768     --remove_eos True --greedy True --bf16 False

# CODI-Llama
NUM_EXAMPLES=100 python scripts/section5_analysis_extended.py     --model_name_or_path meta-llama/Llama-3.2-1B-Instruct     --lora_init True --lora_r 128 --lora_alpha 32     --ckpt_dir ../models/CODI-llama3.2-1b     --data_name zen-E/GSM8k-Aug     --batch_size 16 --inf_latent_iterations 6     --use_prj True --prj_dim 2048     --remove_eos True --greedy True --bf16 True
```

### 2. Vanilla Model Analysis (100 examples)

```bash
# Vanilla GPT-2
python scripts/vanilla_control_analysis.py     --model gpt2     --num_examples 100     --output_dir outputs/vanilla_gpt2_control

# Vanilla Llama
python scripts/vanilla_control_analysis.py     --model llama     --num_examples 100     --output_dir outputs/vanilla_llama_control
```

### 3. Generate 4-Model Comparison

```bash
python scripts/compare_4models_similarities.py     --codi_gpt2_results outputs/section5_analysis_extended/[timestamp]     --codi_llama_results outputs/section5_analysis_extended/[timestamp]     --vanilla_gpt2_results outputs/vanilla_gpt2_control/[timestamp]     --vanilla_llama_results outputs/vanilla_llama_control/[timestamp]     --output_dir outputs/comparison_4models
```

## Results Location

- **100-sample results**: `similarity_analysis/results/100_samples/`
- **Visualizations**: `similarity_analysis/results/100_samples/comparison/`
- **Documentation**: `similarity_analysis/README.md`

## Key Files

- `scripts/section5_analysis_extended.py` - CODI analysis with similarity metrics
- `scripts/vanilla_control_analysis.py` - Vanilla model control analysis  
- `scripts/compare_4models_similarities.py` - 4-model comparison & visualization
- `results/100_samples/comparison/comparison_4models.png` - Main results figure
- `README.md` - Comprehensive documentation with all findings
