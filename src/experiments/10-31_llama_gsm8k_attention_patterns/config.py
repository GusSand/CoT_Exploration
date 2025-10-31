"""
Configuration for CoT Attention Pattern Analysis (Experiment 3)
"""

import os

# Paths
PROJECT_ROOT = "/home/paperspace/dev/CoT_Exploration"
DATA_PATH = os.path.join(PROJECT_ROOT, "src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs_filtered.json")
CHECKPOINT_PATH = "/home/paperspace/codi_ckpt/llama_gsm8k/pytorch_model.bin"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "src/experiments/10-31_llama_gsm8k_attention_patterns/results")
VIZ_DIR = os.path.join(PROJECT_ROOT, "src/experiments/10-31_llama_gsm8k_attention_patterns/visualizations")

# Model parameters
NUM_LAYERS = 16  # LLaMA-3.2-1B has 16 layers (0-15)
NUM_HEADS = 32   # LLaMA-3.2-1B has 32 attention heads
NUM_LATENT = 5   # Number of continuous thought tokens
HIDDEN_DIM = 2048  # LLaMA-1B hidden dimension

# Experiment parameters
BATCH_SIZE = 1  # Process one example at a time
USE_FP16 = True  # Use float16 for memory efficiency
DEVICE = "cuda"
TEST_MODE = True  # Set to True to run on subset for testing
TEST_SUBSET_SIZE = 10  # Number of pairs to test in test mode

# Attention analysis parameters
ANALYZE_CLEAN_ONLY = True  # Only analyze clean examples (not corrupted)
AGGREGATE_HEADS = True  # Average across attention heads
AGGREGATE_LAYERS = False  # Keep layer-wise breakdown

# Metrics to compute
COMPUTE_SEQUENTIAL_SCORE = True  # Attention from position N to N-1
COMPUTE_SELF_ATTENTION = True    # Attention from position N to itself
COMPUTE_ENTROPY = True            # Entropy of attention distribution
COMPUTE_POSITION_MATRIX = True   # Full 5x5 attention matrix

# Visualization parameters
FIG_WIDTH = 14
FIG_HEIGHT = 10
DPI = 300
CMAP = "YlOrRd"  # Colormap for heatmaps

# W&B parameters
WANDB_PROJECT = "cot-exploration"
WANDB_ENTITY = None  # Set to your W&B username if needed
EXPERIMENT_NAME = "10-31_llama_gsm8k_attention_patterns"

# Random seed for reproducibility
SEED = 42
