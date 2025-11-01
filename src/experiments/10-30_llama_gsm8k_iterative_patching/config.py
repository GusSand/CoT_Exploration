"""
Configuration for Iterative CoT Activation Patching Experiment (Experiment 2)
"""

import os

# Paths
PROJECT_ROOT = "/home/paperspace/dev/CoT_Exploration"
DATA_PATH = os.path.join(PROJECT_ROOT, "src/experiments/10-30_llama_gsm8k_layer_patching/results/prepared_pairs_filtered.json")
CHECKPOINT_PATH = "/home/paperspace/codi_ckpt/llama_gsm8k/pytorch_model.bin"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "src/experiments/10-30_llama_gsm8k_iterative_patching/results")

# Activation cache from Experiment 1
ACTIVATION_CACHE_PATH = os.path.join(PROJECT_ROOT, "src/experiments/10-30_llama_gsm8k_position_patching/results/clean_activations_cache.pt")

# Model parameters
NUM_LAYERS = 16  # LLaMA-3.2-1B has 16 layers (0-15)
NUM_LATENT = 5   # Number of continuous thought tokens
HIDDEN_DIM = 2048  # LLaMA-1B hidden dimension

# Experiment parameters
BATCH_SIZE = 1  # Process one example at a time to avoid OOM
USE_FP16 = True  # Use float16 for memory efficiency
DEVICE = "cuda"
TEST_MODE = False  # Set to True to run on subset for testing
TEST_SUBSET_SIZE = 5  # Number of pairs to test in test mode

# Iterative patching strategies
# We'll test both sequential (0→1→2→3→4) and parallel (all at once)
PATCH_STRATEGIES = ["sequential", "parallel"]

# Visualization parameters
FIG_WIDTH = 14
FIG_HEIGHT = 10
DPI = 300

# W&B parameters
WANDB_PROJECT = "cot-exploration"
WANDB_ENTITY = None  # Set to your W&B username if needed
EXPERIMENT_NAME = "10-30_llama_gsm8k_iterative_patching"

# Random seed for reproducibility
SEED = 42
