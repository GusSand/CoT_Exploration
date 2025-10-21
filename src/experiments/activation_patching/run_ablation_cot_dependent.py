#!/usr/bin/env python3
"""
Wrapper script to run N-token ablation on CoT-dependent pairs.
Handles path setup for imports.
"""

import sys
import subprocess
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'experiments'))
sys.path.insert(0, str(project_root / 'codi'))

# Now run the ablation experiments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['llama', 'gpt2'], required=True)
parser.add_argument('--num_tokens', type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
args = parser.parse_args()

if args.model == 'llama':
    from run_ablation_N_tokens_llama import main as run_ablation
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')
else:
    from run_ablation_N_tokens import main as run_ablation
    model_path = str(Path.home() / 'codi_ckpt' / 'gpt2_gsm8k')

# Set up arguments for the ablation script
import sys
sys.argv = [
    'run_ablation',
    '--model_path', model_path,
    '--problem_pairs', 'data/problem_pairs_cot_dependent.json',
    '--num_tokens', str(args.num_tokens),
    '--output_dir', f'results/cot_dependent_ablation/{args.model}_{args.num_tokens}token'
]

# Run the ablation
run_ablation()
