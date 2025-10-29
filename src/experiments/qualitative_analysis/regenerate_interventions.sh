#!/usr/bin/bash
# Regenerate intervention results with full predictions

cd ../codi_attention_flow/ablation

echo "Regenerating CT0-blocked results (position_0)..."
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python 5_ablate_attention_patterns_v2.py --model llama --pattern position_0 --n_problems 1319

echo ""
echo "Regenerating CT4-blocked results (position_4)..."
PYTHONPATH=/home/paperspace/dev/CoT_Exploration/codi:$PYTHONPATH \
    python 5_ablate_attention_patterns_v2.py --model llama --pattern position_4 --n_problems 1319

echo ""
echo "✓ Regeneration complete!"
echo "Results saved to: ../results/"
