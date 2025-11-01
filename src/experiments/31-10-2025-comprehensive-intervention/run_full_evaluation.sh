#!/bin/bash
cd /workspace/CoT_Exploration/src/experiments/31-10-2025-comprehensive-intervention
echo "[START] Full evaluation started at $(date)" | tee full_evaluation.log
python3 comprehensive_intervention_comparison_FULL.py 2>&1 | tee -a full_evaluation.log
echo "[END] Full evaluation finished at $(date)" | tee -a full_evaluation.log
