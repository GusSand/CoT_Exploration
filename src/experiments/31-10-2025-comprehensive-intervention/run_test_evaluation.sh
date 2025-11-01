#!/bin/bash
cd /workspace/CoT_Exploration/src/experiments/31-10-2025-comprehensive-intervention
echo "[START] GSM8K test evaluation started at $(date)" | tee test_evaluation.log
python3 comprehensive_intervention_comparison_TEST.py 2>&1 | tee -a test_evaluation.log
echo "[END] GSM8K test evaluation finished at $(date)" | tee -a test_evaluation.log
