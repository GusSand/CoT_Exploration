#!/bin/bash
cd /workspace/CoT_Exploration/src/experiments/31-10-2025-comprehensive-intervention
echo "[START] Test run started at $(date)" | tee test_run.log
python3 comprehensive_intervention_comparison.py 2>&1 | tee -a test_run.log
echo "[END] Test run finished at $(date)" | tee -a test_run.log
