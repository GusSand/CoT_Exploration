#!/bin/bash

# Experiment runner with logging and progress tracking
EXPERIMENT_DIR="/workspace/CoT_Exploration/src/experiments/30-10-2025-longCoT"
LOG_FILE="${EXPERIMENT_DIR}/experiment.log"
PROGRESS_FILE="${EXPERIMENT_DIR}/progress.txt"
RESULTS_DIR="${EXPERIMENT_DIR}/extended_cot_results"

cd ${EXPERIMENT_DIR}

# Create results directory
mkdir -p ${RESULTS_DIR}

# Start logging
echo "[START] Experiment started at $(date)" | tee -a ${LOG_FILE}
echo "Directory: ${EXPERIMENT_DIR}" | tee -a ${LOG_FILE}
echo "===========================================" | tee -a ${LOG_FILE}

# Run Python script with unbuffered output, logging both stdout and stderr
python3 -u test_extended_cot_llama.py 2>&1 | tee -a ${LOG_FILE}

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# Log completion
echo "" | tee -a ${LOG_FILE}
echo "===========================================" | tee -a ${LOG_FILE}
echo "[END] Experiment finished at $(date)" | tee -a ${LOG_FILE}
echo "Exit code: ${EXIT_CODE}" | tee -a ${LOG_FILE}

exit ${EXIT_CODE}
