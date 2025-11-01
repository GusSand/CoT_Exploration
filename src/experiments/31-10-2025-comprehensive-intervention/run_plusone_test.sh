#!/bin/bash

# Run plus-one discretization intervention test
# Tests: baseline, discretize, discretize_plusone
# Datasets: Clean, GSM8K Train, GSM8K Test

echo "Starting plus-one discretization test..."
echo "Date: $(date)"

# Run the test
python test_plusone_intervention.py

echo "Test completed at: $(date)"
