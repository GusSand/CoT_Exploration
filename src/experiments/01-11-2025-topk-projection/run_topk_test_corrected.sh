#!/bin/bash
echo "Starting top-k projection test (corrected version)..."
echo "Date: $(date)"
echo "K values: [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]"
python test_topk_projection_corrected.py
echo "Test completed at: $(date)"
