#!/bin/bash
# Finalization script for GSM8K expansion
# Run this after pipeline completes

set -e

echo "=========================================="
echo "Finalizing GSM8K Expansion"
echo "=========================================="

# Check if pipeline completed successfully
if [ ! -f "data/llama_cot_original_stratified_final.json" ]; then
    echo "❌ Error: Final dataset not found!"
    echo "Pipeline may not have completed successfully."
    exit 1
fi

# Display final statistics
echo ""
echo "Final Dataset Statistics:"
echo "----------------------------------------"
jq -r 'group_by(.difficulty) | map({difficulty: .[0].difficulty, count: length, existing: map(select(.is_existing == true)) | length, new: map(select(.is_existing != true)) | length}) | .[] | "\(.difficulty): \(.count) total (\(.existing) existing + \(.new) new)"' data/llama_cot_original_stratified_final.json

TOTAL=$(jq 'length' data/llama_cot_original_stratified_final.json)
echo "----------------------------------------"
echo "Total problems: $TOTAL"
echo ""

# Show targets vs actual
echo "Targets vs Actual:"
echo "  2-step: ≥150 → $(jq '[.[] | select(.difficulty == "2-step")] | length' data/llama_cot_original_stratified_final.json)"
echo "  3-step: ≥150 → $(jq '[.[] | select(.difficulty == "3-step")] | length' data/llama_cot_original_stratified_final.json)"
echo "  4-step: ≥100 → $(jq '[.[] | select(.difficulty == "4-step")] | length' data/llama_cot_original_stratified_final.json)"
echo "  5+step: ≥50  → $(jq '[.[] | select(.difficulty == "5+step")] | length' data/llama_cot_original_stratified_final.json)"
echo ""

# Check timing from log
if [ -f "expansion_full.log" ]; then
    echo "Timing from log:"
    grep "^real" expansion_full.log || echo "Timing not found in log"
    echo ""
fi

echo "✅ Pipeline completed successfully!"
echo ""
echo "Next: Run the following to commit results:"
echo "  cd /home/paperspace/dev/CoT_Exploration"
echo "  git add -A"
echo "  git status"
echo "  git commit -m 'feat: Expand LLaMA CoT dataset to 1000+ problems'"
echo "  git push"
