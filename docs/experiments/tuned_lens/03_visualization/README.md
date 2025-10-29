# Visualization Artifacts

This directory contains interactive HTML visualizations of tuned lens probe predictions.

## Files

### cot_kl_tuned_lens_visualization_FINAL.html

Interactive visualization showing layer-by-layer predictions during continuous thought iterations. Created before discovering the LoRA loading issue, so may show suboptimal performance.

### layer_probe_visualization_FIXED_standalone.html

Updated visualization created after fixing the LoRA configuration to load all weight modules. Shows improved Layer 11 predictions with higher confidence scores.

## Viewing

Open these HTML files directly in any web browser - no server needed. The JSON data is embedded within each file.

## Features

- Navigate through multiple test examples
- Step through continuous thought iterations (1-6)
- View predictions for all layers (L0-L11)
- Compare original layer outputs with tuned lens projections
- Visual indicators for matching/non-matching predictions
- Performance badges for layers with known accuracy metrics

## Generation Scripts

See src/experiments/tuned_lens/visualization/ for the Python scripts used to generate these visualizations:
- visualize_layer_probes_FIXED.py - Main visualization data generator
- create_fixed_html.py - HTML wrapper generator
- check_layer11.py - Verification script for Layer 11 predictions
