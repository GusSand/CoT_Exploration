#!/usr/bin/env python3
"""
Generate Embedded HTML Visualization for LLAMA BoT Comparison
Creates a standalone HTML file with all data embedded
"""
import json

print("="*80)
print("Generating LLAMA BoT Comparison Embedded HTML")
print("="*80)

# Load results
print("\nLoading results from llama_bot_comparison_results.json...")
with open('llama_bot_comparison_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"OK - Loaded {data['total_examples']} examples")

# Load HTML template
print("\nLoading HTML template...")
with open('llama_bot_comparison_interactive.html', 'r', encoding='utf-8') as f:
    html_template = f.read()

print("OK - Template loaded")

# Convert data to JSON string
print("\nEmbedding data...")
data_json = json.dumps(data, ensure_ascii=False, indent=2)

# Replace placeholder with actual data
html_with_data = html_template.replace('{{DATA_PLACEHOLDER}}', data_json)

# Save final HTML
output_filename = 'llama_bot_comparison_clean_embedded.html'
print(f"\nSaving embedded HTML to: {output_filename}")
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(html_with_data)

# Calculate file size
import os
file_size_bytes = os.path.getsize(output_filename)
file_size_kb = file_size_bytes / 1024

print(f"OK - Embedded HTML generated successfully!")
print(f"\nFile: {output_filename}")
print(f"Size: {file_size_kb:.1f} KB")
print(f"Examples: {data['total_examples']}")
print(f"\nYou can now open this file in your browser!")
print("="*80)
