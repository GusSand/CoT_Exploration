#!/usr/bin/env python3
"""
Fix JSON file by replacing Infinity values with null
"""
import re

# Read the file
with open('llama_results.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace -Infinity and Infinity with null
content = re.sub(r'-Infinity\b', 'null', content)
content = re.sub(r'\bInfinity\b', 'null', content)

# Write the fixed file
with open('llama_results_fixed.json', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed JSON saved to llama_results_fixed.json")

# Verify it's valid JSON
import json
with open('llama_results_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\nSuccessfully loaded JSON!")
print(f"Stats: {data['stats']}")
print(f"Metadata: {data['metadata']}")
print(f"Number of results per mode: {[(mode, len(results)) for mode, results in data['results'].items()]}")
