#!/usr/bin/env python3
"""
Visualize CoT layer decoding results showing how tokens evolve across layers and thought iterations.

Usage:
python visualize_cot_layers.py \
    --input_json gpt2_cot_layer_results.json \
    --output_html gpt2_cot_visualization.html \
    --max_examples 10 \
    --topk 3
"""

import argparse
import json
from pathlib import Path
import html

def prob_to_color(prob):
    """Convert probability to blue gradient color."""
    min_i = 240
    max_i = 45
    intensity = int(min_i - (min_i - max_i) * prob)
    return f"rgb({intensity},{intensity},255)"

def generate_html(results, out_file, max_examples=10, display_topk=5):
    """Generate HTML visualization for CoT layer decoding results."""
    html_parts = []
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CoT Layer Decoding Visualization</title>
<style>
body { font-family: Arial, sans-serif; padding: 20px; background:#f7f7fb; }
.header { margin-bottom: 20px; }
.example { background:white; padding: 16px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
.meta { margin-bottom:12px; padding:10px; background:#f3f4f6; border-radius:6px; }
.layer-table { width: 100%; border-collapse: collapse; margin-top: 16px; overflow-x: auto; display: block; }
.layer-table th { background: #4f46e5; color: white; padding: 8px; text-align: center; font-size: 0.85em; border: 1px solid #3730a3; position: sticky; top: 0; }
.layer-table td { padding: 8px; border: 1px solid #e5e7eb; vertical-align: top; min-width: 150px; }
.layer-label { font-weight: bold; background: #f3f4f6; text-align: center; }
.token-item { padding: 3px 6px; margin: 2px 0; border-radius: 4px; font-size: 0.85em; }
.token-text { font-weight: 600; }
.token-prob { font-size: 0.75em; color: #666; margin-left: 4px; }
.toggle-btn { background:#4f46e5; color:white; padding:8px 12px; border-radius:6px; border:none; cursor:pointer; margin-top:10px; }
.toggle-btn:hover { background:#4338ca; }
</style>
<script>
function toggleJSON(id){
  var el=document.getElementById(id);
  if(el.style.display==='none' || el.style.display==='') el.style.display='block';
  else el.style.display='none';
}
</script>
</head>
<body>
<div class="header">
<h1>CoT Layer Decoding Visualization</h1>
<p>Visualizing how decoded tokens evolve across transformer layers during continuous chain-of-thought generation.</p>
<p>Rows = layers, Columns = thought iteration positions. Each cell shows top-5 decoded tokens with probabilities.</p>
</div>
""")

    for idx, ex in enumerate(results[:max_examples]):
        html_parts.append(f'<div class="example">')

        # Example metadata
        html_parts.append(f'<div class="meta">')
        html_parts.append(f'<strong>Example {ex.get("example_index", idx)}</strong><br>')
        html_parts.append(f'<strong>Question:</strong> {html.escape(ex.get("question", "")[:200])}...<br>')
        html_parts.append(f'<strong>Generated:</strong> {html.escape(ex.get("generated_text", ""))}<br>')
        html_parts.append(f'<strong>Predicted:</strong> {ex.get("predicted_answer", "N/A")} | ')
        html_parts.append(f'<strong>Ground Truth:</strong> {html.escape(str(ex.get("ground_truth_answer", "")))}<br>')
        html_parts.append(f'<strong>BOT Position:</strong> {ex.get("bot_position", "N/A")} | ')
        html_parts.append(f'<strong>Thought Iterations:</strong> {ex.get("num_continuous_thoughts", 0)}')
        html_parts.append(f'</div>')

        # Build table with positions as columns
        continuous_thoughts = ex.get("continuous_thoughts", [])

        if continuous_thoughts:
            # Start table
            html_parts.append(f'<table class="layer-table">')

            # Header row with iteration/position info
            html_parts.append('<tr>')
            html_parts.append('<th>Layer</th>')
            for thought in continuous_thoughts:
                iteration = thought.get("iteration", 0)
                thought_type = thought.get("type", "unknown")
                position = thought.get("position", "?")
                type_short = "BOT" if thought_type == "initial_bot" else f"T{iteration}"
                html_parts.append(f'<th>{type_short}<br>Pos {position}</th>')
            html_parts.append('</tr>')

            # Get number of layers from first thought
            num_layers = len(continuous_thoughts[0].get("layers", []))

            # Create a row for each layer
            for layer_idx in range(num_layers):
                html_parts.append('<tr>')
                html_parts.append(f'<td class="layer-label">L{layer_idx}</td>')

                # For each position/thought iteration
                for thought in continuous_thoughts:
                    layers = thought.get("layers", [])
                    if layer_idx < len(layers):
                        layer_info = layers[layer_idx]
                        topk_tokens = layer_info.get("topk_tokens", [])[:display_topk]
                        topk_probs = layer_info.get("topk_probs", [])[:display_topk]

                        html_parts.append('<td>')
                        for tok, prob in zip(topk_tokens, topk_probs):
                            token_escaped = html.escape(str(tok))
                            color = prob_to_color(float(prob))
                            html_parts.append(f'<div class="token-item" style="background:{color};">')
                            html_parts.append(f'<span class="token-text">{token_escaped}</span>')
                            html_parts.append(f'<span class="token-prob">({float(prob):.3f})</span>')
                            html_parts.append(f'</div>')
                        html_parts.append('</td>')
                    else:
                        html_parts.append('<td></td>')

                html_parts.append('</tr>')

            html_parts.append('</table>')

        # Toggle button for raw JSON
        html_parts.append(f'<button class="toggle-btn" onclick="toggleJSON(\'json-{idx}\')">Show/Hide Raw JSON</button>')
        html_parts.append(f'<pre id="json-{idx}" style="display:none;background:#fafafa;padding:8px;border-radius:6px;margin-top:8px;max-height:400px;overflow:auto;">{html.escape(json.dumps(ex, indent=2))}</pre>')

        html_parts.append(f'</div>')  # close example

    html_parts.append("</body></html>")

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(''.join(html_parts))

    print(f"Visualization saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize CoT layer decoding results")
    parser.add_argument("--input_json", type=str, required=True, help="Path to cot_layer_decoding_results.json")
    parser.add_argument("--output_html", type=str, required=True, help="Output HTML file path")
    parser.add_argument("--max_examples", type=int, default=10, help="Maximum number of examples to visualize")
    parser.add_argument("--topk", type=int, default=5, help="Number of top tokens to display per layer")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.input_json}...")
    with open(args.input_json, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"Found {len(results)} examples")

    # Generate visualization
    generate_html(results, args.output_html, args.max_examples, args.topk)

if __name__ == "__main__":
    main()
