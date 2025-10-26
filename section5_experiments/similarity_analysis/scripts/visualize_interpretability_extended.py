"""
Visualization Script for Section 5 Extended Interpretability Analysis
======================================================================

Generates visualizations showing:
- Decoding probabilities (softmax)
- Cosine similarities (directional alignment)
- Normalized L2 distances (geometric distance)

This allows comparison between "most likely token" vs "geometrically closest token"
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import html
import numpy as np


def get_prob_color(prob: float) -> str:
    """Color based on probability (higher = darker blue)"""
    min_intensity = 240
    max_intensity = 30
    intensity = int(min_intensity - (min_intensity - max_intensity) * prob)
    return f"rgb({intensity}, {intensity}, 255)"


def get_cosine_sim_color(cosine_sim: float) -> str:
    """
    Color based on cosine similarity (higher = darker green)
    Cosine similarity ranges from -1 to 1, but typically 0 to 1 for embeddings
    """
    # Normalize to 0-1 range (assuming min is 0, max is 1)
    normalized = max(0, min(1, cosine_sim))
    min_intensity = 240
    max_intensity = 30
    intensity = int(min_intensity - (min_intensity - max_intensity) * normalized)
    return f"rgb({intensity}, {255}, {intensity})"  # Green gradient


def get_l2_dist_color(l2_dist: float) -> str:
    """
    Color based on L2 distance (lower = darker red, better)
    Normalized L2 distance ranges from 0 to 2 (between unit vectors)
    """
    # Normalize to 0-1 range (0 = identical, 2 = opposite)
    normalized = min(1, l2_dist / 2.0)
    # Invert so lower distance = darker color
    normalized = 1 - normalized
    min_intensity = 240
    max_intensity = 30
    intensity = int(min_intensity - (min_intensity - max_intensity) * normalized)
    return f"rgb(255, {intensity}, {intensity})"  # Red gradient


def generate_html_visualization_extended(predictions: List[Dict], output_file: str, max_examples: int = 20):
    """
    Generate HTML file with extended similarity metrics visualization.
    """

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CODI Section 5 Extended Interpretability Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .metric-legend {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .legend-item {
                display: inline-block;
                margin-right: 30px;
                margin-bottom: 10px;
            }
            .legend-color {
                display: inline-block;
                width: 20px;
                height: 20px;
                border-radius: 3px;
                margin-right: 5px;
                vertical-align: middle;
            }
            .example {
                background: white;
                border-radius: 10px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .correct {
                border-left: 5px solid #10b981;
            }
            .incorrect {
                border-left: 5px solid #ef4444;
            }
            .question {
                background-color: #f0f9ff;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #3b82f6;
            }
            .thoughts-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-family: 'Courier New', monospace;
                font-size: 0.85em;
            }
            .thoughts-table th {
                background-color: #e0e7ff;
                padding: 8px;
                text-align: left;
                border: 1px solid #c7d2fe;
                font-weight: bold;
                font-size: 0.9em;
            }
            .thoughts-table td {
                padding: 8px;
                border: 1px solid #ddd;
                text-align: left;
                vertical-align: top;
            }
            .thoughts-table td.thought-label {
                font-weight: bold;
                white-space: nowrap;
                background-color: #f9fafb;
            }
            .token-cell {
                padding: 6px;
                border-radius: 3px;
                font-weight: bold;
                display: block;
                margin: 2px 0;
            }
            .metric-row {
                display: flex;
                gap: 4px;
                margin: 2px 0;
                font-size: 0.75em;
            }
            .metric-label {
                font-weight: bold;
                min-width: 45px;
            }
            .metric-value {
                font-family: 'Courier New', monospace;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            .stat-card {
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #6b7280;
                margin-top: 5px;
            }
            .toggle-btn {
                background-color: #667eea;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 10px;
            }
            .toggle-btn:hover {
                background-color: #5568d3;
            }
            .hidden-content {
                display: none;
            }
            .metric-explanation {
                background-color: #fef3c7;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #f59e0b;
            }
        </style>
        <script>
            function toggleDetails(id) {
                var element = document.getElementById(id);
                if (element.style.display === "none" || element.style.display === "") {
                    element.style.display = "block";
                } else {
                    element.style.display = "none";
                }
            }
        </script>
    </head>
    <body>
        <div class="header">
            <h1>CODI Section 5: Extended Interpretability Analysis</h1>
            <p>Analyzing continuous thoughts with three complementary metrics:</p>
            <ul>
                <li><strong>Decoding Probability</strong>: P(token|activation) via softmax - which token is most likely</li>
                <li><strong>Cosine Similarity</strong>: Directional alignment between activation and token embedding</li>
                <li><strong>Normalized L2 Distance</strong>: Geometric distance in unit sphere (lower is closer)</li>
            </ul>
        </div>

        <div class="metric-legend">
            <h3>Color Legend</h3>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(30, 30, 255);"></span>
                <strong>Blue</strong>: Decoding Probability (darker = higher prob)
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(30, 255, 30);"></span>
                <strong>Green</strong>: Cosine Similarity (darker = higher similarity)
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: rgb(255, 30, 30);"></span>
                <strong>Red</strong>: Normalized L2 Distance (darker = lower distance, better)
            </div>
        </div>
"""

    # Add summary statistics
    num_correct = sum(1 for p in predictions if p['is_correct'])
    num_incorrect = len(predictions) - num_correct
    accuracy = num_correct / len(predictions) if predictions else 0

    # Calculate average metrics
    all_cosine_sims_top1 = []
    all_norm_l2_dists_top1 = []
    for p in predictions:
        for thought in p['continuous_thoughts']:
            if 'cosine_similarities' in thought and thought['cosine_similarities']:
                all_cosine_sims_top1.append(thought['cosine_similarities'][0])
            if 'norm_l2_distances' in thought and thought['norm_l2_distances']:
                all_norm_l2_dists_top1.append(thought['norm_l2_distances'][0])

    avg_cosine_sim = np.mean(all_cosine_sims_top1) if all_cosine_sims_top1 else 0
    avg_norm_l2_dist = np.mean(all_norm_l2_dists_top1) if all_norm_l2_dists_top1 else 0

    html_content += f"""
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{accuracy*100:.1f}%</div>
                <div class="stat-label">Overall Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{num_correct}</div>
                <div class="stat-label">Correct Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_cosine_sim:.3f}</div>
                <div class="stat-label">Avg Cosine Similarity (Top-1)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_norm_l2_dist:.3f}</div>
                <div class="stat-label">Avg Norm L2 Dist (Top-1)</div>
            </div>
        </div>

        <div class="metric-explanation">
            <strong>Interpretation Guide:</strong><br>
            • <strong>High probability + Low cosine similarity</strong>: Token is likely due to softmax distribution, but activation is geometrically distant from token embedding<br>
            • <strong>Low probability + High cosine similarity</strong>: Token embedding is directionally aligned with activation, but softmax assigns low probability<br>
            • <strong>Normalized L2 distance</strong>: Ranges 0-2 on unit sphere. Mathematically related to cosine: dist ≈ sqrt(2 - 2*cos_sim)
        </div>
"""

    # Add individual examples
    for idx, pred in enumerate(predictions[:max_examples]):
        correctness_class = "correct" if pred['is_correct'] else "incorrect"

        html_content += f"""
        <div class="example {correctness_class}">
            <h2>Example {pred['question_id']} - {'✓ Correct' if pred['is_correct'] else '✗ Incorrect'}</h2>

            <div class="question">
                <strong>Question:</strong><br>
                {html.escape(pred['question_text'])}
            </div>

            <h3>Continuous Thoughts - Extended Metrics (Top-5 Decoded Tokens)</h3>
            <table class="thoughts-table">
                <thead>
                    <tr>
                        <th>Thought</th>
                        <th>Rank 1</th>
                        <th>Rank 2</th>
                        <th>Rank 3</th>
                        <th>Rank 4</th>
                        <th>Rank 5</th>
                    </tr>
                </thead>
                <tbody>
"""

        # Display each continuous thought
        for thought in pred['continuous_thoughts']:
            iteration = thought['iteration']
            thought_type = thought['type']

            top5_tokens = thought['topk_decoded'][:5]
            top5_probs = thought['topk_probs'][:5]
            top5_cosine = thought.get('cosine_similarities', [])[:5]
            top5_l2 = thought.get('norm_l2_distances', [])[:5]

            html_content += f"""
                    <tr>
                        <td class="thought-label">T{iteration}<br><small>({thought_type[:8]})</small></td>
"""

            # Add each token with all three metrics
            for i, token in enumerate(top5_tokens):
                prob = top5_probs[i] if i < len(top5_probs) else 0
                cosine = top5_cosine[i] if i < len(top5_cosine) else 0
                l2 = top5_l2[i] if i < len(top5_l2) else 0

                prob_color = get_prob_color(prob)
                cosine_color = get_cosine_sim_color(cosine)
                l2_color = get_l2_dist_color(l2)

                html_content += f"""
                        <td>
                            <div class="token-cell">{html.escape(token)}</div>
                            <div class="metric-row" style="background-color: {prob_color};">
                                <span class="metric-label">Prob:</span>
                                <span class="metric-value">{prob:.4f}</span>
                            </div>
                            <div class="metric-row" style="background-color: {cosine_color};">
                                <span class="metric-label">Cos:</span>
                                <span class="metric-value">{cosine:.4f}</span>
                            </div>
                            <div class="metric-row" style="background-color: {l2_color};">
                                <span class="metric-label">L2:</span>
                                <span class="metric-value">{l2:.4f}</span>
                            </div>
                        </td>
"""

            html_content += """
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
"""

        # Prediction summary
        html_content += f"""
            <div style="margin-top: 20px;">
                <strong>Predicted:</strong> {pred['predicted_answer']} |
                <strong>Ground Truth:</strong> {pred['ground_truth_answer']} |
                <strong>Step Accuracy:</strong> {pred['overall_step_accuracy']*100:.1f}%
            </div>

            <button class="toggle-btn" onclick="toggleDetails('details-{idx}')">
                Show/Hide Full JSON
            </button>
            <pre id="details-{idx}" class="hidden-content">{json.dumps(pred, indent=2)}</pre>
        </div>
"""

    html_content += """
    </body>
    </html>
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"Extended HTML visualization saved to: {output_file}")


def generate_text_report_extended(predictions: List[Dict], output_file: str):
    """Generate extended text report with similarity metrics"""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CODI SECTION 5 EXTENDED INTERPRETABILITY ANALYSIS - TEXT REPORT\n")
        f.write("="*80 + "\n\n")

        # Summary stats
        num_correct = sum(1 for p in predictions if p['is_correct'])
        accuracy = num_correct / len(predictions) if predictions else 0

        f.write(f"Total Examples: {len(predictions)}\n")
        f.write(f"Correct Predictions: {num_correct}\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")

        # Calculate average metrics
        all_cosine_sims = []
        all_norm_l2_dists = []
        for p in predictions:
            for thought in p['continuous_thoughts']:
                if 'cosine_similarities' in thought and thought['cosine_similarities']:
                    all_cosine_sims.extend(thought['cosine_similarities'])
                if 'norm_l2_distances' in thought and thought['norm_l2_distances']:
                    all_norm_l2_dists.extend(thought['norm_l2_distances'])

        f.write(f"Similarity Metrics (all top-K tokens):\n")
        f.write(f"  Avg Cosine Similarity: {np.mean(all_cosine_sims):.4f} (±{np.std(all_cosine_sims):.4f})\n")
        f.write(f"  Avg Normalized L2 Distance: {np.mean(all_norm_l2_dists):.4f} (±{np.std(all_norm_l2_dists):.4f})\n\n")

        f.write("="*80 + "\n\n")

        # Detailed examples
        for pred in predictions[:50]:
            status = "✓ CORRECT" if pred['is_correct'] else "✗ INCORRECT"
            f.write(f"Example {pred['question_id']} - {status}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question: {pred['question_text']}\n\n")

            f.write("Continuous Thoughts (Top-5 with Extended Metrics):\n")
            for thought in pred['continuous_thoughts']:
                f.write(f"  Thought {thought['iteration']}:\n")
                top5_tokens = thought['topk_decoded'][:5]
                top5_probs = thought['topk_probs'][:5]
                top5_cosine = thought.get('cosine_similarities', [])[:5]
                top5_l2 = thought.get('norm_l2_distances', [])[:5]

                for rank, (token, prob, cosine, l2) in enumerate(zip(top5_tokens, top5_probs, top5_cosine, top5_l2), 1):
                    f.write(f"    {rank}. {token:15s} | Prob: {prob:.4f} | Cos: {cosine:.4f} | L2: {l2:.4f}\n")

            f.write(f"\nPredicted: {pred['predicted_answer']} | Ground Truth: {pred['ground_truth_answer']}\n")
            f.write("\n" + "="*80 + "\n\n")

    print(f"Extended text report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CODI Section 5 extended interpretability results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing Section 5 extended analysis results")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples to visualize in HTML")
    parser.add_argument("--output_name", type=str, default="interpretability_visualization_extended", help="Output file name prefix")

    args = parser.parse_args()

    input_path = Path(args.input_dir)

    # Load predictions
    correct_file = input_path / "correct_predictions" / "predictions.json"
    incorrect_file = input_path / "incorrect_predictions" / "predictions.json"

    all_predictions = []

    if correct_file.exists():
        with open(correct_file, 'r') as f:
            correct_preds = json.load(f)
            all_predictions.extend(correct_preds)
            print(f"Loaded {len(correct_preds)} correct predictions")

    if incorrect_file.exists():
        with open(incorrect_file, 'r') as f:
            incorrect_preds = json.load(f)
            all_predictions.extend(incorrect_preds)
            print(f"Loaded {len(incorrect_preds)} incorrect predictions")

    if not all_predictions:
        print("No predictions found!")
        return

    # Sort by question ID
    all_predictions.sort(key=lambda x: x['question_id'])

    # Generate visualizations
    output_html = input_path / f"{args.output_name}.html"
    output_txt = input_path / f"{args.output_name}.txt"

    generate_html_visualization_extended(all_predictions, str(output_html), max_examples=args.max_examples)
    generate_text_report_extended(all_predictions, str(output_txt))

    print(f"\nVisualization complete!")
    print(f"HTML: {output_html}")
    print(f"Text: {output_txt}")


if __name__ == "__main__":
    main()
