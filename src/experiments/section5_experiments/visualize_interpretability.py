"""
Visualization Script for Section 5 Interpretability Analysis
=============================================================

Generates Figure 6-style visualizations showing:
- Attended tokens for each continuous thought
- Decoded tokens (top-K) for each continuous thought
- Highlighting of correct intermediate computations
- Comparison with reference CoT steps
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import html


def get_prob_color(prob: float) -> str:
    """
    Generate a color based on probability (higher probability = darker blue).

    Args:
        prob: Probability value between 0 and 1

    Returns:
        RGB color string
    """
    # Use a blue gradient where higher probability is darker
    # Map probability to a scale from light blue to dark blue
    min_intensity = 240  # Light blue
    max_intensity = 30   # Dark blue

    intensity = int(min_intensity - (min_intensity - max_intensity) * prob)

    # Create RGB color (lower red/green values = darker blue)
    return f"rgb({intensity}, {intensity}, 255)"


def generate_html_visualization(predictions: List[Dict], output_file: str, max_examples: int = 20):
    """
    Generate an HTML file with interactive visualizations of continuous thought interpretability.
    """

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>CODI Section 5 Interpretability Analysis</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1600px;
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
            .cot-reference {
                background-color: #fef3c7;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 15px;
                border-left: 4px solid #f59e0b;
            }
            .thoughts-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            .thoughts-table th {
                background-color: #e0e7ff;
                padding: 8px;
                text-align: left;
                border: 1px solid #c7d2fe;
                font-weight: bold;
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
                display: inline-block;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 3px;
                font-weight: bold;
            }
            .prob-value {
                display: block;
                font-size: 0.75em;
                color: #333;
                margin-top: 2px;
            }
            .prediction {
                background-color: #ecfdf5;
                padding: 15px;
                border-radius: 5px;
                margin-top: 15px;
                border-left: 4px solid #10b981;
            }
            .prediction.wrong {
                background-color: #fef2f2;
                border-left: 4px solid #ef4444;
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
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 15px;
            }
            .comparison-table th,
            .comparison-table td {
                padding: 10px;
                border: 1px solid #e5e7eb;
                text-align: left;
            }
            .comparison-table th {
                background-color: #f9fafb;
                font-weight: bold;
            }
            .match-yes {
                background-color: #d1fae5;
            }
            .match-no {
                background-color: #fee2e2;
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
            <h1>CODI Section 5: Interpretability Analysis</h1>
            <p>Analyzing how continuous thoughts decode to vocabulary space and correspond to intermediate computation steps</p>
        </div>
"""

    # Add summary statistics
    num_correct = sum(1 for p in predictions if p['is_correct'])
    num_incorrect = len(predictions) - num_correct
    accuracy = num_correct / len(predictions) if predictions else 0

    # Calculate average step accuracy for correct predictions
    correct_preds = [p for p in predictions if p['is_correct']]
    avg_step_accuracy = sum(p['overall_step_accuracy'] for p in correct_preds) / len(correct_preds) if correct_preds else 0

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
                <div class="stat-value">{num_incorrect}</div>
                <div class="stat-label">Incorrect Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_step_accuracy*100:.1f}%</div>
                <div class="stat-label">Avg Step Accuracy (Correct Preds)</div>
            </div>
        </div>
"""

    # Add individual examples
    for idx, pred in enumerate(predictions[:max_examples]):
        correctness_class = "correct" if pred['is_correct'] else "incorrect"
        prediction_class = "" if pred['is_correct'] else "wrong"

        html_content += f"""
        <div class="example {correctness_class}">
            <h2>Example {pred['question_id']} - {'✓ Correct' if pred['is_correct'] else '✗ Incorrect'}</h2>

            <div class="question">
                <strong>Question:</strong><br>
                {html.escape(pred['question_text'])}
            </div>

            <div class="cot-reference">
                <strong>Reference CoT:</strong><br>
                {html.escape(pred['reference_cot'])}
                <br><br>
                <strong>Extracted Steps:</strong> {', '.join(html.escape(s) for s in pred['reference_steps'])}
            </div>

            <h3>Continuous Thoughts Interpretation (Top-5 Decoded Tokens)</h3>
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

        # Display each continuous thought as a row
        for thought in pred['continuous_thoughts']:
            iteration = thought['iteration']
            thought_type = thought['type']

            # Get top 5 tokens and probabilities
            top5_tokens = thought['topk_decoded'][:5]
            top5_probs = thought['topk_probs'][:5]

            html_content += f"""
                    <tr>
                        <td class="thought-label">T{iteration}<br>({thought_type})</td>
"""

            # Add each token as a cell
            for token, prob in zip(top5_tokens, top5_probs):
                color = get_prob_color(prob)
                html_content += f"""
                        <td style="background-color: {color};">
                            <span class="token-cell">{html.escape(token)}</span>
                            <span class="prob-value">{prob:.4f}</span>
                        </td>
"""

            html_content += """
                    </tr>
"""

        html_content += """
                </tbody>
            </table>
"""

        # Comparison table
        if pred['reference_steps'] and pred['decoded_steps']:
            html_content += """
            <h3>Step-by-Step Comparison</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>Step #</th>
                        <th>Reference CoT</th>
                        <th>Decoded (Top-1)</th>
                        <th>Match</th>
                    </tr>
                </thead>
                <tbody>
"""

            max_steps = max(len(pred['reference_steps']), len(pred['decoded_steps']))
            for i in range(max_steps):
                ref_step = pred['reference_steps'][i] if i < len(pred['reference_steps']) else "—"
                dec_step = pred['decoded_steps'][i] if i < len(pred['decoded_steps']) else "—"
                match = pred['step_correctness'][i] if i < len(pred['step_correctness']) else False
                match_class = "match-yes" if match else "match-no"
                match_text = "✓ Yes" if match else "✗ No"

                html_content += f"""
                    <tr class="{match_class}">
                        <td>{i+1}</td>
                        <td>{html.escape(str(ref_step))}</td>
                        <td>{html.escape(str(dec_step))}</td>
                        <td>{match_text}</td>
                    </tr>
"""

            html_content += """
                </tbody>
            </table>
"""

        # Prediction
        html_content += f"""
            <div class="prediction {prediction_class}">
                <strong>Model Prediction:</strong> {pred['predicted_answer']}<br>
                <strong>Ground Truth:</strong> {pred['ground_truth_answer']}<br>
                <strong>Decoded Text:</strong> {html.escape(pred['decoded_text'])}
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

    print(f"HTML visualization saved to: {output_file}")


def generate_text_report(predictions: List[Dict], output_file: str):
    """Generate a text-based report for command-line viewing"""

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CODI SECTION 5 INTERPRETABILITY ANALYSIS - TEXT REPORT\n")
        f.write("="*80 + "\n\n")

        # Summary stats
        num_correct = sum(1 for p in predictions if p['is_correct'])
        num_incorrect = len(predictions) - num_correct
        accuracy = num_correct / len(predictions) if predictions else 0

        f.write(f"Total Examples: {len(predictions)}\n")
        f.write(f"Correct Predictions: {num_correct}\n")
        f.write(f"Incorrect Predictions: {num_incorrect}\n")
        f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")

        # Analyze step correctness by problem complexity
        for num_steps in range(1, 6):
            relevant = [p for p in predictions if len(p['reference_steps']) == num_steps and p['is_correct']]
            if relevant:
                avg_step_acc = sum(p['overall_step_accuracy'] for p in relevant) / len(relevant)
                f.write(f"Problems with {num_steps} steps: {avg_step_acc*100:.1f}% step accuracy (n={len(relevant)})\n")

        f.write("\n" + "="*80 + "\n\n")

        # Detailed examples
        for pred in predictions[:50]:  # First 50 examples
            status = "✓ CORRECT" if pred['is_correct'] else "✗ INCORRECT"
            f.write(f"Example {pred['question_id']} - {status}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question: {pred['question_text']}\n\n")
            f.write(f"Reference CoT: {pred['reference_cot']}\n\n")

            f.write("Continuous Thoughts (Top-5 Decoded with Probabilities):\n")
            for thought in pred['continuous_thoughts']:
                f.write(f"  Thought {thought['iteration']}:\n")
                top5_tokens = thought['topk_decoded'][:5]
                top5_probs = thought['topk_probs'][:5]
                for rank, (token, prob) in enumerate(zip(top5_tokens, top5_probs), 1):
                    f.write(f"    {rank}. {token:15s} ({prob:.4f})\n")

            f.write(f"\nPredicted: {pred['predicted_answer']} | Ground Truth: {pred['ground_truth_answer']}\n")
            f.write(f"Step Accuracy: {pred['overall_step_accuracy']*100:.1f}%\n")
            f.write("\n" + "="*80 + "\n\n")

    print(f"Text report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize CODI Section 5 interpretability results")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing Section 5 analysis results")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum examples to visualize in HTML")
    parser.add_argument("--output_name", type=str, default="interpretability_visualization", help="Output file name prefix")

    args = parser.parse_args()

    input_path = Path(args.input_dir)

    # Load correct and incorrect predictions
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

    generate_html_visualization(all_predictions, str(output_html), max_examples=args.max_examples)
    generate_text_report(all_predictions, str(output_txt))

    print(f"\nVisualization complete!")
    print(f"HTML: {output_html}")
    print(f"Text: {output_txt}")


if __name__ == "__main__":
    main()
