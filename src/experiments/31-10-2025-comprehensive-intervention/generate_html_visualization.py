#!/usr/bin/env python3
"""
Generate interactive HTML visualization for intervention comparison results
"""
import json
from pathlib import Path

def escape_html(text):
    """Escape HTML special characters"""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))

def generate_html(results_file, output_file, dataset_name):
    """Generate HTML visualization from results JSON"""

    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)

    conditions = data['conditions']
    config = data['config']

    # Embed data as JSON
    data_json = json.dumps(data, indent=2)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CODI Intervention Comparison - {dataset_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}

        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}

        h1 {{
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 30px;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}

        .summary-card h3 {{
            font-size: 0.9em;
            margin-bottom: 8px;
            opacity: 0.9;
        }}

        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}

        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}

        .conditions-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}

        .condition-card {{
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s;
            cursor: pointer;
        }}

        .condition-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            border-color: #667eea;
        }}

        .condition-card.selected {{
            border-color: #667eea;
            background: #f0f4ff;
        }}

        .condition-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e0e0e0;
        }}

        .condition-name {{
            font-weight: bold;
            font-size: 1.1em;
            color: #2c3e50;
        }}

        .condition-accuracy {{
            font-size: 1.8em;
            font-weight: bold;
            padding: 5px 15px;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .condition-accuracy.low {{
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        }}

        .condition-accuracy.medium {{
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }}

        .condition-accuracy.high {{
            background: linear-gradient(135deg, #27ae60 0%, #229954 100%);
        }}

        .condition-stats {{
            display: flex;
            gap: 15px;
            font-size: 0.9em;
            color: #7f8c8d;
        }}

        .examples-section {{
            margin-top: 30px;
            display: none;
        }}

        .examples-section.active {{
            display: block;
        }}

        .examples-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }}

        .example-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 15px;
        }}

        .example-card.correct {{
            border-left: 5px solid #27ae60;
        }}

        .example-card.incorrect {{
            border-left: 5px solid #e74c3c;
        }}

        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}

        .example-status {{
            padding: 5px 15px;
            border-radius: 5px;
            font-weight: bold;
            color: white;
        }}

        .example-status.correct {{
            background: #27ae60;
        }}

        .example-status.incorrect {{
            background: #e74c3c;
        }}

        .question {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-style: italic;
        }}

        .answer-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }}

        .answer-box {{
            padding: 15px;
            border-radius: 8px;
            background: #f8f9fa;
        }}

        .answer-box .label {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        .answer-box .value {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
        }}

        .cot-tokens {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            max-height: 200px;
            overflow-y: auto;
        }}

        .cot-tokens .label {{
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }}

        .token {{
            display: inline-block;
            margin: 3px;
            padding: 5px 10px;
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}

        .token.intervened {{
            background: #fff3cd;
            border-color: #ffc107;
        }}

        .token.number {{
            background: #d1ecf1;
            border-color: #17a2b8;
        }}

        .back-button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 1.1em;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.3s;
        }}

        .back-button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}

        .filters {{
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}

        .filter-group {{
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 10px;
        }}

        .filter-label {{
            font-weight: bold;
            color: #2c3e50;
        }}

        select, input {{
            padding: 8px 15px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 1em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>CODI Intervention Comparison</h1>
        <div class="subtitle">{dataset_name} - {config['num_examples']} examples, {config['num_conditions']} conditions</div>

        <div class="summary-grid">
            <div class="summary-card">
                <h3>Dataset</h3>
                <div class="value">{dataset_name}</div>
            </div>
            <div class="summary-card">
                <h3>Examples</h3>
                <div class="value">{config['num_examples']}</div>
            </div>
            <div class="summary-card">
                <h3>Conditions</h3>
                <div class="value">{config['num_conditions']}</div>
            </div>
            <div class="summary-card">
                <h3>Best Accuracy</h3>
                <div class="value" id="best-accuracy">-</div>
            </div>
        </div>

        <div class="chart-container">
            <canvas id="accuracyChart"></canvas>
        </div>

        <!-- Overview Section -->
        <div id="overview-section">
            <div class="filters">
                <div class="filter-group">
                    <span class="filter-label">Sort by:</span>
                    <select id="sort-select">
                        <option value="default">Default Order</option>
                        <option value="accuracy-desc">Accuracy (High to Low)</option>
                        <option value="accuracy-asc">Accuracy (Low to High)</option>
                        <option value="name">Name (A-Z)</option>
                    </select>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Filter by scope:</span>
                    <select id="scope-filter">
                        <option value="all">All Scopes</option>
                        <option value="none">Baseline</option>
                        <option value="numbers">Numbers Only</option>
                        <option value="all">All Positions</option>
                    </select>
                </div>
            </div>

            <div class="conditions-grid" id="conditions-grid">
                <!-- Will be populated by JavaScript -->
            </div>
        </div>

        <!-- Examples Section -->
        <div id="examples-section" class="examples-section">
            <button class="back-button" onclick="showOverview()">← Back to Overview</button>
            <div class="examples-header" id="examples-header">
                <!-- Will be populated by JavaScript -->
            </div>
            <div id="examples-list">
                <!-- Will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const data = {data_json};

        // Initialize visualization
        let currentConditionIndex = null;

        function init() {{
            renderOverview();
            createChart();
            updateBestAccuracy();
        }}

        function updateBestAccuracy() {{
            const accuracies = data.conditions.map(c => c.accuracy);
            const best = Math.max(...accuracies);
            document.getElementById('best-accuracy').textContent = best.toFixed(1) + '%';
        }}

        function createChart() {{
            const ctx = document.getElementById('accuracyChart').getContext('2d');
            const labels = data.conditions.map(c => {{
                if (c.intervention_scope === 'none') return 'Baseline';
                return `${{c.intervention_type}} (${{c.intervention_scope}})`;
            }});
            const accuracies = data.conditions.map(c => c.accuracy);

            // Color by intervention type
            const colors = data.conditions.map(c => {{
                if (c.intervention_type === 'baseline') return '#27ae60';
                if (c.intervention_type === 'replacement') return '#3498db';
                if (c.intervention_type === 'zero' || c.intervention_type === 'average') return '#e74c3c';
                if (c.intervention_type === 'minus') return '#f39c12';
                if (c.intervention_type.includes('discretize')) return '#9b59b6';
                if (c.intervention_type.includes('proj')) return '#1abc9c';
                return '#95a5a6';
            }});

            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: 'Accuracy (%)',
                        data: accuracies,
                        backgroundColor: colors,
                        borderColor: colors.map(c => c + 'cc'),
                        borderWidth: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 3,
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            max: 100,
                            ticks: {{
                                callback: function(value) {{
                                    return value + '%';
                                }}
                            }}
                        }}
                    }},
                    plugins: {{
                        tooltip: {{
                            callbacks: {{
                                label: function(context) {{
                                    return 'Accuracy: ' + context.parsed.y.toFixed(1) + '%';
                                }}
                            }}
                        }},
                        legend: {{
                            display: false
                        }}
                    }}
                }}
            }});
        }}

        function renderOverview() {{
            const grid = document.getElementById('conditions-grid');
            grid.innerHTML = '';

            data.conditions.forEach((condition, index) => {{
                const card = createConditionCard(condition, index);
                grid.appendChild(card);
            }});
        }}

        function createConditionCard(condition, index) {{
            const div = document.createElement('div');
            div.className = 'condition-card';
            div.onclick = () => showCondition(index);

            const accuracy = condition.accuracy;
            let accuracyClass = '';
            if (accuracy >= 80) accuracyClass = 'high';
            else if (accuracy >= 50) accuracyClass = 'medium';
            else accuracyClass = 'low';

            const name = condition.intervention_scope === 'none' ?
                'Baseline' :
                `${{condition.intervention_type}} (${{condition.intervention_scope}})`;

            const correct = condition.results.filter(r => r.correct).length;
            const total = condition.results.length;

            div.innerHTML = `
                <div class="condition-header">
                    <div class="condition-name">${{name}}</div>
                    <div class="condition-accuracy ${{accuracyClass}}">${{accuracy.toFixed(1)}}%</div>
                </div>
                <div class="condition-stats">
                    <span><strong>${{correct}}</strong> / ${{total}} correct</span>
                </div>
            `;

            return div;
        }}

        function showCondition(index) {{
            currentConditionIndex = index;
            const condition = data.conditions[index];

            // Update header
            const header = document.getElementById('examples-header');
            const name = condition.intervention_scope === 'none' ?
                'Baseline' :
                `${{condition.intervention_type}} (${{condition.intervention_scope}})`;

            const correct = condition.results.filter(r => r.correct).length;
            const total = condition.results.length;

            header.innerHTML = `
                <h2>${{name}}</h2>
                <div style="font-size: 1.2em; margin-top: 10px;">
                    Accuracy: ${{condition.accuracy.toFixed(1)}}% (${{correct}}/${{total}})
                </div>
            `;

            // Render examples
            const examplesList = document.getElementById('examples-list');
            examplesList.innerHTML = '';

            condition.results.forEach((result, idx) => {{
                const exampleCard = createExampleCard(result, idx);
                examplesList.appendChild(exampleCard);
            }});

            // Hide overview, show examples
            document.getElementById('overview-section').style.display = 'none';
            document.getElementById('examples-section').classList.add('active');
        }}

        function createExampleCard(result, index) {{
            const div = document.createElement('div');
            div.className = `example-card ${{result.correct ? 'correct' : 'incorrect'}}`;

            const statusClass = result.correct ? 'correct' : 'incorrect';
            const statusText = result.correct ? '✓ Correct' : '✗ Incorrect';

            // Format decoded tokens
            let tokensHtml = '';
            if (result.decoded_tokens && result.decoded_tokens.length > 0) {{
                const tokens = result.decoded_tokens.slice(0, 10);  // Show first 10
                tokensHtml = tokens.map((token, i) => {{
                    let classes = ['token'];
                    if (result.intervened_positions && result.intervened_positions.includes(token.position)) {{
                        classes.push('intervened');
                    }}
                    if (token.is_number) {{
                        classes.push('number');
                    }}
                    return `<span class="${{classes.join(' ')}}" title="Position ${{token.position}}">${{escapeHtml(token.token)}}</span>`;
                }}).join('');
                if (result.decoded_tokens.length > 10) {{
                    tokensHtml += `<span style="padding: 5px;">... and ${{result.decoded_tokens.length - 10}} more</span>`;
                }}
            }}

            div.innerHTML = `
                <div class="example-header">
                    <div><strong>Example ${{index + 1}}</strong></div>
                    <div class="example-status ${{statusClass}}">${{statusText}}</div>
                </div>
                <div class="question">${{escapeHtml(result.question)}}</div>
                <div class="answer-comparison">
                    <div class="answer-box">
                        <div class="label">Ground Truth</div>
                        <div class="value">${{result.ground_truth}}</div>
                    </div>
                    <div class="answer-box">
                        <div class="label">Predicted</div>
                        <div class="value">${{result.predicted_answer !== null ? result.predicted_answer : 'None'}}</div>
                    </div>
                </div>
                ${{tokensHtml ? `
                <div class="cot-tokens">
                    <div class="label">CoT Tokens (first 10):</div>
                    ${{tokensHtml}}
                </div>
                ` : ''}}
            `;

            return div;
        }}

        function showOverview() {{
            document.getElementById('overview-section').style.display = 'block';
            document.getElementById('examples-section').classList.remove('active');
            currentConditionIndex = null;
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // Event listeners for filters
        document.getElementById('sort-select').addEventListener('change', function() {{
            // Implement sorting logic
            console.log('Sort by:', this.value);
        }});

        document.getElementById('scope-filter').addEventListener('change', function() {{
            // Implement filtering logic
            console.log('Filter by:', this.value);
        }});

        // Initialize on load
        window.addEventListener('load', init);
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[OK] HTML visualization saved to {output_file}")


def main():
    results_dir = Path('./intervention_comparison_results')

    # Generate for clean dataset
    clean_file = results_dir / 'full_results_clean_132_examples.json'
    if clean_file.exists():
        print("\nGenerating HTML visualization for clean dataset...")
        generate_html(clean_file, results_dir / 'visualization_clean.html', 'Clean Dataset')

    # Generate for GSM8K dataset
    gsm8k_file = results_dir / 'full_results_gsm8k_train_132_examples.json'
    if gsm8k_file.exists():
        print("\nGenerating HTML visualization for GSM8K train...")
        generate_html(gsm8k_file, results_dir / 'visualization_gsm8k_train.html', 'GSM8K Train')

    print("\n[OK] All HTML visualizations generated!")


if __name__ == "__main__":
    main()
