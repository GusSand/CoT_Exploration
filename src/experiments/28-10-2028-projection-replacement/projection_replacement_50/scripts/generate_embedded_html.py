import json
from pathlib import Path

# Read the JSON data
json_file = Path("C:/Users/Paper001/Documents/claude/results_correct/llama_intervention_results_50_correct.json")
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Create HTML with embedded data
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CODI-LLaMA: Number Embedding Intervention Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }

        h1 {
            text-align: center;
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .subtitle {
            text-align: center;
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 40px;
        }

        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s;
        }

        .summary-card:hover {
            transform: translateY(-5px);
        }

        .summary-card h3 {
            font-size: 0.9em;
            margin-bottom: 10px;
            opacity: 0.9;
        }

        .summary-card .value {
            font-size: 2.5em;
            font-weight: bold;
        }

        .summary-card .subvalue {
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 5px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }

        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }

        .chart-container {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .chart-container h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.1em;
        }

        .example-list {
            margin-top: 20px;
        }

        .example-card {
            background: #f8f9fa;
            padding: 25px;
            margin-bottom: 25px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
            transition: all 0.3s;
        }

        .example-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .example-card.changed {
            border-left-color: #e74c3c;
            background: #fff5f5;
        }

        .example-card h4 {
            color: #2c3e50;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }

        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }

        .badge.correct {
            background: #2ecc71;
            color: white;
        }

        .badge.incorrect {
            background: #e74c3c;
            color: white;
        }

        .badge.changed {
            background: #f39c12;
            color: white;
        }

        .badge.unchanged {
            background: #95a5a6;
            color: white;
        }

        .question {
            color: #555;
            font-style: italic;
            margin-bottom: 15px;
            line-height: 1.6;
        }

        .answer-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 15px;
        }

        .answer-box {
            padding: 15px;
            border-radius: 8px;
            background: white;
            border: 2px solid #ddd;
        }

        .answer-box h5 {
            color: #7f8c8d;
            font-size: 0.85em;
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .answer-box .answer-text {
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 5px;
            font-size: 0.9em;
        }

        .answer-box .predicted {
            color: #667eea;
            font-size: 1.3em;
            font-weight: bold;
        }

        /* Token Decoding Comparison */
        .token-comparison {
            margin-top: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #667eea;
        }

        .token-comparison h5 {
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1em;
        }

        .token-grid {
            display: grid;
            grid-template-columns: auto 1fr 1fr;
            gap: 10px 15px;
            align-items: center;
            font-size: 0.9em;
        }

        .token-grid-header {
            font-weight: bold;
            color: #2c3e50;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
        }

        .token-position {
            font-weight: bold;
            color: #667eea;
            padding: 8px 12px;
            background: #f0f3ff;
            border-radius: 5px;
            text-align: center;
        }

        .token-value {
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-weight: 600;
        }

        .token-value.number {
            background: #fff3cd;
            color: #856404;
            border: 2px solid #ffc107;
        }

        .token-value.changed {
            background: #ffebee;
            color: #c62828;
            border: 2px solid #e74c3c;
            animation: highlight 0.5s ease;
        }

        @keyframes highlight {
            0% { background: #ff5252; }
            100% { background: #ffebee; }
        }

        .token-value.intervened::after {
            content: " ⚡";
            color: #e74c3c;
        }

        .key-finding {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }

        .key-finding h3 {
            font-size: 1.5em;
            margin-bottom: 15px;
        }

        .key-finding p {
            line-height: 1.8;
            font-size: 1.1em;
        }

        .filter-buttons {
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .filter-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 25px;
            background: #667eea;
            color: white;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s;
        }

        .filter-btn:hover {
            background: #764ba2;
            transform: scale(1.05);
        }

        .filter-btn.active {
            background: #2c3e50;
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .stats-table th,
        .stats-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }

        .stats-table th {
            background: #667eea;
            color: white;
            font-weight: 600;
        }

        .stats-table tr:hover {
            background: #f8f9fa;
        }

        .legend {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            font-size: 0.85em;
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }

        .legend-box {
            width: 20px;
            height: 20px;
            border-radius: 3px;
        }

        @media (max-width: 768px) {
            .answer-comparison {
                grid-template-columns: 1fr;
            }

            .charts-grid {
                grid-template-columns: 1fr;
            }

            .token-grid {
                grid-template-columns: 1fr;
                gap: 5px;
            }

            h1 {
                font-size: 1.8em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 CODI-LLaMA: Number Embedding Intervention Analysis</h1>
        <p class="subtitle">Exploring how number embeddings in continuous chain-of-thought affect final answers on GSM8K</p>

        <!-- Key Finding -->
        <div class="key-finding">
            <h3>🎯 Key Finding</h3>
            <p>
                Numbers in continuous Chain-of-Thought <strong>CAN</strong> affect final answers, but only in a <strong>minority of cases (16%)</strong>.
                The intervention slightly improves accuracy by <strong>+2.0%</strong>, suggesting that while numbers play some role,
                the model is relatively robust to these targeted interventions.
            </p>
        </div>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Total Examples</h3>
                <div class="value">50</div>
                <div class="subvalue">GSM8K Test Set</div>
            </div>
            <div class="summary-card">
                <h3>Accuracy (No Intervention)</h3>
                <div class="value">52.0%</div>
                <div class="subvalue">26 correct</div>
            </div>
            <div class="summary-card">
                <h3>Accuracy (With Intervention)</h3>
                <div class="value">54.0%</div>
                <div class="subvalue">27 correct (+2%)</div>
            </div>
            <div class="summary-card">
                <h3>Answers Changed</h3>
                <div class="value">16%</div>
                <div class="subvalue">8 out of 50</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="section">
            <h2>📊 Visualizations</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <h3>Accuracy Comparison</h3>
                    <canvas id="accuracyChart"></canvas>
                </div>
                <div class="chart-container">
                    <h3>Answer Changes Breakdown</h3>
                    <canvas id="changesChart"></canvas>
                </div>
                <div class="chart-container" style="grid-column: 1 / -1;">
                    <h3>Number Positions in Chain-of-Thought</h3>
                    <canvas id="positionsChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Detailed Breakdown Table -->
        <div class="section">
            <h2>📈 Detailed Breakdown</h2>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Correct → Wrong</strong></td>
                        <td>1</td>
                        <td>2%</td>
                    </tr>
                    <tr>
                        <td><strong>Wrong → Correct</strong></td>
                        <td>2</td>
                        <td>4%</td>
                    </tr>
                    <tr>
                        <td><strong>Wrong → Wrong (different)</strong></td>
                        <td>5</td>
                        <td>10%</td>
                    </tr>
                    <tr>
                        <td><strong>Unchanged</strong></td>
                        <td>42</td>
                        <td>84%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Examples -->
        <div class="section">
            <h2>📝 Example Cases with Token Decoding</h2>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-box" style="background: #fff3cd; border: 2px solid #ffc107;"></div>
                    <span>Number token</span>
                </div>
                <div class="legend-item">
                    <div class="legend-box" style="background: #ffebee; border: 2px solid #e74c3c;"></div>
                    <span>Token changed by intervention</span>
                </div>
                <div class="legend-item">
                    <span>⚡ = Intervention applied at this position</span>
                </div>
            </div>
            <div class="filter-buttons">
                <button class="filter-btn active" onclick="filterExamples('all')">All Examples (50)</button>
                <button class="filter-btn" onclick="filterExamples('changed')">Answer Changed (8)</button>
                <button class="filter-btn" onclick="filterExamples('improved')">Wrong → Correct (2)</button>
                <button class="filter-btn" onclick="filterExamples('worsened')">Correct → Wrong (1)</button>
            </div>
            <div id="examplesList" class="example-list"></div>
        </div>
    </div>

    <script>
        // Embedded data
        const resultsData = """ + json.dumps(data) + """;

        // Initialize on page load
        window.addEventListener('DOMContentLoaded', function() {
            initializeCharts();
            displayExamples('all');
        });

        // Initialize charts
        function initializeCharts() {
            const summary = resultsData.summary;

            // Accuracy Chart
            new Chart(document.getElementById('accuracyChart'), {
                type: 'bar',
                data: {
                    labels: ['No Intervention', 'With Intervention'],
                    datasets: [{
                        label: 'Accuracy (%)',
                        data: [summary.accuracy_no_intervention, summary.accuracy_with_intervention],
                        backgroundColor: ['#3498db', '#e74c3c'],
                        borderColor: ['#2980b9', '#c0392b'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ticks: {
                                callback: function(value) {
                                    return value + '%';
                                }
                            }
                        }
                    }
                }
            });

            // Changes Pie Chart
            new Chart(document.getElementById('changesChart'), {
                type: 'doughnut',
                data: {
                    labels: ['Unchanged', 'Changed'],
                    datasets: [{
                        data: [summary.total - summary.answers_changed, summary.answers_changed],
                        backgroundColor: ['#2ecc71', '#f39c12'],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });

            // Position Chart
            const positionCounts = {};
            resultsData.results.forEach(r => {
                r.number_positions.forEach(pos => {
                    positionCounts[pos] = (positionCounts[pos] || 0) + 1;
                });
            });

            const positions = Object.keys(positionCounts).sort((a, b) => a - b);
            const posLabels = positions.map(p => p == 0 ? 'BoT' : `T${p}`);
            const posCounts = positions.map(p => positionCounts[p]);

            new Chart(document.getElementById('positionsChart'), {
                type: 'bar',
                data: {
                    labels: posLabels,
                    datasets: [{
                        label: 'Number of Examples',
                        data: posCounts,
                        backgroundColor: '#9b59b6',
                        borderColor: '#8e44ad',
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                stepSize: 5
                            }
                        }
                    }
                }
            });
        }

        // Filter and display examples
        function filterExamples(filter) {
            // Update button states
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');

            displayExamples(filter);
        }

        function displayExamples(filter) {
            let examples = resultsData.results;

            // Apply filter
            if (filter === 'changed') {
                examples = examples.filter(r =>
                    r.predicted_with_intervention !== null &&
                    r.predicted_no_intervention !== r.predicted_with_intervention
                );
            } else if (filter === 'improved') {
                examples = examples.filter(r =>
                    !r.correct_no_intervention && r.correct_with_intervention
                );
            } else if (filter === 'worsened') {
                examples = examples.filter(r =>
                    r.correct_no_intervention && !r.correct_with_intervention
                );
            }

            const container = document.getElementById('examplesList');

            if (examples.length === 0) {
                container.innerHTML = '<p style="text-align: center; color: #7f8c8d;">No examples match this filter.</p>';
                return;
            }

            container.innerHTML = examples.map((ex, idx) => {
                const changed = ex.predicted_with_intervention !== null &&
                               ex.predicted_no_intervention !== ex.predicted_with_intervention;

                // Build token comparison grid
                let tokenComparisonHTML = '';
                if (ex.decoded_no_intervention && ex.decoded_with_intervention) {
                    const maxPos = Math.max(ex.decoded_no_intervention.length, ex.decoded_with_intervention.length);

                    tokenComparisonHTML = `
                        <div class="token-comparison">
                            <h5>🔍 Decoded Tokens at Each CoT Position</h5>
                            <div class="token-grid">
                                <div class="token-grid-header">Position</div>
                                <div class="token-grid-header">Without Intervention</div>
                                <div class="token-grid-header">With Intervention</div>
                    `;

                    for (let i = 0; i < maxPos; i++) {
                        const noInt = ex.decoded_no_intervention[i];
                        const withInt = ex.decoded_with_intervention[i];

                        const posLabel = i === 0 ? 'BoT' : `T${i}`;
                        const tokenChanged = noInt && withInt && noInt.token !== withInt.token;
                        const intervened = withInt && withInt.intervened;

                        tokenComparisonHTML += `
                            <div class="token-position">${posLabel}</div>
                            <div class="token-value ${noInt && noInt.is_number ? 'number' : ''}">
                                ${noInt ? noInt.token : 'N/A'}
                            </div>
                            <div class="token-value ${withInt && withInt.is_number ? 'number' : ''} ${tokenChanged ? 'changed' : ''} ${intervened ? 'intervened' : ''}">
                                ${withInt ? withInt.token : 'N/A'}
                            </div>
                        `;
                    }

                    tokenComparisonHTML += `
                            </div>
                        </div>
                    `;
                }

                return `
                    <div class="example-card ${changed ? 'changed' : ''}">
                        <h4>
                            <span>Example ${idx + 1}</span>
                            <div>
                                ${changed ? '<span class="badge changed">Answer Changed</span>' : '<span class="badge unchanged">Unchanged</span>'}
                            </div>
                        </h4>
                        <div class="question">${ex.question}</div>
                        <p><strong>Ground Truth:</strong> ${ex.ground_truth}</p>

                        ${tokenComparisonHTML}

                        <div class="answer-comparison">
                            <div class="answer-box">
                                <h5>Without Intervention</h5>
                                <div class="answer-text">${ex.answer_no_intervention || 'N/A'}</div>
                                <div class="predicted">Predicted: ${ex.predicted_no_intervention || 'N/A'}</div>
                                <span class="badge ${ex.correct_no_intervention ? 'correct' : 'incorrect'}">
                                    ${ex.correct_no_intervention ? '✓ Correct' : '✗ Incorrect'}
                                </span>
                            </div>

                            <div class="answer-box">
                                <h5>With Intervention</h5>
                                <div class="answer-text">${ex.answer_with_intervention || 'N/A'}</div>
                                <div class="predicted">Predicted: ${ex.predicted_with_intervention || 'N/A'}</div>
                                <span class="badge ${ex.correct_with_intervention ? 'correct' : 'incorrect'}">
                                    ${ex.correct_with_intervention ? '✓ Correct' : '✗ Incorrect'}
                                </span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    </script>
</body>
</html>
"""

# Write the HTML file
output_file = Path("C:/Users/Paper001/Documents/claude/results_visualization_embedded.html")
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Generated embedded HTML: {output_file}")
print(f"Total examples: {len(data['results'])}")
print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
