"""
Generate interactive HTML dashboard for TopK SAE feature visualization.

Creates a Neuronpedia-style interface with:
- Feature selector
- Max activating samples
- Token correlation heatmap
- Layer selectivity heatmap
"""

import json
from pathlib import Path


def generate_html_dashboard(feature_data_path, output_path):
    """Generate standalone HTML dashboard."""

    # Load feature data
    with open(feature_data_path, 'r') as f:
        data = json.load(f)

    feature_ids = data['feature_ids']
    features = data['features']

    # Feature names mapping
    feature_names = {
        317: "The Number '12'",
        154: "The Number '50'",
        269: "The Number '2'",
        460: "The Number '10'",
        486: "The Number '45'",
        283: "Identity Statements",
        422: "Multi-step Chains",
        505: "Large Numbers",
        39: "The Number '23'",
        52: "Feature 52",
        120: "Feature 120",
        138: "Feature 138",
        346: "Feature 346",
        349: "Feature 349",
        414: "Feature 414",
        447: "Feature 447",
        457: "Feature 457",
        509: "Feature 509",
        81: "Feature 81",
        247: "Large Round Numbers"
    }

    def get_feature_label(fid):
        """Get descriptive label for feature."""
        name = feature_names.get(fid, f"Feature {fid}")
        return f"F{fid}: {name}"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TopK SAE Feature Explorer - Layer 14, Position 3</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            font-size: 32px;
        }}
        .subtitle {{
            opacity: 0.9;
            font-size: 16px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .controls {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        select {{
            padding: 10px 15px;
            font-size: 16px;
            border: 2px solid #667eea;
            border-radius: 5px;
            min-width: 200px;
            cursor: pointer;
        }}
        select:hover {{
            border-color: #764ba2;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .panel {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .panel h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .sample {{
            background: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        .activation-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
        }}
        .cot-step {{
            font-family: 'Courier New', monospace;
            background: #e8eaf6;
            padding: 5px 8px;
            margin: 3px 2px;
            display: inline-block;
            border-radius: 3px;
            font-size: 14px;
        }}
        .token-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .token-badge {{
            background: #e1f5e1;
            border: 1px solid #4caf50;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 13px;
        }}
        .token-badge .enrichment {{
            font-weight: bold;
            color: #2e7d32;
        }}
        .stats {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin: 15px 0;
        }}
        .stats-item {{
            display: inline-block;
            margin-right: 20px;
        }}
        .stats-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .stats-value {{
            font-size: 20px;
            font-weight: bold;
            color: #333;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 TopK SAE Feature Explorer</h1>
            <div class="subtitle">Layer 14, Position 3 • K=100, d=512 • Interactive Feature Visualization</div>
        </div>

        <div class="controls">
            <label for="featureSelect"><strong>Select Feature:</strong></label>
            <select id="featureSelect" onchange="updateFeature()">
                {' '.join(f'<option value="{fid}">{get_feature_label(fid)}</option>' for fid in feature_ids)}
            </select>
        </div>

        <div id="featureStats" class="panel stats"></div>

        <div class="grid">
            <div class="panel">
                <h2>📊 Token Correlations</h2>
                <div id="tokenCorrelations"></div>
            </div>

            <div class="panel">
                <h2>🎯 Layer Selectivity</h2>
                <div id="layerSelectivity"></div>
            </div>
        </div>

        <div class="panel full-width">
            <h2>💡 Top Activating Samples</h2>
            <div id="maxSamples"></div>
        </div>
    </div>

    <script>
        // Load feature data
        const featureData = {json.dumps(features)};
        const featureIds = {json.dumps(feature_ids)};

        // Feature names
        const featureNames = {json.dumps(feature_names)};

        function updateFeature() {{
            const featureId = document.getElementById('featureSelect').value;
            const data = featureData[featureId];

            // Update stats
            updateStats(featureId, data);

            // Update token correlations
            updateTokenCorrelations(featureId, data);

            // Update layer selectivity
            updateLayerSelectivity(featureId, data);

            // Update max samples
            updateMaxSamples(featureId, data);
        }}

        function updateStats(featureId, data) {{
            const stats = data.correlations;
            const featureName = featureNames[featureId] || `Feature ${{featureId}}`;
            const html = `
                <div class="stats-item">
                    <div class="stats-label">Feature</div>
                    <div class="stats-value">F${{featureId}}: ${{featureName}}</div>
                </div>
                <div class="stats-item">
                    <div class="stats-label">Activation Frequency</div>
                    <div class="stats-value">${{(stats.activation_frequency * 100).toFixed(1)}}%</div>
                </div>
                <div class="stats-item">
                    <div class="stats-label">Active Samples</div>
                    <div class="stats-value">${{stats.num_active_samples}}</div>
                </div>
                <div class="stats-item">
                    <div class="stats-label">Significant Tokens</div>
                    <div class="stats-value">${{stats.top_tokens.length}}</div>
                </div>
            `;
            document.getElementById('featureStats').innerHTML = html;
        }}

        function updateTokenCorrelations(featureId, data) {{
            const tokens = data.correlations.top_tokens.slice(0, 15);

            if (tokens.length === 0) {{
                document.getElementById('tokenCorrelations').innerHTML = '<p>No significant token correlations found.</p>';
                return;
            }}

            const trace = {{
                x: tokens.map(t => (t.enrichment * 100).toFixed(1)),
                y: tokens.map(t => t.token),
                orientation: 'h',
                type: 'bar',
                marker: {{
                    color: tokens.map(t => t.enrichment),
                    colorscale: [
                        [0, '#fff5f5'],
                        [0.5, '#90caf9'],
                        [1, '#1976d2']
                    ],
                    showscale: false
                }},
                text: tokens.map(t => `p=${{t.p_value.toExponential(2)}}`),
                textposition: 'outside',
                hovertemplate:
                    '<b>%{{y}}</b><br>' +
                    'Enrichment: %{{x}}%<br>' +
                    'Active: %{{customdata[0]}}<br>' +
                    'Inactive: %{{customdata[1]}}<br>' +
                    'p-value: %{{customdata[2]:.2e}}<extra></extra>',
                customdata: tokens.map(t => [t.active_count, t.inactive_count, t.p_value])
            }};

            const layout = {{
                xaxis: {{ title: 'Enrichment (%)' }},
                yaxis: {{ autorange: 'reversed' }},
                margin: {{ l: 60, r: 20, t: 20, b: 40 }},
                height: 400
            }};

            Plotly.newPlot('tokenCorrelations', [trace], layout, {{responsive: true}});
        }}

        function updateLayerSelectivity(featureId, data) {{
            const selectivity = data.layer_selectivity;

            if (!selectivity || !selectivity.layer_means) {{
                document.getElementById('layerSelectivity').innerHTML = '<p>Layer selectivity data not available.</p>';
                return;
            }}

            const layers = Object.keys(selectivity.layer_means).map(Number).sort((a,b) => a - b);
            const means = layers.map(l => selectivity.layer_means[l]);

            const trace = {{
                x: layers,
                y: means,
                type: 'bar',
                marker: {{
                    color: means,
                    colorscale: 'Viridis',
                    showscale: false
                }},
                hovertemplate:
                    'Layer %{{x}}<br>' +
                    'Mean Activation: %{{y:.4f}}<extra></extra>'
            }};

            const mostSelective = selectivity.most_selective_layer;
            const selectivityIndex = selectivity.selectivity_index;

            const layout = {{
                xaxis: {{ title: 'Layer', dtick: 1 }},
                yaxis: {{ title: 'Mean Activation' }},
                margin: {{ l: 60, r: 20, t: 40, b: 40 }},
                height: 300,
                annotations: [{{
                    text: `Most Selective: Layer ${{mostSelective}} (SI=${{selectivityIndex.toFixed(3)}})`,
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.5,
                    y: 1,
                    xanchor: 'center',
                    yanchor: 'bottom',
                    showarrow: false,
                    font: {{ size: 12, color: '#666' }}
                }}]
            }};

            Plotly.newPlot('layerSelectivity', [trace], layout, {{responsive: true}});
        }}

        function updateMaxSamples(featureId, data) {{
            const samples = data.max_activations.slice(0, 10);

            if (samples.length === 0) {{
                document.getElementById('maxSamples').innerHTML = '<p>No activating samples found.</p>';
                return;
            }}

            let html = '';
            samples.forEach((sample, idx) => {{
                const cotSteps = sample.cot_sequence.map(step =>
                    `<span class="cot-step">${{step}}</span>`
                ).join(' ');

                html += `
                    <div class="sample">
                        <div><strong>Sample ${{idx + 1}}</strong>
                            <span class="activation-badge">Activation: ${{sample.activation.toFixed(3)}}</span>
                        </div>
                        <div style="margin-top: 10px;">
                            <strong>CoT Sequence:</strong><br>
                            ${{cotSteps}}
                        </div>
                    </div>
                `;
            }});

            document.getElementById('maxSamples').innerHTML = html;
        }}

        // Initialize with first feature
        updateFeature();
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Generated HTML dashboard: {output_path}")


def main():
    """Generate HTML dashboard."""
    feature_data_path = Path('src/experiments/topk_grid_pilot/visualizations/feature_data.json')
    output_path = Path('src/experiments/topk_grid_pilot/visualizations/feature_dashboard.html')

    generate_html_dashboard(feature_data_path, output_path)

    print("\n" + "="*80)
    print("✓ Interactive HTML dashboard generated!")
    print(f"✓ Open in browser: file://{output_path.absolute()}")
    print("="*80)


if __name__ == '__main__':
    main()
