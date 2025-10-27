"""
Create interactive HTML dashboard for GPT-2 feature interpretability.

Features:
- Browse all 15,399 interpretable features
- Filter by layer, position, feature type, monosemanticity
- Sort by enrichment, activation rate, correlation count
- View detailed correlation breakdown
- Model comparison summary
- Top monosemantic features showcase

Output: dashboard.html (standalone, no dependencies)
"""

import json
from pathlib import Path


def load_data():
    """Load all required data."""
    with open('src/experiments/gpt2_feature_interpretability/data/gpt2_labeled_features.json', 'r') as f:
        labeled_data = json.load(f)

    with open('src/experiments/gpt2_feature_interpretability/data/model_comparison.json', 'r') as f:
        comparison_data = json.load(f)

    return labeled_data, comparison_data


def generate_html(labeled_data, comparison_data):
    """Generate interactive HTML dashboard."""

    features = labeled_data['features']
    metadata = labeled_data['metadata']
    gpt2_analysis = comparison_data['gpt2_analysis']

    # Prepare feature list for table
    feature_list = []
    for key, feat in features.items():
        top_corr = feat['top_correlations'][0] if feat['top_correlations'] else {}

        feature_list.append({
            'key': key,
            'layer': feat['layer'],
            'position': feat['position'],
            'feature_id': feat['feature_id'],
            'label': feat['label'],
            'is_monosemantic': feat['is_monosemantic'],
            'explanation': feat['explanation'],
            'activation_rate': feat['activation_rate'],
            'num_correlations': feat['num_correlations'],
            'top_token': top_corr.get('token', 'N/A'),
            'top_enrichment': top_corr.get('enrichment', 0),
            'correlations_json': json.dumps(feat['top_correlations'])
        })

    # Sort by enrichment (descending)
    feature_list.sort(key=lambda x: x['top_enrichment'], reverse=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT-2 Feature Interpretability Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}

        .stat-card h3 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 10px;
        }}

        .stat-card p {{
            color: #666;
            font-size: 0.9em;
        }}

        .filters {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .filter-row {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
        }}

        .filter-group label {{
            font-weight: 600;
            font-size: 0.9em;
            color: #555;
        }}

        .filter-group select,
        .filter-group input {{
            padding: 8px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
            font-size: 0.9em;
        }}

        .filter-group select:focus,
        .filter-group input:focus {{
            outline: none;
            border-color: #667eea;
        }}

        button {{
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }}

        button:hover {{
            background: #5568d3;
        }}

        .table-container {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        thead {{
            background: #667eea;
            color: white;
        }}

        thead th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }}

        thead th:hover {{
            background: #5568d3;
        }}

        tbody tr {{
            border-bottom: 1px solid #f0f0f0;
            transition: background 0.2s;
        }}

        tbody tr:hover {{
            background: #f8f9ff;
        }}

        tbody td {{
            padding: 15px;
        }}

        .mono-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: 600;
        }}

        .mono-yes {{
            background: #10b981;
            color: white;
        }}

        .mono-no {{
            background: #ef4444;
            color: white;
        }}

        .enrichment {{
            font-weight: 700;
            color: #667eea;
        }}

        .view-btn {{
            padding: 6px 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.85em;
        }}

        .view-btn:hover {{
            background: #5568d3;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
        }}

        .modal-content {{
            background: white;
            margin: 5% auto;
            padding: 30px;
            border-radius: 10px;
            width: 80%;
            max-width: 700px;
            max-height: 80vh;
            overflow-y: auto;
        }}

        .close {{
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            color: #999;
        }}

        .close:hover {{
            color: #333;
        }}

        .corr-list {{
            margin-top: 20px;
        }}

        .corr-item {{
            padding: 15px;
            background: #f8f9ff;
            border-left: 4px solid #667eea;
            margin-bottom: 10px;
            border-radius: 4px;
        }}

        .corr-item strong {{
            color: #667eea;
            font-size: 1.1em;
        }}

        .comparison-section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-top: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .comparison-section h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}

        .insight-box {{
            background: #f8f9ff;
            padding: 20px;
            border-left: 4px solid #667eea;
            margin: 15px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>GPT-2 Feature Interpretability Dashboard</h1>
            <p>Exploring {metadata['total_features']:,} interpretable features from 72 TopK SAEs</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>41.8%</h3>
                <p>Interpretability Rate</p>
                <small>15,399 / 36,864 features</small>
            </div>
            <div class="stat-card">
                <h3>72.6%</h3>
                <p>Monosemantic Rate</p>
                <small>11,187 / 15,399 features</small>
            </div>
            <div class="stat-card">
                <h3>49,748</h3>
                <p>Total Correlations</p>
                <small>Avg 3.2 per feature</small>
            </div>
            <div class="stat-card">
                <h3>66.4%</h3>
                <p>Number Features</p>
                <small>Dominant feature type</small>
            </div>
        </div>

        <div class="filters">
            <div class="filter-row">
                <div class="filter-group">
                    <label>Layer</label>
                    <select id="layerFilter">
                        <option value="">All Layers</option>
                        {chr(10).join(f'<option value="{i}">Layer {i}</option>' for i in range(12))}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Position</label>
                    <select id="positionFilter">
                        <option value="">All Positions</option>
                        {chr(10).join(f'<option value="{i}">Position {i}</option>' for i in range(6))}
                    </select>
                </div>
                <div class="filter-group">
                    <label>Type</label>
                    <select id="typeFilter">
                        <option value="">All Types</option>
                        <option value="number">Number</option>
                        <option value="operator">Operator</option>
                        <option value="polysemantic">Polysemantic</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Monosemantic</label>
                    <select id="monoFilter">
                        <option value="">All</option>
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search</label>
                    <input type="text" id="searchInput" placeholder="Search labels...">
                </div>
                <button onclick="applyFilters()">Apply Filters</button>
                <button onclick="resetFilters()">Reset</button>
            </div>
        </div>

        <div class="table-container">
            <table id="featuresTable">
                <thead>
                    <tr>
                        <th onclick="sortTable('key')">Feature ID</th>
                        <th onclick="sortTable('layer')">Layer</th>
                        <th onclick="sortTable('position')">Position</th>
                        <th onclick="sortTable('label')">Label</th>
                        <th onclick="sortTable('mono')">Mono</th>
                        <th onclick="sortTable('enrichment')">Top Enrichment</th>
                        <th onclick="sortTable('token')">Top Token</th>
                        <th onclick="sortTable('correlations')">Correlations</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody id="featuresBody">
                </tbody>
            </table>
        </div>

        <div class="comparison-section">
            <h2>Model Comparison: GPT-2 vs LLaMA</h2>
            <div class="insight-box">
                <h3>Key Finding: Model Capacity vs Interpretability</h3>
                <p><strong>GPT-2 (124M):</strong> 72.6% monosemantic rate, 29.3% sparsity</p>
                <p><strong>LLaMA (1B):</strong> ~50% estimated monosemantic rate, 19.5% sparsity</p>
                <p style="margin-top: 10px;"><em>Smaller models require more specialized, monosemantic features to compensate for limited capacity.</em></p>
            </div>
        </div>
    </div>

    <!-- Modal for feature details -->
    <div id="detailsModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalBody"></div>
        </div>
    </div>

    <script>
        // Feature data
        const features = {json.dumps(feature_list)};
        let filteredFeatures = [...features];

        // Render table
        function renderTable() {{
            const tbody = document.getElementById('featuresBody');
            tbody.innerHTML = '';

            filteredFeatures.forEach(feat => {{
                const row = `
                    <tr>
                        <td>${{feat.key}}</td>
                        <td>${{feat.layer}}</td>
                        <td>${{feat.position}}</td>
                        <td>${{feat.label}}</td>
                        <td><span class="mono-badge ${{feat.is_monosemantic ? 'mono-yes' : 'mono-no'}}">${{feat.is_monosemantic ? 'YES' : 'NO'}}</span></td>
                        <td class="enrichment">${{feat.top_enrichment.toFixed(1)}}x</td>
                        <td>${{feat.top_token}}</td>
                        <td>${{feat.num_correlations}}</td>
                        <td><button class="view-btn" onclick="showDetails('${{feat.key}}')">View</button></td>
                    </tr>
                `;
                tbody.innerHTML += row;
            }});
        }}

        // Apply filters
        function applyFilters() {{
            const layer = document.getElementById('layerFilter').value;
            const position = document.getElementById('positionFilter').value;
            const type = document.getElementById('typeFilter').value;
            const mono = document.getElementById('monoFilter').value;
            const search = document.getElementById('searchInput').value.toLowerCase();

            filteredFeatures = features.filter(feat => {{
                if (layer && feat.layer != layer) return false;
                if (position && feat.position != position) return false;
                if (type && !feat.label.startsWith(type)) return false;
                if (mono && feat.is_monosemantic.toString() !== mono) return false;
                if (search && !feat.label.toLowerCase().includes(search)) return false;
                return true;
            }});

            renderTable();
        }}

        // Reset filters
        function resetFilters() {{
            document.getElementById('layerFilter').value = '';
            document.getElementById('positionFilter').value = '';
            document.getElementById('typeFilter').value = '';
            document.getElementById('monoFilter').value = '';
            document.getElementById('searchInput').value = '';
            filteredFeatures = [...features];
            renderTable();
        }}

        // Sort table
        let sortDirection = {{}};
        function sortTable(column) {{
            sortDirection[column] = !sortDirection[column];
            const dir = sortDirection[column] ? 1 : -1;

            filteredFeatures.sort((a, b) => {{
                let aVal = a[column] || a['top_enrichment'];
                let bVal = b[column] || b['top_enrichment'];

                if (column === 'mono') {{
                    aVal = a.is_monosemantic;
                    bVal = b.is_monosemantic;
                }}

                if (typeof aVal === 'string') {{
                    return dir * aVal.localeCompare(bVal);
                }}
                return dir * (aVal - bVal);
            }});

            renderTable();
        }}

        // Show details modal
        function showDetails(key) {{
            const feat = features.find(f => f.key === key);
            const correlations = JSON.parse(feat.correlations_json);

            let html = `
                <h2>${{feat.key}}: ${{feat.label}}</h2>
                <p style="margin: 15px 0;"><strong>Explanation:</strong> ${{feat.explanation}}</p>
                <p><strong>Activation Rate:</strong> ${{(feat.activation_rate * 100).toFixed(1)}}%</p>
                <p><strong>Monosemantic:</strong> ${{feat.is_monosemantic ? 'Yes' : 'No'}}</p>
                <div class="corr-list">
                    <h3>Top Correlations:</h3>
            `;

            correlations.forEach((corr, idx) => {{
                html += `
                    <div class="corr-item">
                        <strong>#${{idx + 1}}: ${{corr.token}}</strong><br>
                        Enrichment: <span class="enrichment">${{corr.enrichment.toFixed(1)}}x</span> |
                        p-value: ${{corr.p_value.toExponential(2)}}<br>
                        Active with token: ${{corr.active_with_token}} / ${{corr.active_with_token + corr.active_without_token}}
                    </div>
                `;
            }});

            html += '</div>';
            document.getElementById('modalBody').innerHTML = html;
            document.getElementById('detailsModal').style.display = 'block';
        }}

        function closeModal() {{
            document.getElementById('detailsModal').style.display = 'none';
        }}

        // Close modal on outside click
        window.onclick = function(event) {{
            const modal = document.getElementById('detailsModal');
            if (event.target == modal) {{
                modal.style.display = 'none';
            }}
        }}

        // Initial render
        renderTable();
    </script>
</body>
</html>"""

    return html


def create_dashboard():
    """Generate dashboard HTML file."""
    print("="*80)
    print("CREATING INTERACTIVE DASHBOARD")
    print("="*80)

    print("\n[1/2] Loading data...")
    labeled_data, comparison_data = load_data()

    print("[2/2] Generating HTML...")
    html = generate_html(labeled_data, comparison_data)

    output_path = Path('src/experiments/gpt2_feature_interpretability/dashboard.html')
    with open(output_path, 'w') as f:
        f.write(html)

    size_kb = output_path.stat().st_size / 1024
    print(f"✓ Saved to: {output_path} ({size_kb:.1f} KB)")

    print("\n" + "="*80)
    print("DASHBOARD COMPLETE!")
    print("="*80)
    print(f"  Open in browser: file://{output_path.absolute()}")
    print(f"  Features: 15,399 browsable entries")
    print(f"  Filters: Layer, Position, Type, Monosemanticity, Search")
    print("="*80)


if __name__ == '__main__':
    create_dashboard()
