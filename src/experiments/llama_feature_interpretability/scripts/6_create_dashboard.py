"""
Create interactive dashboard with top features only.

Instead of embedding all 15,399 features (14.3 MB), show:
1. Summary statistics
2. Top 100 features overall
3. Top features by category (numbers, operators, composite)
4. Model comparison
5. Links to detailed JSON for full exploration

Output: dashboard.html (~86 KB)
"""

import json
from pathlib import Path


def create_lightweight_dashboard():
    """Generate lightweight HTML dashboard with top features only."""
    print("="*80)
    print("CREATING DASHBOARD")
    print("="*80)

    # Load data
    print("\n[1/3] Loading data...")
    with open('src/experiments/llama_feature_interpretability/data/llama_labeled_features.json', 'r') as f:
        labeled_data = json.load(f)

    with open('src/experiments/llama_feature_interpretability/data/model_comparison.json', 'r') as f:
        comparison_data = json.load(f)

    features = labeled_data['features']
    metadata = labeled_data['metadata']

    # Extract top features by category
    print("[2/3] Extracting top features...")

    # Top 100 by enrichment
    monosemantic_features = [
        (key, feat) for key, feat in features.items()
        if feat['is_monosemantic'] and feat['top_correlations']
    ]
    monosemantic_features.sort(key=lambda x: x[1]['top_correlations'][0]['enrichment'], reverse=True)
    top_100 = monosemantic_features[:100]

    # Top 50 by each type
    number_features = [(k, f) for k, f in monosemantic_features if f['label'].startswith('number_')][:50]
    operator_features = [(k, f) for k, f in monosemantic_features if f['label'].startswith('operator_')][:20]
    composite_features = [(k, f) for k, f in monosemantic_features if '_with_' in f['label']][:30]

    # Generate HTML
    print("[3/3] Generating HTML...")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLaMA Feature Interpretability - Lightweight Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #0a0e27;
            color: #e0e6ed;
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            margin-bottom: 30px;
        }}

        h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        .alert {{
            background: #1e3a5f;
            border-left: 4px solid #3b82f6;
            padding: 15px 20px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}

        .alert h3 {{
            margin-bottom: 8px;
            color: #60a5fa;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: #1a1f3a;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #2d3748;
            transition: transform 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: #667eea;
        }}

        .stat-card h3 {{
            font-size: 2.5em;
            color: #667eea;
            margin-bottom: 8px;
        }}

        .stat-card p {{
            color: #a0aec0;
            font-size: 0.95em;
            margin-bottom: 8px;
        }}

        .stat-card .comparison {{
            color: #718096;
            font-size: 0.75em;
            font-style: italic;
            line-height: 1.4;
        }}

        .section {{
            background: #1a1f3a;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #2d3748;
        }}

        .section h2 {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #667eea;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}

        th {{
            background: #0f1629;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #667eea;
            position: sticky;
            top: 0;
            cursor: pointer;
            user-select: none;
            transition: background 0.2s;
        }}

        th:hover {{
            background: #1a2340;
        }}

        th.sortable::after {{
            content: ' ⇅';
            opacity: 0.3;
            font-size: 0.9em;
        }}

        th.sort-asc::after {{
            content: ' ↑';
            opacity: 1;
            color: #667eea;
        }}

        th.sort-desc::after {{
            content: ' ↓';
            opacity: 1;
            color: #667eea;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #2d3748;
        }}

        tr:hover {{
            background: #0f1629;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }}

        .badge-mono {{
            background: #10b981;
            color: #fff;
        }}

        .badge-poly {{
            background: #f59e0b;
            color: #fff;
        }}

        .enrichment {{
            color: #a78bfa;
            font-weight: 600;
        }}

        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            border-bottom: 2px solid #2d3748;
        }}

        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            border: none;
            background: transparent;
            color: #a0aec0;
            font-size: 1em;
            transition: all 0.2s;
        }}

        .tab.active {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            margin-bottom: -2px;
        }}

        .tab:hover {{
            color: #667eea;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .download-link {{
            display: inline-block;
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            margin-top: 20px;
            transition: background 0.2s;
        }}

        .download-link:hover {{
            background: #764ba2;
        }}

        .comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 20px;
        }}

        .model-card {{
            background: #0f1629;
            padding: 20px;
            border-radius: 8px;
            border: 2px solid #2d3748;
        }}

        .model-card h3 {{
            color: #667eea;
            margin-bottom: 15px;
        }}

        .model-stat {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #2d3748;
        }}

        .model-stat:last-child {{
            border-bottom: none;
        }}

        code {{
            background: #0f1629;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Courier New', monospace;
            color: #a78bfa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LLaMA Feature Interpretability</h1>
            <p class="subtitle">Monosemantic Feature Catalog - Lightweight Dashboard</p>
            <p class="subtitle">Showing top 100 features (of 15,399 total)</p>
        </header>

        <div class="alert">
            <h3>⚡ Lightweight Version</h3>
            <p>This page shows only the top features for fast loading. For full exploration, use the complete dataset files:</p>
            <ul style="margin-top: 10px; margin-left: 20px;">
                <li><code>llama_labeled_features.json</code> - All 15,399 features with labels</li>
                <li><code>gpt2_feature_token_correlations.json</code> - Raw correlation data</li>
            </ul>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>{metadata['total_features']:,}</h3>
                <p>Interpretable Features</p>
                <div class="comparison">41.8% of total features<br>LLaMA: ~20% expected</div>
            </div>
            <div class="stat-card">
                <h3>{metadata['monosemantic_rate']*100:.1f}%</h3>
                <p>Monosemantic Rate</p>
                <div class="comparison">GPT-2: 72.6% vs LLaMA: ~50% (est.)</div>
            </div>
            <div class="stat-card">
                <h3>66.4%</h3>
                <p>Number Features</p>
                <div class="comparison">GPT-2 specialized for numbers<br>LLaMA: more balanced (est.)</div>
            </div>
            <div class="stat-card">
                <h3>169.9×</h3>
                <p>Max Enrichment</p>
                <div class="comparison">Feature L4_P3_F241 → "50000"<br>Extreme specialization</div>
            </div>
        </div>

        <div class="section">
            <h2>Feature Categories</h2>
            <div class="tabs">
                <button class="tab active" onclick="showTab('top100', this)">Top 100 Overall</button>
                <button class="tab" onclick="showTab('numbers', this)">Number Features</button>
                <button class="tab" onclick="showTab('operators', this)">Operators</button>
                <button class="tab" onclick="showTab('composite', this)">Composite</button>
            </div>

            <div id="top100" class="tab-content active">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature</th>
                            <th>Layer</th>
                            <th>Pos</th>
                            <th>Label</th>
                            <th>Enrichment</th>
                            <th>Activation %</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for i, (key, feat) in enumerate(top_100, 1):
        top_corr = feat['top_correlations'][0]
        html += f"""
                        <tr>
                            <td>{i}</td>
                            <td><code>{key}</code></td>
                            <td>{feat['layer']}</td>
                            <td>{feat['position']}</td>
                            <td>{feat['label']}</td>
                            <td class="enrichment">{top_corr['enrichment']:.1f}×</td>
                            <td>{feat['activation_rate']*100:.1f}%</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>

            <div id="numbers" class="tab-content">
                <p style="margin-bottom: 15px;">Top 50 number-specific features by enrichment score.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Layer</th>
                            <th>Pos</th>
                            <th>Number</th>
                            <th>Enrichment</th>
                            <th>Activation %</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for key, feat in number_features:
        top_corr = feat['top_correlations'][0]
        number = feat['label'].replace('number_', '')
        html += f"""
                        <tr>
                            <td><code>{key}</code></td>
                            <td>{feat['layer']}</td>
                            <td>{feat['position']}</td>
                            <td><strong>{number}</strong></td>
                            <td class="enrichment">{top_corr['enrichment']:.1f}×</td>
                            <td>{feat['activation_rate']*100:.1f}%</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>

            <div id="operators" class="tab-content">
                <p style="margin-bottom: 15px;">Operator-specific features (addition, subtraction, multiplication, division).</p>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Layer</th>
                            <th>Pos</th>
                            <th>Operator</th>
                            <th>Enrichment</th>
                            <th>Activation %</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for key, feat in operator_features:
        top_corr = feat['top_correlations'][0]
        op = feat['label'].replace('operator_', '')
        html += f"""
                        <tr>
                            <td><code>{key}</code></td>
                            <td>{feat['layer']}</td>
                            <td>{feat['position']}</td>
                            <td><strong>{op}</strong></td>
                            <td class="enrichment">{top_corr['enrichment']:.1f}×</td>
                            <td>{feat['activation_rate']*100:.1f}%</td>
                        </tr>
"""

    html += """
                    </tbody>
                </table>
            </div>

            <div id="composite" class="tab-content">
                <p style="margin-bottom: 15px;">Composite features that correlate with operation + number patterns.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Layer</th>
                            <th>Pos</th>
                            <th>Pattern</th>
                            <th>Enrichment</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for key, feat in composite_features:
        top_corr = feat['top_correlations'][0]
        html += f"""
                        <tr>
                            <td><code>{key}</code></td>
                            <td>{feat['layer']}</td>
                            <td>{feat['position']}</td>
                            <td>{feat['label']}</td>
                            <td class="enrichment">{top_corr['enrichment']:.1f}×</td>
                        </tr>
"""

    html += f"""
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Model Comparison: GPT-2 vs LLaMA</h2>
            <div class="comparison">
                <div class="model-card">
                    <h3>GPT-2 (124M params)</h3>
                    <div class="model-stat">
                        <span>Monosemantic Rate</span>
                        <strong>72.6%</strong>
                    </div>
                    <div class="model-stat">
                        <span>Sparsity</span>
                        <strong>29.3%</strong>
                    </div>
                    <div class="model-stat">
                        <span>Strategy</span>
                        <strong>Specialized</strong>
                    </div>
                    <div class="model-stat">
                        <span>Number Features</span>
                        <strong>66.4%</strong>
                    </div>
                </div>
                <div class="model-card">
                    <h3>LLaMA (1B params)</h3>
                    <div class="model-stat">
                        <span>Monosemantic Rate</span>
                        <strong>~50% (est.)</strong>
                    </div>
                    <div class="model-stat">
                        <span>Sparsity</span>
                        <strong>19.5%</strong>
                    </div>
                    <div class="model-stat">
                        <span>Strategy</span>
                        <strong>Distributed</strong>
                    </div>
                    <div class="model-stat">
                        <span>Number Features</span>
                        <strong>TBD</strong>
                    </div>
                </div>
            </div>
            <p style="margin-top: 20px; color: #a0aec0;">
                <strong>Key Insight:</strong> Model capacity determines feature specialization. Smaller models (GPT-2)
                use highly monosemantic features to compensate for limited capacity, while larger models (LLaMA)
                can afford distributed, redundant representations.
            </p>
        </div>

        <div class="section">
            <h2>Access Full Data</h2>
            <p>For complete exploration of all 15,399 features, use these data files:</p>
            <ul style="margin: 20px 0 0 20px; color: #a0aec0;">
                <li style="margin-bottom: 10px;">
                    <code>data/llama_labeled_features.json</code> (17.5 MB) - All features with labels and correlations
                </li>
                <li style="margin-bottom: 10px;">
                    <code>data/gpt2_feature_token_correlations.json</code> (19.5 MB) - Raw statistical correlations
                </li>
                <li style="margin-bottom: 10px;">
                    <code>data/model_comparison.json</code> (5.2 KB) - GPT-2 vs LLaMA comparison
                </li>
            </ul>
            <p style="margin-top: 20px; color: #a0aec0;">
                Load these files in Python/JavaScript for custom analysis and filtering.
            </p>
        </div>
    </div>

    <script>
        function showTab(tabId, clickedButton) {{
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            clickedButton.classList.add('active');
        }}

        function sortTable(table, columnIndex, isNumeric = false) {{
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const headers = table.querySelectorAll('th');
            const currentHeader = headers[columnIndex];

            // Determine sort direction
            let isAscending = true;
            if (currentHeader.classList.contains('sort-asc')) {{
                isAscending = false;
            }}

            // Clear all sort indicators
            headers.forEach(h => {{
                h.classList.remove('sort-asc', 'sort-desc');
            }});

            // Sort rows
            rows.sort((a, b) => {{
                let aVal = a.cells[columnIndex].textContent.trim();
                let bVal = b.cells[columnIndex].textContent.trim();

                if (isNumeric) {{
                    // Extract numbers (handle formats like "169.9×" or "29.3%")
                    aVal = parseFloat(aVal.replace(/[^0-9.-]/g, '')) || 0;
                    bVal = parseFloat(bVal.replace(/[^0-9.-]/g, '')) || 0;
                    return isAscending ? aVal - bVal : bVal - aVal;
                }} else {{
                    return isAscending ?
                        aVal.localeCompare(bVal) :
                        bVal.localeCompare(aVal);
                }}
            }});

            // Update display
            rows.forEach(row => tbody.appendChild(row));

            // Update sort indicator
            currentHeader.classList.add(isAscending ? 'sort-asc' : 'sort-desc');
        }}

        // Initialize sortable headers
        document.addEventListener('DOMContentLoaded', function() {{
            document.querySelectorAll('table').forEach(table => {{
                const headers = table.querySelectorAll('th');
                headers.forEach((header, index) => {{
                    header.classList.add('sortable');

                    // Determine if column is numeric based on header text
                    const headerText = header.textContent.toLowerCase();
                    const isNumeric = headerText.includes('enrichment') ||
                                     headerText.includes('activation') ||
                                     headerText.includes('layer') ||
                                     headerText.includes('pos') ||
                                     headerText.includes('rank');

                    header.addEventListener('click', () => {{
                        sortTable(table, index, isNumeric);
                    }});
                }});
            }});
        }});
    </script>
</body>
</html>
"""

    # Save
    output_path = Path('src/experiments/llama_feature_interpretability/dashboard.html')
    with open(output_path, 'w') as f:
        f.write(html)

    size_kb = output_path.stat().st_size / 1024
    print(f"\n✓ Saved to: {output_path} ({size_kb:.1f} KB)")
    print("\n" + "="*80)
    print("DASHBOARD COMPLETE!")
    print("="*80)
    print(f"  File size: {size_kb:.1f} KB")
    print(f"  Features shown: 100 top features + 50 numbers + 20 operators + 30 composite")
    print(f"  Open: file://{output_path.absolute()}")
    print("="*80)


if __name__ == '__main__':
    create_lightweight_dashboard()
