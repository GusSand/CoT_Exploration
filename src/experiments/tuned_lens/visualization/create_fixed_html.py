#!/usr/bin/env python3
import json

# Read the FIXED JSON
with open('layer_probe_visualization_FIXED_20251029_024134.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

print(f"Loaded {len(json_data['results'])} examples")

# Create standalone HTML with embedded JSON
html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Layer Probe Visualization - FIXED LoRA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            padding: 30px;
        }
        h1 {
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2em;
            text-align: center;
        }
        .subtitle {
            color: #718096;
            text-align: center;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .controls {
            background: #f7fafc;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }
        .control-group label {
            font-weight: 600;
            color: #4a5568;
            font-size: 0.9em;
        }
        select {
            padding: 8px 12px;
            border: 2px solid #cbd5e0;
            border-radius: 6px;
            font-size: 1em;
            background: white;
            cursor: pointer;
            transition: border-color 0.2s;
        }
        select:hover { border-color: #667eea; }
        .button-group {
            display: flex;
            gap: 10px;
            margin-left: auto;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            background: #667eea;
            color: white;
        }
        button:hover {
            background: #5568d3;
            transform: translateY(-1px);
        }
        button:active { transform: translateY(0); }
        .question-box {
            background: #edf2f7;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid #667eea;
        }
        .question-box strong { color: #2d3748; }
        .layers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .layer-card {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            padding: 15px;
            transition: all 0.2s;
        }
        .layer-card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transform: translateY(-2px);
        }
        .layer-card.best {
            border-color: #48bb78;
            background: #f0fff4;
        }
        .layer-card.worst {
            border-color: #f56565;
            background: #fff5f5;
        }
        .layer-card.broken {
            border-color: #ed8936;
            background: #fffaf0;
        }
        .final-layer {
            background: #e6fffa;
            border-color: #319795;
        }
        .layer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e2e8f0;
        }
        .layer-title {
            font-size: 1.1em;
            font-weight: 700;
            color: #2d3748;
        }
        .layer-badge {
            font-size: 0.75em;
            padding: 3px 8px;
            border-radius: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }
        .badge-best { background: #48bb78; color: white; }
        .badge-worst { background: #f56565; color: white; }
        .badge-broken { background: #ed8936; color: white; }
        .prediction-section {
            margin-bottom: 15px;
        }
        .prediction-section:last-child {
            margin-bottom: 0;
        }
        .section-title {
            font-size: 0.85em;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .token-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .token-text {
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
            background: #edf2f7;
            padding: 4px 8px;
            border-radius: 4px;
            flex: 1;
            margin-right: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .token-text.match {
            background: #c6f6d5;
            font-weight: 600;
        }
        .prob-bar {
            width: 60px;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-right: 8px;
        }
        .prob-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s;
        }
        .prob-text {
            font-size: 0.85em;
            color: #718096;
            min-width: 45px;
            text-align: right;
        }
        .match-indicator {
            margin-top: 10px;
            padding: 8px;
            border-radius: 6px;
            font-size: 0.9em;
            font-weight: 600;
            text-align: center;
        }
        .match-indicator.match {
            background: #c6f6d5;
            color: #22543d;
        }
        .match-indicator.no-match {
            background: #fed7d7;
            color: #742a2a;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Layer Probe Visualization (FIXED LoRA)</h1>
        <div class="subtitle">2-Chunk Training • Tuned Lens Analysis • All LoRA Weights Loaded</div>

        <div class="controls">
            <div class="control-group">
                <label for="exampleSelect">Example:</label>
                <select id="exampleSelect"></select>
            </div>

            <div class="control-group">
                <label for="iterationSelect">Continuous Thought Iteration:</label>
                <select id="iterationSelect"></select>
            </div>

            <div class="button-group">
                <button id="prevBtn">← Previous</button>
                <button id="nextBtn">Next →</button>
            </div>
        </div>

        <div id="questionBox" class="question-box"></div>

        <div id="layersContainer" class="layers-grid"></div>
    </div>

    <script>
        const data = ''' + json.dumps(json_data) + ''';

        let currentExample = 0;
        let currentIteration = 0;

        const layerMetadata = {
            2: { badge: 'best', acc: 74 },
            6: { badge: 'best', acc: 73 },
            7: { badge: 'best', acc: 72 },
            8: { badge: 'broken', acc: 1 },
            10: { badge: 'worst', acc: 52 }
        };

        function initializeControls() {
            const exampleSelect = document.getElementById('exampleSelect');
            const iterationSelect = document.getElementById('iterationSelect');

            data.results.forEach((result, idx) => {
                const option = document.createElement('option');
                option.value = idx;
                option.textContent = `Example ${idx + 1}`;
                exampleSelect.appendChild(option);
            });

            const numIterations = data.num_iterations;
            for (let i = 0; i < numIterations; i++) {
                const option = document.createElement('option');
                option.value = i;
                option.textContent = `Iteration ${i + 1}`;
                iterationSelect.appendChild(option);
            }

            exampleSelect.addEventListener('change', (e) => {
                currentExample = parseInt(e.target.value);
                render();
            });

            iterationSelect.addEventListener('change', (e) => {
                currentIteration = parseInt(e.target.value);
                render();
            });

            document.getElementById('prevBtn').addEventListener('click', () => {
                if (currentIteration > 0) {
                    currentIteration--;
                    document.getElementById('iterationSelect').value = currentIteration;
                    render();
                }
            });

            document.getElementById('nextBtn').addEventListener('click', () => {
                if (currentIteration < data.num_iterations - 1) {
                    currentIteration++;
                    document.getElementById('iterationSelect').value = currentIteration;
                    render();
                }
            });
        }

        function render() {
            const result = data.results[currentExample];
            const thought = result.continuous_thoughts[currentIteration];

            const questionBox = document.getElementById('questionBox');
            questionBox.innerHTML = `<strong>Question:</strong> ${result.question}`;

            const l11Layer = thought.layers[11];
            const l11Token = l11Layer.original.top1_token;

            const container = document.getElementById('layersContainer');
            container.innerHTML = '';

            thought.layers.forEach((layer, idx) => {
                const layerCard = createLayerCard(layer, l11Token, idx);
                container.appendChild(layerCard);
            });
        }

        function createLayerCard(layer, l11Token, layerIdx) {
            const card = document.createElement('div');
            card.className = 'layer-card';

            if (layerIdx === 11) {
                card.classList.add('final-layer');
            } else if (layerMetadata[layerIdx]) {
                card.classList.add(layerMetadata[layerIdx].badge);
            }

            const header = document.createElement('div');
            header.className = 'layer-header';

            const title = document.createElement('div');
            title.className = 'layer-title';
            title.textContent = `Layer ${layerIdx}`;

            if (layerIdx === 11) {
                title.textContent += ' (Final)';
            }

            header.appendChild(title);

            if (layerMetadata[layerIdx]) {
                const badge = document.createElement('span');
                badge.className = `layer-badge badge-${layerMetadata[layerIdx].badge}`;
                badge.textContent = layerMetadata[layerIdx].badge;
                header.appendChild(badge);
            }

            card.appendChild(header);

            const originalSection = createPredictionSection('Original', layer.original, l11Token);
            card.appendChild(originalSection);

            if (layer.tuned_lens && layerIdx < 11) {
                const tunedSection = createPredictionSection('Tuned Lens', layer.tuned_lens, l11Token);
                card.appendChild(tunedSection);

                const matchDiv = document.createElement('div');
                matchDiv.className = 'match-indicator';
                const matches = layer.tuned_lens.top1_token === l11Token;
                matchDiv.classList.add(matches ? 'match' : 'no-match');
                matchDiv.textContent = matches ? '✓ Matches L11' : '✗ No match';
                card.appendChild(matchDiv);
            }

            return card;
        }

        function createPredictionSection(title, predictions, l11Token) {
            const section = document.createElement('div');
            section.className = 'prediction-section';

            const sectionTitle = document.createElement('div');
            sectionTitle.className = 'section-title';
            sectionTitle.textContent = title;
            section.appendChild(sectionTitle);

            const numShow = Math.min(3, predictions.topk_tokens.length);

            for (let i = 0; i < numShow; i++) {
                const token = predictions.topk_tokens[i];
                const prob = predictions.topk_probs[i];

                const item = document.createElement('div');
                item.className = 'token-item';

                const tokenText = document.createElement('div');
                tokenText.className = 'token-text';
                if (token === l11Token && title !== 'Original' && i === 0) {
                    tokenText.classList.add('match');
                }
                tokenText.textContent = token === '' ? '(empty)' : `"${token}"`;

                const probBar = document.createElement('div');
                probBar.className = 'prob-bar';
                const probFill = document.createElement('div');
                probFill.className = 'prob-fill';
                probFill.style.width = `${prob * 100}%`;
                probBar.appendChild(probFill);

                const probText = document.createElement('div');
                probText.className = 'prob-text';
                probText.textContent = `${(prob * 100).toFixed(1)}%`;

                item.appendChild(tokenText);
                item.appendChild(probBar);
                item.appendChild(probText);

                section.appendChild(item);
            }

            return section;
        }

        initializeControls();
        render();
    </script>
</body>
</html>'''

# Write standalone HTML
with open('layer_probe_visualization_FIXED_standalone.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("✓ Standalone HTML created successfully!")
print("File: layer_probe_visualization_FIXED_standalone.html")
