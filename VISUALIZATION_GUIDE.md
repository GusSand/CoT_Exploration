# How to Use the Section 5 Interactive Visualization

This guide explains how to access and use the interactive HTML visualization generated from the CODI Section 5 interpretability analysis.

---

## Quick Start

### Option 1: View Locally (Recommended)

```bash
# From your local machine, navigate to the results directory
cd /workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/

# Open the HTML file in your default browser
# On Linux:
xdg-open interpretability_visualization.html

# On macOS:
open interpretability_visualization.html

# On Windows:
start interpretability_visualization.html

# Or manually: Right-click the file and select "Open with" → your browser
```

### Option 2: Python HTTP Server

If you're working on a remote server:

```bash
# Navigate to the output directory
cd /workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/

# Start a simple HTTP server
python -m http.server 8000

# Then access in your browser at:
# http://localhost:8000/interpretability_visualization.html
# (or replace localhost with your server IP if remote)
```

### Option 3: Copy to Local Machine

If you're working on a remote server and want to view locally:

```bash
# From your LOCAL machine (not the server):
scp user@server:/workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/interpretability_visualization.html ~/Downloads/

# Then open the downloaded file in your browser
```

---

## Understanding the Visualization

### Header Section

At the top, you'll see:
- **Title**: "CODI Section 5: Interpretability Analysis"
- **Summary Statistics Cards**:
  - Overall Accuracy (43.21%)
  - Correct Predictions (570)
  - Incorrect Predictions (749)
  - Average Step Accuracy for correct predictions

### Example Cards

Each example is displayed in a card with:

#### 1. **Status Indicator**
- **Green left border**: ✓ Correct prediction
- **Red left border**: ✗ Incorrect prediction
- Header shows: "Example [ID] - [Status]"

#### 2. **Question Section** (Blue background)
- The original math word problem
- Easy to read format

#### 3. **Reference CoT Section** (Yellow/Gold background)
- The ground truth chain-of-thought from the dataset
- Format: `<<step1>> <<step2>> ...`
- Below: "Extracted Steps" showing parsed intermediate steps

#### 4. **Continuous Thoughts Interpretation**
- **One section per continuous thought** (7 total: initial + 6 iterations)
- Each thought shows:
  - **Thought number**: e.g., "Thought 0 (initial)" or "Thought 1 (continuous_thought)"
  - **Top-K decoded tokens**: Shows up to 10 tokens the model "thought"
  - **Token highlighting**: Top-1 token has darker blue background and bold text
  - **Probability scores**: Hover over tokens to see exact probabilities

#### 5. **Step-by-Step Comparison Table**
For problems with reference steps, shows:
- **Column 1**: Step number
- **Column 2**: Reference CoT step (from ground truth)
- **Column 3**: Decoded step (from top-1 continuous thought)
- **Column 4**: Match indicator (✓ Yes / ✗ No)
- **Row colors**:
  - Green background: Step matches
  - Red background: Step doesn't match

#### 6. **Model Prediction Section**
- **Green background**: Correct prediction
- **Red background**: Incorrect prediction
- Shows:
  - Model's predicted answer
  - Ground truth answer
  - Full decoded text from model

#### 7. **Show/Hide Full JSON Button**
- Click to expand/collapse the complete JSON data for that example
- Useful for debugging or detailed analysis

---

## Features and Interactivity

### 1. **Scrolling Through Examples**
- Scroll down to see more examples
- First 100 examples are displayed by default (configurable)

### 2. **Color-Coded Information**
- **Blue boxes**: Questions (input)
- **Yellow boxes**: Reference CoT (ground truth)
- **Gray boxes**: Continuous thoughts (model internals)
- **Green boxes**: Correct predictions
- **Red boxes**: Incorrect predictions
- **Purple headers**: Continuous thought sections

### 3. **Token Analysis**
Each decoded token shows:
- The actual token text (may include spaces like `' 9'` or `'9'`)
- Lighter blue: Lower-ranked tokens (less probable)
- Darker blue + bold: Top-1 token (most probable)
- Hover to see probability score

### 4. **Quick Navigation**
- Use browser's Find function (Ctrl+F / Cmd+F) to search for:
  - Specific question IDs: "Example 42"
  - Specific patterns: "Correct" or "Incorrect"
  - Specific tokens: "' 9'"

### 5. **JSON Inspection**
Click "Show/Hide Full JSON" to see:
- All continuous thoughts with full top-10 rankings
- Exact probability scores
- Step correctness booleans
- Complete metadata

---

## Interpreting the Visualizations

### Understanding Continuous Thoughts

**What you're seeing**: When the model "thinks" in continuous space, we decode those thoughts back to vocabulary to see what they might represent.

**Example**:
```
Thought 0 (initial): [' 13', '13', ' 12']
Thought 1: ['-', ' than', ' instead']
Thought 2: [' 9', '9', ' 8']
```

This means:
- After seeing the question, the model's first continuous thought most strongly corresponds to the token `' 13'`
- Second thought corresponds to `'-'` (an operator)
- Third thought corresponds to `' 9'` (an intermediate result)

### Common Patterns You'll See

1. **Numbers**: `' 9'`, `'9'`, `' 13'`, `'12'`
   - May represent intermediate computation results
   - Note: Space prefix matters (` 9` vs `9` are different tokens)

2. **Operators/Connectives**: `'-'`, `'+'`, `' than'`, `' instead'`
   - May represent operations or logical connections
   - Sometimes appear between numerical thoughts

3. **Repeated Patterns**: Many examples show similar tokens
   - This is an interesting finding!
   - Suggests thoughts may be position-dependent or semantic markers
   - Different from expecting literal intermediate results

### Step Correctness

**Green rows** = Decoded thought matches reference CoT step
**Red rows** = No match

**Important**: Low match rates (2-7%) don't mean the model is wrong - final answers are still correct (43.21% accuracy). This suggests:
- Continuous thoughts encode reasoning **semantically** (meaning-based)
- Not **literally** (exact token matches)
- Different from how explicit CoT works

---

## Advanced Usage

### Filtering Examples

**In Browser Console** (F12 → Console):

```javascript
// Hide all correct predictions
document.querySelectorAll('.correct').forEach(el => el.style.display = 'none');

// Hide all incorrect predictions
document.querySelectorAll('.incorrect').forEach(el => el.style.display = 'none');

// Show all again
document.querySelectorAll('.example').forEach(el => el.style.display = 'block');

// Find examples with specific thought tokens
Array.from(document.querySelectorAll('.example')).filter(el =>
  el.textContent.includes("' 9'")
);
```

### Extracting Data

**Copy JSON for Analysis**:
1. Click "Show/Hide Full JSON" on any example
2. Right-click the JSON text
3. Select "Copy"
4. Paste into your analysis tool

**Or load the source files directly**:
```python
import json

# Load all correct predictions
with open('correct_predictions/predictions.json', 'r') as f:
    correct = json.load(f)

# Analyze specific aspects
for pred in correct[:10]:
    print(f"Q{pred['question_id']}: Top thoughts = {[t['topk_decoded'][0] for t in pred['continuous_thoughts']]}")
```

---

## Alternative Viewing Options

### 1. **Text Report** (Terminal)

For command-line viewing:
```bash
less interpretability_visualization.txt

# Or with color:
cat interpretability_visualization.txt | less -R

# Search within:
grep -A 20 "Example 0" interpretability_visualization.txt
```

### 2. **CSV Analysis** (Spreadsheet)

Open `interpretability_analysis.csv` in:
- **Excel**: File → Open → Select CSV
- **Google Sheets**: File → Import → Upload file
- **pandas**:
  ```python
  import pandas as pd
  df = pd.read_csv('interpretability_analysis.csv')

  # Filter correct predictions
  correct_df = df[df['is_correct'] == True]

  # Analyze by step count
  df.groupby('num_reference_steps')['step_accuracy'].mean()
  ```

### 3. **JSON Analysis** (Programmatic)

```python
import json

with open('correct_predictions/predictions.json', 'r') as f:
    data = json.load(f)

# Example: Find problems where model decoded ' 9' in first thought
interesting_cases = [
    p for p in data
    if ' 9' in p['continuous_thoughts'][1]['topk_decoded'][0]
]

print(f"Found {len(interesting_cases)} examples with ' 9' in first thought")
```

---

## Troubleshooting

### Visualization Doesn't Load

**Problem**: HTML file opens but shows blank page
**Solution**:
- Check browser console (F12) for errors
- Try a different browser (Chrome, Firefox recommended)
- The file is self-contained (no external dependencies), so should work offline

### File Too Large

**Problem**: Browser struggles with large file
**Solution**:
- Our default shows 100 examples (manageable)
- To show fewer:
  ```bash
  python visualize_interpretability.py --input_dir [DIR] --max_examples 25
  ```

### Can't Access on Remote Server

**Problem**: Working on remote server without X forwarding
**Solution**:
- Use Python HTTP server (see Option 2 above)
- Or copy file to local machine (see Option 3)
- Or use text report: `less interpretability_visualization.txt`

### Tokens Look Strange

**Problem**: Tokens like `' 9'` with leading spaces
**Explanation**:
- This is correct! Tokenizers distinguish between:
  - `' 9'` = space followed by 9 (beginning of word)
  - `'9'` = just the digit 9 (middle/end of word)
- Both are valid and mean slightly different things

---

## Customizing the Visualization

### Generate with Different Settings

```bash
cd /workspace/CoT_Exploration/codi

python visualize_interpretability.py \
    --input_dir outputs/section5_analysis/section5_run_20251016_144501 \
    --max_examples 25 \
    --output_name custom_viz
```

Parameters:
- `--max_examples`: Number of examples to visualize (default: 50)
- `--output_name`: Output file prefix (default: interpretability_visualization)

### Modify Styles

The HTML uses inline CSS. To customize:
1. Open `visualize_interpretability.py`
2. Find the `<style>` section in `generate_html_visualization()`
3. Modify colors, fonts, layouts
4. Regenerate: `python visualize_interpretability.py ...`

---

## Tips for Analysis

### 1. **Start with Correct Predictions**
- Easier to understand what "good" continuous thoughts look like
- Filter: Use browser Find (Ctrl+F) → "✓ Correct"

### 2. **Compare Similar Problems**
- Find problems with same number of steps
- See if decoded tokens show patterns

### 3. **Focus on Top-1 Tokens**
- These are the most probable interpretations
- Highlighted with darker background

### 4. **Look for Semantic Patterns**
- Don't expect exact numerical matches
- Look for conceptual patterns:
  - Numbers appearing in relevant thoughts
  - Operators between numerical thoughts
  - Position-dependent patterns

### 5. **Cross-Reference with CSV**
- CSV gives quick overview
- HTML gives detailed inspection
- Use CSV to find interesting cases, then inspect in HTML

---

## Example Workflow

### Finding Interesting Examples

1. **Open CSV in spreadsheet**:
   - Sort by `step_accuracy` descending
   - Find examples with high step accuracy

2. **Note the `question_id`** values

3. **Open HTML visualization**:
   - Use Ctrl+F to search "Example [ID]"
   - Inspect the continuous thoughts

4. **Analyze patterns**:
   - What tokens appear?
   - Do they relate to the problem?
   - Are there positional patterns?

5. **Export for paper/presentation**:
   - Screenshot interesting examples
   - Or copy JSON for detailed analysis

---

## Getting Help

### Where to Find More Information

- **Implementation Details**: `section5_experiments/README.md`
- **Experimental Findings**: `docs/experiments/10-16_gpt2_gsm8k_section5_reproduction.md`
- **Research Journal**: `docs/research_journal.md`
- **Original Paper**: `docs/codi.pdf` (Section 5, page 8-9)

### Questions About Results

**Why are step accuracies so low (2-7%)?**
- See detailed analysis in `docs/experiments/10-16_gpt2_gsm8k_section5_reproduction.md`
- TL;DR: Likely measuring something different than the paper
- Final answers are still correct (43.21% accuracy)
- Suggests semantic vs. literal interpretation

**Why do all examples show similar tokens?**
- This is an interesting finding requiring further investigation
- May indicate position-based encoding
- Or semantic markers rather than literal results
- Worth exploring with different decoding strategies

---

## Summary

**To view**: Just open `interpretability_visualization.html` in any web browser
**Best for**: Interactive exploration, finding interesting patterns
**Alternatives**: CSV (spreadsheet), text report (terminal), JSON (programming)

The visualization makes it easy to:
- ✅ See which predictions were correct/incorrect
- ✅ Inspect continuous thought patterns
- ✅ Compare decoded thoughts to reference CoT
- ✅ Export data for further analysis

Happy analyzing! 🔍
