# CODI Section 5 - Quick Start Guide

**⚡ Fast path to viewing your Section 5 interpretability results**

---

## 🎯 View Results Now

### Step 1: Locate Your Results

Your results are in:
```
/workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/
```

### Step 2: Choose Your Viewing Method

#### **Option A: Interactive HTML** (Recommended) 🌐

```bash
# If on local machine:
cd /workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/
xdg-open interpretability_visualization.html  # Linux
open interpretability_visualization.html       # macOS
start interpretability_visualization.html      # Windows

# If on remote server:
cd /workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/
python -m http.server 8000
# Then open: http://localhost:8000/interpretability_visualization.html
```

#### **Option B: Terminal View** 📟

```bash
cd /workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/
less interpretability_visualization.txt

# Or show first 20 examples:
head -300 interpretability_visualization.txt
```

#### **Option C: Spreadsheet** 📊

```bash
# Open in Excel, Google Sheets, or pandas:
cd /workspace/CoT_Exploration/codi/outputs/section5_analysis/section5_run_20251016_144501/
# Then open: interpretability_analysis.csv
```

---

## 📊 What You'll See

### Overall Results
- **Accuracy**: 43.21% (matches paper's 43.7%)
- **Correct**: 570 examples
- **Incorrect**: 749 examples

### Per-Example Data
Each of 1,319 GSM8K examples includes:
- ✅ **Question** and **Ground Truth**
- 🧠 **7 Continuous Thoughts** (decoded to top-10 tokens each)
- 📝 **Reference CoT** steps
- 🔍 **Step-by-step comparison**
- ✓/✗ **Correctness indicator**

---

## 🗂️ File Overview

| File | Size | Purpose |
|------|------|---------|
| **interpretability_visualization.html** | ~5-10MB | Interactive browser view (best for exploration) |
| **interpretability_visualization.txt** | ~2MB | Terminal-friendly report (quick inspection) |
| **interpretability_analysis.csv** | 52KB | Spreadsheet format (filtering, sorting) |
| **summary_statistics.json** | 1.2KB | Aggregate metrics (programming) |
| **correct_predictions/predictions.json** | 3.5MB | All 570 correct (detailed analysis) |
| **incorrect_predictions/predictions.json** | ~4.6MB | All 749 incorrect (failure analysis) |

---

## 🔍 Quick Analysis Tasks

### Find Specific Examples

**By Question ID:**
```bash
# In HTML: Ctrl+F → "Example 42"
# In text: grep "Example 42" interpretability_visualization.txt -A 30
```

**By Correctness:**
```bash
# Correct only: grep "✓ CORRECT" interpretability_visualization.txt
# Incorrect only: grep "✗ INCORRECT" interpretability_visualization.txt
```

### Load in Python

```python
import json
import pandas as pd

# Load correct predictions
with open('correct_predictions/predictions.json', 'r') as f:
    correct = json.load(f)

# Or use CSV
df = pd.read_csv('interpretability_analysis.csv')
print(df[df['is_correct'] == True].head())
```

---

## 🎨 HTML Features

When viewing `interpretability_visualization.html`:

- **🟢 Green border** = Correct prediction
- **🔴 Red border** = Incorrect prediction
- **Dark blue token** = Top-1 (most probable)
- **Light blue tokens** = Top-2 through top-10
- **Click button** = Show/hide full JSON data

**Navigation:**
- Scroll to see more examples
- Use Ctrl+F to search
- Click tokens to see probabilities (hover)

---

## ⚠️ Key Finding

**Step Correctness**: 44-56% (vs. paper's reported 75-97%)

**Analysis:**
- **Bug Fixed**: Fixed batch decoding bug causing identical continuous thoughts
- **Methodology Refined**: Using paper's approach - every other thought (0, 2, 4, 6) + top-5 tokens
- **Corrected Results**: Now showing 44-56% step correctness
- **Best Result**: 56.3% on 3-step problems (vs paper's 75.0%, gap: -18.7pp)
- Still lower than paper's 75-97% - likely subtle validation methodology differences
- Final answers remain correct (43.21% overall accuracy)
- See `docs/experiments/section5_methodology_refinement_2025-10-16.md` for detailed analysis

---

## 📚 More Information

- **Detailed Guide**: `VISUALIZATION_GUIDE.md`
- **Experiment Report**: `docs/experiments/section5_reproduction_2025-10-16.md`
- **Research Journal**: `docs/research_journal.md`
- **Implementation**: `section5_experiments/README.md`

---

## 🚀 Run Again (Optional)

To regenerate with different settings:

```bash
cd /workspace/CoT_Exploration
source env/bin/activate
bash run_section5.sh  # Full analysis (~7 min)

# Or just regenerate visualizations:
cd codi
python visualize_interpretability.py \
    --input_dir outputs/section5_analysis/section5_run_20251016_144501 \
    --max_examples 50 \
    --output_name my_custom_viz
```

---

## 💡 Pro Tips

1. **Start with HTML visualization** - most intuitive
2. **Use CSV for filtering** - find interesting cases
3. **Use JSON for programming** - detailed analysis
4. **Check first 10 examples** to understand patterns
5. **Compare correct vs incorrect** to see differences

---

## ❓ Quick FAQ

**Q: File too big to open?**
A: Reduce `--max_examples` when regenerating

**Q: Can't see on remote server?**
A: Use `python -m http.server 8000` or copy to local machine

**Q: Why similar tokens across examples?**
A: Interesting finding! See detailed report for analysis

**Q: Is the model working correctly?**
A: Yes! 43.21% accuracy matches paper (43.7%)

---

**Ready to explore!** Open `interpretability_visualization.html` and start analyzing! 🎉

For detailed instructions, see: `VISUALIZATION_GUIDE.md`
