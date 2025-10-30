# Extended CoT Experiment: Tracking Answer Emergence

**Script**: `test_extended_cot_llama.py`
**Purpose**: Test whether the correct answer appears in decoded tokens during extended chain-of-thought reasoning

---

## Experiment Design

### Hypothesis
In CODI-LLaMA's continuous chain-of-thought, the correct numerical answer may appear as a decoded token at intermediate positions during extended reasoning, beyond the standard 6 iterations.

### Method
1. Run CoT for **50 iterations** (instead of standard 6)
2. Decode the last layer token at each position
3. Track when/if the ground truth answer appears in decoded tokens
4. Record all appearance positions

### Key Differences from Original Script
- **NO intervention**: Pure vanilla CoT (removed all projection replacement code)
- **Extended iterations**: 50 CoT tokens instead of 6
- **Answer tracking**: New function `find_answer_in_cot()` searches for ground truth
- **Small test set**: 5 examples for initial testing

---

## Script Components

### 1. Extended CoT Generation
```python
def run_extended_cot(model, tokenizer, training_args, question, max_iterations=50):
    # Runs BoT + max_iterations CoT tokens (default 50)
    # Decodes token at each position
    # Returns: decoded_tokens list with position, token, is_number info
```

### 2. Answer Detection
```python
def find_answer_in_cot(decoded_tokens, ground_truth_answer):
    # Searches decoded tokens for ground truth answer
    # Handles both integer and float matching
    # Returns: List of positions where answer appears
```

### 3. Result Tracking
For each example, records:
- `decoded_tokens`: All decoded tokens (positions 0-50)
- `answer_appears_in_cot`: Boolean flag
- `answer_positions`: List of all positions where answer found
- `first_appearance`: Position of first occurrence (or None)
- `num_appearances`: Total count of appearances
- `correct`: Whether final predicted answer is correct

---

## Output Format

### Per-Example Results
```json
{
  "question": "...",
  "ground_truth": 18,
  "max_cot_iterations": 50,
  "decoded_tokens": [
    {"position": 0, "token": "16", "is_number": true, "token_id": 123},
    {"position": 1, "token": "3", "is_number": true, "token_id": 456},
    ...
    {"position": 18, "token": "18", "is_number": true, "token_id": 789}
  ],
  "answer_text": "18",
  "predicted_answer": 18.0,
  "correct": true,
  "answer_appears_in_cot": true,
  "answer_positions": [18, 24, 35],
  "first_appearance": 18,
  "num_appearances": 3
}
```

### Summary Statistics
- Total examples tested
- Accuracy (correct final answers)
- **Answer appearance rate**: % of examples where answer appeared in CoT
- **Position statistics**: Mean, median, min, max of first appearances
- Distribution by example

---

## Research Questions

### Primary Questions
1. **Does the answer appear?** In what % of examples does the ground truth answer appear as a decoded token during extended CoT?

2. **When does it appear?** At what position(s) does it typically first appear?
   - Early (positions 0-10)?
   - Middle (positions 10-30)?
   - Late (positions 30-50)?

3. **Is appearance correlated with correctness?** Do correct predictions show answer in CoT more often than incorrect predictions?

### Secondary Questions
4. How many times does the answer appear (once vs. multiple times)?
5. Is there a pattern to appearance positions?
6. Does appearance position vary by problem difficulty?

---

## Usage

### Initial Test (5 examples)
```bash
python test_extended_cot_llama.py
```

This will:
- Load CODI-LLaMA model
- Test on first 5 clean variant examples
- Run 50 CoT iterations per example
- Save results to `./extended_cot_results/extended_cot_test_5_examples.json`
- Print detailed per-example analysis

### Expected Runtime
- ~5-10 seconds per example (50 iterations + answer generation)
- Total: ~30-60 seconds for 5 examples

### Output
Terminal will show:
- Per-example decoded tokens (first 20 positions)
- Answer appearance status
- First appearance position
- Summary statistics

---

## Configuration

### Adjustable Parameters
```python
MAX_COT_ITERATIONS = 50  # Number of CoT iterations (line 312)
num_test_examples = 5     # Number of examples to test (line 298)
```

### For Full-Scale Testing
To test on more examples:
1. Change `num_test_examples = 5` to desired number (e.g., 50, 132)
2. Adjust `MAX_COT_ITERATIONS` if needed (test different lengths)

---

## Expected Output Example

```
================================================================================
Example 1/5 (pair_id=0)
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast...
Ground truth answer: 18.0
================================================================================
Running extended CoT for 50 iterations...

Predicted answer: 18.0
Correct: True
Answer appears in CoT: True
First appearance at position: 14
Total appearances: 3
All positions: [14, 28, 42]

Decoded CoT tokens (positions 0-50):
  Pos  0: '16' [NUM]
  Pos  1: '3' [NUM]
  Pos  2: '4' [NUM]
  ...
  Pos 14: '18' [NUM] <-- ANSWER
  ...
  Pos 28: '18' [NUM] <-- ANSWER
  ...
```

---

## Files Generated

1. **Results JSON**: `extended_cot_results/extended_cot_test_5_examples.json`
   - Complete results for all examples
   - Configuration parameters
   - Summary statistics

2. **This Documentation**: `EXTENDED_COT_EXPERIMENT.md`

---

## Next Steps After Initial Test

Once the 5-example test confirms the script works:

1. **Scale up**: Test on 50-100 examples
2. **Vary iteration length**: Try 20, 50, 100 iterations
3. **Compare datasets**: Test on both clean and corrupted variants
4. **Analyze patterns**: Look for systematic relationships between:
   - Problem difficulty and appearance position
   - Reasoning steps and answer emergence
   - Correct vs incorrect predictions

---

## Technical Notes

### Model Configuration
- Base: meta-llama/Llama-3.2-1B
- LoRA checkpoint: `/workspace/CoT_Exploration/models/CODI-llama3.2-1b`
- Standard 6-iteration CoT is overridden to 50 iterations
- No intervention applied (pure vanilla reasoning)

### Data Source
- Path: `/workspace/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_all.json`
- Filter: `variant == "clean"` (132 examples available)
- Initial test: First 5 examples

### Number Detection
Uses regex `r'^\s?\d+'` to identify number tokens, same as intervention experiments.

---

## Status

✅ **Script Ready**
- Tested locally for syntax
- Based on proven `scale_intervention_llama_clean.py` template
- Intervention code removed, extended CoT logic added
- Answer tracking implemented

🔜 **Awaiting**: Server SSH credentials to run on GPU

---

**Created**: 2025-10-29
**Author**: Claude Code
