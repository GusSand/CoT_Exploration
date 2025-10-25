# Corruption Strategy Analysis - Why We Got So Few Target Cases

**Date**: October 18, 2025
**Issue**: Out of 45 problem pairs, only 9 (20%) were valid intervention targets
**Root Cause**: Corruption strategy was too weak

---

## TL;DR

**The Question**: Why did 22/45 (48.9%) clean baselines fail?

**The Answer**: They DIDN'T - clean accuracy (51%) is actually BETTER than expected (43%)!

**The REAL Problem**: Of 23 cases where clean was correct, the model STILL solved 14 (60.9%) correctly even with corruption. Our corruption strategy is too weak!

---

## The Numbers

### Expected vs Observed

| Metric | Expected | Observed | Status |
|--------|----------|----------|--------|
| **Clean Accuracy** | ~43% | 51.1% (23/45) | ✓ Better than expected |
| **Corruption Effectiveness** | Should break model | Only broke 9/23 (39%) | ❌ Too weak |

### Outcome Breakdown

Out of 45 total pairs:

| Category | Count | % | Interpretation |
|----------|-------|---|----------------|
| Both correct (Clean ✓, Corrupted ✓) | 14 | 31.1% | **Corruption failed!** |
| TARGET (Clean ✓, Corrupted ✗) | 9 | 20.0% | **Valid targets** ✓ |
| Invalid (Clean ✗, Corrupted ✗) | 20 | 44.4% | Clean baseline failed |
| Reversed (Clean ✗, Corrupted ✓) | 2 | 4.4% | Rare anomaly |

**Key Finding**: Of the 23 solvable problems (clean ✓), corruption only broke 9 (39.1%). This means **60.9% of corruptions were ineffective**.

---

## Current Corruption Strategy

### Implementation (generate_pairs.py:48-53)

```python
# Find all numbers in question
numbers = re.findall(r'\d+', question)

# Change first number (usually an operand)
original_num = numbers[0]
corrupted_num = str(int(original_num) + 1)  # Just add 1!

# Replace only the first occurrence
corrupted_question = question.replace(original_num, corrupted_num, 1)
```

**Strategy**: Add +1 to the first number in the problem

**Problem**: This is TOO WEAK and doesn't substantially change the problem

---

## Examples of Failed Corruptions

### Example 1: Pair 55
```
CLEAN:
Q: Jean has 30 lollipops. Jean eats 2...
A: 14 ✓

CORRUPTED: (30 → 32)
Q: Jean has 32 lollipops. Jean eats 2...
A: 15 ✓ (model still got it right!)
```

**Why it failed**: Adding 2 lollipops doesn't change problem structure. Same operations, slightly different numbers.

### Example 2: Pair 59
```
CLEAN:
Q: A raspberry bush has 6 clusters of 20 fruit each and 67 individual fruit...
A: 187 ✓

CORRUPTED: (6 → 7)
Q: A raspberry bush has 7 clusters of 20 fruit each and 67 individual fruit...
A: 207 ✓ (model still got it right!)
```

**Why it failed**: Model just calculated `7*20 + 67` instead of `6*20 + 67`. Same reasoning process.

### Example 3: Pair 3
```
CLEAN:
Q: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint...
A: 540 ✓

CORRUPTED: (first "3" → 4)
Q: James decides to run 4 sprints 3 times a week. He runs 60 meters each sprint...
A: 720 ✓ (model still got it right!)
```

**Why it failed**: Changing 3→4 doesn't break the multiplication structure. Model just computed `4*3*60` instead of `3*3*60`.

---

## Why Weak Corruption Hurts Us

### Impact on Sample Size

```
45 total pairs
├─ 23 valid (clean ✓)
│  ├─ 14 both correct → ❌ WASTED (corruption failed)
│  └─ 9 targets → ✓ Usable for intervention
└─ 22 invalid (clean ✗)
```

**Net result**: Only **9/45 (20%)** of our effort produced usable target cases!

**If corruption was effective** (80% break rate):
```
45 total pairs
├─ 23 valid (clean ✓)
│  ├─ 4 both correct → Minor waste
│  └─ 19 targets → ✓ Usable
└─ 22 invalid (clean ✗)
```

**We could have 19 targets instead of 9!** (2.1x more data)

### Impact on Statistical Power

| Scenario | n targets | Required n | Have enough? |
|----------|-----------|------------|--------------|
| Current (weak corruption) | 9 | 634 | ❌ 70x short |
| Better corruption (80% break) | 19 | 634 | ❌ 33x short |
| Best corruption (95% break) | 22 | 634 | ❌ 29x short |

**Lesson**: Even with better corruption, we still need way more pairs. But it would help!

---

## Better Corruption Strategies

### Strategy 1: Change Operation-Critical Numbers

Instead of:
```python
# Change first number
original_num = numbers[0]
corrupted_num = str(int(original_num) + 1)
```

Do:
```python
# Change the number that's most critical to the operation
# For "A has X apples, gives away Y, how many left?"
# → Change Y (what's given away), not X (what they start with)

# Use heuristics:
# - If "eats", "gives", "spends" → change that number
# - If multiplication, change one operand significantly
# - If division, change divisor (more disruptive)
```

**Example**:
```
Original: "Janet has 16 eggs, eats 3..."
Weak corruption: "Janet has 17 eggs, eats 3..." (model still solves)
Better corruption: "Janet has 16 eggs, eats 10..." (more disruptive!)
```

### Strategy 2: Larger Magnitude Changes

Instead of +1:
```python
# Add +5 to +10 for small numbers (<20)
# Add +20% to +50% for large numbers

if int(original_num) < 20:
    corrupted_num = str(int(original_num) + random.randint(5, 10))
else:
    corrupted_num = str(int(int(original_num) * random.uniform(1.2, 1.5)))
```

### Strategy 3: Change Multiple Numbers

```python
# Change 2-3 numbers in the problem, not just one
# This makes it harder for model to "ignore" the change

numbers_to_change = random.sample(numbers, min(3, len(numbers)))
for num in numbers_to_change:
    # Apply corruption...
```

### Strategy 4: Structural Changes

```python
# Don't just change numbers - change relationships

# Example transformations:
# - "X more than Y" → "X less than Y"
# - "bought X, sold Y" → "sold X, bought Y"
# - "increased by X%" → "decreased by X%"
```

This is more complex but much more effective!

### Strategy 5: Validate Corruption

```python
def is_corruption_effective(clean_ans, corrupted_ans, threshold=0.2):
    """Check if corruption changed answer enough."""
    if clean_ans == 0:
        return corrupted_ans != 0

    relative_change = abs(corrupted_ans - clean_ans) / abs(clean_ans)
    return relative_change > threshold  # Answer should change by >20%
```

Only keep pairs where corruption actually changed the answer substantially!

---

## Recommended Improvements

### Priority 1: Filter Out Ineffective Corruptions

**Immediate fix** (no code changes needed):
- Run model on all 45 pairs (clean + corrupted)
- Keep only pairs where: `clean_correct AND NOT corrupted_correct`
- This is what we're already doing! ✓

**But**: Generate WAY more pairs (500+) to get enough targets

### Priority 2: Improve Corruption Strategy

For next round of data generation:

1. **Change operation-critical numbers** (not just first number)
2. **Larger magnitude changes** (+5 to +10, or +20-50%)
3. **Validate corruption effectiveness** before manual review
4. **Target 80% break rate** (model gets wrong after corruption)

### Priority 3: Problem Selection

Currently: Random problems from GSM8K

Better:
```python
# Filter for problems that are:
# 1. Simple enough that model solves correctly (40-60% difficulty)
# 2. Sensitive to number changes (avoid "robust" problems)
# 3. Have clear operational structure (addition, multiplication, etc.)
```

---

## Cost-Benefit Analysis

### Current Approach
```
45 pairs generated
→ 23 clean correct (51%)
→ 9 targets (20% of total, 39% of clean correct)
→ Effort-to-target ratio: 5:1
```

### With Better Corruption (80% break rate)
```
45 pairs generated
→ 23 clean correct (51%)
→ 18 targets (40% of total, 78% of clean correct)
→ Effort-to-target ratio: 2.5:1
```

**Benefit**: 2x improvement in data efficiency!

### With Better Corruption + More Pairs
```
500 pairs generated
→ 255 clean correct (51%)
→ 204 targets (41% of total, 80% of clean correct)
→ Have enough for n=204 (still 3x short of 634, but much better!)
```

---

## Action Items

### Short Term (Use Current Data)

✅ **Already doing this**: Filter to only use valid intervention cases
- This is why we have n=9 instead of n=45
- Correctly identifies the 9 usable targets

❌ **Can't improve** current n=9 without more data

### Medium Term (Next Experiment)

1. **Generate 500+ pairs** (not 45)
2. **Improve corruption strategy**:
   - Change operation-critical numbers
   - Larger magnitude changes (+5 to +10)
   - Validate 20% answer change before keeping pair
3. **Pre-filter** before manual review:
   - Run model on both clean and corrupted
   - Keep only pairs where clean ✓ AND corrupted ✗
   - Aim for 100-200 targets

### Long Term (Scaling Up)

1. **Automated corruption pipeline**:
   - Generate 5,000 candidates
   - Auto-filter to ~1,000 valid targets
   - Manual review sample for quality
2. **Multiple corruption types**:
   - Number changes (current)
   - Operation changes ("more" → "less")
   - Entity swaps ("John gives to Mary" → "Mary gives to John")
3. **Difficulty calibration**:
   - Target problems with 40-60% base accuracy
   - Avoid too-easy and too-hard problems

---

## Conclusion

**Question**: Why did so many clean baselines fail?

**Answer**: They didn't! Clean accuracy (51%) is normal. The real issue is:
1. **Weak corruption strategy** (just +1 to first number)
2. **60.9% of corruptions were ineffective** (model still solved them)
3. This reduced our targets from 23 → 9 (2.6x loss)

**Solution**:
- **Short term**: Accept n=9, acknowledge statistical limitations
- **Medium term**: Generate 500+ pairs with better corruption strategy
- **Long term**: Automated pipeline targeting n≥634

**Key Insight**: The problem isn't data quality - it's data **quantity** and **corruption effectiveness**. We need both more pairs AND better corruption to reach adequate statistical power.

---

## References

**Mechanistic Interpretability**:
- Meng et al. (2022) - Uses 1000s of examples, not 9
- Need to match sample sizes from literature!

**GSM8K Dataset**:
- 7,473 training problems available
- 1,319 test problems available
- We've only used 45 (0.6% of training set!)

**Statistical Power**:
- Current: n=9, need n=634 (70x short)
- With better corruption: Could get n≈200 from 500 pairs
- Still 3x short, but moving in right direction!
