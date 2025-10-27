# Dependency Map & Critical Path Analysis

**Project:** CODI Mechanistic Interpretability
**Created:** 2025-10-26

---

## Visual Dependency Graph

```
                                    START
                                      |
                    +-----------------+------------------+
                    |                                    |
                MECH-01                              DECEP-01
          (Data Prep, 8h)                      (Prompts, 4h)
                    |                                    |
         +----------+----------+                     DECEP-02
         |                     |                (Generate, 8h)
     MECH-02              MECH-03                        |
(Step Import, 12h)  (Features, 6h)                  DECEP-03
         |                     |                    (QA, 6h)
         +----------+----------+                         |
                    |                                    |
                MECH-04                              DECEP-04
          (Correlation, 10h)                    (CODI Gen, 3h)
                    |                                    |
         +----------+----------+                     DECEP-05
         |          |          |                  (SAE Extract, 2h)
         |      MECH-05    MECH-06                       |
         |   (Viz, 4h)  (Interv Infra, 14h)         DECEP-06
         |          |          |                 (Diff Analysis, 8h)
         |          |      MECH-07                       |
         |          |   (Sweep, 16h)                 DECEP-07
         |          |          |                  (Classifier, 6h)
         |          |      MECH-08                       |
         |          |   (Viz, 5h)                        |
         |          |          |                         |
         +----------+----------+-------------------------+
                              |
                         PRESENT-01
                    (Consolidate, 6h)
                              |
                         PRESENT-02
                      (Deck, 10h)
                              |
                            END


                    MECH-09 (Redundancy, 8h)
                         ↑
                    [Optional - can run anytime after MECH-06]
```

---

## Critical Path Analysis

### Primary Critical Path (Longest)
**Total: 71 hours**

1. **MECH-01** → 2. **MECH-02** → 3. **MECH-04** → 4. **MECH-06** → 5. **MECH-07** → 6. **MECH-08** → 7. **PRESENT-01** → 8. **PRESENT-02**

**Breakdown:**
- MECH-01: 8h
- MECH-02: 12h
- MECH-04: 10h (requires MECH-02 + MECH-03, but MECH-03 finishes first)
- MECH-06: 14h
- MECH-07: 16h
- MECH-08: 5h
- PRESENT-01: 6h (requires both MECH-08 and DECEP-07)
- PRESENT-02: 10h

**This path determines minimum project duration**

---

## Parallel Workstreams

### Workstream A: Mechanistic Interpretability - Correlation Branch
```
MECH-01 (8h) → MECH-02 (12h) → MECH-04 (10h) → MECH-05 (4h)
                                              ↓
                                         PRESENT-01
```
**Total: 34h sequential**

### Workstream B: Mechanistic Interpretability - Feature Extraction
```
MECH-01 (8h) → MECH-03 (6h) → MECH-04 (10h)
```
**Total: 24h sequential**
**Note:** Merges with Workstream A at MECH-04

### Workstream C: Mechanistic Interpretability - Intervention Branch
```
MECH-01 (8h) → MECH-04 (10h via A/B) → MECH-06 (14h) → MECH-07 (16h) → MECH-08 (5h)
                                                                              ↓
                                                                         PRESENT-01
```
**Total: 53h sequential**

### Workstream D: Deception Detection (Fully Parallel)
```
DECEP-01 (4h) → DECEP-02 (8h) → DECEP-03 (6h) → DECEP-04 (3h) → DECEP-05 (2h) → DECEP-06 (8h) → DECEP-07 (6h)
                                                                                                       ↓
                                                                                                  PRESENT-01
```
**Total: 37h sequential**
**Can run completely in parallel with Workstream A/B/C until PRESENT-01**

### Workstream E: Optional Exploration
```
MECH-01 (8h) → MECH-04 (10h via A/B) → MECH-06 (14h) → MECH-09 (8h)
```
**Total: 40h sequential**
**Note:** MECH-09 is optional and doesn't block anything

---

## Parallelization Strategy

### Phase 1: Foundation (Week 1, Days 1-2)
**Parallel execution:** Start both foundations
- **Engineer 1:** MECH-01 (8h) - Data prep
- **Engineer 2:** DECEP-01 (4h) - Prompt design → DECEP-02 (start, 8h async)

**Duration:** 8h (bottleneck: MECH-01)

### Phase 2: Feature Analysis (Week 1, Days 3-5)
**Parallel execution:**
- **Engineer 1:** MECH-02 (12h) - Step importance
- **Engineer 2:** MECH-03 (6h) - Feature extraction → Wait or help with analysis
- **Background:** DECEP-02 completes (async API calls)

**Duration:** 12h (bottleneck: MECH-02)

### Phase 3: Correlation & QA (Week 2, Days 1-2)
**Parallel execution:**
- **Engineer 1:** MECH-04 (10h) - Correlation analysis
- **Engineer 2:** DECEP-03 (6h) - QA pipeline → DECEP-04 (3h) - CODI gen → DECEP-05 (2h) - SAE extract

**Duration:** 11h (bottleneck: DECEP-03 + DECEP-04 + DECEP-05 = 11h)

### Phase 4: Interventions & Deception Analysis (Week 2, Days 3-5)
**Parallel execution:**
- **Engineer 1:** MECH-06 (14h) - Intervention infrastructure → MECH-07 (16h async overnight)
- **Engineer 2:** MECH-05 (4h) - Correlation viz → DECEP-06 (8h) - Differential analysis → DECEP-07 (6h) - Classifier

**Duration:** 30h wall-clock, but with overnight job (MECH-07)

### Phase 5: Results & Visualization (Week 3, Days 1-2)
**Sequential execution:**
- **Engineer 1:** MECH-08 (5h) - Intervention viz
- **Both:** PRESENT-01 (6h) - Consolidate findings

**Duration:** 11h

### Phase 6: Presentation (Week 3, Days 3-4)
**Sequential execution:**
- **Both:** PRESENT-02 (10h) - Create deck, iterate

**Duration:** 10h

---

## Optimized Timeline

### Best Case (2 Engineers, Aggressive Parallelization)
**Total: ~15-16 work days (~3 weeks)**

- **Week 1:** Foundation + Feature Analysis (Phase 1-2)
- **Week 2:** Correlation + Interventions + Deception (Phase 3-4)
- **Week 3:** Results + Presentation (Phase 5-6)

### Conservative (1 Engineer, Sequential)
**Total: 136 hours ÷ 8h/day = 17 days (~3.5 weeks)**

### Realistic (1 Engineer + Overnight Jobs)
**Total: ~20 work days (~4 weeks)**

---

## Bottleneck Analysis

### Time Bottlenecks (Stories That Block the Most)
1. **MECH-01** (8h) - Blocks all MECH stories
2. **MECH-04** (10h) - Blocks MECH-05, MECH-06, MECH-07, MECH-08
3. **MECH-06** (14h) - Blocks MECH-07, MECH-08, MECH-09
4. **MECH-07** (16h) - Longest single story, blocks MECH-08
5. **DECEP-03** (6h) - Blocks all downstream deception work

### Risk Bottlenecks (Stories That Might Fail)
1. **DECEP-02** - API rate limits, quality issues → Could require regeneration
2. **DECEP-03** - Low retention rate → May need more generation
3. **MECH-02** - Step importance pattern validation → Core methodology
4. **MECH-06** - Causal validation → If interventions don't work, major issue
5. **DECEP-07** - Classifier metrics → If too low, weakens conclusion

---

## Mitigation Strategies

### For Time Bottlenecks
1. **MECH-01:** Start immediately, validate thoroughly before proceeding
2. **MECH-07:** Run overnight with checkpointing
3. **DECEP-02:** Run overnight async
4. **Parallelize:** Run DECEP workstream completely parallel to MECH

### For Risk Bottlenecks
1. **DECEP-02/03:** Budget for 2 rounds of generation if needed
2. **MECH-02:** Validate on small sample (100 problems) before full run
3. **MECH-06:** Extensive unit testing before MECH-07 sweep
4. **DECEP-07:** Have backup analysis if classifier fails (manual feature analysis)

---

## Dependency Matrix

| Story | Depends On | Blocks |
|-------|-----------|--------|
| MECH-01 | - | MECH-02, MECH-03, MECH-06, DECEP-04 |
| MECH-02 | MECH-01 | MECH-04 |
| MECH-03 | MECH-01 | MECH-04 |
| MECH-04 | MECH-02, MECH-03 | MECH-05, MECH-06, DECEP-06 |
| MECH-05 | MECH-04 | PRESENT-01 (soft) |
| MECH-06 | MECH-01, MECH-04 | MECH-07, MECH-09 |
| MECH-07 | MECH-06, MECH-04 | MECH-08 |
| MECH-08 | MECH-07 | PRESENT-01 |
| MECH-09 | MECH-06, MECH-04 | - (optional) |
| DECEP-01 | - | DECEP-02 |
| DECEP-02 | DECEP-01 | DECEP-03 |
| DECEP-03 | DECEP-02 | DECEP-04 |
| DECEP-04 | DECEP-03, MECH-01 | DECEP-05 |
| DECEP-05 | DECEP-04, MECH-01 | DECEP-06 |
| DECEP-06 | DECEP-05, MECH-04 | DECEP-07 |
| DECEP-07 | DECEP-06 | PRESENT-01 |
| PRESENT-01 | MECH-08, DECEP-07 | PRESENT-02 |
| PRESENT-02 | PRESENT-01 | - (final deliverable) |

---

## Resource Requirements by Phase

### Compute Resources
- **GPU (A100 or equivalent):**
  - MECH-02: 2-3 hours
  - MECH-03: 30 min
  - MECH-06: Testing only (1 hour)
  - MECH-07: 8-10 hours
  - DECEP-04: 20 min
  - DECEP-05: 10 min

  **Total GPU time: ~12-15 hours**

### API Costs
- **DECEP-02:**
  - Claude 3.5 Sonnet: 150 pairs × 2 completions = 300 calls
  - GPT-4o: 100 pairs × 2 completions = 200 calls
  - **Estimated cost: $3-4**

### Storage
- **HDF5 files:**
  - feature_activations_train.h5: ~5-10 GB
  - deception_continuous_thoughts.h5: ~200 MB
  - deception_feature_activations.h5: ~50 MB

  **Total: ~10-15 GB**

### Human Resources
- **Optimal:** 2 engineers for 2-3 weeks
- **Minimum:** 1 engineer for 3.5-4 weeks

---

## Recommended Execution Order

### Sprint 1 (Week 1): Foundation & Features
1. MECH-01 (critical path)
2. DECEP-01 (parallel)
3. MECH-02 (critical path) + DECEP-02 (parallel overnight)
4. MECH-03 (parallel with MECH-02 if 2 engineers)
5. DECEP-03 (after DECEP-02 completes)

### Sprint 2 (Week 2): Analysis & Interventions
1. MECH-04 (critical path)
2. MECH-05 (parallel)
3. DECEP-04, DECEP-05, DECEP-06 (parallel, sequential within)
4. MECH-06 (critical path)
5. MECH-07 (overnight)
6. DECEP-07 (parallel)

### Sprint 3 (Week 3): Results & Presentation
1. MECH-08 (after MECH-07 completes)
2. PRESENT-01 (both engineers)
3. PRESENT-02 (both engineers)
4. [Optional] MECH-09 if time permits

---

## Key Milestones

1. **Milestone 1: Data Ready** (End of Day 2)
   - MECH-01 complete
   - Deception prompts ready

2. **Milestone 2: Core Analysis Complete** (End of Week 2, Day 2)
   - MECH-04 complete (correlation analysis)
   - Key mechanistic insights discovered

3. **Milestone 3: Interventions Complete** (End of Week 2, Day 5)
   - MECH-07 complete (intervention sweep)
   - DECEP-07 complete (classifier trained)

4. **Milestone 4: Presentation Ready** (End of Week 3, Day 4)
   - PRESENT-02 complete
   - Ready to present to Neel Nanda

---

## Success Criteria by Milestone

### Milestone 1
- ✓ 7,473 training problems loaded
- ✓ 1,000 stratified test problems
- ✓ SAE models load and encode/decode correctly
- ✓ Deception prompts validated

### Milestone 2
- ✓ Step importance scores computed for all problems
- ✓ Feature activations extracted
- ✓ At least 5 features per position show |r| > 0.3
- ✓ F1412, F1377 rank in top features

### Milestone 3
- ✓ Interventions show position-specific causal effects
- ✓ F1412 impacts position 0-1, F1377 impacts position 4-5
- ✓ Deception classifier achieves >80% accuracy
- ✓ ROC-AUC > 0.85

### Milestone 4
- ✓ 8-10 slide deck complete
- ✓ Speaker notes written
- ✓ Presentation practiced
- ✓ Findings clearly communicated

---

## Notes

- **MECH-09** is optional and can be descoped if time is limited
- **MECH-07** is the longest single task (16h) - plan accordingly
- **DECEP workstream** can run completely parallel - leverage this!
- **Overnight jobs:** MECH-02, MECH-07, DECEP-02 are good overnight candidates
- **Checkpointing is critical** for long-running jobs
