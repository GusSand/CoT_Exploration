# User Stories: CODI Mechanistic Interpretability Project

**Project Goal:** Understand how CODI's continuous thoughts work mechanistically and detect deception in latent reasoning.

**Target Audience:** Neel Nanda (15-minute presentation)

**Project Owner:** Research Team

**Created:** 2025-10-26

---

## Epic 1: Mechanistic Interpretability - Understanding Continuous Thoughts

### Story MECH-01: Data Preparation & Sampling
**ID:** MECH-01
**Priority:** CRITICAL (Must have first)
**Effort:** 8 hours
**Dependencies:** None

**User Story:**
As a **mechanistic interpretability researcher**, I want to **prepare a stratified test dataset from GSM8K with loaded SAE models** so that **I can run reproducible experiments on diverse problem types**.

**Acceptance Criteria:**
- [ ] Load full GSM8K dataset (7,473 training + 1,319 test problems)
- [ ] Create stratified test set of 1,000 problems with diversity across:
  - Operation types (addition, multiplication, division, multi-step)
  - Difficulty levels (2-5 step problems)
  - Number ranges (small/large numbers, decimals)
- [ ] Load 6 position-specific SAE models (sae_position_0.pt through sae_position_5.pt)
- [ ] Verify SAE models encode/decode correctly with test cases
- [ ] Generate metadata for each problem (operation types, num_steps, difficulty, etc.)
- [ ] Output files created:
  - `full_train_problems.json` (7,473 problems)
  - `stratified_test_problems.json` (1,000 problems)
  - `data_split_metadata.json`
- [ ] Print and validate summary statistics showing balanced distribution
- [ ] All tests pass

**Definition of Done:**
- Files exist and are properly formatted JSON
- SAE models load without errors
- Test set shows balanced stratification (chi-square test p > 0.05)
- Documentation updated in DATA_INVENTORY.md

**Risks:**
- SAE model files may not exist or be corrupted
- GSM8K data format may differ from expected

**Notes:**
- This is a foundational story - all MECH stories depend on it
- Add WandB logging for data statistics

---

### Story MECH-02: Resampling Infrastructure for Step Importance
**ID:** MECH-02
**Priority:** CRITICAL
**Effort:** 12 hours
**Dependencies:** MECH-01

**User Story:**
As a **mechanistic interpretability researcher**, I want to **measure the causal importance of each continuous thought step** so that **I can understand which positions matter most for reasoning**.

**Acceptance Criteria:**
- [ ] Implement `decode_from_position(problem, start_step, with_context)` function
- [ ] Implement `measure_step_importance()` using KL divergence methodology
- [ ] Handle edge cases: step 0, step 5, corrupted activations
- [ ] Efficient batching for A100 GPU (batch size 32-64)
- [ ] Progress bar with ETA
- [ ] Checkpointing every 500 problems
- [ ] Validate on 100 test problems first showing expected pattern (early > late importance)
- [ ] Run on full 7,473 training problems
- [ ] Output files created:
  - `step_importance_scores.json`
  - `step_importance_summary_stats.json`
- [ ] Performance targets met:
  - Throughput: >250 problems/hour
  - Memory: <40GB GPU RAM
  - Total time: 2-3 hours on A100

**Definition of Done:**
- Infrastructure code passes all tests
- Early steps show higher average importance than later steps (validates hypothesis)
- Position 0 has non-zero importance
- Results reproducible with fixed random seed
- WandB logging shows compute metrics

**Risks:**
- KL divergence may be numerically unstable
- GPU memory issues with large batches
- Compute time may exceed estimates

**Notes:**
- This is the core methodology for understanding step roles
- Need approval before running full dataset (8-10 hour job)

---

### Story MECH-03: Feature Activation Extraction
**ID:** MECH-03
**Priority:** HIGH
**Effort:** 6 hours
**Dependencies:** MECH-01

**User Story:**
As a **mechanistic interpretability researcher**, I want to **extract SAE feature activations at all continuous thought positions** so that **I can analyze which features are active when**.

**Acceptance Criteria:**
- [ ] Extract SAE activations at all 6 positions for each problem
- [ ] Store in HDF5 format with sparse representation
- [ ] Include metadata linking to interpretable features
- [ ] Compute activation statistics (mean, std, sparsity per feature)
- [ ] Output files created:
  - `feature_activations_train.h5` (compressed, <10GB)
  - `activation_metadata.json`
- [ ] Validation checks pass:
  - No NaN or Inf values
  - L0 norms match expected values (19-56 active features per position)
  - File size <10GB with gzip compression
- [ ] Print summary statistics

**Definition of Done:**
- HDF5 file created with correct structure and compression
- Activation statistics match expected sparsity patterns
- Documentation shows how to load and use the data
- Update DATA_INVENTORY.md

**Risks:**
- File size may be larger than expected
- SAE models may have inconsistent output dimensions

**Notes:**
- Can run in parallel with MECH-02
- Critical for MECH-04 correlation analysis

---

### Story MECH-04: Correlation Analysis (Feature × Step × Importance)
**ID:** MECH-04
**Priority:** CRITICAL
**Effort:** 10 hours
**Dependencies:** MECH-02, MECH-03

**User Story:**
As a **mechanistic interpretability researcher**, I want to **correlate SAE features with step importance scores** so that **I can discover which features are causally important at which positions**.

**Acceptance Criteria:**
- [ ] Compute Spearman correlation for each (feature, position) pair
- [ ] Statistical significance testing with Bonferroni correction (12,288 comparisons)
- [ ] Calculate effect sizes (Cohen's d)
- [ ] Identify top-20 features per position
- [ ] Compare against random baseline (permutation test)
- [ ] Stratify by problem type for robustness
- [ ] Output files created:
  - `feature_step_correlations.json`
  - `top_features_by_position.json`
  - `correlation_summary_stats.json`
- [ ] Validation criteria met:
  - Random baseline shows r ≈ 0, p > 0.05
  - At least 5 features per position with |r| > 0.3
  - Known features (F1412, F1377) rank highly

**Definition of Done:**
- All correlations computed with proper statistical testing
- Top features identified and validated
- Known interpretable features appear in top rankings
- Results documented in research journal

**Risks:**
- Multiple comparison correction may be too conservative
- Effect sizes may be smaller than expected

**Notes:**
- This is the key analytical story that reveals mechanistic insights
- Results drive MECH-05 visualizations and MECH-06 interventions

---

### Story MECH-05: Correlation Heatmap Visualizations
**ID:** MECH-05
**Priority:** HIGH
**Effort:** 4 hours
**Dependencies:** MECH-04

**User Story:**
As a **mechanistic interpretability researcher**, I want to **create publication-quality heatmaps of feature-step correlations** so that **I can effectively communicate findings in the presentation**.

**Acceptance Criteria:**
- [ ] Create 4 heatmap figures:
  1. Basic: Top 20 features × 6 positions (correlation values)
  2. With overlay: Same + average step importance curve
  3. Grouped: Features grouped by interpretation type
  4. Stratified: Addition vs. multiplication problems
- [ ] Export as high-res PNG (300 DPI) and vector PDF
- [ ] Use colorblind-friendly palette (viridis/cividis)
- [ ] Design requirements met:
  - Readable text at 8.5×11" print size
  - Asterisks for significance levels
  - Feature interpretations as labels
  - Red box highlighting position 0 anomaly
- [ ] Output files created:
  - `correlation_heatmap_basic.png`
  - `correlation_heatmap_with_importance.png`
  - `correlation_heatmap_grouped.png`
  - `correlation_heatmap_stratified.png`
  - `figures_for_presentation.pdf`

**Definition of Done:**
- All 4 figures created and validated
- Color scheme works in grayscale
- File sizes <5MB per PNG
- Figures reviewed and approved for presentation

**Risks:**
- Figure design may require iterations
- Text may be too small in complex heatmaps

**Notes:**
- These are key presentation assets
- May need design review before finalizing

---

### Story MECH-06: Intervention Infrastructure (Ablation/Boost)
**ID:** MECH-06
**Priority:** CRITICAL
**Effort:** 14 hours
**Dependencies:** MECH-01, MECH-04

**User Story:**
As a **mechanistic interpretability researcher**, I want to **ablate or boost SAE features at specific positions** so that **I can prove features are causally important, not just correlational**.

**Acceptance Criteria:**
- [ ] Implement `ablate_feature_at_step(problem, feature_id, step_position)`
- [ ] Implement `boost_feature_at_step(problem, feature_id, step_position, alpha)`
- [ ] Implement `ablate_multiple_features(problem, feature_ids, step_position)`
- [ ] Measure impact: accuracy delta, answer KL divergence, correctness
- [ ] Efficient batching for A100 (32 problems at once)
- [ ] Validation tests pass:
  - Test 1: F1412 (addition) at position 0 on addition problems → accuracy drops
  - Test 2: F1412 at position 5 → minimal impact
  - Test 3: Random dead feature → no impact
- [ ] Performance targets met:
  - Throughput: >100 interventions/minute on A100
  - Memory: <40GB GPU RAM

**Definition of Done:**
- Infrastructure code implemented and tested
- All 3 validation tests pass with expected results
- Code is efficient and ready for large-scale sweep
- Documentation includes usage examples

**Risks:**
- Interventions may not show expected causal effects
- Numerical instability in SAE decode step
- GPU memory constraints

**Notes:**
- This proves causality, not just correlation
- Critical validation step before MECH-07 sweep

---

### Story MECH-07: Large-Scale Intervention Sweep
**ID:** MECH-07
**Priority:** HIGH
**Effort:** 16 hours (mostly compute time)
**Dependencies:** MECH-06, MECH-04

**User Story:**
As a **mechanistic interpretability researcher**, I want to **run systematic ablations across top features and all positions** so that **I can build comprehensive intervention profiles showing where each feature matters**.

**Acceptance Criteria:**
- [ ] Ablate top 10 features per position (60 features total) at each of 6 positions
- [ ] Include 5 control features (random, dead)
- [ ] Test on 1,000 problems from stratified test set
- [ ] Total: 65 features × 6 positions × 1,000 problems = 390,000 interventions
- [ ] Checkpoint every 5,000 interventions
- [ ] Progress tracking with ETA
- [ ] Output files created:
  - `intervention_results_raw.json`
  - `intervention_summary.json`
- [ ] Compute time: 8-10 hours on A100
- [ ] Results show position-specific causal effects

**Definition of Done:**
- All interventions completed successfully
- Results show clear position-specific patterns
- Control features show minimal/flat impact
- Data ready for visualization (MECH-08)
- Results documented in research journal

**Risks:**
- Very long compute time (8-10 hours)
- Checkpoint recovery may be needed
- May discover unexpected null results

**Notes:**
- This is a long-running job - needs approval to start
- Can run overnight
- Critical data for final presentation

---

### Story MECH-08: Intervention Profile Visualizations
**ID:** MECH-08
**Priority:** HIGH
**Effort:** 5 hours
**Dependencies:** MECH-07

**User Story:**
As a **mechanistic interpretability researcher**, I want to **create intervention profile plots** so that **I can show where features have causal impact in the presentation**.

**Acceptance Criteria:**
- [ ] Create 3 figures:
  1. Line plot: Δaccuracy vs. step for F1412, F1377, F1893, control
  2. Heatmap: All 60 features × 6 positions (Δaccuracy)
  3. Bar plot: Step-specific causal leverage
- [ ] Include error bars (95% confidence intervals)
- [ ] Annotations for key findings
- [ ] Output files created:
  - `intervention_profiles_key_features.png`
  - `intervention_heatmap_all.png`
  - `step_sensitivity_barplot.png`
- [ ] Design meets presentation standards (300 DPI, colorblind-friendly)

**Definition of Done:**
- All 3 figures created and validated
- Key findings clearly annotated
- Figures approved for presentation
- Shows F1412 high impact at position 0-1, F1377 at position 4-5

**Risks:**
- Results may be noisy requiring smoothing
- Key patterns may not be as clear as expected

**Notes:**
- These are critical presentation figures
- Shows causal validation of correlation findings

---

### Story MECH-09: Multi-Feature Redundancy Analysis
**ID:** MECH-09
**Priority:** MEDIUM
**Effort:** 8 hours
**Dependencies:** MECH-06, MECH-04

**User Story:**
As a **mechanistic interpretability researcher**, I want to **test whether feature redundancy is position-specific** so that **I can understand backup mechanisms in continuous thoughts**.

**Acceptance Criteria:**
- [ ] For each position 0-5, ablate 1, 2, 3, 4, 5 top features
- [ ] Measure accuracy and MSE degradation
- [ ] Compute compound ratio: mse[3] / mse[1]
- [ ] Run on 500 test problems
- [ ] Compare to global ablation baseline
- [ ] Output files created:
  - `redundancy_by_position.json`
  - `redundancy_curves.png`
- [ ] Hypothesis validation:
  - Position 0: Low redundancy (planning critical)
  - Position 1-4: High redundancy (calculation backups)
  - Position 5: Low redundancy (output critical)

**Definition of Done:**
- Redundancy analysis completed for all positions
- Results show position-specific patterns
- Visualization clearly shows redundancy differences
- Findings documented

**Risks:**
- Redundancy patterns may be uniform (null result)
- Multiple feature ablations may have non-additive effects

**Notes:**
- This is an exploratory story (nice-to-have, not critical)
- Can be deprioritized if time is limited

---

## Epic 2: Deception Detection in Latent Reasoning

### Story DECEP-01: Design Generation Prompts
**ID:** DECEP-01
**Priority:** CRITICAL
**Effort:** 4 hours
**Dependencies:** None

**User Story:**
As a **AI safety researcher**, I want to **design prompts for generating honest vs. concealed reasoning pairs** so that **I can create a dataset to test deception detection**.

**Acceptance Criteria:**
- [ ] Create prompt template for honest reasoning (explicit steps)
- [ ] Create prompt templates for 3 concealment strategies:
  - skip_intermediate: Omit middle steps
  - vague_language: "I calculated..." without showing work
  - result_only: Jump directly to answer
- [ ] Create validation prompt (check answers match)
- [ ] Include diversity controls and quality checks
- [ ] Output file created: `deception_prompts.json`
- [ ] Manual validation: Generate 5 examples, verify quality
- [ ] Quality criteria met:
  - Answers match exactly
  - Concealment is realistic (not gibberish)
  - Reasoning at GSM8K difficulty level

**Definition of Done:**
- Prompts file created with all templates
- 5 manual examples generated and validated
- Examples show clear difference between honest/concealed
- Team approves prompt quality

**Risks:**
- LLM may not follow concealment instructions consistently
- Prompt engineering may require iteration

**Notes:**
- Foundation for DECEP-02
- Quality here determines dataset quality

---

### Story DECEP-02: Generate Synthetic Deception Dataset
**ID:** DECEP-02
**Priority:** CRITICAL
**Effort:** 8 hours (includes 3-4 hours API time)
**Dependencies:** DECEP-01

**User Story:**
As a **AI safety researcher**, I want to **generate 250 honest/concealed problem pairs using Claude and GPT-4 APIs** so that **I can have a dataset to train and test deception detection**.

**Acceptance Criteria:**
- [ ] Generate 250 pairs total:
  - 150 using Claude 3.5 Sonnet
  - 100 using GPT-4o
- [ ] Stratify: 33% skip_intermediate, 33% vague_language, 33% result_only
- [ ] Async execution with retry logic and rate limiting
- [ ] Checkpoint every 50 pairs
- [ ] Progress tracking with ETA
- [ ] Output files created:
  - `deception_dataset_raw.json` (250 pairs)
  - `generation_log.txt`
- [ ] Cost estimate: $3-4 for API calls
- [ ] Time estimate: 3-4 hours

**Definition of Done:**
- All 250 pairs generated successfully
- Strategy distribution is balanced
- Generation log shows successful completion
- Data ready for QA (DECEP-03)
- Update DATA_INVENTORY.md

**Risks:**
- API rate limits or failures
- Cost overruns
- Quality issues requiring regeneration

**Notes:**
- Long-running async job
- Can run overnight
- Need API keys configured

---

### Story DECEP-03: Quality Assurance Pipeline
**ID:** DECEP-03
**Priority:** CRITICAL
**Effort:** 6 hours
**Dependencies:** DECEP-02

**User Story:**
As a **AI safety researcher**, I want to **filter and validate generated deception pairs** so that **I ensure high-quality data for downstream analysis**.

**Acceptance Criteria:**
- [ ] Automated checks on all 250 pairs:
  - Answers match
  - Non-empty solutions
  - Reasonable length (20-200 tokens)
  - Honest has operations, concealed doesn't
  - Both solutions coherent
  - Actually concealed (similarity < 0.8)
- [ ] Manual review sampling (30 random pairs)
- [ ] Flag suspicious pairs
- [ ] Target: 150+ pairs retained (60% retention)
- [ ] Output files created:
  - `deception_dataset_clean.json` (150+ pairs)
  - `deception_dataset_rejected.json` (with reasons)
  - `qa_report.json`

**Definition of Done:**
- QA pipeline implemented and run
- 150+ high-quality pairs retained
- Rejection reasons documented
- Manual review completed
- Clean dataset ready for DECEP-04
- Update DATA_INVENTORY.md

**Risks:**
- Retention rate may be lower than 60%
- May need to generate more data

**Notes:**
- Quality is critical for valid results
- Budget time for manual review

---

### Story DECEP-04: Generate Continuous Thoughts for Deception Data
**ID:** DECEP-04
**Priority:** HIGH
**Effort:** 3 hours
**Dependencies:** DECEP-03, MECH-01

**User Story:**
As a **AI safety researcher**, I want to **generate continuous thoughts for honest and concealed reasoning** so that **I can extract features for deception detection**.

**Acceptance Criteria:**
- [ ] Run CODI on all clean pairs (150+ × 2 = 300+ inferences)
- [ ] Extract continuous thoughts (6 positions × 2048 dimensions)
- [ ] Verify outputs valid (no NaN)
- [ ] Save in HDF5 format with metadata
- [ ] Output files created:
  - `deception_continuous_thoughts.h5`
  - `deception_ct_statistics.json`
- [ ] Validation checks pass:
  - No NaN or Inf values
  - L2 norms in range (15-35)
  - Difference between honest/concealed is non-zero
- [ ] Compute time: ~15-20 minutes

**Definition of Done:**
- HDF5 file created with proper structure
- Statistics show expected patterns
- Difference between honest/concealed is measurable
- Ready for SAE encoding (DECEP-05)
- Update DATA_INVENTORY.md

**Risks:**
- CODI model may not be accessible
- Output statistics may not show expected differences

**Notes:**
- Quick task once infrastructure is in place
- Critical for DECEP-05

---

### Story DECEP-05: Extract SAE Activations for Deception Data
**ID:** DECEP-05
**Priority:** HIGH
**Effort:** 2 hours
**Dependencies:** DECEP-04, MECH-01

**User Story:**
As a **AI safety researcher**, I want to **extract SAE feature activations for honest and concealed continuous thoughts** so that **I can identify deception signatures**.

**Acceptance Criteria:**
- [ ] Extract features for all honest continuous thoughts
- [ ] Extract features for all concealed continuous thoughts
- [ ] Maintain alignment (pair_id ↔ version ↔ position ↔ features)
- [ ] Use same SAE models as MECH-03
- [ ] Save in sparse HDF5 format
- [ ] Output files created:
  - `deception_feature_activations.h5`
  - `deception_feature_statistics.json`
- [ ] Validation checks pass:
  - L0 norms match expected (19-56 per position)
  - Sparsity is high (70-90% zeros)
  - Difference is non-zero
- [ ] Compute time: ~5-10 minutes

**Definition of Done:**
- HDF5 file created with proper structure
- Statistics match expected sparsity patterns
- Ready for differential analysis (DECEP-06)
- Update DATA_INVENTORY.md

**Risks:**
- Feature differences may be too subtle
- Data alignment issues

**Notes:**
- Fast task leveraging existing SAE infrastructure
- Critical for DECEP-06

---

### Story DECEP-06: Differential Feature Analysis
**ID:** DECEP-06
**Priority:** CRITICAL
**Effort:** 8 hours
**Dependencies:** DECEP-05, MECH-04

**User Story:**
As a **AI safety researcher**, I want to **identify features that activate differently for honest vs. concealed reasoning** so that **I can discover deception signatures in continuous thoughts**.

**Acceptance Criteria:**
- [ ] Compute Δfeature = concealed - honest for each position
- [ ] Statistical testing (paired t-test, Wilcoxon)
- [ ] Effect size calculation (Cohen's d)
- [ ] Identify top deception signatures (largest |Δ| with p < 0.01)
- [ ] Correlate with known interpretable features
- [ ] Stratify by concealment strategy
- [ ] Create heatmap visualization
- [ ] Output files created:
  - `deception_signature_features.json`
  - `deception_delta_by_position.json`
  - `deception_heatmap_by_strategy.png`
- [ ] Expected findings validated:
  - Position 2-4: Strongest signatures
  - F1412 suppressed in concealed chains
  - Novel concealment-specific features identified

**Definition of Done:**
- All statistical tests completed
- Top deception signatures identified
- Visualization created and validated
- Findings show position-specific patterns
- Results documented in research journal

**Risks:**
- Effect sizes may be too small to detect
- Deception signatures may be strategy-specific

**Notes:**
- This is the key analytical story for deception detection
- Results drive DECEP-07 classifier

---

### Story DECEP-07: Train Deception Detection Classifier
**ID:** DECEP-07
**Priority:** HIGH
**Effort:** 6 hours
**Dependencies:** DECEP-06

**User Story:**
As a **AI safety researcher**, I want to **train a classifier to detect deception from SAE features** so that **I can demonstrate practical monitoring capability**.

**Acceptance Criteria:**
- [ ] Train logistic regression on top 120 features (20 per position)
- [ ] Train/test split (80/20, stratified)
- [ ] Report metrics: precision, recall, F1, ROC-AUC
- [ ] Baseline comparisons (random, majority class)
- [ ] Feature importance analysis
- [ ] Ablation study (5, 10, 20, 50 features)
- [ ] Output files created:
  - `deception_classifier_results.json`
  - `deception_classifier_model.pkl`
  - `roc_curve.png`
  - `feature_importance.png`
- [ ] Target metrics met:
  - Accuracy > 0.80
  - F1 > 0.80
  - ROC-AUC > 0.85

**Definition of Done:**
- Classifier trained and evaluated
- All metrics exceed targets
- Visualizations created
- Ablation study shows feature efficiency
- Results demonstrate practical monitoring viability
- Results documented in research journal

**Risks:**
- Classifier may not reach target metrics
- Overfitting on small dataset
- Features may not generalize

**Notes:**
- This is the proof-of-concept for practical monitoring
- Key result for presentation implications

---

## Epic 3: Presentation Deliverables

### Story PRESENT-01: Consolidate Findings
**ID:** PRESENT-01
**Priority:** HIGH
**Effort:** 6 hours
**Dependencies:** MECH-08, DECEP-07

**User Story:**
As a **research team member**, I want to **synthesize all experimental results into a coherent narrative** so that **we can create an effective presentation for Neel Nanda**.

**Acceptance Criteria:**
- [ ] Identify 3-5 key findings (ranked by importance)
- [ ] Write 1-paragraph summary per finding
- [ ] Create summary statistics table
- [ ] Select top 8 figures for presentation
- [ ] Highlight surprising/unexpected results
- [ ] Output files created:
  - `findings_summary.md`
  - `key_statistics.json`
  - `key_figures/` (folder with 8 figures)

**Definition of Done:**
- All key findings documented with evidence
- Statistics table completed
- 8 best figures selected and copied
- Narrative arc is clear and compelling
- Team review and approval

**Risks:**
- Findings may not tell coherent story
- May lack surprising insights

**Notes:**
- This is critical synthesis work before presentation
- May require team discussion

---

### Story PRESENT-02: Create Presentation Deck
**ID:** PRESENT-02
**Priority:** CRITICAL
**Effort:** 10 hours
**Dependencies:** PRESENT-01

**User Story:**
As a **research team member**, I want to **create a slide deck for Neel Nanda** so that **we can effectively communicate our findings in 15 minutes**.

**Acceptance Criteria:**
- [ ] Create 8-10 slides total
- [ ] Clear narrative arc: anomaly → mechanism → validation → implications
- [ ] High-quality figures (300 DPI)
- [ ] Concise text (max 5 bullets per slide, max 10 words per bullet)
- [ ] Speaker notes for each slide
- [ ] Output files created:
  - `presentation_neel_nanda.pdf`
  - `presentation_notes.md`
- [ ] Slide structure:
  1. Title
  2. Research Question
  3. The Anomaly (Position 0)
  4. Method
  5. Finding 1 - Feature Specialization
  6. Finding 2 - Deception Signatures
  7. Finding 3 - Detection Works
  8. Implications
  9. Next Steps
  10. Thank You
- [ ] Design requirements met:
  - Clean, minimal design
  - Colorblind-friendly colors
  - Readable at presentation size

**Definition of Done:**
- Presentation deck completed
- Speaker notes written
- Team review and approval
- Practice run completed
- Ready for delivery to Neel Nanda

**Risks:**
- Fitting all findings into 15 minutes
- Slides may need multiple revisions

**Notes:**
- This is the final deliverable
- Allow time for iterations and practice

---

## Summary Statistics

**Total Stories:** 18
- Epic 1 (MECH): 9 stories
- Epic 2 (DECEP): 7 stories
- Epic 3 (PRESENT): 2 stories

**Total Estimated Effort:** 136 hours (~3.5 weeks at 40 hours/week)

**Priority Breakdown:**
- CRITICAL: 10 stories (foundational work, key analyses, final deliverable)
- HIGH: 7 stories (important supporting work)
- MEDIUM: 1 story (exploratory/nice-to-have)

**Longest Running Jobs:**
- MECH-07: 16 hours (mostly compute)
- MECH-06: 14 hours (infrastructure + validation)
- MECH-02: 12 hours (core methodology)
- PRESENT-02: 10 hours (presentation creation)

---

## Notes for Development

### Critical Path
```
MECH-01 → MECH-02 → MECH-04 → MECH-07 → MECH-08 → PRESENT-01 → PRESENT-02
       → MECH-03 ↗
```

### Parallel Workstreams
- **Workstream 1:** MECH-01 → MECH-02 → MECH-04
- **Workstream 2:** MECH-01 → MECH-03 → MECH-04
- **Workstream 3:** DECEP-01 → DECEP-02 → DECEP-03 → DECEP-04 → DECEP-05 → DECEP-06 → DECEP-07

Workstreams 1+2 and 3 can run in parallel until PRESENT-01.

### Long-Running Jobs (Need Approval)
- MECH-02: ~2-3 hours (7.5K problems)
- MECH-07: ~8-10 hours (390K interventions)
- DECEP-02: ~3-4 hours (API generation)

### Key Validation Points
- After MECH-01: Verify data quality and SAE loading
- After MECH-02: Validate step importance patterns
- After MECH-04: Check if known features rank highly
- After MECH-06: Validate causal interventions work
- After DECEP-03: Verify dataset quality before proceeding
- After DECEP-07: Check if classifier meets target metrics

### WandB Integration Points
All stories should log to WandB:
- Data statistics
- Compute metrics (GPU usage, throughput)
- Experimental results
- Model training metrics

---

## Next Steps for PM

1. **Review and Prioritization:** Review these user stories with team and confirm priorities
2. **Sprint Planning:** Group stories into 2-week sprints
3. **Risk Mitigation:** Plan contingencies for long-running jobs
4. **Resource Allocation:** Confirm GPU availability for compute-heavy stories
5. **Cost Approval:** Get budget approval for API costs (~$3-4)
6. **Timeline:** Create Gantt chart showing critical path and parallel work
