# Project Charter: CODI Mechanistic Interpretability Study

**Project Name:** Understanding and Detecting Deception in CODI Continuous Thoughts
**Project Owner:** Research Team
**Target Audience:** Neel Nanda (15-minute presentation)
**Created:** 2025-10-26
**Status:** Planning Complete, Ready for Development

---

## Executive Summary

This project aims to understand the mechanistic interpretability of CODI's continuous thought reasoning and develop methods to detect deception in latent reasoning. The work addresses a critical AI safety question: **Can we monitor models that reason in hidden continuous spaces?**

### Key Deliverables
1. **Mechanistic Analysis:** Understanding which SAE features matter at which reasoning steps
2. **Deception Detection:** Classifier to identify concealed reasoning (target: >80% accuracy)
3. **Presentation:** 15-minute deck for Neel Nanda with compelling findings and implications

### Success Criteria
- Identify position-specific feature importance patterns
- Validate causal relationships through interventions
- Achieve >80% deception detection accuracy
- Deliver professional presentation with publication-quality figures

---

## Project Scope

### In Scope
- **Mechanistic Interpretability (Epic 1):**
  - Data preparation from GSM8K (7,473 training + 1,000 test problems)
  - Step-level importance analysis using resampling methodology
  - SAE feature extraction and correlation analysis
  - Causal validation through ablation/boost interventions
  - Comprehensive intervention sweep (390,000 interventions)
  - Publication-quality visualizations

- **Deception Detection (Epic 2):**
  - Synthetic dataset generation (250 honest/concealed pairs)
  - Quality assurance pipeline
  - Differential feature analysis identifying deception signatures
  - Binary classifier training and evaluation

- **Presentation (Epic 3):**
  - Findings consolidation
  - Slide deck creation for Neel Nanda
  - Speaker notes and practice

### Out of Scope
- Other models beyond CODI
- Datasets beyond GSM8K and synthetic deception data
- Real-world deception scenarios (only synthetic)
- Production deployment of monitoring system
- Multi-feature redundancy analysis (MECH-09 is optional)

---

## Business Case

### Problem Statement
Models are increasingly reasoning in continuous/latent spaces (e.g., OpenAI o1, CODI) rather than natural language. This raises a critical AI safety question: **If models reason in hidden spaces, can we still monitor for deceptive reasoning?**

### Opportunity
- First study to examine CODI mechanistic interpretability at step level
- Novel methodology for position-specific feature importance
- Practical demonstration of latent reasoning monitoring
- Potential for publication and collaboration with leading researchers

### Value Proposition
1. **Research Value:** Novel insights into continuous thought mechanisms
2. **Safety Value:** Proof-of-concept for monitoring latent reasoning
3. **Career Value:** Presentation to Neel Nanda, publication potential
4. **Infrastructure Value:** Reusable analysis pipeline for future models

---

## Project Structure

### Epics (3)
1. **Epic 1: Mechanistic Interpretability** (9 stories, 83 hours)
2. **Epic 2: Deception Detection** (7 stories, 37 hours)
3. **Epic 3: Presentation** (2 stories, 16 hours)

### Total Scope
- **18 user stories** (17 required + 1 optional)
- **136 hours** estimated effort
- **3-4 weeks** timeline (single engineer)
- **2-3 weeks** timeline (two engineers with parallelization)

---

## Timeline & Milestones

### Sprint 1 (Week 1): Foundation & Features
**Duration:** 5 days
**Effort:** 38 hours

**Stories:**
- MECH-01: Data Preparation (8h) - CRITICAL PATH
- DECEP-01: Design Prompts (4h)
- MECH-02: Step Importance Infrastructure (12h) - CRITICAL PATH
- DECEP-02: Generate Dataset (8h, async overnight)
- MECH-03: Feature Extraction (6h)

**Milestone 1: Data Ready**
- ✓ 7,473 training problems loaded
- ✓ 1,000 stratified test problems
- ✓ SAE models validated
- ✓ Deception prompts ready

### Sprint 2 (Week 2): Analysis & Interventions
**Duration:** 5 days
**Effort:** 60 hours

**Stories:**
- DECEP-03: QA Pipeline (6h)
- MECH-04: Correlation Analysis (10h) - CRITICAL PATH
- MECH-05: Correlation Viz (4h)
- DECEP-04: CODI Generation (3h)
- DECEP-05: SAE Extraction (2h)
- DECEP-06: Differential Analysis (8h)
- MECH-06: Intervention Infrastructure (14h) - CRITICAL PATH
- DECEP-07: Deception Classifier (6h)
- MECH-07: Intervention Sweep (16h, overnight) - CRITICAL PATH

**Milestone 2: Core Analysis Complete**
- ✓ Step importance patterns discovered
- ✓ Feature correlations computed
- ✓ Interventions validated

**Milestone 3: Interventions Complete**
- ✓ Large-scale intervention sweep done
- ✓ Deception classifier trained
- ✓ Key findings identified

### Sprint 3 (Week 3): Results & Presentation
**Duration:** 4 days
**Effort:** 21 hours

**Stories:**
- MECH-08: Intervention Visualizations (5h)
- PRESENT-01: Consolidate Findings (6h)
- PRESENT-02: Create Presentation (10h)

**Milestone 4: Presentation Ready**
- ✓ 8-10 slide deck complete
- ✓ Speaker notes written
- ✓ Ready to present

### Optional: Exploration
**If Time Permits**

**Stories:**
- MECH-09: Multi-Feature Redundancy (8h)

---

## Budget & Resources

### Budget Summary
| Category | Amount | % of Total |
|----------|--------|------------|
| Labor | $20,400 - $34,000 | 95-99% |
| Compute (GPU) | $150 - $200 | 0.5-1% |
| API Costs | $3 - $4 | <0.1% |
| Storage | $0 - $5 | <0.1% |
| **Total** | **$20,553 - $34,209** | **100%** |

**Recommended Budget:** $25,000 (includes 15% contingency buffer)
**Expected Cost:** $21,812 (probabilistic estimate)

### Resource Requirements

#### Human Resources
- **Optimal:** 2 ML engineers/researchers for 2-3 weeks
- **Minimum:** 1 ML engineer/researcher for 3.5-4 weeks
- **Skills Required:**
  - PyTorch/deep learning
  - Mechanistic interpretability familiarity
  - Statistical analysis
  - Data visualization
  - Scientific writing/presentation

#### Compute Resources
- **GPU:** NVIDIA A100 or equivalent
  - Total: ~15 hours
  - Largest job: 8 hours (MECH-07)
- **CPU:** Standard development machine
- **Storage:** ~15 GB for datasets and results

#### API Access
- Anthropic Claude API (~$1)
- OpenAI GPT-4 API (~$1)
- Buffer for retries (~$2)

---

## Risk Assessment

### High-Priority Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| MECH-02 methodology validation fails | Medium | Critical | Validate on 100 problems first before full run |
| DECEP-03 QA retention <60% | Medium | High | Budget for additional generation round |
| MECH-07 intervention sweep shows null results | Low | Critical | Validate MECH-06 thoroughly first |
| DECEP-07 classifier doesn't meet 80% target | Medium | High | Have backup manual analysis ready |
| MECH-07 compute takes >10 hours | Medium | Medium | Use checkpointing, run overnight |
| API rate limits during DECEP-02 | Low | Low | Implement retry logic with backoff |
| Presentation doesn't fit in 15 minutes | Medium | Medium | Practice and iterate, prioritize key findings |

### Risk Mitigation Plan
1. **Early validation:** Run small-scale tests before large jobs
2. **Checkpointing:** All long-running jobs must checkpoint
3. **Parallel execution:** DECEP workstream fully parallel to reduce calendar risk
4. **Descoping option:** MECH-09 is optional, can be cut if needed
5. **Budget buffer:** 15% contingency for rework

---

## Success Metrics

### Research Outcomes
1. **Feature Discovery**
   - Target: Identify >5 features per position with |r| > 0.3
   - Success: Known features (F1412, F1377) rank in top 20

2. **Causal Validation**
   - Target: F1412 ablation at position 0 causes >10% accuracy drop
   - Success: Clear position-specific intervention profiles

3. **Deception Detection**
   - Target: Classifier accuracy >80%, ROC-AUC >0.85
   - Success: Practical monitoring demonstrated

### Deliverable Quality
1. **Presentation**
   - 8-10 slides, fits in 15 minutes
   - Publication-quality figures (300 DPI)
   - Clear narrative arc

2. **Documentation**
   - All results documented in research journal
   - Reproducible experiments
   - Code commented and clean

### Project Management
1. **Timeline:** Complete within 4 weeks
2. **Budget:** Stay within $25,000 budget
3. **Quality:** All validation tests pass

---

## Stakeholders

### Primary Stakeholder
- **Neel Nanda** - Recipient of presentation, potential collaborator

### Project Team
- **Research Lead** - Overall project direction, presentation delivery
- **ML Engineer(s)** - Implementation of stories
- **Advisor/PI** - Budget approval, strategic guidance

### Success Criteria by Stakeholder
- **Neel Nanda:** Learns novel insights, finds work compelling enough for feedback/collaboration
- **Research Team:** Completes project on time, gains insights, potential publication
- **Advisor/PI:** Project stays on budget, demonstrates research capability

---

## Dependencies & Assumptions

### External Dependencies
- Access to CODI model weights
- Access to SAE models (sae_position_0.pt through sae_position_5.pt)
- GSM8K dataset availability
- GPU allocation confirmed
- API keys for Claude and GPT-4

### Assumptions
1. CODI and SAE models work as documented
2. GPU availability when needed (no scheduling conflicts)
3. Step importance methodology is sound (will validate early)
4. Deception can be detected from feature activations
5. Neel Nanda is interested in this research area
6. Team has necessary ML/interpretability skills

### Constraints
1. **Timeline:** Must complete in 3-4 weeks (presentation deadline)
2. **Budget:** ~$20-25K maximum
3. **Compute:** Limited GPU hours, must use efficiently
4. **Presentation:** Must fit in 15 minutes

---

## Communication Plan

### Weekly Status Updates
**Format:** Written report + optional standup

**Content:**
- Stories completed this week
- Stories in progress
- Blockers/risks
- Budget tracking (actual vs estimated)
- Next week priorities

### Key Decision Points
1. **After MECH-01:** Proceed with MECH-02 full run? (GPU approval)
2. **After MECH-02:** Are step importance patterns as expected?
3. **After DECEP-03:** Is dataset quality sufficient or need more generation?
4. **After MECH-06:** Proceed with MECH-07 full sweep? (8h GPU approval)
5. **After DECEP-07:** Do results support strong conclusion?

### Escalation Path
- **Technical blockers:** Research lead → Advisor
- **Budget overruns:** Research lead → Advisor → Funding source
- **Timeline risks:** Descope MECH-09, streamline presentation

---

## Quality Assurance

### Code Quality
- Unit tests for core functions (resampling, interventions)
- Validation tests with known features
- Code review before merging
- Clear documentation and comments

### Data Quality
- Automated QA checks (DECEP-03)
- Manual review sampling (30 random pairs)
- No NaN/Inf values in computed results
- Reproducibility with fixed random seeds

### Analysis Quality
- Statistical significance testing with multiple comparison correction
- Effect size reporting (not just p-values)
- Baseline comparisons (random, control features)
- Stratification by problem type for robustness

### Presentation Quality
- Clear narrative arc
- Publication-quality figures
- Fits in time limit
- Practice run before delivery

---

## Change Management

### Scope Change Process
1. Identify change and impact on timeline/budget
2. Assess if change is critical or nice-to-have
3. If critical, adjust priorities and descope elsewhere
4. Document change and rationale
5. Update user stories and estimates

### Approval Required For
- Adding new stories (epic-level changes)
- Changing success criteria
- Budget increases >10%
- Timeline extension >1 week
- Descopring CRITICAL priority stories

### Fast-Track Approval For
- Minor story refinements
- Descoping MEDIUM priority stories (MECH-09)
- Technical implementation details
- Visualization design iterations

---

## Project Governance

### Project Sponsor
- **Role:** Budget approval, strategic direction
- **Involvement:** Weekly status review, key decision points

### Project Lead
- **Role:** Day-to-day execution, technical decisions
- **Responsibilities:** Task assignment, quality assurance, stakeholder communication

### Development Team
- **Role:** Implementation of user stories
- **Responsibilities:** Code quality, validation testing, documentation

### Review Board
- **When:** After each milestone
- **Purpose:** Go/no-go decisions, risk assessment
- **Attendees:** Sponsor, Project Lead, key team members

---

## Next Steps (Immediate Actions)

### Before Starting Development
1. [ ] **Review & approve this charter** with team and advisor
2. [ ] **Secure budget approval** ($25,000)
3. [ ] **Confirm GPU allocation** (15 hours A100 or equivalent)
4. [ ] **Obtain API keys** (Claude, OpenAI with $5 limit)
5. [ ] **Verify data access** (CODI models, SAE models, GSM8K)
6. [ ] **Set up WandB project** for experiment tracking
7. [ ] **Create project Kanban board** with all 18 stories
8. [ ] **Schedule weekly status meetings**

### Week 1 Kickoff
1. [ ] Assign MECH-01 to engineer (critical path)
2. [ ] Assign DECEP-01 to engineer (can run parallel)
3. [ ] Set up development environment
4. [ ] Create data directory structure
5. [ ] Initialize git branch for development

### Success Checklist
- [ ] Budget approved
- [ ] Timeline approved
- [ ] Resources allocated
- [ ] Stakeholders aligned
- [ ] Charter reviewed and accepted
- [ ] **READY TO START DEVELOPMENT**

---

## Approval Signatures

**Project Charter Approved By:**

| Name | Role | Date | Signature |
|------|------|------|-----------|
| | Research Lead | | |
| | Advisor/PI | | |
| | Budget Authority | | |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-26 | PM Team | Initial charter creation |

---

## Related Documents

- [User Stories](/home/paperspace/dev/CoT_Exploration/docs/project/user_stories.md)
- [Dependency Map](/home/paperspace/dev/CoT_Exploration/docs/project/dependency_map.md)
- [Cost Estimation](/home/paperspace/dev/CoT_Exploration/docs/project/cost_estimation.md)
- [Data Inventory](/home/paperspace/dev/CoT_Exploration/docs/DATA_INVENTORY.md)
- [Research Journal](/home/paperspace/dev/CoT_Exploration/docs/research_journal.md)

---

**End of Project Charter**
