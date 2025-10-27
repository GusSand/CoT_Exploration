# Cost Estimation: CODI Mechanistic Interpretability Project

**Project:** CODI Mechanistic Interpretability
**Created:** 2025-10-26
**Estimation Basis:** Historical data, technical specifications, vendor pricing

---

## Summary

| Cost Category | Estimated Cost | Notes |
|--------------|---------------|-------|
| **Human Resources** | $20,400 - $34,000 | 136 hours at $150-250/hour |
| **Compute (GPU)** | $150 - $200 | 12-15 hours A100 at ~$12-15/hour |
| **API Costs** | $3 - $4 | Claude + GPT-4 generation |
| **Storage** | $0 - $5 | 10-15 GB, likely within free tier |
| **Total** | **$20,553 - $34,209** | Academic/research rate |

**Timeline:** 3-4 weeks (single engineer) or 2-3 weeks (two engineers)

---

## Detailed Human Resource Costs

### By Story

| Story ID | Hours | @ $150/hr | @ $250/hr | Priority |
|----------|-------|-----------|-----------|----------|
| **Epic 1: Mechanistic Interpretability** |
| MECH-01 | 8 | $1,200 | $2,000 | CRITICAL |
| MECH-02 | 12 | $1,800 | $3,000 | CRITICAL |
| MECH-03 | 6 | $900 | $1,500 | HIGH |
| MECH-04 | 10 | $1,500 | $2,500 | CRITICAL |
| MECH-05 | 4 | $600 | $1,000 | HIGH |
| MECH-06 | 14 | $2,100 | $3,500 | CRITICAL |
| MECH-07 | 16 | $2,400 | $4,000 | HIGH |
| MECH-08 | 5 | $750 | $1,250 | HIGH |
| MECH-09 | 8 | $1,200 | $2,000 | MEDIUM |
| **Epic 1 Subtotal** | **83h** | **$12,450** | **$20,750** | |
| **Epic 2: Deception Detection** |
| DECEP-01 | 4 | $600 | $1,000 | CRITICAL |
| DECEP-02 | 8 | $1,200 | $2,000 | CRITICAL |
| DECEP-03 | 6 | $900 | $1,500 | CRITICAL |
| DECEP-04 | 3 | $450 | $750 | HIGH |
| DECEP-05 | 2 | $300 | $500 | HIGH |
| DECEP-06 | 8 | $1,200 | $2,000 | CRITICAL |
| DECEP-07 | 6 | $900 | $1,500 | HIGH |
| **Epic 2 Subtotal** | **37h** | **$5,550** | **$9,250** | |
| **Epic 3: Presentation** |
| PRESENT-01 | 6 | $900 | $1,500 | HIGH |
| PRESENT-02 | 10 | $1,500 | $2,500 | CRITICAL |
| **Epic 3 Subtotal** | **16h** | **$2,400** | **$4,000** | |
| **TOTALS** | **136h** | **$20,400** | **$34,000** | |

### Labor Rate Assumptions
- **$150/hour:** Mid-level ML engineer (academic/startup rate)
- **$250/hour:** Senior ML engineer / researcher (market rate)
- **Academic setting:** May be lower ($50-100/hour for PhD student/postdoc)
- **Internal project:** Cost = salary + overhead (~1.5x salary)

---

## Detailed Compute Costs

### GPU Compute (NVIDIA A100 or equivalent)

#### Pricing Reference
- **Cloud providers:** $10-15/hour (AWS, GCP, Azure)
- **Lambda Labs:** ~$1.10/hour (if available)
- **Paperspace:** ~$2.30/hour
- **Estimate used:** $12/hour (mid-range cloud)

#### GPU Usage by Story

| Story ID | GPU Hours | Cost @ $12/hr | Notes |
|----------|-----------|---------------|-------|
| MECH-01 | 0.5 | $6 | SAE model validation |
| MECH-02 | 2.5 | $30 | 7.5K problems, step importance |
| MECH-03 | 0.5 | $6 | Feature extraction (fast) |
| MECH-04 | 0.2 | $2 | Correlation compute (CPU-heavy) |
| MECH-06 | 1.0 | $12 | Intervention validation |
| MECH-07 | 8.0 | $96 | 390K interventions (longest job) |
| MECH-09 | 1.5 | $18 | Redundancy analysis |
| DECEP-04 | 0.3 | $4 | CODI generation (165 pairs × 2) |
| DECEP-05 | 0.2 | $2 | SAE encoding (fast) |
| **Total** | **14.7h** | **$176** | |

#### Budget Scenarios
- **Best case (Lambda/Paperspace):** 14.7h × $1.50/hr = **$22**
- **Expected (mid-tier cloud):** 14.7h × $12/hr = **$176**
- **Worst case (on-demand AWS):** 14.7h × $15/hr = **$221**

**Recommendation:** Use existing Paperspace allocation if available (~$22 vs $176 savings)

---

## API Costs (LLM Inference)

### DECEP-02: Synthetic Data Generation

#### Claude 3.5 Sonnet (150 pairs)
- **Input tokens per pair:** ~300 tokens (problem + prompt)
- **Output tokens per pair:** ~200 tokens (honest solution) + ~150 tokens (concealed solution) = 350 tokens
- **Total per pair:** 300 input + 350 output = 650 tokens
- **150 pairs:** 150 × 650 = 97,500 tokens

**Claude 3.5 Sonnet Pricing (as of Oct 2024):**
- Input: $3.00 per million tokens
- Output: $15.00 per million tokens

**Calculation:**
- Input: 45,000 tokens × $3/M = $0.14
- Output: 52,500 tokens × $15/M = $0.79
- **Claude total: $0.93**

#### GPT-4o (100 pairs)
- **Input tokens per pair:** ~300 tokens
- **Output tokens per pair:** ~350 tokens
- **100 pairs:** 100 × 650 = 65,000 tokens

**GPT-4o Pricing (as of Oct 2024):**
- Input: $2.50 per million tokens
- Output: $10.00 per million tokens

**Calculation:**
- Input: 30,000 tokens × $2.50/M = $0.08
- Output: 35,000 tokens × $10/M = $0.35
- **GPT-4o total: $0.43**

#### Total API Cost
- Claude 3.5 Sonnet: $0.93
- GPT-4o: $0.43
- **Buffer (20% for retries/validation):** $0.27
- **Total: ~$1.63**

**Note:** Original estimate of $3-4 was conservative. Actual cost likely **$2-3** with retries.

#### Contingency: If QA Retention is Low
- If DECEP-03 retention is 60% (150/250 pairs), we're fine
- If retention is <60%, may need to generate additional 50-100 pairs
- **Contingency budget:** +$1.00 (additional generation)

**Total API Budget with Contingency: $3-4** ✓ (matches original estimate)

---

## Storage Costs

### HDF5 Data Files

| File | Size (Compressed) | Purpose | Story |
|------|------------------|---------|-------|
| feature_activations_train.h5 | ~8 GB | SAE features (7.5K problems) | MECH-03 |
| deception_continuous_thoughts.h5 | ~200 MB | CODI outputs (165 pairs × 2) | DECEP-04 |
| deception_feature_activations.h5 | ~50 MB | SAE features (165 pairs × 2) | DECEP-05 |
| intervention_results_raw.json | ~500 MB | Intervention sweep results | MECH-07 |
| checkpoints/ | ~2 GB | Temporary checkpoint files | Various |
| **Total** | **~11 GB** | | |

### Storage Pricing
- **Local storage:** Free (existing allocation)
- **Cloud storage (S3/GCS):** ~$0.023/GB/month
- **Monthly cost:** 11 GB × $0.023 = **$0.25/month**
- **Project duration (1 month):** **$0.25**

**Recommendation:** Use local storage (free). Only upload final results to cloud if needed for sharing.

---

## Risk-Adjusted Cost Estimates

### Scenario 1: Everything Goes Smoothly
**Probability: 40%**

| Category | Cost |
|----------|------|
| Labor | 136h × $150 = $20,400 |
| GPU | 14.7h × $12 = $176 |
| API | $2 |
| Storage | $0 (local) |
| **Total** | **$20,578** |

### Scenario 2: Minor Issues (Most Likely)
**Probability: 50%**

**Issues:**
- DECEP-03 QA retention lower than expected → +50 pairs generation
- MECH-06 validation requires debugging → +4 hours
- PRESENT-02 requires 2 revision rounds → +4 hours
- MECH-07 needs to be rerun once due to bug → +8h GPU

| Category | Cost |
|----------|------|
| Labor | 144h × $150 = $21,600 |
| GPU | 22.7h × $12 = $272 |
| API | $3 |
| Storage | $0 |
| **Total** | **$21,875** |

### Scenario 3: Significant Rework Required
**Probability: 10%**

**Issues:**
- MECH-02 methodology validation fails, needs redesign → +16 hours
- DECEP-02 quality issues, need full regeneration → +8 hours, +$2 API
- DECEP-07 classifier doesn't meet targets → +8 hours rework
- Multiple presentation iterations → +6 hours

| Category | Cost |
|----------|------|
| Labor | 174h × $150 = $26,100 |
| GPU | 26.7h × $12 = $320 |
| API | $5 |
| Storage | $0 |
| **Total** | **$26,425** |

### Expected Value Calculation
- Scenario 1: $20,578 × 0.40 = $8,231
- Scenario 2: $21,875 × 0.50 = $10,938
- Scenario 3: $26,425 × 0.10 = $2,643
- **Expected Total: $21,812**

**Recommended Budget: $25,000** (includes 15% buffer above expected value)

---

## Cost Optimization Strategies

### Labor Cost Reduction
1. **Use less expensive compute for development/testing**
   - Use T4 GPU for testing ($0.50/hr vs $12/hr) → Save ~$50
   - Only use A100 for final runs

2. **Descope optional work**
   - MECH-09 (redundancy analysis) is optional → Save 8h = $1,200

3. **Parallelize aggressively**
   - 2 engineers working in parallel reduces calendar time
   - Faster to result = less overhead

4. **Reuse existing infrastructure**
   - MECH-03 and DECEP-05 use same SAE infrastructure
   - MECH-06 infrastructure reused by MECH-07, MECH-09

### Compute Cost Reduction
1. **Use Paperspace or Lambda Labs** (if available)
   - $1.50/hr vs $12/hr → Save ~$150 (85% reduction)

2. **Optimize batch sizes**
   - Maximize GPU utilization
   - Reduce total compute time by 10-20% → Save $20-35

3. **Run overnight jobs efficiently**
   - MECH-02, MECH-07 can run unattended
   - No wasted GPU time waiting for manual steps

4. **Checkpoint and resume**
   - Avoid losing work to preemption/errors
   - Use spot instances (70% cheaper) → Save ~$120

### API Cost Reduction
1. **Use Claude only** (vs Claude + GPT-4)
   - Generate all 250 pairs with Claude
   - Reduces model comparison complexity
   - Save: ~$0.50 (marginal, but simpler)

2. **Optimize prompts for shorter outputs**
   - Concealed solutions should be naturally shorter
   - Could reduce output tokens by 20% → Save ~$0.30

3. **Batch API calls efficiently**
   - Reduce overhead and request volume

---

## Budget Allocation by Phase

### Phase 1: Foundation (Week 1)
| Cost Type | Amount |
|-----------|--------|
| Labor | 28h × $150 = $4,200 |
| GPU | 1h × $12 = $12 |
| API | $2 |
| **Total** | **$4,214** |

**Stories:** MECH-01, DECEP-01, DECEP-02, MECH-02, MECH-03, DECEP-03

### Phase 2: Analysis (Week 2, Part 1)
| Cost Type | Amount |
|-----------|--------|
| Labor | 21h × $150 = $3,150 |
| GPU | 1h × $12 = $12 |
| API | $0 |
| **Total** | **$3,162** |

**Stories:** MECH-04, MECH-05, DECEP-04, DECEP-05, DECEP-06

### Phase 3: Interventions (Week 2, Part 2)
| Cost Type | Amount |
|-----------|--------|
| Labor | 35h × $150 = $5,250 |
| GPU | 10h × $12 = $120 |
| API | $0 |
| **Total** | **$5,370** |

**Stories:** MECH-06, MECH-07, MECH-08, DECEP-07

### Phase 4: Presentation (Week 3)
| Cost Type | Amount |
|-----------|--------|
| Labor | 16h × $150 = $2,400 |
| GPU | $0 |
| API | $0 |
| **Total** | **$2,400** |

**Stories:** PRESENT-01, PRESENT-02

### Phase 5: Optional (If Time)
| Cost Type | Amount |
|-----------|--------|
| Labor | 8h × $150 = $1,200 |
| GPU | 2h × $12 = $24 |
| API | $0 |
| **Total** | **$1,224** |

**Stories:** MECH-09

---

## ROI Analysis

### Value Delivered
1. **Research Insights**
   - Understanding of CODI mechanistic interpretability
   - Novel findings on step-specific feature roles
   - Deception detection methodology
   - **Value:** High (foundational research)

2. **Reusable Infrastructure**
   - SAE analysis pipeline
   - Intervention framework
   - Deception detection classifier
   - **Value:** Medium (can be applied to future models)

3. **Presentation to Neel Nanda**
   - 15-minute presentation
   - Potential for collaboration/publication
   - **Value:** High (networking, credibility)

4. **Publication Potential**
   - Workshop paper at NeurIPS/ICML (~8 pages)
   - Full paper potential if results are strong
   - **Value:** Very High (academic career)

### Cost-Benefit
- **Cost:** ~$21,000 (expected value)
- **Benefit:**
  - Publication: Priceless (for academic career)
  - Collaboration with Neel Nanda: High value
  - Reusable infrastructure: $5-10K value (time savings on future projects)

**ROI: Positive** (especially for academic/research context)

---

## Funding Sources

### Academic Setting
1. **Research grants** (NSF, DARPA, etc.)
2. **University discretionary funds**
3. **Lab budget allocation**
4. **Student stipend/research credits**

### Industry Setting
1. **R&D budget**
2. **ML team quarterly allocation**
3. **Innovation/exploration fund**

### Startup/Independent
1. **Personal investment** (~$200 for compute + API)
2. **Labor is sweat equity**
3. **Use free-tier/academic credits where possible**

---

## Approvals Required

### Before Starting
- [ ] **Budget approval:** $25,000 (or equivalent labor allocation)
- [ ] **GPU allocation:** 15 hours A100 (or equivalent)
- [ ] **API keys:** Claude API, OpenAI API (with $5 credit limit)
- [ ] **Timeline approval:** 3-4 weeks dedicated work

### During Execution
- [ ] **MECH-02 full run:** Approval before 7.5K problem run (2.5h GPU)
- [ ] **DECEP-02 generation:** Approval before API calls ($2-3)
- [ ] **MECH-07 intervention sweep:** Approval before 390K interventions (8h GPU)

### Cost Control Checkpoints
- **After Week 1:** Review actual vs estimated costs
- **After MECH-06:** Decide if MECH-07 full sweep is warranted
- **After DECEP-03:** Decide if more data generation needed

---

## Summary & Recommendations

### Recommended Budget
- **Total Budget:** $25,000 (conservative, includes buffer)
- **Expected Cost:** $21,812 (probabilistic estimate)
- **Minimum Cost:** $20,578 (best case)

### Key Cost Drivers
1. **Labor:** 85-90% of total cost
2. **GPU (MECH-07):** 8 hours = ~$96 (largest single compute job)
3. **API:** Minimal (<1% of total)

### Cost Optimization Priority
1. **Use cheaper GPU providers** (Paperspace, Lambda) → Save $150
2. **Descope MECH-09 if needed** → Save $1,200
3. **Parallelize to reduce calendar time** → Save overhead
4. **Efficient checkpointing** → Avoid rerun costs

### Go/No-Go Decision Criteria
- **Go if:** Budget >$20K, GPU allocation confirmed, 3-4 weeks available
- **No-go if:** Budget <$10K, no GPU access, <2 weeks available

### Final Recommendation
✅ **PROCEED** - Expected ROI is positive for research/academic context
- Ensure GPU allocation secured (Paperspace if available)
- Get approval for MECH-07 before running (largest compute cost)
- Budget time for 1-2 revision rounds

---

## Tracking & Reporting

### Actual vs Estimated Tracking Template

| Story | Est. Hours | Act. Hours | Δ | Est. GPU | Act. GPU | Δ | Notes |
|-------|-----------|-----------|---|----------|----------|---|-------|
| MECH-01 | 8 | | | 0.5 | | | |
| MECH-02 | 12 | | | 2.5 | | | |
| ... | | | | | | | |

### Weekly Cost Report Template

**Week [N] Cost Report**
- Labor hours this week: [X] hours
- GPU hours this week: [Y] hours
- API costs this week: $[Z]
- Total week cost: $[Total]
- Cumulative cost: $[Cumulative]
- Budget remaining: $[Remaining]
- On track: [Yes/No]

---

## Appendix: Pricing References (Oct 2025)

### GPU Pricing
- **AWS EC2 p4d.24xlarge (8× A100):** ~$32/hour (~$4/A100)
- **GCP a2-highgpu-1g (1× A100):** ~$3.67/hour
- **Azure NC A100 v4:** ~$3.67/hour
- **Paperspace A100:** ~$2.30/hour
- **Lambda Labs A100:** ~$1.10/hour (if available)

### LLM API Pricing
- **Claude 3.5 Sonnet:** $3 input, $15 output per M tokens
- **GPT-4o:** $2.50 input, $10 output per M tokens
- **GPT-4o-mini:** $0.15 input, $0.60 output per M tokens

### Storage Pricing
- **AWS S3 Standard:** $0.023/GB/month
- **GCS Standard:** $0.020/GB/month
- **Azure Blob Hot:** $0.018/GB/month

Last updated: 2025-10-26
