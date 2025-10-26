# Liars-Bench CODI Experiment

**Project**: Train CODI models (GPT-2 & LLaMA) on liars-bench Instructed Deception task, then train deception detection probes on continuous thought activations.

**Date Started**: 2025-10-24
**Status**: 🏗️ In Progress (Story 1.1)

---

## Goal

1. **Train CODI** on liars-bench Instructed Deception (ID) dataset for both GPT-2 and LLaMA
2. **Validate performance** on honest examples (~90%+ accuracy target)
3. **Train linear probes** on continuous thought activations to detect deception
4. **Compare** with Apollo Research baselines

---

## Project Structure

```
liars_bench_codi/
├── data/
│   ├── raw/                    # Raw dataset from HuggingFace
│   └── processed/              # CODI-formatted datasets
├── scripts/
│   ├── 1_download_dataset.py   # Download liars-bench from HF
│   ├── 2_preprocess_data.py    # Format for CODI training
│   ├── 3_analyze_baseline.py   # Dataset analysis
│   ├── train_gpt2.sh           # GPT-2 training script
│   ├── train_llama.sh          # LLaMA training script
│   └── ...
├── results/                    # Evaluation results
├── notebooks/                  # Analysis notebooks
└── README.md                   # This file
```

---

## Quick Start

### 1. Download Dataset

```bash
cd src/experiments/liars_bench_codi/scripts

# Requires HuggingFace token with access to Cadenza-Labs/liars-bench
python 1_download_dataset.py --hf-token YOUR_TOKEN_HERE
```

### 2. Preprocess Data

```bash
python 2_preprocess_data.py
```

### 3. Train CODI Models

```bash
# GPT-2
bash train_gpt2.sh

# LLaMA
bash train_llama.sh
```

---

## Success Criteria

### Stage 1: CODI Validation
- ✅ GPT-2 achieves ≥90% accuracy on honest examples
- ✅ LLaMA achieves ≥90% accuracy on honest examples
- ✅ Continuous thoughts show interpretability (not random)

### Stage 2: Probe Training (if Stage 1 passes)
- ✅ Probes achieve above-chance deception detection
- ✅ Identify which layers/tokens encode deception
- ✅ Compare favorably to Apollo Research baselines

---

## Progress Tracking

- [x] Story 1.1: Dataset download script created
- [ ] Story 1.1: Dataset downloaded
- [ ] Story 1.2: Data preprocessing
- [ ] Story 1.3: Baseline analysis
- [ ] Story 2.1-2.3: GPT-2 training & validation
- [ ] Story 3.1-3.3: LLaMA training & validation
- [ ] Story 4.1: Interpretability check
- [ ] Story 5.1-5.3: Probe training
- [ ] Story 6.1-6.3: Analysis & documentation

---

## References

- **Dataset**: [Cadenza-Labs/liars-bench](https://huggingface.co/datasets/Cadenza-Labs/liars-bench)
- **Paper**: [Detecting Strategic Deception Using Linear Probes](https://arxiv.org/pdf/2502.03407)
- **Code**: [ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection)
- **CODI Paper**: [docs/codi.pdf](../../../docs/codi.pdf)
