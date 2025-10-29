# Adversarial Attack Case Studies

This directory contains qualitative analysis of adversarial attacks on CODI vs Plain LLaMA models.

## Contents

- **case_studies.md**: Detailed case studies showing:
  - Examples of attacks that succeed vs fail on both models
  - CODI's critical vulnerability (Number Perturbation Moderate - causes 0% accuracy)
  - CODI's robustness advantage (Structure Disruption - +24pp better than Plain LLaMA)
  - Plain LLaMA's response to number perturbation attacks

## Related Experiments

- **Main Results**: [docs/experiments/10-29_llama_gsm8k_adversarial_attacks_codi.md](/home/paperspace/dev/CoT_Exploration/docs/experiments/10-29_llama_gsm8k_adversarial_attacks_codi.md)
- **Raw Data**: `/data/adversarial_attacks/`
- **Attack Implementations**: `/src/experiments/adversarial_attacks/strategies/`

## Key Findings

1. **CODI's Critical Vulnerability**: Number Perturbation Moderate (3-5 numbers) causes complete collapse (0% accuracy, -54pp drop)
2. **CODI's Advantage**: Structure Disruption shows CODI is MORE robust than Plain LLaMA (+24pp accuracy)
3. **Plain LLaMA**: More resilient to number attacks (-26pp vs CODI's -54pp) but fails on structure disruption

## Case Study Examples

### Number Perturbation (CODI Vulnerability)
- Fundraising problem: Injecting "(30 total)", "(note: 76)", "(costs $89)" breaks CODI completely
- CODI: 1860 (wrong) vs Plain: 320 (wrong) vs Gold: 2280

### Structure Disruption (CODI Advantage)
- TV/Reading problem: Shuffling all sentences
- CODI: 36 (correct) vs Plain: 84 (wrong) vs Gold: 36

## Usage

These case studies provide concrete examples for:
- Understanding attack mechanisms
- Developing defenses
- Identifying deployment risks
- Designing adversarial training data
