# Viewing Experiments by Date

## Quick Commands

### See today's work (Oct 25):
```bash
ls -l docs/experiments/10-25*
```

### See yesterday (Oct 24):
```bash
ls -l docs/experiments/10-24*
```

### See last week (Oct 16-20):
```bash
ls -l docs/experiments/10-1[6-9]* docs/experiments/10-20*
```

### All GSM8K experiments:
```bash
ls -l docs/experiments/*gsm8k*
```

### All Liars-Bench experiments:
```bash
ls -l docs/experiments/*liars_bench*
```

### By model:
```bash
ls -l docs/experiments/*gpt2*
ls -l docs/experiments/*llama*
```

### Chronological view:
```bash
ls -lt docs/experiments/10-* | head -20  # Most recent 20
```

## File Organization

- **Format**: `MM-DD_<model>_<dataset>_<experiment>.md`
- **Sorts chronologically** when using `ls`
- **Model comes SECOND** for quick scanning (gpt2, llama, both)
- **Dataset visible** third position (gsm8k, liars_bench, commonsense)

## Examples

- `10-25_gpt2_gsm8k_attention_visualization.md` - Oct 25, GPT-2, GSM8K, attention viz
- `10-24_llama_gsm8k_sae_error_analysis.md` - Oct 24, LLaMA, GSM8K, SAE error analysis
- `10-21_both_gsm8k_cot_necessity_ablation.md` - Oct 21, Both models, GSM8K, CoT ablation
- `10-25_gpt2_liars_bench_deception_detection.md` - Oct 25, GPT-2, Liars-Bench, deception
