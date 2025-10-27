# GPT-2 TopK SAE Usage Guide

**Last Updated**: 2025-10-27
**Model**: GPT-2 (124M params, 768 hidden dims)
**SAE Config**: d=512, K=150 (sweet spot)
**Total SAEs**: 72 (12 layers × 6 positions)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Loading SAE Models](#loading-sae-models)
3. [Extracting Features](#extracting-features)
4. [Using Features for Analysis](#using-features-for-analysis)
5. [File Structure](#file-structure)
6. [Common Tasks](#common-tasks)
7. [Performance Benchmarks](#performance-benchmarks)

---

## Quick Start

### Load a Single SAE

```python
import torch
from pathlib import Path
import sys

# Add TopK SAE to path
sys.path.insert(0, 'src/experiments/topk_grid_pilot')
from topk_sae import TopKAutoencoder

# Load SAE for Position 3, Layer 8 (middle layer)
checkpoint_path = Path('src/experiments/gpt2_sae_training/results/sweet_spot_all/gpt2_sweet_spot_pos3_layer8.pt')
checkpoint = torch.load(checkpoint_path, weights_only=False)

# Initialize model
sae = TopKAutoencoder(
    input_dim=checkpoint['config']['input_dim'],      # 768 (GPT-2 hidden dim)
    latent_dim=checkpoint['config']['latent_dim'],    # 512 (sweet spot)
    k=checkpoint['config']['k']                       # 150 (sweet spot)
)

# Load trained weights
sae.load_state_dict(checkpoint['model_state_dict'])
sae.eval()

print(f"✓ Loaded SAE for Position {checkpoint['config']['position']}, Layer {checkpoint['config']['layer']}")
print(f"  Explained Variance: {checkpoint['metrics']['explained_variance']:.3f}")
print(f"  Feature Death Rate: {checkpoint['metrics']['feature_death_rate']:.3f}")
```

### Extract Features from Activations

```python
import torch

# Example: Extract features from GPT-2 activations
# activations shape: (batch_size, 768)
activations = torch.randn(10, 768)  # Replace with real GPT-2 activations

with torch.no_grad():
    reconstruction, sparse_features, metrics = sae(activations)

print(f"Input shape: {activations.shape}")
print(f"Sparse features shape: {sparse_features.shape}")  # (10, 512)
print(f"Active features per sample: {metrics['l0_mean']:.1f}")  # Should be ~150
print(f"Feature sparsity: {(sparse_features != 0).sum(dim=-1).float().mean().item():.1f}")
```

---

## Loading SAE Models

### Load All SAEs (All Layers × Positions)

```python
import torch
from pathlib import Path

def load_all_saes(base_dir='src/experiments/gpt2_sae_training/results/sweet_spot_all'):
    """Load all 72 SAEs into a dictionary."""
    base_path = Path(base_dir)
    saes = {}

    for layer in range(12):
        for position in range(6):
            checkpoint_path = base_path / f'gpt2_sweet_spot_pos{position}_layer{layer}.pt'
            checkpoint = torch.load(checkpoint_path, weights_only=False)

            # Initialize SAE
            sae = TopKAutoencoder(
                input_dim=checkpoint['config']['input_dim'],
                latent_dim=checkpoint['config']['latent_dim'],
                k=checkpoint['config']['k']
            )
            sae.load_state_dict(checkpoint['model_state_dict'])
            sae.eval()

            # Store in dictionary
            key = (layer, position)
            saes[key] = {
                'model': sae,
                'config': checkpoint['config'],
                'metrics': checkpoint['metrics']
            }

    return saes

# Usage
all_saes = load_all_saes()
print(f"Loaded {len(all_saes)} SAEs")

# Access specific SAE
sae_l8_p3 = all_saes[(8, 3)]['model']
print(f"SAE for Layer 8, Position 3: {sae_l8_p3}")
```

### Load SAEs for Specific Layer

```python
def load_layer_saes(layer, base_dir='src/experiments/gpt2_sae_training/results/sweet_spot_all'):
    """Load all SAEs for a specific layer (all 6 positions)."""
    saes = {}

    for position in range(6):
        checkpoint_path = Path(base_dir) / f'gpt2_sweet_spot_pos{position}_layer{layer}.pt'
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        sae = TopKAutoencoder(
            input_dim=checkpoint['config']['input_dim'],
            latent_dim=checkpoint['config']['latent_dim'],
            k=checkpoint['config']['k']
        )
        sae.load_state_dict(checkpoint['model_state_dict'])
        sae.eval()

        saes[position] = sae

    return saes

# Usage
layer_8_saes = load_layer_saes(8)
print(f"Loaded {len(layer_8_saes)} SAEs for Layer 8")
```

---

## Extracting Features

### From Pre-extracted Activations

If you have pre-extracted GPT-2 activations:

```python
import torch

# Load GPT-2 activations (Position 3, Layer 8)
train_data = torch.load('src/experiments/gpt2_sae_training/data/gpt2_full_train_activations.pt', weights_only=False)

# Extract specific position and layer
import numpy as np
positions = np.array(train_data['metadata']['positions'])
layers = np.array(train_data['metadata']['layers'])
mask = (positions == 3) & (layers == 8)
activations = train_data['activations'][mask]

print(f"Activations shape: {activations.shape}")  # (800, 768)

# Extract features using SAE
sae = all_saes[(8, 3)]['model']
with torch.no_grad():
    reconstruction, sparse_features, metrics = sae(activations)

print(f"Sparse features shape: {sparse_features.shape}")  # (800, 512)
print(f"Non-zero features per sample: {(sparse_features != 0).sum(dim=-1).float().mean().item():.1f}")
```

### From Live GPT-2 Inference

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments, DataArguments
from transformers import HfArgumentParser
from peft import LoraConfig, TaskType

def extract_gpt2_activations(question, model_path="~/codi_ckpt/gpt2_gsm8k/", target_layer=8):
    """Extract continuous thought activations from GPT-2 CODI model."""

    # Load GPT-2 CODI model
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=[
            '--model_name_or_path', 'gpt2',
            '--output_dir', './tmp',
            '--num_latent', '6',
            '--use_lora', 'True',
            '--ckpt_dir', model_path,
            '--use_prj', 'True',
            '--prj_dim', '768',
            '--lora_r', '128',
            '--lora_alpha', '32',
            '--lora_init', 'True',
        ]
    )

    model_args.train = False
    training_args.greedy = True

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=['c_attn', 'c_proj', 'c_fc'],
        init_lora_weights=True,
    )

    model = CODI(model_args, training_args, lora_config)

    import os
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()
    model.float()
    model.eval()

    # Extract activations
    tokenizer = model.tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    with torch.no_grad():
        inputs = tokenizer(question, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Get input embeddings
        input_embd = model.get_embd(model.codi, model.model_name)(input_ids)

        # Forward through model
        outputs = model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = outputs.past_key_values

        # Get BOT embedding
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        # Extract activations for all 6 positions
        layer_activations = []
        latent_embd = bot_emb

        for position in range(6):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Get activation from target layer
            activation = outputs.hidden_states[target_layer + 1][:, -1, :].cpu()
            layer_activations.append(activation.squeeze(0))

            # Update for next position
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            if model.use_prj:
                latent_embd = model.prj(latent_embd)

        return torch.stack(layer_activations)  # (6, 768)

# Usage
question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

activations = extract_gpt2_activations(question, target_layer=8)
print(f"Extracted activations: {activations.shape}")  # (6, 768)

# Extract SAE features for each position
for position in range(6):
    sae = all_saes[(8, position)]['model']
    with torch.no_grad():
        reconstruction, features, metrics = sae(activations[position:position+1])

    # Get top-K active features
    active_mask = features[0] != 0
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    active_values = features[0][active_mask]

    print(f"\nPosition {position}:")
    print(f"  Active features: {len(active_indices)}")
    print(f"  Top 5 features: {active_indices[:5].tolist()}")
    print(f"  Top 5 values: {active_values[:5].tolist()}")
```

---

## Using Features for Analysis

### Feature Catalog Analysis

```python
def analyze_feature_activation_patterns(saes, activations_dict):
    """
    Analyze which features activate for which problems.

    Args:
        saes: Dictionary of (layer, position) -> SAE model
        activations_dict: Dictionary of (layer, position) -> activations tensor

    Returns:
        Dictionary mapping (layer, position, feature_id) -> activation statistics
    """
    feature_stats = {}

    for (layer, position), sae in saes.items():
        activations = activations_dict[(layer, position)]

        with torch.no_grad():
            reconstruction, features, metrics = sae(activations)

        # Analyze each feature
        for feature_id in range(features.shape[1]):
            feature_activations = features[:, feature_id]
            active_samples = (feature_activations != 0).sum().item()

            if active_samples > 0:
                feature_stats[(layer, position, feature_id)] = {
                    'active_samples': active_samples,
                    'activation_rate': active_samples / len(activations),
                    'mean_activation': feature_activations[feature_activations != 0].mean().item(),
                    'max_activation': feature_activations.max().item(),
                }

    return feature_stats

# Usage
feature_stats = analyze_feature_activation_patterns(all_saes, activations_by_layer_pos)
print(f"Analyzed {len(feature_stats)} feature activation patterns")
```

### Error Prediction with SAE Features

```python
def predict_errors_with_sae_features(sae, activations, labels):
    """
    Use SAE features to predict if GPT-2 will make an error.

    Args:
        sae: TopK SAE model
        activations: Input activations (N, 768)
        labels: Binary labels (0=correct, 1=incorrect)

    Returns:
        accuracy: Classification accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Extract SAE features
    with torch.no_grad():
        reconstruction, features, metrics = sae(activations)

    features_np = features.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Train simple classifier
    X_train, X_test, y_train, y_test = train_test_split(
        features_np, labels_np, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    return accuracy, clf

# Usage example
# Assuming you have activations and error labels
accuracy, classifier = predict_errors_with_sae_features(
    sae=all_saes[(8, 3)]['model'],
    activations=activations,
    labels=error_labels
)
print(f"Error prediction accuracy: {accuracy:.3f}")
```

---

## File Structure

### Directory Layout

```
src/experiments/gpt2_sae_training/
├── data/
│   ├── gpt2_full_train_activations.pt        # Train activations (57,600 samples)
│   └── gpt2_full_val_activations.pt          # Val activations (14,400 samples)
│
├── results/
│   ├── sweet_spot_all/
│   │   ├── gpt2_sweet_spot_pos{0-5}_layer{0-11}.pt  # 72 SAE checkpoints
│   │   └── sweet_spot_metrics_all.json               # Metrics for all SAEs
│   │
│   ├── analysis_summary.json                 # Parameter sweep results
│   ├── gpt2_grid_metrics_pos3_layer8_config{0-7}.json  # Individual config metrics
│   ├── gpt2_sweet_spot_reconstruction_loss.png       # Heatmap visualization
│   └── gpt2_sweet_spot_feature_death_rate.png        # Heatmap visualization
│
└── scripts/
    ├── convert_gpt2_data.py                  # Convert JSON to PT format
    ├── train_gpt2_grid.py                    # Train parameter sweep
    ├── analyze_results.py                    # Analyze sweep results
    ├── train_sweet_spot_all_layers_positions.py  # Train all 72 SAEs
    └── visualize_sweet_spot.py               # Create heatmaps
```

### Checkpoint Structure

Each `.pt` file contains:

```python
checkpoint = {
    'model_state_dict': OrderedDict,  # SAE weights
    'config': {
        'input_dim': 768,              # GPT-2 hidden dimension
        'latent_dim': 512,             # Dictionary size (sweet spot)
        'k': 150,                      # Active features (sweet spot)
        'position': int,               # Position index (0-5)
        'layer': int,                  # Layer index (0-11)
        'model': 'gpt2'
    },
    'metrics': {
        'explained_variance': float,   # Reconstruction quality (0-1)
        'feature_death_rate': float,   # Unused features (0-1)
        'reconstruction_loss': float,  # MSE loss
        'l0_mean': float               # Average active features (~150)
    }
}
```

---

## Common Tasks

### Task 1: Compare Features Across Layers

```python
def compare_features_across_layers(position=3):
    """Compare which features activate at different layers for the same position."""
    layer_features = {}

    for layer in range(12):
        sae = all_saes[(layer, position)]['model']

        # Load activations for this layer-position
        activations = load_activations(position, layer)

        with torch.no_grad():
            reconstruction, features, metrics = sae(activations)

        # Count active features
        active_features = (features != 0).sum(dim=0)
        layer_features[layer] = active_features

    return layer_features

# Usage
features_by_layer = compare_features_across_layers(position=3)

# Find features that are consistently active across layers
import torch
all_features = torch.stack([features_by_layer[l] for l in range(12)])
consistent_features = (all_features > 0).sum(dim=0) == 12  # Active in all layers
print(f"Features active in all layers: {consistent_features.sum().item()}")
```

### Task 2: Extract Top Features for Interpretability

```python
def extract_top_features(sae, activations, top_k=10):
    """Extract the most frequently activated features."""
    with torch.no_grad():
        reconstruction, features, metrics = sae(activations)

    # Count how often each feature activates
    feature_counts = (features != 0).sum(dim=0)

    # Get top K features
    top_k_indices = feature_counts.argsort(descending=True)[:top_k]
    top_k_counts = feature_counts[top_k_indices]

    # Get average activation magnitude for top features
    top_k_magnitudes = []
    for idx in top_k_indices:
        active_mask = features[:, idx] != 0
        if active_mask.any():
            avg_mag = features[active_mask, idx].abs().mean().item()
        else:
            avg_mag = 0.0
        top_k_magnitudes.append(avg_mag)

    return {
        'feature_ids': top_k_indices.tolist(),
        'activation_counts': top_k_counts.tolist(),
        'avg_magnitudes': top_k_magnitudes
    }

# Usage
top_features = extract_top_features(
    sae=all_saes[(8, 3)]['model'],
    activations=activations,
    top_k=20
)

print("Top 20 Most Active Features:")
for i, (feat_id, count, mag) in enumerate(zip(
    top_features['feature_ids'],
    top_features['activation_counts'],
    top_features['avg_magnitudes']
)):
    print(f"  {i+1}. Feature {feat_id}: {count} activations (avg magnitude: {mag:.3f})")
```

### Task 3: Reconstruct Activations and Measure Quality

```python
def evaluate_reconstruction_quality(sae, activations):
    """Evaluate how well SAE reconstructs activations."""
    with torch.no_grad():
        reconstruction, features, metrics = sae(activations)

    # Compute metrics
    mse = torch.nn.functional.mse_loss(reconstruction, activations)

    # Explained variance
    residual_var = (activations - reconstruction).var()
    total_var = activations.var()
    explained_variance = 1 - (residual_var / total_var)

    # Cosine similarity
    cosine_sim = torch.nn.functional.cosine_similarity(
        activations, reconstruction, dim=-1
    ).mean()

    return {
        'mse': mse.item(),
        'explained_variance': explained_variance.item(),
        'cosine_similarity': cosine_sim.item(),
        'l0_norm': metrics['l0_mean'],
        'feature_death_rate': metrics.get('feature_death_rate', None)
    }

# Usage
quality = evaluate_reconstruction_quality(
    sae=all_saes[(8, 3)]['model'],
    activations=activations
)

print("Reconstruction Quality:")
for metric, value in quality.items():
    if value is not None:
        print(f"  {metric}: {value:.4f}")
```

---

## Performance Benchmarks

### Inference Speed

```python
import time

# Benchmark single SAE forward pass
sae = all_saes[(8, 3)]['model'].cuda()
activations = torch.randn(1000, 768).cuda()

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = sae(activations)

# Benchmark
torch.cuda.synchronize()
start = time.time()

for _ in range(100):
    with torch.no_grad():
        reconstruction, features, metrics = sae(activations)

torch.cuda.synchronize()
elapsed = time.time() - start

samples_per_sec = (100 * 1000) / elapsed
print(f"Inference speed: {samples_per_sec:.0f} samples/sec")
print(f"Latency per sample: {elapsed / (100 * 1000) * 1000:.3f} ms")
```

**Expected Performance** (A100 GPU):
- Inference: ~50,000 samples/sec
- Latency: ~0.02 ms per sample
- Batch processing: Highly efficient for large batches

### Memory Usage

```python
def estimate_memory_usage():
    """Estimate memory usage for all SAEs."""

    # Single SAE parameters
    input_dim = 768
    latent_dim = 512

    # Encoder: (768, 512) + bias (512)
    # Decoder: (512, 768) + bias (768)
    encoder_params = input_dim * latent_dim + latent_dim
    decoder_params = latent_dim * input_dim + input_dim
    total_params = encoder_params + decoder_params

    # Memory in MB (float32 = 4 bytes)
    memory_mb = (total_params * 4) / (1024 ** 2)

    print(f"Single SAE:")
    print(f"  Parameters: {total_params:,}")
    print(f"  Memory: {memory_mb:.2f} MB")
    print()
    print(f"All 72 SAEs:")
    print(f"  Total parameters: {total_params * 72:,}")
    print(f"  Total memory: {memory_mb * 72:.2f} MB")

estimate_memory_usage()
```

**Expected Output**:
```
Single SAE:
  Parameters: 787,968
  Memory: 3.00 MB

All 72 SAEs:
  Total parameters: 56,733,696
  Total memory: 216.38 MB
```

---

## Best Practices

1. **Always use `eval()` mode**: Set SAE to evaluation mode before inference
2. **Batch processing**: Process multiple samples together for efficiency
3. **GPU acceleration**: Move models and data to GPU for 100x speedup
4. **Cache SAEs**: Load SAEs once and reuse them across multiple inference runs
5. **Monitor memory**: Loading all 72 SAEs requires ~220 MB GPU memory
6. **Feature analysis**: Focus on frequently activated features (top 10-20%) for interpretability

---

## Troubleshooting

### Issue: SAE output has wrong shape

**Solution**: Ensure input activations have correct shape `(batch_size, 768)`

```python
# Correct
activations = torch.randn(10, 768)  # ✓

# Incorrect
activations = torch.randn(768, 10)  # ✗ Wrong dimensions
activations = torch.randn(768)      # ✗ Missing batch dimension
```

### Issue: L0 norm is not exactly K

**Solution**: This is expected due to TopK implementation. Check that L0 ≈ K ± small epsilon.

```python
# Should be ~150
print(f"L0 norm: {metrics['l0_mean']:.1f}")  # e.g., 150.0 or 149.8
```

### Issue: High feature death rate

**Solution**: Check if you're using the correct checkpoint. Sweet spot should have ~5% death rate.

```python
# Check loaded checkpoint metrics
print(f"Feature death rate: {checkpoint['metrics']['feature_death_rate']:.3f}")
# Should be ~0.04-0.05 for sweet spot
```

---

## Related Documentation

- **Experiment Report**: `docs/experiments/10-27_gpt2_gsm8k_topk_sae_sweep.md`
- **Data Inventory**: `docs/DATA_INVENTORY.md` (Section 18)
- **Research Journal**: `docs/research_journal.md` (2025-10-27b entry)
- **TopK SAE Implementation**: `src/experiments/topk_grid_pilot/topk_sae.py`

---

## Citation

If you use these SAEs in your research, please cite:

```
GPT-2 TopK SAE Parameter Sweep (2025)
CoT Exploration Project
Sweet Spot Configuration: d=512, K=150
Dataset: 1,000 GPT-2 CODI predictions on GSM8K
```

---

**Questions or Issues?** Check the detailed experiment report or consult the research journal for additional context.
