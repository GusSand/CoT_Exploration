import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
from scipy import stats
import sys

sys.path.append(str(Path(__file__).parent.parent.parent / "operation_circuits"))
from sae_model import SparseAutoencoder

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models_full_dataset"
VIZ_DIR = BASE_DIR / "analysis" / "visualizations" / "full_dataset_7473_samples"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
test_data = torch.load(BASE_DIR / "results" / "enriched_test_data_with_cot.pt", weights_only=False)
activations = test_data['hidden_states'].to(device)
metadata = test_data['metadata']

def viz_feature_clean(position, feature_id):
    print(f"\nVisualizing Position {position}, Feature {feature_id}")
    
    sae = SparseAutoencoder(input_dim=2048, n_features=2048, l1_coefficient=0.0005).to(device)
    sae.load_state_dict(torch.load(MODELS_DIR / f"pos_{position}_final.pt", map_location=device))
    sae.eval()
    
    pos_indices = [i for i, p in enumerate(metadata['positions']) if p == position]
    pos_activations = activations[pos_indices]
    pos_cot_token_ids = [metadata['cot_token_ids'][i] for i in pos_indices]
    
    with torch.no_grad():
        _, features = sae(pos_activations)
    
    feature_acts = features[:, feature_id].cpu().numpy()
    threshold = np.percentile(feature_acts, 90)
    top_samples = np.where(feature_acts > threshold)[0]
    
    # Count tokens
    token_counts_top = Counter()
    for idx in top_samples:
        tokens = pos_cot_token_ids[idx]
        if isinstance(tokens, list):
            for token in tokens:
                token_counts_top[token] += 1
    
    token_counts_all = Counter()
    for tokens in pos_cot_token_ids:
        if isinstance(tokens, list):
            for token in tokens:
                token_counts_all[token] += 1
    
    # Calculate enrichment and p-values
    enriched = []
    n_samples = len(pos_activations)
    for token, count_top in token_counts_top.most_common(10):
        count_all = token_counts_all[token]
        freq_top = count_top / len(top_samples) if len(top_samples) > 0 else 0
        freq_all = count_all / n_samples
        enrich = (freq_top / freq_all - 1) * 100 if freq_all > 0 else 0  # % enrichment
        
        # Fisher's exact test
        a = count_top
        b = len(top_samples) - a
        c = count_all - a
        d = n_samples - len(top_samples) - c
        
        if a >= 0 and b >= 0 and c >= 0 and d >= 0:
            try:
                _, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
            except:
                p_value = 1.0
        else:
            p_value = 1.0
        
        tok_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
        enriched.append({
            "token": tok_str,
            "enrichment": enrich,
            "p_value": p_value,
            "neg_log_p": -np.log10(p_value) if p_value > 0 else 100
        })
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    tokens = [e['token'] for e in enriched]
    enrichments = [e['enrichment'] for e in enriched]
    neg_log_ps = [e['neg_log_p'] for e in enriched]
    
    # Left panel: Enrichment
    colors_enrich = ['darkgreen' if e > 30 else 'yellowgreen' if e > 10 else 'lightsalmon' for e in enrichments]
    bars1 = ax1.barh(range(len(tokens)), enrichments, color=colors_enrich, edgecolor='black', linewidth=1)
    ax1.set_yticks(range(len(tokens)))
    ax1.set_yticklabels([f"'{t}'" for t in tokens], fontsize=11)
    ax1.set_xlabel('Enrichment (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Feature {feature_id} (Position {position})\nToken Enrichment', 
                  fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars1, enrichments)):
        ax1.text(val, i, f' {val:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # Right panel: Statistical significance
    colors_sig = ['darkred' if p > 50 else 'red' if p > 20 else 'orange' if p > 10 else 'yellow' for p in neg_log_ps]
    bars2 = ax2.barh(range(len(tokens)), neg_log_ps, color=colors_sig, edgecolor='black', linewidth=1)
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels([f"'{t}'" for t in tokens], fontsize=11)
    ax2.set_xlabel('-log10(p-value)', fontsize=12, fontweight='bold')
    ax2.set_title('Statistical Significance\n(Higher = Stronger Association)', 
                  fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add p-value labels
    for i, (bar, val, pval) in enumerate(zip(bars2, neg_log_ps, [e['p_value'] for e in enriched])):
        if val > 50:
            label = f' p<{pval:.0e}'
        else:
            label = f' p={pval:.2e}'
        ax2.text(val, i, label, va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / f"feature_enrichment_pos{position}_f{feature_id}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: feature_enrichment_pos{position}_f{feature_id}.png")
    plt.close()

viz_feature_clean(5, 1377)
viz_feature_clean(0, 1412)
print("\n✓ Complete!")
