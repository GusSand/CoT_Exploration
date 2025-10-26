import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer
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

def viz_feature(position, feature_id):
    print(f"\n{'='*80}\nPosition {position}, Feature {feature_id}\n{'='*80}")
    
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
    
    print(f"Samples: {len(pos_activations)}, Top: {len(top_samples)}, Threshold: {threshold:.4f}")
    print(f"Mean: {feature_acts.mean():.4f}, Max: {feature_acts.max():.4f}")
    
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
    
    enriched = []
    for token, count_top in token_counts_top.most_common(15):
        count_all = token_counts_all[token]
        freq_top = count_top / len(top_samples) if len(top_samples) > 0 else 0
        freq_all = count_all / len(pos_activations)
        enrich = freq_top / freq_all if freq_all > 0 else 0
        tok_str = tokenizer.decode([token]) if isinstance(token, int) else str(token)
        enriched.append({"token": tok_str, "enrichment": enrich, "freq_top": freq_top})
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(feature_acts, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'90th: {threshold:.3f}')
    ax1.set_xlabel('Activation', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title(f'Pos {position}, F{feature_id}: Activation Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[1, 0])
    tokens = [t['token'] for t in enriched[:10]]
    enrichments = [t['enrichment'] * 100 for t in enriched[:10]]
    ax2.barh(range(len(tokens)), enrichments, color='coral', alpha=0.8, edgecolor='black')
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10)
    ax2.set_xlabel('Enrichment %', fontsize=11, fontweight='bold')
    ax2.set_title('Top 10 Enriched Tokens', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    for i, val in enumerate(enrichments):
        ax2.text(val, i, f' {val:.0f}%', va='center', fontsize=9, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, 1])
    freqs = [t['freq_top'] * 100 for t in enriched[:10]]
    ax3.barh(range(len(tokens)), freqs, color='lightgreen', alpha=0.8, edgecolor='black')
    ax3.set_yticks(range(len(tokens)))
    ax3.set_yticklabels([f"'{t}'" for t in tokens], fontsize=10)
    ax3.set_xlabel('Frequency (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Top Sample Frequency', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    for i, val in enumerate(freqs):
        ax3.text(val, i, f' {val:.1f}%', va='center', fontsize=9, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[2, :])
    top_15 = np.argsort(feature_acts)[-15:][::-1]
    top_15_acts = feature_acts[top_15]
    ax4.bar(range(15), top_15_acts, color='mediumpurple', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Rank', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Activation', fontsize=11, fontweight='bold')
    ax4.set_title('Top 15 Activations', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(15))
    ax4.set_xticklabels([f'#{i+1}' for i in range(15)], fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    for i, val in enumerate(top_15_acts):
        ax4.text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'Feature {feature_id} (Position {position}, Full Dataset)', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(VIZ_DIR / f"feature_detail_pos{position}_f{feature_id}.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: feature_detail_pos{position}_f{feature_id}.png")
    plt.close()
    
    print("\n🔝 Top 5 Enriched:")
    for i, t in enumerate(enriched[:5], 1):
        print(f"  {i}. '{t['token']}': {t['enrichment']*100:.1f}%")

viz_feature(5, 1377)
viz_feature(0, 1412)
print("\n" + "="*80 + "\nCOMPLETE!\n" + "="*80)
