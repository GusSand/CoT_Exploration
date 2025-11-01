"""
CODI Full Circuit Analysis - Master Script
Combines all analysis techniques to create a comprehensive circuit understanding

This script:
1. Runs intervention propagation analysis
2. Extracts attention patterns
3. Decomposes pathway contributions
4. Creates a unified circuit diagram showing all information flows
"""

import torch
import sys
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
number_regex = re.compile(r'^\s?\d+')

def load_llama_model():
    """Load CODI-LLaMA model"""
    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B", attn_implementation="eager",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/CoT_Exploration/models/CODI-llama3.2-1b",
        full_precision=True,
        token=None
    )

    llama_training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=2048,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    llama_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=llama_model_args.lora_r,
        lora_alpha=llama_model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        init_lora_weights=True,
    )

    llama_model = CODI(llama_model_args, llama_training_args, llama_lora_config)
    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    return llama_model, llama_tokenizer, llama_training_args


def comprehensive_circuit_trace(model, tokenizer, training_args, question):
    """
    Comprehensive trace capturing all circuit information:
    - Hidden states at each layer and position
    - Attention patterns
    - Projection contributions
    - Token predictions
    """
    batch_size = 1
    questions = [question]

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer(questions, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    question_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].cpu().tolist())

    circuit_trace = {
        'question': question,
        'question_tokens': question_tokens,
        'question_length': len(question_tokens),
        'cot_positions': []
    }

    with torch.no_grad():
        # Initial encoding (position 0: BoT)
        past_key_values = None
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            output_attentions=True,
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        all_hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        latent_embd = all_hidden_states[-1][:, -1:, :]
        first_layer = all_hidden_states[0][:, -1:, :]

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])

        # Compute top-k logits for "logit lens" style analysis
        top_k = 5
        top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices[0]]

        pos_data = {
            'position': 0,
            'token': token_str,
            'token_id': token_id,
            'top_k_tokens': top_tokens,
            'top_k_logits': top_logits[0].cpu().tolist(),
            'layer_norms': [torch.norm(hs[:, -1:, :]).item() for hs in all_hidden_states],
            'first_to_last_delta': torch.norm(latent_embd - first_layer).item(),
            'attention_to_question': 0.0,  # Will compute below
            'attention_to_cot': 0.0,
        }

        # Compute average attention to question
        avg_attn = []
        for attn in attentions:
            attn_from_last = attn[0, :, -1, :].mean(dim=0).cpu().numpy()  # Average across heads
            avg_attn.append(attn_from_last)
        avg_attn = np.mean(avg_attn, axis=0)  # Average across layers
        pos_data['attention_to_question'] = float(np.sum(avg_attn[:len(question_tokens)]))

        # Store projection info
        if training_args.use_prj:
            pre_prj = latent_embd.clone()
            latent_embd = model.prj(latent_embd)
            pos_data['projection_delta'] = torch.norm(latent_embd - pre_prj).item()
            pos_data['projection_relative'] = pos_data['projection_delta'] / torch.norm(pre_prj).item()

        circuit_trace['cot_positions'].append(pos_data)

        # Chain-of-Thought iterations
        for i in range(training_args.inf_latent_iterations):
            input_to_iteration = latent_embd.clone()

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states
            attentions = outputs.attentions

            latent_embd = all_hidden_states[-1][:, -1:, :]
            first_layer = all_hidden_states[0][:, -1:, :]

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])

            top_logits, top_indices = torch.topk(logits, top_k, dim=-1)
            top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices[0]]

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'token_id': token_id,
                'top_k_tokens': top_tokens,
                'top_k_logits': top_logits[0].cpu().tolist(),
                'layer_norms': [torch.norm(hs[:, -1:, :]).item() for hs in all_hidden_states],
                'first_to_last_delta': torch.norm(latent_embd - first_layer).item(),
                'attention_contribution': torch.norm(latent_embd - input_to_iteration).item(),
            }

            # Compute attention to question vs previous CoT
            avg_attn = []
            for attn in attentions:
                attn_from_last = attn[0, :, -1, :].mean(dim=0).cpu().numpy()
                avg_attn.append(attn_from_last)
            avg_attn = np.mean(avg_attn, axis=0)

            pos_data['attention_to_question'] = float(np.sum(avg_attn[:circuit_trace['question_length']]))
            pos_data['attention_to_cot'] = float(np.sum(avg_attn[circuit_trace['question_length']:]))

            # Store projection info
            if training_args.use_prj:
                pre_prj = latent_embd.clone()
                latent_embd = model.prj(latent_embd)
                pos_data['projection_delta'] = torch.norm(latent_embd - pre_prj).item()
                pos_data['projection_relative'] = pos_data['projection_delta'] / torch.norm(pre_prj).item()

            circuit_trace['cot_positions'].append(pos_data)

    return circuit_trace


def create_unified_circuit_diagram(circuit_trace, output_path):
    """
    Create a comprehensive circuit diagram showing:
    - Token flow through CoT positions
    - Attention pathways (to question and previous CoT)
    - Direct projection pathways
    - Layer-wise processing depth
    """
    fig = plt.figure(figsize=(24, 16))

    # Create grid layout
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    # 1. Main Circuit Flow Diagram (top, spanning all columns)
    ax_circuit = fig.add_subplot(gs[0:2, :])
    ax_circuit.set_xlim(0, 10)
    ax_circuit.set_ylim(0, 10)
    ax_circuit.axis('off')
    ax_circuit.set_title('CODI Chain-of-Thought Circuit Diagram', fontsize=20, fontweight='bold', pad=20)

    # Draw question box
    question_box = FancyBboxPatch((0.5, 8), 1.5, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='lightblue', linewidth=2)
    ax_circuit.add_patch(question_box)
    ax_circuit.text(1.25, 8.6, 'Question', ha='center', va='center', fontsize=14, fontweight='bold')

    # Draw CoT positions
    cot_positions = circuit_trace['cot_positions']
    num_positions = len(cot_positions)
    x_start = 3
    x_spacing = 1.0

    position_centers = []

    for i, pos_data in enumerate(cot_positions):
        x = x_start + i * x_spacing
        y = 7

        # Draw CoT node
        color = 'lightgreen' if not bool(number_regex.match(pos_data['token'])) else 'lightyellow'
        cot_box = FancyBboxPatch((x - 0.3, y - 0.4), 0.6, 0.8, boxstyle="round,pad=0.05",
                                  edgecolor='black', facecolor=color, linewidth=2)
        ax_circuit.add_patch(cot_box)

        token_display = pos_data['token'][:4]  # Truncate for display
        ax_circuit.text(x, y + 0.1, f"CoT{i}", ha='center', va='center', fontsize=10, fontweight='bold')
        ax_circuit.text(x, y - 0.2, token_display, ha='center', va='center', fontsize=12, fontweight='bold')

        position_centers.append((x, y))

        # Draw attention arrows to question
        attn_to_q = pos_data.get('attention_to_question', 0)
        if attn_to_q > 0.1:
            arrow = FancyArrowPatch((1.25, 8), (x, y + 0.4),
                                   arrowstyle='->', mutation_scale=20,
                                   color='blue', linewidth=1 + attn_to_q * 3, alpha=0.6)
            ax_circuit.add_patch(arrow)

        # Draw attention arrows to previous CoT
        if i > 0:
            attn_to_cot = pos_data.get('attention_to_cot', 0)
            if attn_to_cot > 0.05:
                for j in range(max(0, i-2), i):  # Show connections to previous 2 positions
                    prev_x, prev_y = position_centers[j]
                    arrow = FancyArrowPatch((prev_x, prev_y - 0.4), (x, y + 0.4),
                                           arrowstyle='->', mutation_scale=15,
                                           color='purple', linewidth=0.5 + attn_to_cot * 2,
                                           alpha=0.4, linestyle='dashed')
                    ax_circuit.add_patch(arrow)

        # Draw projection pathway (direct connection to next position)
        if i < num_positions - 1:
            proj_delta = pos_data.get('projection_delta', 0)
            next_x = x_start + (i + 1) * x_spacing

            # Draw projection as thick red arrow
            arrow = FancyArrowPatch((x, y - 0.5), (next_x, y - 0.5),
                                   arrowstyle='->', mutation_scale=25,
                                   color='red', linewidth=2 + proj_delta * 0.5, alpha=0.8)
            ax_circuit.add_patch(arrow)

            # Label
            mid_x = (x + next_x) / 2
            ax_circuit.text(mid_x, y - 0.8, 'PRJ', ha='center', fontsize=9,
                           fontweight='bold', color='red')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', edgecolor='black', label='Question Input'),
        mpatches.Patch(facecolor='lightyellow', edgecolor='black', label='Number Token'),
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Operation Token'),
        mpatches.FancyArrowPatch((0, 0), (0.5, 0), color='blue', linewidth=2, label='Attention to Question'),
        mpatches.FancyArrowPatch((0, 0), (0.5, 0), color='purple', linewidth=2,
                                linestyle='dashed', label='Attention to Previous CoT'),
        mpatches.FancyArrowPatch((0, 0), (0.5, 0), color='red', linewidth=3, label='Direct Projection'),
    ]
    ax_circuit.legend(handles=legend_elements, loc='lower left', fontsize=10)

    # 2. Logit Lens (top-k predictions at each position)
    ax_logit = fig.add_subplot(gs[2, :])

    positions = [p['position'] for p in cot_positions]
    tokens = [p['token'] for p in cot_positions]

    # Create text table showing top predictions
    ax_logit.axis('off')
    ax_logit.set_title('Logit Lens: Top-5 Token Predictions at Each CoT Position',
                       fontsize=14, fontweight='bold', pad=10)

    table_data = []
    table_data.append(['Position'] + [f"CoT{i}" for i in positions])
    table_data.append(['Predicted'] + tokens)

    for k_idx in range(min(3, len(cot_positions[0]['top_k_tokens']))):
        row = [f'Top-{k_idx+1}']
        for pos_data in cot_positions:
            if k_idx < len(pos_data['top_k_tokens']):
                row.append(pos_data['top_k_tokens'][k_idx])
            else:
                row.append('-')
        table_data.append(row)

    table = ax_logit.table(cellText=table_data, cellLoc='center', loc='center',
                          colWidths=[0.12] * (len(positions) + 1))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Highlight predicted tokens
    for i in range(1, len(positions) + 1):
        table[(1, i)].set_facecolor('lightgreen')
        table[(1, i)].set_text_props(weight='bold')

    # 3. Information Flow Metrics
    ax_metrics = fig.add_subplot(gs[3, 0])

    projection_deltas = [p.get('projection_delta', 0) for p in cot_positions]
    attention_contribs = [p.get('attention_contribution', 0) for p in cot_positions]

    ax_metrics.bar(positions, projection_deltas, alpha=0.7, label='Projection Δ', color='red')
    ax_metrics.plot(positions, attention_contribs, marker='o', linewidth=2,
                   label='Attention Contribution', color='blue')

    ax_metrics.set_xlabel('CoT Position', fontsize=11, fontweight='bold')
    ax_metrics.set_ylabel('Contribution Magnitude', fontsize=11, fontweight='bold')
    ax_metrics.set_title('Information Flow Components', fontsize=12, fontweight='bold')
    ax_metrics.legend()
    ax_metrics.grid(True, alpha=0.3)

    # 4. Attention Distribution
    ax_attn_dist = fig.add_subplot(gs[3, 1])

    attn_to_q = [p.get('attention_to_question', 0) for p in cot_positions]
    attn_to_cot = [p.get('attention_to_cot', 0) for p in cot_positions]

    width = 0.35
    ax_attn_dist.bar([p - width/2 for p in positions], attn_to_q, width,
                     label='To Question', alpha=0.7, color='blue')
    ax_attn_dist.bar([p + width/2 for p in positions], attn_to_cot, width,
                     label='To Prev CoT', alpha=0.7, color='purple')

    ax_attn_dist.set_xlabel('CoT Position', fontsize=11, fontweight='bold')
    ax_attn_dist.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
    ax_attn_dist.set_title('Attention Distribution', fontsize=12, fontweight='bold')
    ax_attn_dist.legend()
    ax_attn_dist.grid(True, alpha=0.3, axis='y')

    # 5. Layer Processing Depth
    ax_layers = fig.add_subplot(gs[3, 2])

    layer_deltas = [p['first_to_last_delta'] for p in cot_positions]

    ax_layers.plot(positions, layer_deltas, marker='s', linewidth=2, markersize=8, color='darkgreen')

    ax_layers.set_xlabel('CoT Position', fontsize=11, fontweight='bold')
    ax_layers.set_ylabel('Layer Processing Δ', fontsize=11, fontweight='bold')
    ax_layers.set_title('Transformer Depth Usage', fontsize=12, fontweight='bold')
    ax_layers.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive circuit diagram saved to {output_path}")
    plt.close()


def generate_circuit_summary(circuit_trace):
    """
    Generate a textual summary of the circuit analysis
    """
    summary = []
    summary.append("="*80)
    summary.append("CODI CHAIN-OF-THOUGHT CIRCUIT ANALYSIS SUMMARY")
    summary.append("="*80)
    summary.append("")

    summary.append(f"Question: {circuit_trace['question'][:100]}...")
    summary.append(f"Question Length: {circuit_trace['question_length']} tokens")
    summary.append(f"CoT Length: {len(circuit_trace['cot_positions'])} positions")
    summary.append("")

    summary.append("CoT Token Sequence:")
    tokens = [p['token'] for p in circuit_trace['cot_positions']]
    summary.append("  " + " → ".join(tokens))
    summary.append("")

    summary.append("Key Circuit Findings:")
    summary.append("")

    # Analyze attention patterns
    avg_attn_to_q = np.mean([p.get('attention_to_question', 0)
                              for p in circuit_trace['cot_positions']])
    avg_attn_to_cot = np.mean([p.get('attention_to_cot', 0)
                                for p in circuit_trace['cot_positions'][1:]])

    summary.append(f"1. Attention Patterns:")
    summary.append(f"   - Average attention to question: {avg_attn_to_q:.3f}")
    summary.append(f"   - Average attention to previous CoT: {avg_attn_to_cot:.3f}")

    if avg_attn_to_q > avg_attn_to_cot:
        summary.append(f"   → The model primarily attends to the QUESTION rather than previous CoT")
    else:
        summary.append(f"   → The model primarily attends to PREVIOUS COT rather than the question")
    summary.append("")

    # Analyze projection contribution
    avg_proj = np.mean([p.get('projection_delta', 0) for p in circuit_trace['cot_positions']])
    avg_attn_contrib = np.mean([p.get('attention_contribution', 0)
                                for p in circuit_trace['cot_positions'][1:]])

    summary.append(f"2. Information Flow Pathways:")
    summary.append(f"   - Average projection contribution: {avg_proj:.3f}")
    summary.append(f"   - Average attention contribution: {avg_attn_contrib:.3f}")

    if avg_proj > avg_attn_contrib:
        summary.append(f"   → DIRECT PROJECTION is the dominant pathway (ratio: {avg_proj/avg_attn_contrib:.2f}x)")
    else:
        summary.append(f"   → ATTENTION is the dominant pathway (ratio: {avg_attn_contrib/avg_proj:.2f}x)")
    summary.append("")

    # Analyze layer processing
    avg_layer_delta = np.mean([p['first_to_last_delta'] for p in circuit_trace['cot_positions']])
    summary.append(f"3. Transformer Processing Depth:")
    summary.append(f"   - Average layer progression: {avg_layer_delta:.3f}")
    summary.append(f"   → The model uses {'DEEP' if avg_layer_delta > 100 else 'SHALLOW'} processing")
    summary.append("")

    # Token type analysis
    num_tokens = [p for p in circuit_trace['cot_positions']
                  if bool(number_regex.match(p['token']))]
    op_tokens = [p for p in circuit_trace['cot_positions']
                 if not bool(number_regex.match(p['token']))]

    summary.append(f"4. Token Type Distribution:")
    summary.append(f"   - Number tokens: {len(num_tokens)} / {len(circuit_trace['cot_positions'])}")
    summary.append(f"   - Operation tokens: {len(op_tokens)} / {len(circuit_trace['cot_positions'])}")
    summary.append("")

    summary.append("="*80)

    return "\n".join(summary)


def main():
    print("="*80)
    print("CODI COMPREHENSIVE CIRCUIT ANALYSIS")
    print("="*80)

    model, tokenizer, training_args = load_llama_model()

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print("\nPerforming comprehensive circuit trace...")
    circuit_trace = comprehensive_circuit_trace(model, tokenizer, training_args, question)

    # Save trace data
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "comprehensive_circuit_trace.json", 'w') as f:
        json.dump(circuit_trace, f, indent=2)
    print(f"✓ Circuit trace saved to {output_dir / 'comprehensive_circuit_trace.json'}")

    # Create unified diagram
    print("\nCreating unified circuit diagram...")
    create_unified_circuit_diagram(circuit_trace, output_dir / "codi_circuit_diagram.png")

    # Generate summary
    print("\nGenerating circuit summary...")
    summary = generate_circuit_summary(circuit_trace)
    print("\n" + summary)

    with open(output_dir / "circuit_summary.txt", 'w') as f:
        f.write(summary)
    print(f"\n✓ Summary saved to {output_dir / 'circuit_summary.txt'}")

    print("\n" + "="*80)
    print("COMPREHENSIVE CIRCUIT ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {output_dir / 'codi_circuit_diagram.png'}")
    print(f"  2. {output_dir / 'comprehensive_circuit_trace.json'}")
    print(f"  3. {output_dir / 'circuit_summary.txt'}")


if __name__ == "__main__":
    main()
