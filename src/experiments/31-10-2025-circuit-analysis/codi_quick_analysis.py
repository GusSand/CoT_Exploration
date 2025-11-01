"""
CODI Quick Circuit Analysis - Simplified version that works with SDPA attention
"""

import torch
import sys
import re
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

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
    print("="*80)
    print("Loading CODI-LLaMA Model")
    print("="*80)

    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B",
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

    # Force eager attention implementation
    llama_model.codi.config._attn_implementation = 'eager'

    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    print("✓ Model loaded successfully")
    return llama_model, llama_tokenizer, llama_training_args


def analyze_cot_circuit(model, tokenizer, training_args, question):
    """Analyze CoT computation focusing on projection vs attention pathways"""
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

    analysis = {
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
            past_key_values=past_key_values,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        all_hidden_states = outputs.hidden_states

        latent_embd = all_hidden_states[-1][:, -1:, :]
        first_layer = all_hidden_states[0][:, -1:, :]

        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])

        pos_data = {
            'position': 0,
            'token': token_str,
            'token_id': token_id,
            'layer_norms': [torch.norm(hs[:, -1:, :]).item() for hs in all_hidden_states],
            'first_to_last_delta': torch.norm(latent_embd - first_layer).item(),
        }

        if training_args.use_prj:
            pre_prj = latent_embd.clone()
            latent_embd = model.prj(latent_embd)
            pos_data['projection_delta'] = torch.norm(latent_embd - pre_prj).item()
            pos_data['projection_ratio'] = pos_data['projection_delta'] / torch.norm(pre_prj).item()

        analysis['cot_positions'].append(pos_data)

        # Chain-of-Thought iterations
        for i in range(training_args.inf_latent_iterations):
            input_to_iteration = latent_embd.clone()

            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            all_hidden_states = outputs.hidden_states

            latent_embd = all_hidden_states[-1][:, -1:, :]
            first_layer = all_hidden_states[0][:, -1:, :]

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id])

            pos_data = {
                'position': i + 1,
                'token': token_str,
                'token_id': token_id,
                'layer_norms': [torch.norm(hs[:, -1:, :]).item() for hs in all_hidden_states],
                'first_to_last_delta': torch.norm(latent_embd - first_layer).item(),
                'attention_contribution': torch.norm(latent_embd - input_to_iteration).item(),
            }

            if training_args.use_prj:
                pre_prj = latent_embd.clone()
                latent_embd = model.prj(latent_embd)
                pos_data['projection_delta'] = torch.norm(latent_embd - pre_prj).item()
                pos_data['projection_ratio'] = pos_data['projection_delta'] / torch.norm(pre_prj).item()

                # Compare projection vs attention contribution
                if pos_data['attention_contribution'] > 0:
                    pos_data['proj_vs_attn_ratio'] = pos_data['projection_delta'] / pos_data['attention_contribution']

            analysis['cot_positions'].append(pos_data)

    return analysis


def print_analysis_summary(analysis):
    """Print a summary of the circuit analysis"""
    print("\n" + "="*80)
    print("CODI CIRCUIT ANALYSIS SUMMARY")
    print("="*80)

    print(f"\nQuestion: {analysis['question'][:80]}...")
    print(f"CoT Length: {len(analysis['cot_positions'])} positions\n")

    print("CoT Token Sequence:")
    tokens = [p['token'] for p in analysis['cot_positions']]
    print("  " + " → ".join(tokens))

    print("\n" + "-"*80)
    print("Position | Token | Projection Δ | Attention Δ | Proj/Attn Ratio")
    print("-"*80)

    for pos in analysis['cot_positions']:
        proj_delta = pos.get('projection_delta', 0)
        attn_delta = pos.get('attention_contribution', 0)
        ratio = pos.get('proj_vs_attn_ratio', 0)

        print(f"{pos['position']:8} | {pos['token']:5} | {proj_delta:12.3f} | {attn_delta:11.3f} | {ratio:15.3f}")

    print("-"*80)

    # Calculate averages
    avg_proj = np.mean([p.get('projection_delta', 0) for p in analysis['cot_positions']])
    avg_attn = np.mean([p.get('attention_contribution', 0) for p in analysis['cot_positions'][1:]])

    print(f"\nAverage Projection Contribution: {avg_proj:.3f}")
    print(f"Average Attention Contribution:  {avg_attn:.3f}")

    if avg_proj > avg_attn:
        print(f"\n→ DIRECT PROJECTION pathway dominates ({avg_proj/avg_attn:.2f}x stronger)")
    else:
        print(f"\n→ ATTENTION pathway dominates ({avg_attn/avg_proj:.2f}x stronger)")

    print("="*80)


def main():
    print("="*80)
    print("CODI CIRCUIT ANALYSIS")
    print("="*80)

    model, tokenizer, training_args = load_llama_model()

    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    print("\nAnalyzing CoT circuit...")
    analysis = analyze_cot_circuit(model, tokenizer, training_args, question)

    # Save results
    output_dir = Path("./circuit_analysis_results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "circuit_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Analysis saved to {output_dir / 'circuit_analysis.json'}")

    # Print summary
    print_analysis_summary(analysis)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
