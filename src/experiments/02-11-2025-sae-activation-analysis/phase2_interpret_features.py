#!/usr/bin/env python3
"""
Phase 2: Interpret SAE features - what do they represent?

Since CODI generates latent CoT (not text), we can't directly see "which tokens"
the model is thinking. Instead, we analyze:

1. **Decode hidden states**: Pass hidden states through LM head to see top predicted tokens
2. **Question patterns**: Does the feature fire for certain types of questions?
3. **Answer correlation**: Does the feature correlate with correct/incorrect answers?
4. **Reference CoT patterns**: Which parts of reference solutions appear in questions
   where the feature fires strongly?

This gives us semantic interpretation of what each SAE feature represents.
"""

import torch
import sys
import os
import json
import numpy as np
import re
from pathlib import Path
from collections import defaultdict, Counter
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_cot_and_decode(model, tokenizer, training_args, question, layer_idx=14):
    """
    Run CoT and decode hidden states to see what the model is "thinking".
    Returns decoded top tokens for each CoT position.
    """
    batch_size = 1

    if training_args.remove_eos:
        bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch_size, 1).to(device)
    else:
        bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id],
                                  dtype=torch.long).expand(batch_size, 2).to(device)

    inputs = tokenizer([question], return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

    decoded_thoughts = []

    with torch.no_grad():
        # Initial encoding
        outputs = model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # CoT iterations - decode what the model is thinking
        for i in range(training_args.inf_latent_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values

            # Get hidden state from specified layer
            hidden_state = outputs.hidden_states[layer_idx][:, -1, :]

            # Decode to tokens
            logits = model.codi.lm_head(hidden_state)
            probs = torch.softmax(logits[:, :model.codi.config.vocab_size-1], dim=-1)
            top_probs, top_indices = torch.topk(probs, 5)

            top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices[0]]
            top_probs_list = top_probs[0].float().cpu().numpy().tolist()

            decoded_thoughts.append({
                'position': i,
                'top_tokens': list(zip(top_tokens, top_probs_list))
            })

            # Continue projection
            latent_embd = outputs.hidden_states[-1][:, -1:, :]
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return decoded_thoughts


def analyze_feature_with_decoding(feature_id, layer_name, phase1_data, gsm8k_train,
                                   model, tokenizer, training_args, n_examples=200):
    """
    Analyze a feature by:
    1. Finding where it fires
    2. Re-running those examples and decoding what the model is thinking
    3. Extracting patterns from questions and reference CoTs
    """
    print(f"\n{'='*80}")
    print(f"Analyzing Feature {feature_id} in {layer_name.upper()} layer")
    print(f"{'='*80}\n")

    layer_idx = {'early': 4, 'middle': 8, 'late': 14}[layer_name]

    # Collect examples where this feature fires
    firing_examples = []

    for example_idx, example in enumerate(phase1_data['results'][:n_examples]):
        for pos_idx, position_data in enumerate(example['activations']):
            layer_data = position_data['layers'][layer_name]

            if feature_id in layer_data['firing_indices']:
                idx_in_list = layer_data['firing_indices'].index(feature_id)
                activation_value = layer_data['firing_values'][idx_in_list]

                firing_examples.append({
                    'example_idx': example_idx,
                    'cot_position': pos_idx,
                    'activation': activation_value,
                    'question': example['question'],
                    'answer': gsm8k_train[example_idx]['answer']
                })

    if len(firing_examples) == 0:
        print(f"Feature {feature_id} never fired in {n_examples} examples.")
        return None

    print(f"Feature fired {len(firing_examples)} times across {n_examples} examples\n")

    # Sort by activation strength
    firing_examples.sort(key=lambda x: x['activation'], reverse=True)

    # Analyze top 10 strongest activations
    print(f"{'='*80}")
    print(f"Top 10 Strongest Activations:")
    print(f"{'='*80}\n")

    detailed_analysis = []

    for rank, ex in enumerate(firing_examples[:10], 1):
        print(f"\n--- Rank {rank} (Activation: {ex['activation']:.4f}) ---")
        print(f"Example {ex['example_idx']}, CoT Position {ex['cot_position']}")
        print(f"\nQuestion: {ex['question']}")
        print(f"\nReference CoT: {ex['answer']}")

        # Re-run this example and decode thoughts
        print(f"\nModel's decoded thoughts at each CoT position:")
        decoded_thoughts = extract_cot_and_decode(
            model, tokenizer, training_args,
            ex['question'], layer_idx=layer_idx
        )

        for thought in decoded_thoughts:
            pos_marker = " <-- FIRES HERE" if thought['position'] == ex['cot_position'] else ""
            print(f"  Position {thought['position']}{pos_marker}:")
            for token, prob in thought['top_tokens'][:3]:
                print(f"    '{token}' ({prob:.3f})")

        detailed_analysis.append({
            'rank': rank,
            'example_idx': ex['example_idx'],
            'cot_position': ex['cot_position'],
            'activation': ex['activation'],
            'question': ex['question'],
            'reference_answer': ex['answer'],
            'decoded_thoughts': decoded_thoughts
        })

    # Extract patterns
    print(f"\n{'='*80}")
    print(f"Pattern Analysis:")
    print(f"{'='*80}\n")

    # Position distribution
    position_counts = Counter([ex['cot_position'] for ex in firing_examples])
    print(f"Position distribution:")
    for pos in sorted(position_counts.keys()):
        count = position_counts[pos]
        pct = 100 * count / len(firing_examples)
        print(f"  Position {pos}: {count:4d} ({pct:5.1f}%)")

    # Extract reference CoT keywords where feature fires
    all_reference_cots = ' '.join([ex['answer'] for ex in firing_examples])
    # Extract numbers, operation words, etc.
    numbers = re.findall(r'\d+', all_reference_cots)
    number_counts = Counter(numbers).most_common(10)

    operations = re.findall(r'(\+|-|\*|/|=|<<|>>)', all_reference_cots)
    operation_counts = Counter(operations).most_common()

    print(f"\nMost common numbers in reference CoTs where feature fires:")
    for num, count in number_counts:
        print(f"  '{num}': {count} times")

    print(f"\nMost common operations in reference CoTs where feature fires:")
    for op, count in operation_counts:
        print(f"  '{op}': {count} times")

    return {
        'feature_id': feature_id,
        'layer': layer_name,
        'n_activations': len(firing_examples),
        'position_distribution': dict(position_counts),
        'common_numbers': number_counts,
        'common_operations': operation_counts,
        'detailed_analysis': detailed_analysis
    }


def main():
    print("="*80)
    print("PHASE 2: SAE Feature Interpretation with Decoding")
    print("="*80)

    # Load Phase 1 data
    phase1_data_path = Path('./phase1_sae_activations_full.json')
    print(f"\nLoading Phase 1 data from: {phase1_data_path}")
    with open(phase1_data_path, 'r') as f:
        phase1_data = json.load(f)

    # Load GSM8K for reference CoTs
    print(f"Loading GSM8K training set...")
    gsm8k_dataset = load_dataset("gsm8k", "main")
    gsm8k_train = gsm8k_dataset['train']

    # Load model
    print("\n" + "="*80)
    print("Loading CODI-LLAMA")
    print("="*80)

    llama_model_args = ModelArguments(
        model_name_or_path="meta-llama/Llama-3.2-1B",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir="/workspace/.cache/huggingface/hub/models--zen-E--CODI-llama3.2-1b-Instruct/snapshots/b2c88ba224b06b12b52ef39b87f794b98a6eb1c8",
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

    # Analyze specific feature (user's example: 3682)
    result = analyze_feature_with_decoding(
        feature_id=3682,
        layer_name='late',
        phase1_data=phase1_data,
        gsm8k_train=gsm8k_train,
        model=llama_model,
        tokenizer=llama_tokenizer,
        training_args=llama_training_args,
        n_examples=500
    )

    # Save results
    output_path = Path('./phase2_feature_3682_interpretation.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
