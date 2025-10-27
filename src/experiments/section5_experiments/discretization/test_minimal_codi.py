#!/usr/bin/env python3
"""
Minimal CODI test using the actual CODI codebase
Based on probe_latent_token.py but simplified for local CPU testing
"""
import torch
import transformers
from peft import LoraConfig, TaskType
import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import CODI, ModelArguments, DataArguments, TrainingArguments

args = argparse.ArgumentParser()
args.add_argument("--data_name", type=str, default="gsm8k")
args = args.parse_args()
if args.data_name != "single-gsm8k": from datasets import load_dataset, concatenate_datasets

# Test question
test_question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
test_answer = "18"

question_name = "question"
answer_name = "answer"
if args.data_name == "gsm8k":
    dataset = load_dataset("gsm8k", "main")
    test_set = dataset['test']
elif args.data_name == "multi-arith":
    dataset = load_dataset("ChilleD/MultiArith")
    test_set = dataset['test']
elif args.data_name == "svamp":
    dataset = load_dataset("ChilleD/SVAMP")
    test_set = concatenate_datasets([dataset["train"], dataset["test"]])
elif args.data_name == "commonsense":
    dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")
    test_set = dataset['validation']
elif args.data_name == "gsm-hard":
    dataset = load_dataset("juyoung-trl/gsm-hard")
    test_set = dataset['train']
elif args.data_name == "gsm-hard-test":
    dataset = load_dataset("juyoung-trl/gsm-hard")
    test_set = dataset['test']
elif args.data_name == "single-gsm8k":
    test_set = [{"question": test_question, "answer": test_answer}]
else:
    raise ValueError(f"Invalid data name: {args.data_name}")


def main():
    print("="*60)
    print("CODI Minimal Test with Official Code")
    print("="*60)

    # Create arguments
    model_args = ModelArguments(
        model_name_or_path="gpt2",
        lora_init=True,
        lora_r=128,
        lora_alpha=32,
        ckpt_dir=".\\checkpoints\\CODI-gpt2",
        full_precision=True
    )

    data_args = DataArguments(
        data_name="gsm8k",
        batch_size=1
    )

    training_args = TrainingArguments(
        output_dir="./outputs",
        model_max_length=512,
        inf_latent_iterations=6,
        use_prj=True,
        prj_dim=768,
        remove_eos=True,
        greedy=True,
        bf16=False,
        inf_num_iterations=1
    )

    # Create LoRA config
    target_modules = ["c_attn", "c_proj", 'c_fc']
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights=True,
    )

    print("\nInitializing CODI model...")
    model = CODI(model_args, training_args, lora_config)

    # Load checkpoint
    print(f"Loading checkpoint from {model_args.ckpt_dir}...")
    checkpoint_path = os.path.join(model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    print("Model loaded successfully!")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id

    model.eval()

    # Set model to float32 for CPU
    model = model.float()
    accuracies = []
    for example in test_set:
        print(f"\nQuestion: {example['question']}")
        print(f"Ground Truth: {example[answer_name]}\n")

        # Run inference modes
        for mode in ["vanilla", "discretized"]:
            print(f"\n{'='*60}")
            print(f"Mode: {mode.upper()}")
            print(f"{'='*60}\n")

            discretize = (mode == "discretized")
            
            
            if model.bot_id is not None:
                bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(1, 1)
            else:
                bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(1, 2)
            inputs = tokenizer([example['question']], return_tensors="pt", padding="longest")
            inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)

            # Encode question
            past_key_values = None
            outputs = model.codi(
                input_ids=inputs["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values,
                attention_mask=inputs["attention_mask"]
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # T-2: Before initial projection (line 188-191 in probe_latent_token.py)
            probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            top5_values, top5_indices = torch.topk(probs, k=3, dim=2)
            tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices[0, 0]]
            probs_list = [p.item() for p in top5_values[0, 0]]
            print(f"T-1 (before prj): {tokens} {[f'{p:.4f}' for p in probs_list]}")

            # Apply initial projection (line 193-194 in probe_latent_token.py)
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # T-1: After projection - this is NOT in probe_latent_token.py
            # The code goes directly into the thought loop
            # So T-1 doesn't exist in the original - it goes straight to T0
            # Let's track it for completeness but note it's not standard
            #probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
            #top5_values, top5_indices = torch.topk(probs, k=3, dim=2)
            #tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices[0, 0]]
            #probs_list = [p.item() for p in top5_values[0, 0]]
            #print(f"T-1 (after prj): {tokens} {[f'{p:.4f}' for p in probs_list]} [NOT standard - for comparison only]")

            # Thought iterations
            for i in range(training_args.inf_latent_iterations):
                # Decode latent embeddings
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Probe BEFORE projection (ALWAYS show this for both modes)
                probs = torch.nn.functional.softmax(model.codi.lm_head(latent_embd), dim=-1)
                top5_values, top5_indices = torch.topk(probs, k=3, dim=2)
                tokens_before = [tokenizer.decode([idx.item()]) for idx in top5_indices[0, 0]]
                probs_list_before = [p.item() for p in top5_values[0, 0]]

                print(f"T{i} (before prj): {tokens_before} {[f'{p:.4f}' for p in probs_list_before]}")

                # For discretized mode, replace continuous embedding with discrete token BEFORE projection
                # Only discretize at ODD positions (T1, T3, T5) - keep even positions (T0, T2, T4) continuous
                if discretize and i % 2 == 1:
                    # Get top token from the continuous vector
                    top_token = torch.argmax(probs[0, 0], dim=-1)
                    chosen_token = tokenizer.decode([top_token.item()])

                    print(f"T{i} DISCRETIZING: Replacing continuous vector with token '{chosen_token}'")

                    # Replace continuous embedding with the token's embedding
                    latent_embd = model.get_embd(model.codi, model.model_name)(top_token.unsqueeze(0)).unsqueeze(0)
                elif discretize and i % 2 == 0:
                    print(f"T{i} KEEPING CONTINUOUS (even position)")

                # Apply projection (now operates on either continuous vector or discrete token embedding)
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd)

            # Generate final answer
            print("\nGenerating final answer...")
            if training_args.remove_eos:
                eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long)
            else:
                eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]], dtype=torch.long)

            eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor)
            output = eot_emb

            gen_kwargs = {
                "max_new_tokens": 256,
                "temperature": 0.1,
                "top_k": 40,
                "top_p": 0.95,
                "do_sample": False,  # Greedy for consistency
            }

            pred_tokens = []
            for i in range(gen_kwargs["max_new_tokens"]):
                out = model.codi(
                    inputs_embeds=output,
                    output_hidden_states=False,
                    attention_mask=None,
                    use_cache=True,
                    output_attentions=False,
                    past_key_values=past_key_values
                )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

                if training_args.greedy or not gen_kwargs["do_sample"]:
                    next_token_id = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

                pred_tokens.append(next_token_id.item())

                if next_token_id.item() == tokenizer.eos_token_id:
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_id.unsqueeze(0)).unsqueeze(0)

            decoded_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            print(f"Generated answer: {decoded_answer}")
            print()
            print(f"Ground Truth: {example[answer_name]}")
            accuracy = (decoded_answer == example[answer_name])
            accuracies.append(accuracy)



    accuracy = sum(accuracies) / len(accuracies) * 100
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
