#!/usr/bin/env python3
"""
Evaluate LLaMA CODI model on GSM8K dataset
Compare performance WITH vs WITHOUT BoT token
Record continuous thought tokens at each step
"""
import torch
import sys
import re
import os
import json
from tqdm import tqdm

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

print("="*80)
print("LLAMA CODI: BoT Token Comparison Evaluation")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")

# Login to HuggingFace
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv('HF_TOKEN')

if hf_token:
    from huggingface_hub import login
    login(token=hf_token)
    print("✓ Logged in to HuggingFace")

# Load dataset
dataset_path = "/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/projection_replacement_clean/llama_cot_clean.json"
import argparse
from datasets import load_dataset

# Parse command line arguments for loading subset of GSM8K test set
parser = argparse.ArgumentParser(description="Evaluate LLaMA CODI model on GSM8K dataset")
parser.add_argument("--num_examples", type=int, default=None, help="Number of test examples to evaluate (0: all; default: load examples known to require CoT for CODI-LLAMA)")
args, unknown = parser.parse_known_args()


if args.num_examples is not None and args.num_examples > 0:
    # Load GSM8K test set from HuggingFace
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    dataset = gsm8k.select(range(args.num_examples))
    print(f"Using first {args.num_examples} examples from GSM8K test set")
elif args.num_examples == 0:
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    dataset = gsm8k
    print(f"Using all {len(dataset)} examples from GSM8K test set")
else:
    print(f"\nLoading dataset from: {dataset_path}")
    with open(dataset_path, 'r') as f:  
        dataset = json.load(f)

print(f"✓ Loaded {len(dataset)} examples\n")

# Load model
print("="*80)
print("Loading CODI-LLaMA Model")
print("="*80)

checkpoint_path = "/workspace/CoT_Exploration/models/CODI-llama3.2-1b/pytorch_model.bin"

llama_model_args = ModelArguments(
    model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
    lora_init=True,
    lora_r=128,
    lora_alpha=32,
    ckpt_dir="/workspace/CoT_Exploration/models/CODI-llama3.2-1b",
    full_precision=True,
    token=hf_token
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

# Create model
llama_model = CODI(llama_model_args, llama_training_args, llama_lora_config)

# Load checkpoint
print(f"\nLoading checkpoint from: {checkpoint_path}")
state_dict = torch.load(checkpoint_path, map_location='cpu')
missing_keys, unexpected_keys = llama_model.load_state_dict(state_dict, strict=False)
print(f"✓ Checkpoint loaded successfully!")

# Tie weights
llama_model.codi.tie_weights()

# Move to device
llama_model = llama_model.to(device)
llama_model = llama_model.to(torch.float32)
llama_model.eval()

print(f"✓ Model ready on {device}\n")

# Load tokenizer
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
llama_tokenizer.pad_token = llama_tokenizer.eos_token

print(f"BoT Token ID: {llama_model.bot_id}")
print(f"EoT Token ID: {llama_model.eot_id}")
print(f"Continuous Thought Iterations: {llama_training_args.inf_latent_iterations}\n")

# Helper function to run inference
def run_inference(question, ground_truth, use_bot_token=True):
    """Run inference and return results"""
    with torch.no_grad():
        # Prepare inputs
        if use_bot_token:
            if llama_training_args.remove_eos:
                bot_tensor = torch.tensor([llama_model.bot_id], dtype=torch.long).unsqueeze(0).to(device)
            else:
                bot_tensor = torch.tensor([llama_tokenizer.eos_token_id, llama_model.bot_id],
                                         dtype=torch.long).unsqueeze(0).to(device)

        inputs = llama_tokenizer([question], return_tensors="pt", padding="longest")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Append BoT token if needed
        if use_bot_token:
            inputs["input_ids"] = torch.cat((inputs["input_ids"], bot_tensor), dim=1)
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(bot_tensor)), dim=1)

        # Encode question
        outputs = llama_model.codi(
            input_ids=inputs["input_ids"],
            use_cache=True,
            output_hidden_states=True,
            past_key_values=None,
            attention_mask=inputs["attention_mask"]
        )
        past_key_values = outputs.past_key_values
        latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

        # Apply initial projection
        if llama_training_args.use_prj:
            latent_embd = llama_model.prj(latent_embd)

        # Generate continuous thoughts
        num_latent = llama_training_args.inf_latent_iterations
        thought_tokens = []
        thought_probs = []

        for i in range(num_latent):
            # Feed latent embedding
            outputs = llama_model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Decode BEFORE applying projection
            probs = torch.nn.functional.softmax(llama_model.codi.lm_head(latent_embd), dim=-1)

            # Get top token
            top_prob, top_idx = torch.topk(probs, k=1, dim=2)
            top_token = llama_tokenizer.decode([top_idx[0, 0].item()])

            thought_tokens.append(top_token)
            thought_probs.append(top_prob[0, 0].item())

            # Apply projection for next iteration
            if llama_training_args.use_prj:
                latent_embd = llama_model.prj(latent_embd)

        # Generate final answer
        eot_tensor = torch.tensor([[llama_model.eot_id]], dtype=torch.long).to(device)
        eot_embd = llama_model.get_embd(llama_model.codi, llama_model.model_name)(eot_tensor)

        outputs = llama_model.codi(
            inputs_embeds=eot_embd,
            use_cache=True,
            output_hidden_states=False,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values

        # Generate answer tokens
        answer_tokens = []
        max_new_tokens = 50
        output = eot_embd

        for _ in range(max_new_tokens):
            outputs = llama_model.codi(
                inputs_embeds=output,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=False
            )
            past_key_values = outputs.past_key_values

            # Get logits for vocabulary tokens only
            logits = outputs.logits[:, -1, :llama_model.codi.config.vocab_size-1]

            if llama_training_args.greedy:
                next_token = torch.argmax(logits, dim=-1).item()
            else:
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1).item()

            answer_tokens.append(next_token)

            if next_token == llama_tokenizer.eos_token_id:
                break

            output = llama_model.get_embd(llama_model.codi, llama_model.model_name)(
                torch.tensor([[next_token]], device=device)
            )

        final_answer = llama_tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Extract predicted number
        predicted_number = None
        numbers = re.findall(r'\d+', final_answer)
        if numbers:
            predicted_number = int(numbers[-1])

        # Check correctness
        correct = (predicted_number == ground_truth)

        return {
            "thought_tokens": thought_tokens,
            "thought_probs": thought_probs,
            "final_answer": final_answer,
            "predicted_number": predicted_number,
            "correct": correct
        }

# Run evaluation
print("="*80)
print("Running Evaluation")
print("="*80)

results_with_bot = []
results_without_bot = []

for example in tqdm(dataset, desc="Evaluating"):
    question = example['question']
    ground_truth = example['answer']

    # Run WITH BoT token
    result_with = run_inference(question, ground_truth, use_bot_token=True)
    results_with_bot.append({
        "question": question,
        "ground_truth": ground_truth,
        "thought_tokens": result_with["thought_tokens"],
        "thought_probs": result_with["thought_probs"],
        "final_answer": result_with["final_answer"],
        "predicted_number": result_with["predicted_number"],
        "correct": result_with["correct"]
    })

    # Run WITHOUT BoT token
    result_without = run_inference(question, ground_truth, use_bot_token=False)
    results_without_bot.append({
        "question": question,
        "ground_truth": ground_truth,
        "thought_tokens": result_without["thought_tokens"],
        "thought_probs": result_without["thought_probs"],
        "final_answer": result_without["final_answer"],
        "predicted_number": result_without["predicted_number"],
        "correct": result_without["correct"]
    })

# Calculate statistics
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

total = len(dataset)
correct_with = sum(1 for r in results_with_bot if r["correct"])
correct_without = sum(1 for r in results_without_bot if r["correct"])

print(f"\nTotal examples: {total}")
print(f"\nWITH BoT Token:")
print(f"  Correct: {correct_with}/{total} ({correct_with/total*100:.1f}%)")
print(f"\nWITHOUT BoT Token:")
print(f"  Correct: {correct_without}/{total} ({correct_without/total*100:.1f}%)")
print(f"\nDifference: {correct_with - correct_without} ({(correct_with - correct_without)/total*100:.1f}%)")

# Analyze thought tokens
number_regex = re.compile(r'^\s?\d+')

numeric_with = []
numeric_without = []

for r in results_with_bot:
    count = sum(1 for token in r["thought_tokens"] if number_regex.match(token))
    numeric_with.append(count)

for r in results_without_bot:
    count = sum(1 for token in r["thought_tokens"] if number_regex.match(token))
    numeric_without.append(count)

avg_numeric_with = sum(numeric_with) / len(numeric_with)
avg_numeric_without = sum(numeric_without) / len(numeric_without)

print(f"\nNumeric Thought Tokens (out of 6):")
print(f"  WITH BoT: {avg_numeric_with:.2f} ({avg_numeric_with/6*100:.1f}%)")
print(f"  WITHOUT BoT: {avg_numeric_without:.2f} ({avg_numeric_without/6*100:.1f}%)")

# Save results
output_path = f"/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/llama_bot_comparison_results{args.num_examples}.json"
output_data = {
    "total_examples": total,
    "accuracy_with_bot": correct_with / total,
    "accuracy_without_bot": correct_without / total,
    "avg_numeric_tokens_with_bot": avg_numeric_with,
    "avg_numeric_tokens_without_bot": avg_numeric_without,
    "results_with_bot": results_with_bot,
    "results_without_bot": results_without_bot
}

with open(output_path, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"\n✓ Results saved to: {output_path}")

# Show some examples
print("\n" + "="*80)
print("Sample Results (first 3 examples)")
print("="*80)

for i in range(min(3, len(results_with_bot))):
    r_with = results_with_bot[i]
    r_without = results_without_bot[i]

    print(f"\nExample {i+1}:")
    print(f"Question: {r_with['question'][:80]}...")
    print(f"Ground Truth: {r_with['ground_truth']}")
    print(f"\nWITH BoT:")
    print(f"  Thoughts: {' → '.join(r_with['thought_tokens'])}")
    print(f"  Answer: {r_with['predicted_number']} {'✓' if r_with['correct'] else '✗'}")
    print(f"\nWITHOUT BoT:")
    print(f"  Thoughts: {' → '.join(r_without['thought_tokens'])}")
    print(f"  Answer: {r_without['predicted_number']} {'✓' if r_without['correct'] else '✗'}")

print("\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)
