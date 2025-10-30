import torch
import sys
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from datetime import datetime

# Load environment variables and login to HuggingFace
load_dotenv("/workspace/.env")
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(token=hf_token)
    print("[OK] Logged in to HuggingFace")

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Number detection regex
number_regex = re.compile(r'^\s?\d+')

# Results directory for incremental saves
RESULTS_DIR = Path('./extended_cot_results')
RESULTS_DIR.mkdir(exist_ok=True)

def save_intermediate_results(results, config, summary):
    """Save results after each example to prevent data loss"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'intermediate_results_{len(results)}_examples.json'

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': config,
            'results': results,
            'summary': summary
        }, f, indent=2)

    return results_file

def load_llama_model():
    """Load CODI-LLaMA model"""
    print("="*80)
    print("Loading CODI-LLaMA from Local Checkpoint")
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
    checkpoint_path = os.path.join(llama_model_args.ckpt_dir, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    llama_model.load_state_dict(state_dict, strict=False)
    llama_model.codi.tie_weights()
    llama_model = llama_model.to(device)
    llama_model = llama_model.to(torch.bfloat16)
    llama_model.eval()

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    return llama_model, llama_tokenizer, llama_training_args


def run_extended_cot(model, tokenizer, training_args, question, max_iterations=50):
    """Run CoT for extended iterations and decode tokens at each position"""
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

    decoded_tokens = []

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
        latent_embd = outputs.hidden_states[-1][:, -1:, :]

        # Decode BoT
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id]).strip()
        is_number = bool(number_regex.match(token_str))

        decoded_tokens.append({
            'position': 0,
            'token': token_str,
            'is_number': is_number,
            'token_id': token_id
        })

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Extended Chain-of-Thought iterations
        for i in range(max_iterations):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1:, :]

            logits = model.codi.lm_head(latent_embd.squeeze(1))
            token_id = torch.argmax(logits, dim=-1).item()
            token_str = tokenizer.decode([token_id]).strip()
            is_number = bool(number_regex.match(token_str))

            decoded_tokens.append({
                'position': i + 1,
                'token': token_str,
                'is_number': is_number,
                'token_id': token_id
            })

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return past_key_values, decoded_tokens


def generate_answer(model, tokenizer, training_args, past_key_values, max_length=128):
    """Generate final answer using past_key_values from CoT"""
    batch_size = 1

    with torch.no_grad():
        if training_args.remove_eos:
            eot_tensor = torch.tensor([[model.eot_id]], dtype=torch.long).expand(batch_size, 1, 1).to(device)
        else:
            eot_tensor = torch.tensor([[tokenizer.eos_token_id, model.eot_id]],
                                       dtype=torch.long).expand(batch_size, 1, 2).to(device)

        eot_emb = model.get_embd(model.codi, model.model_name)(eot_tensor).squeeze(1)
        output = eot_emb

        pred_tokens = []

        for step in range(max_length):
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
            next_token_id = torch.argmax(logits, dim=-1).item()

            pred_tokens.append(next_token_id)

            current_token_str = tokenizer.decode([next_token_id])

            if next_token_id == tokenizer.eos_token_id:
                break

            if number_regex.match(current_token_str.strip()):
                break

            if step >= 49:
                break

            output = model.get_embd(model.codi, model.model_name)(
                torch.tensor([[next_token_id]], device=device)
            )

        full_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)

        # Extract numerical answer
        text = full_answer.replace(',', '')
        numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
        predicted_number = float(numbers[-1]) if numbers else None

        return full_answer, predicted_number


def find_answer_in_cot(decoded_tokens, ground_truth_answer):
    """Find positions where ground truth answer appears in decoded CoT tokens"""
    answer_positions = []

    # Convert ground truth to string
    if ground_truth_answer == int(ground_truth_answer):
        answer_str = str(int(ground_truth_answer))
    else:
        answer_str = str(ground_truth_answer)

    for token_info in decoded_tokens:
        token = token_info['token'].strip()

        # Direct match
        if token == answer_str:
            answer_positions.append(token_info['position'])

        # Check if token is a number matching the answer
        if token_info['is_number']:
            try:
                token_value = float(token)
                if token_value == ground_truth_answer:
                    answer_positions.append(token_info['position'])
            except ValueError:
                pass

    return answer_positions


def process_example(model, tokenizer, training_args, question, answer, max_cot_iterations=50):
    """Process a single example with extended CoT"""
    result = {
        'question': question,
        'ground_truth': answer,
        'max_cot_iterations': max_cot_iterations
    }

    # Run extended CoT
    past_kv, decoded_tokens = run_extended_cot(
        model, tokenizer, training_args, question, max_iterations=max_cot_iterations
    )

    # Generate final answer
    answer_text, predicted_answer = generate_answer(
        model, tokenizer, training_args, past_kv
    )

    # Find where ground truth appears in CoT
    answer_positions = find_answer_in_cot(decoded_tokens, answer)

    result['decoded_tokens'] = decoded_tokens
    result['answer_text'] = answer_text
    result['predicted_answer'] = predicted_answer
    result['correct'] = (predicted_answer == answer) if predicted_answer is not None else False
    result['answer_appears_in_cot'] = len(answer_positions) > 0
    result['answer_positions'] = answer_positions
    result['first_appearance'] = answer_positions[0] if answer_positions else None
    result['num_appearances'] = len(answer_positions)

    return result


def main():
    model, tokenizer, training_args = load_llama_model()

    print("\nLoading clean variant examples...")
    data_path = "/workspace/CoT_Exploration/src/experiments/activation_patching/data/llama_cot_all.json"

    with open(data_path, 'r') as f:
        all_data = json.load(f)

    # Filter for clean variants only
    dataset = [ex for ex in all_data if ex.get('variant') == 'clean']

    print(f"[OK] Loaded {len(dataset)} clean variant examples")

    # Test on first 5 examples
    num_test_examples = 5
    test_dataset = dataset[:num_test_examples]

    print(f"\nTesting on {num_test_examples} examples with extended CoT...")
    print("="*80)

    results = []
    MAX_COT_ITERATIONS = 50

    config = {
        'num_examples': num_test_examples,
        'max_cot_iterations': MAX_COT_ITERATIONS,
        'default_cot_iterations': 6
    }

    for idx, example in enumerate(test_dataset):
        question = example['question']
        answer = float(example['answer'])
        pair_id = example.get('pair_id', idx)

        print(f"\n{'='*80}")
        print(f"Example {idx+1}/{num_test_examples} (pair_id={pair_id})")
        print(f"Question: {question[:80]}...")
        print(f"Ground truth answer: {answer}")
        print(f"{'='*80}")

        try:
            result = process_example(
                model, tokenizer, training_args,
                question, answer,
                max_cot_iterations=MAX_COT_ITERATIONS
            )

            # Add metadata
            result['pair_id'] = pair_id
            result['variant'] = example.get('variant', 'clean')
            result['example_index'] = idx

            results.append(result)

            # Print summary for this example
            print(f"\nPredicted answer: {result['predicted_answer']}")
            print(f"Correct: {result['correct']}")
            print(f"Answer appears in CoT: {result['answer_appears_in_cot']}")
            if result['answer_appears_in_cot']:
                print(f"First appearance at position: {result['first_appearance']}")
                print(f"Total appearances: {result['num_appearances']}")
                print(f"All positions: {result['answer_positions']}")
            else:
                print(f"Ground truth answer NEVER appeared in extended CoT")

            # Show first 20 decoded tokens
            print(f"\nDecoded CoT tokens (first 20 of {len(result['decoded_tokens'])}):")
            for token_info in result['decoded_tokens'][:20]:
                marker = " <-- ANSWER" if token_info['position'] in result['answer_positions'] else ""
                num_flag = " [NUM]" if token_info['is_number'] else ""
                print(f"  Pos {token_info['position']:2d}: '{token_info['token']}'{num_flag}{marker}")

            if len(result['decoded_tokens']) > 20:
                print(f"  ... ({len(result['decoded_tokens'])-20} more positions)")

            # Save intermediate results after each example
            total = len(results)
            correct = sum(1 for r in results if r['correct'])
            answer_appeared = sum(1 for r in results if r['answer_appears_in_cot'])

            summary = {
                'total': total,
                'correct': correct,
                'accuracy': correct/total*100 if total > 0 else 0,
                'answer_appeared': answer_appeared,
                'appearance_rate': answer_appeared/total*100 if total > 0 else 0
            }

            saved_file = save_intermediate_results(results, config, summary)
            print(f"\n[SAVED] Intermediate results: {saved_file}")

        except Exception as e:
            print(f"\nError at example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    answer_appeared = sum(1 for r in results if r['answer_appears_in_cot'])

    print(f"Total examples tested: {total}")
    print(f"Correct predictions: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Answer appeared in CoT: {answer_appeared}/{total} ({answer_appeared/total*100:.1f}%)")

    if answer_appeared > 0:
        first_appearances = [r['first_appearance'] for r in results if r['first_appearance'] is not None]
        print(f"\nFirst appearance statistics:")
        print(f"  Mean position: {np.mean(first_appearances):.1f}")
        print(f"  Median position: {np.median(first_appearances):.1f}")
        print(f"  Min position: {min(first_appearances)}")
        print(f"  Max position: {max(first_appearances)}")

        print(f"\nPosition distribution:")
        for r in results:
            if r['answer_appears_in_cot']:
                print(f"  Pair {r['pair_id']}: First at pos {r['first_appearance']}, appeared {r['num_appearances']} times")

    # Save final results
    results_file = RESULTS_DIR / f"extended_cot_test_{num_test_examples}_examples_FINAL.json"
    with open(results_file, 'w') as f:
        json.dump({
            'config': config,
            'results': results,
            'summary': summary
        }, f, indent=2)

    print(f"\n[OK] Final results saved to {results_file}")
    print(f"\nExperiment complete!")


if __name__ == "__main__":
    main()
