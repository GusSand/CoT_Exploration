import torch
import sys
import re
import os
import json
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login

# Load environment variables and login to HuggingFace
load_dotenv()
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(token=hf_token)
    print("✓ Logged in to HuggingFace")

sys.path.insert(0, "/workspace/CoT_Exploration/codi")
from src.model import CODI, ModelArguments, TrainingArguments
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Number detection regex (from notebook)
number_regex = re.compile(r'^\s?\d+')

def extract_answer_number(text):
    """Extract numerical answer from generated text"""
    text = text.replace(',', '')
    numbers = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    if not numbers:
        return None
    return float(numbers[-1])

def load_llama_model():
    """Load CODI-LLaMA model (from notebook)"""
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

def run_cot_without_intervention(model, tokenizer, training_args, question):
    """Run CoT WITHOUT intervention (from notebook)"""
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
    number_positions = []

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
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))
        decoded_tokens.append({'position': 0, 'token': token_str, 'is_number': is_number})
        if is_number:
            number_positions.append(0)

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Chain-of-Thought iterations
        for i in range(training_args.inf_latent_iterations):
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
            token_str = tokenizer.decode([token_id])
            is_number = bool(number_regex.match(token_str))
            decoded_tokens.append({'position': i+1, 'token': token_str, 'is_number': is_number})
            if is_number:
                number_positions.append(i+1)

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return past_key_values, decoded_tokens, number_positions

def run_cot_with_causal_intervention(model, tokenizer, training_args, question, target_token='5', k=3):
    """Run CoT WITH CAUSAL intervention (from notebook)"""
    batch_size = 1
    questions = [question]

    # Get target embedding
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    embedding_layer = model.codi.get_input_embeddings()
    target_embd = embedding_layer(torch.tensor([target_token_id], device=device))

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
    intervened_positions = []

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

        # Check and intervene at BoT
        logits = model.codi.lm_head(latent_embd.squeeze(1))
        token_id = torch.argmax(logits, dim=-1).item()
        token_str = tokenizer.decode([token_id])
        is_number = bool(number_regex.match(token_str))

        if is_number:
            predicted_embd = embedding_layer(torch.tensor([token_id], device=device))
            A = latent_embd.squeeze(1)

            E_pred_norm = predicted_embd / torch.norm(predicted_embd, dim=-1, keepdim=True)
            E_target_norm = target_embd / torch.norm(target_embd, dim=-1, keepdim=True)

            proj_predicted = torch.sum(A * E_pred_norm, dim=-1, keepdim=True) * E_pred_norm
            proj_target = torch.norm(proj_predicted, dim=-1, keepdim=True) * E_target_norm
            A_modified = A - proj_predicted + k * proj_target

            # CAUSALLY apply intervention
            latent_embd = A_modified.unsqueeze(1)

            logits_modified = model.codi.lm_head(A_modified)
            new_token_id = torch.argmax(logits_modified, dim=-1).item()
            new_token_str = tokenizer.decode([new_token_id])
            decoded_tokens.append({'position': 0, 'token': new_token_str, 'is_number': True, 'intervened': True})
            intervened_positions.append(0)
        else:
            decoded_tokens.append({'position': 0, 'token': token_str, 'is_number': False, 'intervened': False})

        if training_args.use_prj:
            latent_embd = model.prj(latent_embd)

        # Chain-of-Thought iterations with causal intervention
        for i in range(training_args.inf_latent_iterations):
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
            token_str = tokenizer.decode([token_id])
            is_number = bool(number_regex.match(token_str))

            if is_number:
                predicted_embd = embedding_layer(torch.tensor([token_id], device=device))
                A = latent_embd.squeeze(1)

                E_pred_norm = predicted_embd / torch.norm(predicted_embd, dim=-1, keepdim=True)
                E_target_norm = target_embd / torch.norm(target_embd, dim=-1, keepdim=True)

                proj_predicted = torch.sum(A * E_pred_norm, dim=-1, keepdim=True) * E_pred_norm
                proj_target = torch.sum(A * E_target_norm, dim=-1, keepdim=True) * E_target_norm
                A_modified = A - proj_predicted + k * proj_target

                # CAUSALLY apply intervention
                latent_embd = A_modified.unsqueeze(1)

                logits_modified = model.codi.lm_head(A_modified)
                new_token_id = torch.argmax(logits_modified, dim=-1).item()
                new_token_str = tokenizer.decode([new_token_id])
                decoded_tokens.append({'position': i+1, 'token': new_token_str, 'is_number': True, 'intervened': True})
                intervened_positions.append(i+1)
            else:
                decoded_tokens.append({'position': i+1, 'token': token_str, 'is_number': False, 'intervened': False})

            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

    return past_key_values, decoded_tokens, intervened_positions

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
        predicted_number = extract_answer_number(full_answer)

        return full_answer, predicted_number

def process_example(model, tokenizer, training_args, question, answer, target_token='5', k=3):
    """Process a single GSM8K example with and without intervention"""
    result = {
        'question': question,
        'ground_truth': answer,
        'target_token': target_token,
        'k': k
    }

    # Run WITHOUT intervention
    past_kv_no_int, decoded_no_int, num_pos = run_cot_without_intervention(
        model, tokenizer, training_args, question
    )
    answer_no_int, pred_no_int = generate_answer(
        model, tokenizer, training_args, past_kv_no_int
    )

    result['decoded_no_intervention'] = decoded_no_int
    result['number_positions'] = num_pos
    result['answer_no_intervention'] = answer_no_int
    result['predicted_no_intervention'] = pred_no_int
    result['correct_no_intervention'] = (pred_no_int == answer) if pred_no_int is not None else False

    # Run WITH intervention
    if len(num_pos) > 0:
        past_kv_with_int, decoded_with_int, int_pos = run_cot_with_causal_intervention(
            model, tokenizer, training_args, question, target_token, k
        )
        answer_with_int, pred_with_int = generate_answer(
            model, tokenizer, training_args, past_kv_with_int
        )

        result['decoded_with_intervention'] = decoded_with_int
        result['intervened_positions'] = int_pos
        result['answer_with_intervention'] = answer_with_int
        result['predicted_with_intervention'] = pred_with_int
        result['correct_with_intervention'] = (pred_with_int == answer) if pred_with_int is not None else False
    else:
        result['decoded_with_intervention'] = None
        result['intervened_positions'] = []
        result['answer_with_intervention'] = None
        result['predicted_with_intervention'] = None
        result['correct_with_intervention'] = False

    return result

def main():
    model, tokenizer, training_args = load_llama_model()

    print("\nLoading GSM8K test set...")
    dataset = load_dataset("gsm8k", "main", split="test")
    dataset = dataset.select(range(50))
    print(f"✓ Loaded {len(dataset)} examples (limited to 50)")

    results = []
    correct_no_int = 0
    correct_with_int = 0
    total = 0

    print("\nProcessing examples...")
    for idx, example in enumerate(tqdm(dataset)):
        question = example['question']
        answer_str = example['answer']
        answer = extract_answer_number(answer_str)

        try:
            result = process_example(model, tokenizer, training_args, question, answer)
            results.append(result)

            if result['correct_no_intervention']:
                correct_no_int += 1
            if result['correct_with_intervention']:
                correct_with_int += 1
            total += 1

            if (idx + 1) % 10 == 0:
                acc_no = correct_no_int / total * 100
                acc_with = correct_with_int / total * 100
                print(f"\n[{idx+1}/{len(dataset)}] Acc NO_INT={acc_no:.1f}%, WITH_INT={acc_with:.1f}%")

        except Exception as e:
            print(f"\nError at example {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    acc_no_int = correct_no_int / total * 100
    acc_with_int = correct_with_int / total * 100

    print("\n" + "="*80)
    print("Final Results:")
    print("="*80)
    print(f"Total examples: {total}")
    print(f"Accuracy WITHOUT intervention: {acc_no_int:.2f}% ({correct_no_int}/{total})")
    print(f"Accuracy WITH intervention: {acc_with_int:.2f}% ({correct_with_int}/{total})")
    print(f"Difference: {acc_with_int - acc_no_int:.2f}%")

    # Check how many answers changed
    answers_changed = sum(1 for r in results if r['predicted_with_intervention'] is not None and
                         r['predicted_no_intervention'] != r['predicted_with_intervention'])
    print(f"\nAnswers that changed due to intervention: {answers_changed}/{total}")

    output_dir = Path("/workspace/CoT_Exploration/src/experiments/28-10-2028-projection-replacement/results")
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "llama_intervention_results_50_correct.json"
    with open(results_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'total': total,
                'correct_no_intervention': correct_no_int,
                'correct_with_intervention': correct_with_int,
                'accuracy_no_intervention': acc_no_int,
                'accuracy_with_intervention': acc_with_int,
                'answers_changed': answers_changed
            }
        }, f, indent=2)

    print(f"\n✓ Results saved to {results_file}")

if __name__ == "__main__":
    main()
