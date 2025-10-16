"""
Section 5 Analysis Script for CODI Paper Reproduction
======================================================

This script extends the original probe_latent_token.py to:
1. Save model outputs separately for correct/incorrect predictions
2. Analyze how often decoded outputs correspond to correct intermediate computation steps
3. Provide detailed interpretability analysis with visualizations
4. Export results in structured JSON/CSV formats

Based on: CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation
Section 5: Further Analysis - Interpretability Analysis
"""

import logging
import math
import re
import os
import json
import csv
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, List, Tuple
from pathlib import Path
from datetime import datetime

import torch
import transformers
from torch.nn import functional as F

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from accelerate.utils import set_seed
from safetensors.torch import load_file

import numpy as np

from src.model import (
    CODI,
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# Configuration
PROBE_TOPK = 10  # Increased from 5 to get more decoding options
SAVE_ATTENTION = True  # Whether to extract and save attention patterns
VERBOSE_LOGGING = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class PredictionOutput:
    """Structured output for each prediction"""
    question_id: int
    question_text: str
    reference_cot: str
    ground_truth_answer: float
    predicted_answer: float
    is_correct: bool
    decoded_tokens: List[str]
    decoded_text: str

    # Continuous thought interpretability
    continuous_thoughts: List[Dict]  # Each dict contains decoded tokens, attention
    num_continuous_thoughts: int

    # Intermediate computation analysis
    reference_steps: List[str]  # Extracted from CoT
    decoded_steps: List[str]    # Decoded from continuous thoughts
    step_correctness: List[bool]  # Whether each step matches reference
    overall_step_accuracy: float

    # Attention patterns
    attention_to_question_tokens: Optional[List[float]] = None
    attention_to_previous_thoughts: Optional[List[float]] = None


def extract_answer_number(sentence: str) -> float:
    """Extract numerical answer from generated text"""
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    return float(pred[-1])


def extract_intermediate_steps(cot_text: str) -> List[str]:
    """
    Extract intermediate computation steps from CoT text.

    For GSM8K-Aug format: «10÷5=2»«2×2=4»«6×4=24»
    For GSM8K-Aug-NL format: "20% + 30% = 50%. So, the remaining ... is 100% - 50% = 50%."
    """
    steps = []

    # Try structured format first (GSM8K-Aug)
    structured_steps = re.findall(r'«([^»]+)»|<<([^>]+)>>', cot_text)
    if structured_steps:
        for match in structured_steps:
            step = match[0] if match[0] else match[1]
            steps.append(step.strip())
        return steps

    # Try natural language format (GSM8K-Aug-NL)
    # Look for equations like "X = Y" or "X + Y = Z"
    nl_steps = re.findall(r'[\d\.\+\-\*/\(\)]+\s*=\s*[\d\.]+', cot_text)
    if nl_steps:
        return [s.strip() for s in nl_steps]

    return steps


def extract_intermediate_results(cot_text: str) -> List[float]:
    """Extract numerical results from intermediate steps"""
    steps = extract_intermediate_steps(cot_text)
    results = []

    for step in steps:
        # Extract the result after '='
        if '=' in step:
            result_str = step.split('=')[-1].strip()
            try:
                results.append(float(result_str.replace(',', '')))
            except ValueError:
                continue

    return results


def decode_continuous_thought(
    hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_idx: int,
    topk: int = 10
) -> Dict:
    """
    Decode a continuous thought token to vocabulary space for a specific batch item.

    Args:
        hidden_state: Hidden state tensor [batch_size, seq_len, hidden_dim]
        lm_head: Language model head
        tokenizer: Tokenizer for decoding
        batch_idx: Index of the batch item to decode
        topk: Number of top tokens to return

    Returns:
        Dict with 'topk_tokens', 'topk_probs', 'topk_decoded'
    """
    with torch.no_grad():
        logits = lm_head(hidden_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)

        # Decode indices to tokens for the specific batch item
        topk_decoded = []
        for idx in topk_indices[batch_idx, 0]:
            token = tokenizer.decode([idx.item()])
            topk_decoded.append(token)

        return {
            'topk_indices': topk_indices[batch_idx, 0].cpu().tolist(),
            'topk_probs': topk_probs[batch_idx, 0].cpu().tolist(),
            'topk_decoded': topk_decoded
        }


def validate_intermediate_computation(
    decoded_steps: List[List[str]],  # Now expects list of top-K token lists
    reference_steps: List[str],
    tolerance: float = 0.01
) -> Tuple[List[bool], float]:
    """
    Validate if decoded intermediate results match reference CoT.
    Following paper methodology: check if reference value appears in top-5 decoded tokens.

    Args:
        decoded_steps: List of top-K decoded token lists for each step
        reference_steps: List of reference CoT steps
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        Tuple of (step_correctness_list, overall_accuracy)
    """
    if not reference_steps:
        return [], 0.0

    reference_results = extract_intermediate_results(''.join(f'«{s}»' for s in reference_steps))

    # Compare results
    correctness = []
    for i, (ref, topk_tokens) in enumerate(zip(reference_results, decoded_steps)):
        # Check if reference value appears in any of the top-K decoded tokens
        found_match = False
        for token in topk_tokens:
            numbers = re.findall(r'-?\d+\.?\d*', token)
            if numbers:
                try:
                    decoded_val = float(numbers[-1])
                    if abs(ref - decoded_val) < tolerance or (ref != 0 and abs((ref - decoded_val) / ref) < tolerance):
                        found_match = True
                        break
                except ValueError:
                    continue
        correctness.append(found_match)

    # Overall accuracy
    if correctness:
        accuracy = sum(correctness) / len(correctness)
    else:
        accuracy = 0.0

    return correctness, accuracy


def evaluation_section5(model_args, data_args, training_args, output_dir: str):
    """
    Main evaluation function for Section 5 analysis.

    Generates:
    - correct_predictions.json: All correctly predicted samples with full analysis
    - incorrect_predictions.json: All incorrectly predicted samples
    - summary_statistics.json: Aggregate statistics
    - interpretability_analysis.csv: Per-sample analysis for easy inspection
    """

    # Create output directory structure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"section5_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    correct_dir = run_dir / "correct_predictions"
    incorrect_dir = run_dir / "incorrect_predictions"
    correct_dir.mkdir()
    incorrect_dir.mkdir()

    print(f"Output directory: {run_dir}")

    # Load model
    print("Loading model...")
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
    else:
        raise NotImplementedError

    model = CODI(model_args, training_args, lora_config)

    try:
        state_dict = load_file(os.path.join(model_args.ckpt_dir, "model.safetensors"))
    except Exception:
        state_dict = torch.load(os.path.join(model_args.ckpt_dir, "pytorch_model.bin"))

    model.load_state_dict(state_dict, strict=False)
    model.codi.tie_weights()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    model = model.to('cuda')
    model.to(torch.bfloat16)

    # Load dataset
    print("Loading dataset...")
    question_name = "question"
    answer_name = "answer"
    cot_name = "cot"

    if "zen-E/GSM8k-Aug" in data_args.data_name:
        dataset = load_dataset(data_args.data_name)
        test_set = dataset['test']
    elif "gsm8k" == data_args.data_name:
        dataset = load_dataset("gsm8k", "main")
        test_set = dataset['test']
        cot_name = "answer"  # In standard GSM8K, CoT is in answer field
    else:
        raise NotImplementedError(f"Dataset {data_args.data_name} not supported")

    questions = []
    answers = []
    cots = []

    for example in test_set:
        questions.append(example[question_name].strip().replace('  ', ' '))

        # Extract answer
        answer_text = example[answer_name]
        if "####" in answer_text:
            ans = answer_text.split('####')[-1]
        else:
            ans = answer_text
        ans = ans.replace(',', '')
        try:
            answers.append(float(ans))
        except ValueError:
            answers.append(float('inf'))

        # Extract CoT
        if cot_name in example:
            cots.append(example[cot_name])
        else:
            cots.append("")

    print(f"Total examples: {len(questions)}")

    # Tokenize questions
    eval_step = math.ceil(len(questions) / data_args.batch_size)
    question_data = []

    for i in range(eval_step):
        if i < eval_step - 1:
            batch_questions = questions[i*data_args.batch_size: (i+1)*data_args.batch_size]
        else:
            batch_questions = questions[i*data_args.batch_size:]

        batch = tokenizer(
            batch_questions,
            return_tensors="pt",
            padding="longest",
        )

        if training_args.remove_eos:
            bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
        else:
            bot_tensor = torch.tensor([tokenizer.eos_token_id, model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 2)

        batch["input_ids"] = torch.cat((batch["input_ids"], bot_tensor), dim=1)
        batch["attention_mask"] = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch.to(device))

    # Evaluation
    model.eval()
    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1 if not training_args.greedy else 1.0,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": not training_args.greedy,
    }

    all_predictions = []
    correct_predictions = []
    incorrect_predictions = []

    question_idx = 0

    print("Running evaluation...")
    for step, batch in enumerate(question_data):
        batch_size = batch["input_ids"].size(0)

        with torch.no_grad():
            # Encode question
            outputs = model.codi(
                input_ids=batch["input_ids"],
                use_cache=True,
                output_hidden_states=True,
                past_key_values=None,
                attention_mask=batch["attention_mask"]
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            # Store continuous thoughts for each sample in batch
            batch_continuous_thoughts = [[] for _ in range(batch_size)]

            # Decode initial thought (before any continuous iterations)
            for b in range(batch_size):
                decoded_initial = decode_continuous_thought(
                    latent_embd,
                    model.codi.lm_head,
                    tokenizer,
                    batch_idx=b,
                    topk=PROBE_TOPK
                )
                batch_continuous_thoughts[b].append({
                    'iteration': 0,
                    'type': 'initial',
                    **decoded_initial
                })

            # Apply projection if enabled
            if training_args.use_prj:
                latent_embd = model.prj(latent_embd)

            # Iterate through continuous thoughts
            inf_latent_iterations = training_args.inf_latent_iterations
            for i in range(inf_latent_iterations):
                outputs = model.codi(
                    inputs_embeds=latent_embd,
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=past_key_values
                )
                past_key_values = outputs.past_key_values
                latent_embd_pre_proj = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                # Decode before projection for each batch item
                for b in range(batch_size):
                    decoded = decode_continuous_thought(
                        latent_embd_pre_proj,
                        model.codi.lm_head,
                        tokenizer,
                        batch_idx=b,
                        topk=PROBE_TOPK
                    )
                    batch_continuous_thoughts[b].append({
                        'iteration': i + 1,
                        'type': 'continuous_thought',
                        **decoded
                    })

                # Apply projection for next iteration
                if training_args.use_prj:
                    latent_embd = model.prj(latent_embd_pre_proj)
                else:
                    latent_embd = latent_embd_pre_proj

            # Generate answer
            if training_args.remove_eos:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)
            else:
                eot_emb = model.get_embd(model.codi, model.model_name)(
                    torch.tensor([model.eot_id, tokenizer.eos_token_id], dtype=torch.long, device='cuda')
                ).unsqueeze(0).to(device)

            eot_emb = eot_emb.expand(batch_size, -1, -1)
            output = eot_emb

            finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
            pred_tokens = [[] for _ in range(batch_size)]

            for gen_i in range(gen_kwargs["max_new_tokens"]):
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

                if training_args.greedy:
                    next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    logits /= gen_kwargs["temperature"]
                    if gen_kwargs["top_k"] > 1:
                        top_k_values, _ = torch.topk(logits, gen_kwargs["top_k"], dim=-1)
                        min_top_k_value = top_k_values[:, -1].unsqueeze(-1)
                        logits[logits < min_top_k_value] = -float("inf")

                    if gen_kwargs["top_p"] < 1.0:
                        sorted_logit, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logit, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > gen_kwargs["top_p"]
                        if sorted_indices_to_remove.any():
                            sorted_indices_to_remove = sorted_indices_to_remove.roll(1, dims=-1)
                            sorted_indices_to_remove[:, 0] = False
                            for b in range(logits.size(0)):
                                logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = -float("inf")

                    probs = F.softmax(logits, dim=-1)
                    next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

                for b in range(batch_size):
                    if not finished[b]:
                        pred_tokens[b].append(next_token_ids[b].item())
                        if next_token_ids[b] == tokenizer.eos_token_id:
                            finished[b] = True

                if finished.all():
                    break

                output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1).to(device)

            # Process predictions for this batch
            for b in range(batch_size):
                q_idx = question_idx + b
                if q_idx >= len(questions):
                    break

                decoded_text = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                predicted_answer = extract_answer_number(decoded_text)
                ground_truth = answers[q_idx]
                is_correct = (predicted_answer == ground_truth)

                # Extract intermediate steps from CoT
                reference_steps = extract_intermediate_steps(cots[q_idx])

                # Extract decoded steps from continuous thoughts
                # Following paper methodology: use only every other thought (even iterations)
                # Skip thought 0 (initial) and odd iterations, use only even iterations (2, 4, 6)
                # Use top-5 decoded tokens per step (as in paper's Table 3)
                decoded_steps = []
                for thought in batch_continuous_thoughts[b]:
                    # Use only even-numbered iterations (actual computation steps)
                    # Skip iteration 0 (initial) and odd iterations (transitional states)
                    if thought['iteration'] > 0 and thought['iteration'] % 2 == 0:
                        if thought['topk_decoded']:
                            # Use top-5 tokens as in the paper
                            decoded_steps.append(thought['topk_decoded'][:5])

                # Validate intermediate computations
                step_correctness, step_accuracy = validate_intermediate_computation(
                    decoded_steps,
                    reference_steps
                )

                # Create structured output
                prediction = PredictionOutput(
                    question_id=q_idx,
                    question_text=questions[q_idx],
                    reference_cot=cots[q_idx],
                    ground_truth_answer=ground_truth,
                    predicted_answer=predicted_answer,
                    is_correct=is_correct,
                    decoded_tokens=pred_tokens[b],
                    decoded_text=decoded_text,
                    continuous_thoughts=batch_continuous_thoughts[b],
                    num_continuous_thoughts=len(batch_continuous_thoughts[b]),
                    reference_steps=reference_steps,
                    decoded_steps=decoded_steps,
                    step_correctness=step_correctness,
                    overall_step_accuracy=step_accuracy
                )

                all_predictions.append(prediction)

                if is_correct:
                    correct_predictions.append(prediction)
                else:
                    incorrect_predictions.append(prediction)

                # Log progress
                if VERBOSE_LOGGING and q_idx % 100 == 0:
                    print(f"Processed {q_idx}/{len(questions)} questions")

        question_idx += batch_size

    # Save outputs
    print("\nSaving outputs...")

    # Save correct predictions
    with open(correct_dir / "predictions.json", "w") as f:
        json.dump([asdict(p) for p in correct_predictions], f, indent=2)

    # Save incorrect predictions
    with open(incorrect_dir / "predictions.json", "w") as f:
        json.dump([asdict(p) for p in incorrect_predictions], f, indent=2)

    # Calculate statistics
    total = len(all_predictions)
    num_correct = len(correct_predictions)
    num_incorrect = len(incorrect_predictions)
    accuracy = num_correct / total if total > 0 else 0.0

    # Analyze step correctness by number of steps
    step_analysis = {}
    for num_steps in range(1, 6):  # Analyze problems with 1-5 steps
        relevant_preds = [p for p in correct_predictions if len(p.reference_steps) == num_steps]
        if relevant_preds:
            avg_step_acc = np.mean([p.overall_step_accuracy for p in relevant_preds])
            step_analysis[f"{num_steps}_steps"] = {
                "count": len(relevant_preds),
                "avg_step_accuracy": float(avg_step_acc)
            }

    # Summary statistics
    summary = {
        "experiment_info": {
            "timestamp": timestamp,
            "model": model_args.model_name_or_path,
            "checkpoint": model_args.ckpt_dir,
            "dataset": data_args.data_name,
            "num_continuous_thoughts": training_args.inf_latent_iterations,
            "use_projection": training_args.use_prj,
            "greedy_decoding": training_args.greedy
        },
        "overall_results": {
            "total_examples": total,
            "correct_predictions": num_correct,
            "incorrect_predictions": num_incorrect,
            "accuracy": accuracy
        },
        "step_correctness_analysis": step_analysis,
        "output_locations": {
            "correct_predictions": str(correct_dir / "predictions.json"),
            "incorrect_predictions": str(incorrect_dir / "predictions.json")
        }
    }

    with open(run_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create CSV for easy inspection
    csv_data = []
    for p in all_predictions:
        csv_data.append({
            'question_id': p.question_id,
            'is_correct': p.is_correct,
            'ground_truth': p.ground_truth_answer,
            'predicted': p.predicted_answer,
            'num_reference_steps': len(p.reference_steps),
            'num_decoded_steps': len(p.decoded_steps),
            'step_accuracy': p.overall_step_accuracy,
            'top1_continuous_thought_0': p.continuous_thoughts[1]['topk_decoded'][0] if len(p.continuous_thoughts) > 1 else '',
            'top1_continuous_thought_1': p.continuous_thoughts[2]['topk_decoded'][0] if len(p.continuous_thoughts) > 2 else '',
            'top1_continuous_thought_2': p.continuous_thoughts[3]['topk_decoded'][0] if len(p.continuous_thoughts) > 3 else '',
        })

    with open(run_dir / "interpretability_analysis.csv", "w", newline='') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"\n{'='*60}")
    print(f"Section 5 Analysis Complete!")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {num_correct}/{total}")
    print(f"Incorrect: {num_incorrect}/{total}")
    print(f"\nStep Correctness Analysis (for correct predictions):")
    for key, val in step_analysis.items():
        print(f"  {key}: {val['avg_step_accuracy']*100:.1f}% (n={val['count']})")
    print(f"\nOutputs saved to: {run_dir}")
    print(f"{'='*60}\n")

    return accuracy, summary


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = "outputs/section5_analysis"
    accuracy, summary = evaluation_section5(model_args, data_args, training_args, output_dir)
