"""
Section 5 Analysis Script - EXTENDED WITH SIMILARITY METRICS
============================================================

This script extends section5_analysis.py to add geometric similarity metrics:
1. Cosine similarity between activations and token embeddings
2. Normalized Euclidean distance (L2 distance after normalizing to unit length)

Motivation: Decoding probability shows which token is most likely, but doesn't
quantify geometric similarity. An activation can decode to token X (high probability)
while being geometrically distant from X's embedding.

Key metrics:
- Decoding probability: P(token|activation) via softmax
- Cosine similarity: cos(θ) = dot(activation, token_emb) / (||activation|| * ||token_emb||)
- Normalized L2 distance: ||activation_norm - token_emb_norm||_2 where both are unit vectors

Mathematical relationship: norm_l2_dist = sqrt(2 - 2*cos_sim)
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
PROBE_TOPK = 10
SAVE_ATTENTION = True
VERBOSE_LOGGING = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@dataclass
class PredictionOutput:
    """Structured output for each prediction with extended similarity metrics"""
    question_id: int
    question_text: str
    reference_cot: str
    ground_truth_answer: float
    predicted_answer: float
    is_correct: bool
    decoded_tokens: List[str]
    decoded_text: str

    # Continuous thought interpretability
    continuous_thoughts: List[Dict]  # Now includes cosine_sim and norm_l2_dist
    num_continuous_thoughts: int

    # Intermediate computation analysis
    reference_steps: List[str]
    decoded_steps: List[str]
    step_correctness: List[bool]
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
    """Extract intermediate computation steps from CoT text"""
    steps = []

    # Try structured format first (GSM8K-Aug)
    structured_steps = re.findall(r'«([^»]+)»|<<([^>]+)>>', cot_text)
    if structured_steps:
        for match in structured_steps:
            step = match[0] if match[0] else match[1]
            steps.append(step.strip())
        return steps

    # Try natural language format (GSM8K-Aug-NL)
    nl_steps = re.findall(r'[\d\.\+\-\*/\(\)]+\s*=\s*[\d\.]+', cot_text)
    if nl_steps:
        return [s.strip() for s in nl_steps]

    return steps


def extract_intermediate_results(cot_text: str) -> List[float]:
    """Extract numerical results from intermediate steps"""
    steps = extract_intermediate_steps(cot_text)
    results = []

    for step in steps:
        if '=' in step:
            result_str = step.split('=')[-1].strip()
            try:
                results.append(float(result_str.replace(',', '')))
            except ValueError:
                continue

    return results


def compute_similarity_metrics(
    activation: torch.Tensor,  # [hidden_dim]
    token_embeddings: torch.Tensor,  # [topk, hidden_dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute cosine similarity and normalized Euclidean distance.

    Args:
        activation: Continuous activation vector [hidden_dim]
        token_embeddings: Top-K token embeddings [topk, hidden_dim]

    Returns:
        cosine_similarities: [topk] - cosine similarity for each token
        norm_l2_distances: [topk] - normalized L2 distance for each token
    """
    # Normalize activation to unit length
    activation_norm = F.normalize(activation.unsqueeze(0), p=2, dim=1)  # [1, hidden_dim]

    # Normalize token embeddings to unit length
    token_embeddings_norm = F.normalize(token_embeddings, p=2, dim=1)  # [topk, hidden_dim]

    # Cosine similarity = dot product of normalized vectors
    cosine_similarities = torch.mm(activation_norm, token_embeddings_norm.t()).squeeze(0)  # [topk]

    # Normalized Euclidean distance
    # ||a_norm - b_norm||_2 = sqrt(2 - 2*cos_sim) (but compute directly for numerical stability)
    differences = activation_norm - token_embeddings_norm  # [topk, hidden_dim]
    norm_l2_distances = torch.norm(differences, p=2, dim=1)  # [topk]

    return cosine_similarities, norm_l2_distances


def decode_continuous_thought_extended(
    hidden_state: torch.Tensor,
    lm_head: torch.nn.Module,
    embedding_layer: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    batch_idx: int,
    topk: int = 10
) -> Dict:
    """
    Decode continuous thought with extended similarity metrics.

    Args:
        hidden_state: Hidden state tensor [batch_size, seq_len, hidden_dim]
        lm_head: Language model head
        embedding_layer: Token embedding layer (to get token embeddings)
        tokenizer: Tokenizer for decoding
        batch_idx: Index of the batch item to decode
        topk: Number of top tokens to return

    Returns:
        Dict with 'topk_indices', 'topk_probs', 'topk_decoded',
        'cosine_similarities', 'norm_l2_distances'
    """
    with torch.no_grad():
        # Get logits and probabilities
        logits = lm_head(hidden_state)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=topk, dim=-1)

        # Get activation for this batch item
        activation = hidden_state[batch_idx, 0, :]  # [hidden_dim]

        # Get token embeddings for top-K tokens
        topk_token_ids = topk_indices[batch_idx, 0]  # [topk]
        token_embeddings = embedding_layer(topk_token_ids)  # [topk, hidden_dim]

        # Compute similarity metrics
        cosine_sims, norm_l2_dists = compute_similarity_metrics(activation, token_embeddings)

        # Decode indices to tokens
        topk_decoded = []
        for idx in topk_indices[batch_idx, 0]:
            token = tokenizer.decode([idx.item()])
            topk_decoded.append(token)

        return {
            'topk_indices': topk_indices[batch_idx, 0].cpu().tolist(),
            'topk_probs': topk_probs[batch_idx, 0].cpu().tolist(),
            'topk_decoded': topk_decoded,
            'cosine_similarities': cosine_sims.cpu().tolist(),
            'norm_l2_distances': norm_l2_dists.cpu().tolist(),
        }


def validate_intermediate_computation(
    decoded_steps: List[List[str]],
    reference_steps: List[str],
    tolerance: float = 0.01
) -> Tuple[List[bool], float]:
    """Validate if decoded intermediate results match reference CoT"""
    if not reference_steps:
        return [], 0.0

    reference_results = extract_intermediate_results(''.join(f'«{s}»' for s in reference_steps))

    correctness = []
    for i, (ref, topk_tokens) in enumerate(zip(reference_results, decoded_steps)):
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

    accuracy = sum(correctness) / len(correctness) if correctness else 0.0
    return correctness, accuracy


def evaluation_section5_extended(model_args, data_args, training_args, output_dir: str):
    """
    Main evaluation function for Section 5 analysis with extended similarity metrics.
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_path / f"section5_extended_{timestamp}"
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

    # Get embedding layer
    embedding_layer = model.get_embd(model.codi, model.model_name)

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
        cot_name = "answer"
    else:
        raise NotImplementedError(f"Dataset {data_args.data_name} not supported")
    
    # Limit to first N examples if needed
    num_examples = int(os.getenv("NUM_EXAMPLES", "99999"))
    if num_examples < len(test_set):
        test_set = test_set.select(range(num_examples))

    questions = []
    answers = []
    cots = []

    for example in test_set:
        questions.append(example[question_name].strip().replace('  ', ' '))

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

    print("Running evaluation with extended similarity metrics...")
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

            batch_continuous_thoughts = [[] for _ in range(batch_size)]

            # Decode initial thought with extended metrics
            for b in range(batch_size):
                decoded_initial = decode_continuous_thought_extended(
                    latent_embd,
                    model.codi.lm_head,
                    embedding_layer,
                    tokenizer,
                    batch_idx=b,
                    topk=PROBE_TOPK
                )
                batch_continuous_thoughts[b].append({
                    'iteration': 0,
                    'type': 'initial',
                    **decoded_initial
                })

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

                # Decode with extended metrics
                for b in range(batch_size):
                    decoded = decode_continuous_thought_extended(
                        latent_embd_pre_proj,
                        model.codi.lm_head,
                        embedding_layer,
                        tokenizer,
                        batch_idx=b,
                        topk=PROBE_TOPK
                    )
                    batch_continuous_thoughts[b].append({
                        'iteration': i + 1,
                        'type': 'continuous_thought',
                        **decoded
                    })

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

            # Process predictions
            for b in range(batch_size):
                q_idx = question_idx + b
                if q_idx >= len(questions):
                    break

                decoded_text = tokenizer.decode(pred_tokens[b], skip_special_tokens=True)
                predicted_answer = extract_answer_number(decoded_text)
                ground_truth = answers[q_idx]
                is_correct = (predicted_answer == ground_truth)

                reference_steps = extract_intermediate_steps(cots[q_idx])

                decoded_steps = []
                for thought in batch_continuous_thoughts[b]:
                    if thought['iteration'] % 2 == 0:
                        if thought['topk_decoded']:
                            decoded_steps.append(thought['topk_decoded'][:5])

                step_correctness, step_accuracy = validate_intermediate_computation(
                    decoded_steps,
                    reference_steps
                )

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

                if VERBOSE_LOGGING and q_idx % 100 == 0:
                    print(f"Processed {q_idx}/{len(questions)} questions")

        question_idx += batch_size

    # Save outputs
    print("\nSaving outputs...")

    with open(correct_dir / "predictions.json", "w") as f:
        json.dump([asdict(p) for p in correct_predictions], f, indent=2)

    with open(incorrect_dir / "predictions.json", "w") as f:
        json.dump([asdict(p) for p in incorrect_predictions], f, indent=2)

    # Calculate statistics
    total = len(all_predictions)
    num_correct = len(correct_predictions)
    num_incorrect = len(incorrect_predictions)
    accuracy = num_correct / total if total > 0 else 0.0

    step_analysis = {}
    for num_steps in range(1, 6):
        relevant_preds = [p for p in correct_predictions if len(p.reference_steps) == num_steps]
        if relevant_preds:
            avg_step_acc = np.mean([p.overall_step_accuracy for p in relevant_preds])
            step_analysis[f"{num_steps}_steps"] = {
                "count": len(relevant_preds),
                "avg_step_accuracy": float(avg_step_acc)
            }

    # Analyze similarity metrics
    similarity_analysis = analyze_similarity_metrics(all_predictions)

    summary = {
        "experiment_info": {
            "timestamp": timestamp,
            "model": model_args.model_name_or_path,
            "checkpoint": model_args.ckpt_dir,
            "dataset": data_args.data_name,
            "num_continuous_thoughts": training_args.inf_latent_iterations,
            "use_projection": training_args.use_prj,
            "greedy_decoding": training_args.greedy,
            "extended_metrics": ["cosine_similarity", "normalized_l2_distance"]
        },
        "overall_results": {
            "total_examples": total,
            "correct_predictions": num_correct,
            "incorrect_predictions": num_incorrect,
            "accuracy": accuracy
        },
        "step_correctness_analysis": step_analysis,
        "similarity_analysis": similarity_analysis,
        "output_locations": {
            "correct_predictions": str(correct_dir / "predictions.json"),
            "incorrect_predictions": str(incorrect_dir / "predictions.json")
        }
    }

    with open(run_dir / "summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Create enhanced CSV
    csv_data = []
    for p in all_predictions:
        # Get average similarity metrics for top-1 decoded token across thoughts
        avg_cosine_sim = np.mean([thought['cosine_similarities'][0] for thought in p.continuous_thoughts if 'cosine_similarities' in thought])
        avg_norm_l2 = np.mean([thought['norm_l2_distances'][0] for thought in p.continuous_thoughts if 'norm_l2_distances' in thought])

        csv_data.append({
            'question_id': p.question_id,
            'is_correct': p.is_correct,
            'ground_truth': p.ground_truth_answer,
            'predicted': p.predicted_answer,
            'step_accuracy': p.overall_step_accuracy,
            'avg_cosine_sim_top1': avg_cosine_sim,
            'avg_norm_l2_dist_top1': avg_norm_l2,
            'top1_continuous_thought_0': p.continuous_thoughts[1]['topk_decoded'][0] if len(p.continuous_thoughts) > 1 else '',
        })

    with open(run_dir / "interpretability_analysis_extended.csv", "w", newline='') as f:
        if csv_data:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)

    print(f"\n{'='*60}")
    print(f"Section 5 Extended Analysis Complete!")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy*100:.2f}%")
    print(f"Correct: {num_correct}/{total}")
    print(f"\nSimilarity Metrics Summary:")
    print(f"  Avg Cosine Similarity (top-1): {similarity_analysis['avg_cosine_sim_top1']:.4f}")
    print(f"  Avg Normalized L2 Distance (top-1): {similarity_analysis['avg_norm_l2_dist_top1']:.4f}")
    print(f"\nOutputs saved to: {run_dir}")
    print(f"{'='*60}\n")

    return accuracy, summary


def analyze_similarity_metrics(predictions: List[PredictionOutput]) -> Dict:
    """Analyze similarity metrics across all predictions"""
    all_cosine_sims = []
    all_norm_l2_dists = []

    for pred in predictions:
        for thought in pred.continuous_thoughts:
            if 'cosine_similarities' in thought and thought['cosine_similarities']:
                all_cosine_sims.extend(thought['cosine_similarities'])
            if 'norm_l2_distances' in thought and thought['norm_l2_distances']:
                all_norm_l2_dists.extend(thought['norm_l2_distances'])

    return {
        "avg_cosine_sim_all": float(np.mean(all_cosine_sims)) if all_cosine_sims else 0.0,
        "std_cosine_sim_all": float(np.std(all_cosine_sims)) if all_cosine_sims else 0.0,
        "avg_cosine_sim_top1": float(np.mean([sims[0] for pred in predictions for thought in pred.continuous_thoughts if 'cosine_similarities' in thought and thought['cosine_similarities'] for sims in [thought['cosine_similarities']]])) if predictions else 0.0,
        "avg_norm_l2_dist_all": float(np.mean(all_norm_l2_dists)) if all_norm_l2_dists else 0.0,
        "std_norm_l2_dist_all": float(np.std(all_norm_l2_dists)) if all_norm_l2_dists else 0.0,
        "avg_norm_l2_dist_top1": float(np.mean([dists[0] for pred in predictions for thought in pred.continuous_thoughts if 'norm_l2_distances' in thought and thought['norm_l2_distances'] for dists in [thought['norm_l2_distances']]])) if predictions else 0.0,
    }


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    output_dir = "outputs/section5_analysis_extended"
    accuracy, summary = evaluation_section5_extended(model_args, data_args, training_args, output_dir)
