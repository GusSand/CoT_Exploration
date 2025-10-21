"""
Manual Testing Script for CODI
Allows you to test individual problems one at a time.

Usage:
    python manual_test.py

Then paste your question when prompted.
"""

import sys
from pathlib import Path
import torch
import os
from safetensors.torch import load_file

# Add CODI to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "codi"))

from src.model import CODI, ModelArguments, TrainingArguments
from transformers import AutoTokenizer
from peft import LoraConfig

def extract_answer_number(sentence: str):
    """Extract answer from model output (same as evaluation code)."""
    import re
    sentence = sentence.replace(',', '')

    # Try to find "The answer is: X" pattern
    if "The answer is:" in sentence:
        pred = sentence.split("The answer is:")[-1].strip()
        # Try to extract first number
        numbers = re.findall(r'-?\d+\.?\d*', pred)
        if numbers:
            return numbers[0]

    # Otherwise find last number in entire response
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if pred:
        return pred[-1]

    return None


def load_model(model_path):
    """Load CODI model."""
    print(f"Loading model from: {model_path}")
    print("This may take a minute...")

    # Set up model arguments (matching the trained model configuration)
    model_args = ModelArguments(
        model_name_or_path="gpt2",
        lora_r=128,
        lora_alpha=32,
        lora_init=True,
        train=False,  # Set to False for inference
        ckpt_dir=model_path,
    )

    # Set up training arguments (needed for model configuration)
    training_args = TrainingArguments(
        output_dir="/tmp/codi_manual_test",
        model_max_length=512,
        seed=11,
        bf16=False,
        num_latent=6,
        use_lora=True,
        greedy=True,
        use_prj=True,
        prj_dim=768,
        prj_no_ln=False,
        prj_dropout=0.0,
        remove_eos=True,
    )

    # Create LoRA config
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create model
    model = CODI(model_args, training_args, lora_config)

    # Load checkpoint
    checkpoint_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(checkpoint_path):
        state_dict = load_file(checkpoint_path)
    else:
        # Try pytorch_model.bin as fallback
        checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
        state_dict = torch.load(checkpoint_path)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Move to GPU if available and ensure consistent dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Convert projection layer to match model dtype (float16)
    if hasattr(model, 'prj') and model.use_prj:
        model.prj = model.prj.half()

    tokenizer = model.tokenizer

    print("✓ Model loaded!")
    return model, tokenizer


def run_problem(model, tokenizer, question_text):
    """Run a single problem through CODI."""
    device = next(model.parameters()).device

    with torch.no_grad():
        # Tokenize
        inputs = tokenizer(question_text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Get embeddings
        input_embd = model.get_embd(model.codi, model.model_name)(input_ids)

        # Forward through model
        outputs = model.codi(
            inputs_embeds=input_embd,
            use_cache=True,
            output_hidden_states=True
        )
        past_key_values = outputs.past_key_values

        # BOT token (start of latent thinking)
        bot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.bot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        latent_embd = bot_emb

        # Latent thought tokens (6 [THINK] tokens)
        for _ in range(model.num_latent):
            outputs = model.codi(
                inputs_embeds=latent_embd,
                use_cache=True,
                output_hidden_states=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

            if model.use_prj:
                latent_embd = model.prj(latent_embd)

        # EOT token (end of latent thinking)
        eot_emb = model.get_embd(model.codi, model.model_name)(
            torch.tensor([model.eot_id], dtype=torch.long, device=device)
        ).unsqueeze(0)

        output_emb = eot_emb

        # Generate answer (max 200 tokens)
        pred_tokens = []
        for _ in range(200):
            out = model.codi(
                inputs_embeds=output_emb,
                use_cache=True,
                past_key_values=past_key_values
            )

            past_key_values = out.past_key_values
            logits = out.logits[:, -1, :model.codi.config.vocab_size-1]

            # Greedy decoding
            next_token_id = torch.argmax(logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            pred_tokens.append(next_token_id.item())
            output_emb = model.get_embd(model.codi, model.model_name)(
                next_token_id
            ).unsqueeze(1)

        # Decode answer
        answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        return answer


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manual CODI testing")
    parser.add_argument('--model_path', type=str,
                       default='/home/paperspace/codi_ckpt/gpt2_gsm8k',
                       help='Path to CODI model')
    args = parser.parse_args()

    # Load model once
    model, tokenizer = load_model(args.model_path)

    print("\n" + "=" * 80)
    print("MANUAL TESTING MODE")
    print("=" * 80)
    print("\nPaste your question below and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        print("-" * 80)
        question = input("Question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        print("\nRunning model...")
        answer = run_problem(model, tokenizer, question)

        print(f"\nModel Output: {answer}")

        # Extract answer
        extracted = extract_answer_number(answer)
        print(f"Extracted Answer: {extracted}")
        print()


if __name__ == "__main__":
    main()
