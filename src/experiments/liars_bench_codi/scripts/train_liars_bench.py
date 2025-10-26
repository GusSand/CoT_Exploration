"""
Custom CODI training script for liars-bench dataset.

This is a lightweight wrapper around the main CODI training script that adds
support for loading liars-bench data.
"""

import sys
import json
import os
from pathlib import Path

# Add CODI directory to path
codi_path = Path(__file__).parent.parent.parent.parent.parent / "codi"
sys.path.insert(0, str(codi_path))

# Now we can import from CODI
from train import *

# Override the dataset loading section
original_train = train

def train_liars_bench():
    """Modified train function that supports liars-bench dataset."""
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup Peft/LoRA
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon", "qwen"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["gpt2"]):
            target_modules = ["c_attn", "c_proj", 'c_fc']
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, GPT-2, but got {model_args.model_name_or_path}.")

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )

    model = CODI(model_args, training_args, lora_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token_id = model.pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')

    # Dataset loading - add liars-bench support
    if "liars" in data_args.data_name.lower():
        # Load liars-bench dataset from our processed files
        script_dir = Path(__file__).parent
        data_dir = script_dir.parent / "data" / "processed"

        with open(data_dir / "train.json") as f:
            dataset = json.load(f)

        print(f"✅ Loaded liars-bench training data: {len(dataset)} examples")

        # Create dataset using commonsense format (most flexible for Q&A)
        # The SupervisedDataset class will handle this with data_name="liars-commonsense"
        train_dataset = SupervisedDataset(
            data_name="commonsense",  # Use commonsense parsing logic
            raw_data=dataset,
            tokenizer=tokenizer,
            bot=model.bot_id,
            eot=model.eot_id
        )

    elif "icot" in data_args.data_name:
        if 'full' in data_args.data_name:
            dataset = load_dataset("zen-E/GSM8k-Aug-NL")["train"]
        else:
            dataset = load_dataset("zen-E/GSM8k-Aug")["train"]
        train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)

    elif "strategy" in data_args.data_name:
        dataset = load_dataset("zen-E/StrategyQA_CoT_GPT4o")["train"]
        train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)

    elif "commonsense" in data_args.data_name:
        dataset = load_dataset("zen-E/CommonsenseQA-GPT4omini")["train"]
        train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)

    elif "prontoqa" in data_args.data_name:
        with open("/home/ubuntu/coconut/data/prontoqa_train.json") as f:
            dataset = json.load(f)
        train_dataset = SupervisedDataset(data_name=data_args.data_name, raw_data=dataset, tokenizer=tokenizer, bot=model.bot_id, eot=model.eot_id)

    else:
        raise NotImplementedError(f"Dataset {data_args.data_name} is not supported.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # Training
    trainer = CustomTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    print("=" * 80)
    print("🚀 Starting CODI Training on Liars-Bench")
    print("=" * 80)
    print(f"  Model: {model_args.model_name_or_path}")
    print(f"  Dataset: {data_args.data_name}")
    print(f"  Training examples: {len(train_dataset)}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Num latent tokens: {training_args.num_latent}")
    print("=" * 80)

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"  Model saved to: {training_args.output_dir}")
    print(f"  Next: Run evaluation on test set")

if __name__ == "__main__":
    train_liars_bench()
