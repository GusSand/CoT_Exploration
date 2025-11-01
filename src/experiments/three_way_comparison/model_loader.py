#!/usr/bin/env python3
"""
Story 1: Unified Model Loading Infrastructure
Load CODI models for Personal Relations, GSM8K, and CommonsenseQA.
"""
import json
import sys
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / 'codi'))

# Import CODI model
from src.model import CODI
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from dataclasses import dataclass


@dataclass
class ModelArgs:
    model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    use_lora: bool = False
    lora_r: int = 128
    lora_alpha: int = 32
    use_prj: bool = True
    prj_dim: int = 2048
    prj_dropout: float = 0.0
    full_precision: bool = True
    train: bool = False


@dataclass
class TrainingArgs:
    num_latent: int = 6
    model_max_length: int = 512
    bf16: bool = True
    distill_loss_div_std: bool = True
    remove_eos: bool = True
    distill_loss_factor: int = 20
    use_lora: bool = False
    use_prj: bool = True
    prj_dim: int = 2048
    prj_dropout: float = 0.0
    prj_no_ln: bool = False
    print_loss: bool = False
    ref_loss_factor: float = 1.0
    distill_loss_type: str = "smooth_l1"
    fix_attn_mask: bool = False
    restore_from: str = ""


class CODIModelLoader:
    """Unified loader for all three CODI models."""

    def __init__(self, config_path: str = None):
        """
        Initialize model loader with configuration.

        Args:
            config_path: Path to config.json (default: same directory)
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'config.json'

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.current_task = None
        self.model = None
        self.tokenizer = None
        self.device = self.config['device']

    def load_model(self, task: str) -> Tuple[Any, Any, Dict]:
        """
        Load CODI model for specified task.

        Args:
            task: One of 'personal_relations', 'gsm8k', 'commonsense'

        Returns:
            (model, tokenizer, metadata)
        """
        if task not in self.config['tasks']:
            raise ValueError(f"Unknown task: {task}. Must be one of {self.config['tasks']}")

        print(f"\n{'='*80}")
        print(f"Loading {task.upper()} CODI Model")
        print(f"{'='*80}")

        model_path = self.config['model_paths'][task]
        print(f"Model path: {model_path}")

        # Task-specific loading
        if task == 'personal_relations':
            model, tokenizer = self._load_personal_relations(model_path)
        elif task == 'gsm8k':
            model, tokenizer = self._load_gsm8k(model_path)
        elif task == 'commonsense':
            model, tokenizer = self._load_commonsense(model_path)

        self.model = model
        self.tokenizer = tokenizer
        self.current_task = task

        # Create metadata
        metadata = {
            'task': task,
            'model_path': model_path,
            'device': self.device,
            'n_layers': 16,  # LLaMA-1B has 16 layers
            'n_tokens': 6,   # All models use 6 CT tokens
            'hidden_dim': 2048,  # LLaMA-1B hidden dimension
            'model_type': type(model).__name__
        }

        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {metadata['model_type']}")
        print(f"  Device: {metadata['device']}")
        print(f"  Layers: {metadata['n_layers']}, CT Tokens: {metadata['n_tokens']}")

        # Validation: run a test forward pass
        self._validate_model(task)

        return model, tokenizer, metadata

    def _load_personal_relations(self, model_path: str) -> Tuple[Any, Any]:
        """Load Personal Relations model (requires LoRA)."""
        print("  Loading Personal Relations model with LoRA...")

        # Create model args with LoRA
        model_args = ModelArgs(
            model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
            use_lora=True,
            lora_r=128,
            lora_alpha=32,
            full_precision=True  # CRITICAL for Personal Relations
        )

        training_args = TrainingArgs(
            num_latent=6,
            use_lora=True,
            use_prj=True,
            prj_dim=2048,
        )

        # Initialize LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )

        # Load model
        model = CODI(model_args, training_args, lora_config)
        checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(self.device)

        # Convert projection layer to bfloat16 to match hidden states
        if model.use_prj and hasattr(model, 'prj'):
            model.prj = model.prj.to(torch.bfloat16)

        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    def _load_gsm8k(self, model_path: str) -> Tuple[Any, Any]:
        """Load GSM8K model (uses LoRA)."""
        print("  Loading GSM8K model with LoRA...")

        # GSM8K model was trained with LoRA
        model_args = ModelArgs(
            model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
            full_precision=True,
            use_lora=True,  # FIXED: GSM8K uses LoRA
            lora_r=128,
            lora_alpha=32
        )

        training_args = TrainingArgs(
            num_latent=6,
            use_prj=True,
            prj_dim=2048,
            use_lora=True  # FIXED: Enable LoRA
        )

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )

        # Load model
        model = CODI(model_args, training_args, lora_config)
        checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(self.device)

        # Convert projection layer to bfloat16 to match hidden states
        if model.use_prj and hasattr(model, 'prj'):
            model.prj = model.prj.to(torch.bfloat16)

        model.eval()

        # Load tokenizer (use base model tokenizer)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

        return model, tokenizer

    def _load_commonsense(self, model_path: str) -> Tuple[Any, Any]:
        """Load CommonsenseQA model (uses LoRA)."""
        print("  Loading CommonsenseQA model with LoRA...")

        # CommonsenseQA model was trained with LoRA
        model_args = ModelArgs(
            model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
            full_precision=True,
            use_lora=True,  # FIXED: CommonsenseQA uses LoRA
            lora_r=128,
            lora_alpha=32
        )

        training_args = TrainingArgs(
            num_latent=6,
            use_prj=True,
            prj_dim=2048,
            use_lora=True  # FIXED: Enable LoRA
        )

        # Create LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )

        # Load model
        model = CODI(model_args, training_args, lora_config)
        checkpoint = torch.load(f"{model_path}/pytorch_model.bin", map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
        model = model.to(self.device)

        # Convert projection layer to bfloat16 to match hidden states
        if model.use_prj and hasattr(model, 'prj'):
            model.prj = model.prj.to(torch.bfloat16)

        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return model, tokenizer

    def _validate_model(self, task: str):
        """Run a test forward pass to validate model."""
        print("  Running validation forward pass...")

        # Task-specific test inputs
        test_inputs = {
            'personal_relations': "Given the following relationships:\n\nAlice's parent = Bob\n\nQuestion: Who is Alice's parent?\nReasoning:",
            'gsm8k': "Question: Janet has 5 apples. She buys 3 more. How many apples does she have?\nReasoning:",
            'commonsense': "Question: What do people typically do when they are tired?\nChoices:\nA: sleep\nB: run\nC: jump\nD: swim\nE: eat\nReasoning:"
        }

        test_input = test_inputs[task]

        try:
            with torch.no_grad():
                inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
                # Use the internal codi model for generation
                outputs = self.model.codi.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"  ✓ Validation successful! Generated {len(output_text)} characters")
        except Exception as e:
            print(f"  ✗ Validation failed: {e}")
            raise

    def format_input(self, task: str, example: Dict) -> str:
        """
        Format input according to task requirements.

        Args:
            task: Task name
            example: Raw example dict

        Returns:
            Formatted input string
        """
        if task == 'personal_relations':
            return self._format_personal_relations(example)
        elif task == 'gsm8k':
            return self._format_gsm8k(example)
        elif task == 'commonsense':
            return self._format_commonsense(example)

    def _format_personal_relations(self, example: Dict) -> str:
        """Format Personal Relations input."""
        # Example already has proper format from dataset
        question = example.get('question', example.get('input', ''))
        return f"{question}\nReasoning:"

    def _format_gsm8k(self, example: Dict) -> str:
        """Format GSM8K input."""
        question = example.get('question', '')
        return f"Question: {question}\nReasoning:"

    def _format_commonsense(self, example: Dict) -> str:
        """
        Format CommonsenseQA input.

        UPDATED: Matches original CODI evaluation (codi/test.py:137)
        - RAW question text ONLY (no "Reasoning:" suffix)
        - Question already includes formatted choices in zen-E dataset
        """
        # zen-E/CommonsenseQA-GPT4omini format: question already includes choices
        if 'choices' not in example and 'answer' in example:
            # Return raw question text (no suffix) - matches original CODI
            question = example['question'].strip().replace('  ', ' ')
            return question

        # Old format (for backward compatibility)
        question = example['question']
        choices = example['choices']

        formatted = f"Question: {question}\nChoices:\n"
        for label, text in zip(choices['label'], choices['text']):
            formatted += f"{label}: {text}\n"

        return formatted

    def extract_answer(self, task: str, output: str, example: Dict) -> str:
        """
        Extract answer from model output.

        Args:
            task: Task name
            output: Model output text
            example: Original example (for reference)

        Returns:
            Extracted answer string
        """
        if task == 'personal_relations':
            return self._extract_personal_relations_answer(output)
        elif task == 'gsm8k':
            return self._extract_gsm8k_answer(output)
        elif task == 'commonsense':
            return self._extract_commonsense_answer(output)

    def _extract_personal_relations_answer(self, output: str) -> str:
        """Extract Personal Relations answer."""
        output = output.strip()

        # Look for "answer is:" format
        if "answer is:" in output.lower():
            answer_part = output.lower().split("answer is:")[-1].strip()
            answer = answer_part.split()[0] if answer_part.split() else answer_part
            return answer.replace('.', '').replace(',', '').strip().title()

        # Skip "=" prefix if present
        if output.startswith('='):
            output = output[1:].strip()

        # Extract first word
        answer = output.split()[0] if output.split() else output
        return answer.replace('.', '').replace(',', '').strip()

    def _extract_gsm8k_answer(self, output: str) -> str:
        """Extract GSM8K answer (numerical)."""
        output = output.strip()

        # Look for "answer is:" pattern
        if "answer is:" in output.lower():
            answer_part = output.lower().split("answer is:")[-1].strip()
            # Extract first number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer_part)
            if numbers:
                return numbers[0]

        # Fallback: extract last number in output
        import re
        numbers = re.findall(r'-?\d+\.?\d*', output)
        if numbers:
            return numbers[-1]

        return "INVALID"

    def _extract_commonsense_answer(self, output: str) -> str:
        """
        Extract CommonsenseQA answer (A-E).

        UPDATED: Matches original CODI evaluation (codi/test.py:334-337)
        - Split on "The answer is:" (case-insensitive)
        - Take first character after split
        - Default to "C" if invalid (not A-E)
        """
        # Try to find "The answer is:" (case-insensitive)
        output_lower = output.lower()

        if "the answer is:" in output_lower:
            # Split and take last occurrence
            pred = output_lower.split("the answer is:")[-1].strip()

            # Check if first character is valid (A-E)
            if pred and pred[0].upper() in 'ABCDE':
                return pred[0].upper()

        # Default to C (like original CODI)
        return "C"

    def unload_model(self):
        """Unload current model and clear cache."""
        if self.model is not None:
            print(f"\nUnloading {self.current_task} model...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.current_task = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("  ✓ Model unloaded and cache cleared")


def test_model_loader():
    """Test the model loader with all three tasks."""
    print("\n" + "="*80)
    print("TESTING UNIFIED MODEL LOADER")
    print("="*80)

    loader = CODIModelLoader()

    for task in ['personal_relations', 'gsm8k', 'commonsense']:
        try:
            # Load model
            model, tokenizer, metadata = loader.load_model(task)

            # Test successful
            print(f"\n✓ {task.upper()} - PASSED")

            # Unload
            loader.unload_model()

        except Exception as e:
            print(f"\n✗ {task.upper()} - FAILED")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "="*80)
    print("ALL MODELS LOADED SUCCESSFULLY!")
    print("="*80)
    return True


if __name__ == '__main__':
    success = test_model_loader()
    sys.exit(0 if success else 1)
