"""
Utility functions for Tuned Lens experiment.

This module provides helper functions for configuration loading, logging,
data processing, and common operations.
"""

import os
import yaml
import torch
import logging
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load and validate configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is missing required fields
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['model', 'data', 'training', 'tuned_lens']
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValueError(f"Config missing required fields: {missing_fields}")

    # Expand paths
    config = expand_paths(config)

    # Set defaults for optional fields
    config = set_defaults(config)

    return config


def expand_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Expand ~ and environment variables in file paths.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with expanded paths
    """
    # Expand model checkpoint path
    if 'model' in config and 'checkpoint_path' in config['model']:
        config['model']['checkpoint_path'] = os.path.expanduser(
            config['model']['checkpoint_path']
        )

    # Expand dataset path
    if 'data' in config and 'dataset_path' in config['data']:
        config['data']['dataset_path'] = os.path.expanduser(
            config['data']['dataset_path']
        )

    return config


def set_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set default values for optional configuration fields.

    Args:
        config: Configuration dictionary

    Returns:
        Configuration with defaults filled in
    """
    # WandB defaults
    if 'wandb' not in config:
        config['wandb'] = {}
    config['wandb'].setdefault('enabled', True)
    config['wandb'].setdefault('project', 'codi-tuned-lens')
    config['wandb'].setdefault('entity', None)
    config['wandb'].setdefault('log_interval', 10)

    # Compute defaults
    if 'compute' not in config:
        config['compute'] = {}
    config['compute'].setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    config['compute'].setdefault('num_workers', 4)

    # Output defaults
    if 'output' not in config:
        config['output'] = {}
    config['output'].setdefault('verbose', True)

    # Debug defaults
    if 'debug' not in config:
        config['debug'] = {}
    config['debug'].setdefault('enabled', False)

    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('tuned_lens')
    logger.setLevel(logging.DEBUG if config['output']['verbose'] else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if 'log_file' in config['output']:
        log_file = config['output']['log_file'].replace(
            '{timestamp}',
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(config: Dict[str, Any]) -> torch.device:
    """Get torch device based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        torch.device instance
    """
    device_str = config['compute']['device']

    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    # Verify CUDA is available if requested
    if device.type == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device('cpu')

    return device


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


def get_model_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get model information from config.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with model information
    """
    model_config = config['model']

    return {
        'name': model_config['name'],
        'hidden_size': model_config['hidden_size'],
        'num_layers': model_config['num_layers'],
        'num_ct_tokens': model_config['num_ct_tokens'],
        'vocab_size': model_config['vocab_size']
    }


def validate_checkpoint_exists(config: Dict[str, Any]) -> bool:
    """Validate that model checkpoint exists.

    Args:
        config: Configuration dictionary

    Returns:
        True if checkpoint exists, False otherwise
    """
    checkpoint_path = Path(config['model']['checkpoint_path'])

    if not checkpoint_path.exists():
        return False

    # Check for model files
    has_safetensors = (checkpoint_path / "model.safetensors").exists()
    has_pytorch = (checkpoint_path / "pytorch_model.bin").exists()

    return has_safetensors or has_pytorch


def create_output_directories(config: Dict[str, Any]):
    """Create output directories if they don't exist.

    Args:
        config: Configuration dictionary
    """
    directories = [
        config['output']['output_dir'],
        config['output']['models_dir'],
        config['output']['figures_dir'],
        config['output']['decoded_problems_dir'],
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_usage() -> Optional[Dict[str, float]]:
    """Get current GPU memory usage.

    Returns:
        Dictionary with allocated and cached memory in GB, or None if CUDA not available
    """
    if not torch.cuda.is_available():
        return None

    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'cached_gb': torch.cuda.memory_reserved() / 1024**3
    }


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(results: Dict[str, Any], output_path: str):
    """Save results dictionary to JSON file.

    Args:
        results: Results dictionary
        output_path: Output file path
    """
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, torch.Tensor):
            results_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            results_serializable[key] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            results_serializable[key] = value

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)


def load_results(results_path: str) -> Dict[str, Any]:
    """Load results from JSON file.

    Args:
        results_path: Path to results file

    Returns:
        Results dictionary
    """
    import json

    with open(results_path, 'r') as f:
        results = json.load(f)

    return results


# Configuration validation functions

def validate_data_config(config: Dict[str, Any]) -> List[str]:
    """Validate data configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    data_config = config.get('data', {})

    # Check dataset path exists
    dataset_path = Path(data_config.get('dataset_path', ''))
    if not dataset_path.exists():
        errors.append(f"Dataset not found: {dataset_path}")

    # Check split ratios
    train_split = data_config.get('train_split', 0.8)
    test_split = data_config.get('test_split', 0.2)
    if abs(train_split + test_split - 1.0) > 1e-6:
        errors.append(f"Train/test splits must sum to 1.0, got {train_split + test_split}")

    # Check num_problems
    num_problems = data_config.get('num_problems', 1000)
    if num_problems <= 0 and num_problems != -1:
        errors.append("num_problems must be positive or -1 (for all)")

    return errors


def validate_training_config(config: Dict[str, Any]) -> List[str]:
    """Validate training configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    training_config = config.get('training', {})

    # Check learning rate
    lr = training_config.get('learning_rate', 1e-3)
    if lr <= 0 or lr > 0.1:
        errors.append(f"Learning rate {lr} seems unusual (should be 1e-5 to 1e-2)")

    # Check batch size
    batch_size = training_config.get('batch_size', 32)
    if batch_size <= 0:
        errors.append("Batch size must be positive")

    # Check epochs
    num_epochs = training_config.get('num_epochs', 50)
    if num_epochs <= 0:
        errors.append("Number of epochs must be positive")

    return errors


def validate_full_config(config: Dict[str, Any]) -> List[str]:
    """Run all configuration validations.

    Args:
        config: Configuration dictionary

    Returns:
        List of all validation errors (empty if valid)
    """
    errors = []
    errors.extend(validate_data_config(config))
    errors.extend(validate_training_config(config))

    return errors
