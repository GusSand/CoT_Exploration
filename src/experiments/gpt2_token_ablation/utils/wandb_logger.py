"""WandB logging utilities for position ablation experiment."""

import wandb
from typing import Dict, Any
import matplotlib.pyplot as plt


def init_experiment(config: Dict[str, Any], run_name: str = None):
    """Initialize WandB experiment.

    Args:
        config: Experiment configuration dictionary
        run_name: Optional custom run name
    """
    wandb.init(
        project="gpt2-llama-position-analysis",
        config=config,
        name=run_name,
        tags=["position-ablation", "token-decoding"]
    )
    print(f"✓ WandB initialized: {wandb.run.name}")


def log_metrics(metrics_dict: Dict[str, float], step: int = None):
    """Log metrics to WandB.

    Args:
        metrics_dict: Dictionary of metric names and values
        step: Optional step number
    """
    wandb.log(metrics_dict, step=step)


def log_figure(fig: plt.Figure, name: str):
    """Log matplotlib figure to WandB.

    Args:
        fig: Matplotlib figure
        name: Name for the logged figure
    """
    wandb.log({name: wandb.Image(fig)})


def finish_experiment():
    """Finish WandB run."""
    wandb.finish()
    print("✓ WandB run finished")
