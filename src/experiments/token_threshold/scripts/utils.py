#!/usr/bin/env python3
"""
Shared utilities for token threshold experiments.

Includes:
- WandB integration
- Common helper functions
- Shared constants
"""
import wandb
from typing import Dict, Any, Optional


class WandBLogger:
    """Manages WandB experiment tracking."""

    def __init__(self, project: str = "codi-token-threshold",
                 experiment_name: str = None,
                 config: Dict[str, Any] = None,
                 tags: list = None):
        """
        Initialize WandB logger.

        Args:
            project: WandB project name
            experiment_name: Name of this specific experiment run
            config: Configuration dictionary to log
            tags: List of tags for this run
        """
        self.project = project
        self.run = None

        if experiment_name:
            self.run = wandb.init(
                project=project,
                name=experiment_name,
                config=config or {},
                tags=tags or [],
                reinit=True
            )

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        """Log data to WandB."""
        if self.run:
            if step is not None:
                wandb.log(data, step=step)
            else:
                wandb.log(data)

    def log_experiment(self, problem_id: str, experiment_type: str,
                      result: Dict[str, Any]):
        """
        Log a single experiment result.

        Args:
            problem_id: Identifier for the problem
            experiment_type: Type of experiment (e.g., 'threshold', 'enhancement')
            result: Result dictionary with metrics
        """
        log_data = {
            'problem_id': problem_id,
            'experiment_type': experiment_type,
            **result
        }
        self.log(log_data)

    def finish(self):
        """Finish WandB run."""
        if self.run:
            wandb.finish()


def get_corruption_positions(n_tokens: int, corruption_level: int) -> list:
    """
    Get strategic corruption positions for a given corruption level.

    Args:
        n_tokens: Total number of tokens (6 for CODI)
        corruption_level: Number of tokens to corrupt (1-6)

    Returns:
        List of position tuples to corrupt
    """
    if corruption_level == 1:
        # All 6 individual positions
        return [(i,) for i in range(n_tokens)]

    elif corruption_level == 2:
        # 3 strategic samples: sequential pairs
        return [(0, 1), (2, 3), (4, 5)]

    elif corruption_level == 3:
        # 3 samples: first half, second half, distributed
        return [(0, 1, 2), (3, 4, 5), (0, 2, 4)]

    elif corruption_level == 4:
        # 6 samples: skip each token individually
        # This tests which single token is most critical
        return [
            tuple(i for i in range(n_tokens) if i != skip)
            for skip in range(n_tokens)
        ]

    elif corruption_level == 5:
        # 6 samples: keep each token individually
        # This tests minimum viable token
        return [(i,) for i in range(n_tokens)]

    elif corruption_level == 6:
        # Complete ablation
        return [tuple(range(n_tokens))]

    else:
        raise ValueError(f"Invalid corruption level: {corruption_level}")


def get_position_label(positions: tuple) -> str:
    """
    Get human-readable label for corruption positions.

    Args:
        positions: Tuple of position indices

    Returns:
        String label (e.g., "0,1" or "skip_3")
    """
    n_total = 6

    if len(positions) == n_total:
        return "all"
    elif len(positions) == n_total - 1:
        # Skip pattern (4 tokens corrupted, skip 1)
        skipped = set(range(n_total)) - set(positions)
        return f"skip_{list(skipped)[0]}"
    elif len(positions) == 1 and len(positions) < n_total - 1:
        # Keep pattern (5 tokens corrupted, keep 1)
        return f"keep_{positions[0]}"
    else:
        return ",".join(map(str, positions))


# Shared constants
LAYER_CONFIG = {
    'early': 4,
    'middle': 8,
    'late': 14
}

DEFAULT_CORRUPTION_METHODS = ['zero', 'gauss_1.0']
DEFAULT_ENHANCEMENT_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0]
