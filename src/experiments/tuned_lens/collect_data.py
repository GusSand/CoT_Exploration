"""
Data Collection for Tuned Lens (Story 1).

This script extracts continuous thought activations from CODI LLaMA model
for training Tuned Lens transformations.

Key Features:
- Loads 1,000 stratified GSM8K problems where LLaMA needs CoT
- Splits into train (800) and test (200) with stratification preserved
- Extracts continuous thought activations from all 16 layers
- Saves memory by storing target token IDs instead of full logit distributions
- Reuses proven ContinuousThoughtExtractor from operation_circuits

Usage:
    python collect_data.py --config config.yaml
    python collect_data.py --num-problems 100 --debug  # Test on small dataset
"""

import sys
import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "operation_circuits"))

from utils import (
    load_config, setup_logging, set_seed, get_device,
    validate_checkpoint_exists, create_output_directories,
    clear_gpu_cache
)

# Reuse proven extraction infrastructure
from extract_continuous_thoughts import ContinuousThoughtExtractor


def load_dataset(dataset_path: str, num_problems: int = -1) -> List[Dict[str, Any]]:
    """Load GSM8K problems from stratified dataset.

    Args:
        dataset_path: Path to llama_cot_original_stratified_1000.json
        num_problems: Number of problems to load (-1 for all)

    Returns:
        List of problem dictionaries
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        problems = json.load(f)

    # Limit number of problems if specified
    if num_problems > 0 and num_problems < len(problems):
        problems = problems[:num_problems]

    return problems


def stratified_split(problems: List[Dict[str, Any]],
                     train_ratio: float = 0.8,
                     random_seed: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split problems into train/test with stratification by difficulty.

    Args:
        problems: List of problem dictionaries
        train_ratio: Ratio of training data (0.8 = 80%)
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_problems, test_problems)
    """
    # Extract difficulty labels
    difficulties = [p['difficulty'] for p in problems]

    # Stratified split
    train_problems, test_problems = train_test_split(
        problems,
        train_size=train_ratio,
        stratify=difficulties,
        random_state=random_seed
    )

    return train_problems, test_problems


def verify_stratification(problems: List[Dict[str, Any]], split_name: str):
    """Verify stratification is preserved in split.

    Args:
        problems: List of problem dictionaries
        split_name: Name of split (e.g., "train", "test")
    """
    from collections import Counter

    difficulties = [p['difficulty'] for p in problems]
    counts = Counter(difficulties)

    print(f"\n{split_name.capitalize()} split ({len(problems)} problems):")
    for difficulty in sorted(counts.keys()):
        count = counts[difficulty]
        percentage = 100 * count / len(problems)
        print(f"  {difficulty}: {count} ({percentage:.1f}%)")


def extract_activations(problems: List[Dict[str, Any]],
                       extractor: ContinuousThoughtExtractor,
                       layers: List[int],
                       representation: str,
                       config: Dict[str, Any],
                       logger) -> Dict[str, Any]:
    """Extract continuous thought activations for all problems.

    Args:
        problems: List of problem dictionaries
        extractor: ContinuousThoughtExtractor instance
        layers: List of layer indices to extract (e.g., [0, 1, ..., 15])
        representation: "pre_mlp" or "post_mlp" (CODI post-processes with MLP projection)
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dictionary containing:
        - hidden_states: (N, hidden_size) tensor of activations
        - target_token_ids: (N,) tensor of target token IDs
        - metadata: problem IDs, layers, positions, difficulties
    """
    hidden_states_list = []
    target_token_ids_list = []
    metadata = {
        'problem_ids': [],
        'layers': [],
        'positions': [],
        'difficulties': []
    }

    num_ct_tokens = config['model']['num_ct_tokens']
    clear_cache_interval = config['compute'].get('clear_cache_every_n_batches', 10)

    logger.info(f"Extracting activations for {len(problems)} problems...")
    logger.info(f"Layers: {layers}")
    logger.info(f"Representation: {representation}")
    logger.info(f"Continuous thought tokens per layer: {num_ct_tokens}")

    for problem_idx, problem in enumerate(tqdm(problems, desc="Extracting")):
        try:
            # Extract continuous thoughts using proven extractor
            question = problem['question']

            with torch.no_grad():
                # Tokenize input
                inputs = extractor.tokenizer(question, return_tensors="pt").to(extractor.device)
                input_ids = inputs["input_ids"]

                # Get initial embeddings
                input_embd = extractor.model.get_embd(extractor.model.codi, extractor.model.model_name)(input_ids)

                # Forward through model to get context
                outputs = extractor.model.codi(
                    inputs_embeds=input_embd,
                    use_cache=True,
                    output_hidden_states=True
                )

                past_key_values = outputs.past_key_values

                # Get BOT (Beginning of Thought) embedding
                bot_emb = extractor.model.get_embd(extractor.model.codi, extractor.model.model_name)(
                    torch.tensor([extractor.model.bot_id], dtype=torch.long, device=extractor.device)
                ).unsqueeze(0)

                # Process all latent tokens (continuous thoughts)
                latent_embd = bot_emb

                for latent_step in range(num_ct_tokens):
                    # Forward through model for this latent token
                    outputs = extractor.model.codi(
                        inputs_embeds=latent_embd,
                        use_cache=True,
                        output_hidden_states=True,
                        past_key_values=past_key_values
                    )

                    past_key_values = outputs.past_key_values

                    # Extract hidden states from all requested layers
                    for layer in layers:
                        # hidden_states is a tuple: (layer_0, layer_1, ..., layer_N)
                        # Extract the last token's hidden state
                        hidden_state = outputs.hidden_states[layer][:, -1, :].cpu()

                        # Store activation
                        hidden_states_list.append(hidden_state.squeeze(0))

                        # For target, use the model's final output token
                        # We'll compute this once and reuse
                        if layer == layers[0] and latent_step == 0:
                            # Get model's prediction to use as target
                            # Use final hidden state to compute logits
                            final_hidden = outputs.hidden_states[-1][:, -1, :]
                            logits = extractor.model.codi.lm_head(final_hidden)
                            target_token_id = logits.argmax(dim=-1).item()

                        # Store metadata
                        metadata['problem_ids'].append(problem.get('gsm8k_id', f"problem_{problem_idx}"))
                        metadata['layers'].append(layer)
                        metadata['positions'].append(latent_step)  # Position in latent sequence (0-5)
                        metadata['difficulties'].append(problem['difficulty'])

                    # Update latent embedding for next iteration
                    latent_embd = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)

                    # Apply projection if model uses it (post-MLP representation)
                    if representation == "post_mlp" and hasattr(extractor.model, 'prj') and extractor.model.use_prj:
                        latent_embd = extractor.model.prj(latent_embd)

                # Store target token ID for all samples from this problem
                for _ in range(len(layers) * num_ct_tokens):
                    target_token_ids_list.append(target_token_id)

            # Clear cache periodically to prevent memory buildup
            if (problem_idx + 1) % clear_cache_interval == 0:
                clear_gpu_cache()

        except Exception as e:
            logger.error(f"Failed to process problem {problem_idx} ({problem.get('gsm8k_id', 'unknown')}): {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Convert lists to tensors
    hidden_states = torch.stack(hidden_states_list)
    target_token_ids = torch.tensor(target_token_ids_list, dtype=torch.long)

    logger.info(f"\nExtraction complete!")
    logger.info(f"Total samples: {len(hidden_states)}")
    logger.info(f"Hidden states shape: {hidden_states.shape}")
    logger.info(f"Target token IDs shape: {target_token_ids.shape}")

    # Verify counts
    expected_samples = len(problems) * len(layers) * num_ct_tokens
    logger.info(f"Expected samples: {expected_samples} ({len(problems)} problems × {len(layers)} layers × {num_ct_tokens} tokens)")
    if len(hidden_states) < expected_samples:
        logger.warning(f"Some samples were skipped due to errors!")

    return {
        'hidden_states': hidden_states,
        'target_token_ids': target_token_ids,
        'metadata': metadata,
        'config': {
            'model': config['model']['name'],
            'representation': representation,
            'hidden_size': config['model']['hidden_size'],
            'num_layers': config['model']['num_layers'],
            'num_ct_tokens': num_ct_tokens
        }
    }


def save_dataset(data: Dict[str, Any], output_path: str, logger):
    """Save dataset to file.

    Args:
        data: Dataset dictionary
        output_path: Output file path
        logger: Logger instance
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PyTorch file
    torch.save(data, output_path)

    # Calculate file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"\nDataset saved to: {output_path}")
    logger.info(f"File size: {file_size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Collect training data for Tuned Lens")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--num-problems', type=int, default=-1,
                       help='Number of problems to use (-1 for all from config)')
    parser.add_argument('--representation', type=str, choices=['pre_mlp', 'post_mlp'],
                       help='Override representation from config')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode (small dataset)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.num_problems > 0:
        config['data']['num_problems'] = args.num_problems
    if args.representation:
        config['data']['representation'] = args.representation
    if args.debug:
        config['debug']['enabled'] = True
        config['data']['num_problems'] = config['debug'].get('num_debug_problems', 10)

    # Setup logging
    logger = setup_logging(config)
    logger.info("="*70)
    logger.info("TUNED LENS DATA COLLECTION (Story 1)")
    logger.info("="*70)

    # Set random seed
    set_seed(config['data']['random_seed'])
    logger.info(f"Random seed: {config['data']['random_seed']}")

    # Validate checkpoint exists
    if not validate_checkpoint_exists(config):
        logger.error(f"CODI checkpoint not found: {config['model']['checkpoint_path']}")
        logger.error("Please check the checkpoint path in config.yaml")
        sys.exit(1)

    # Create output directories
    create_output_directories(config)

    # Load dataset
    logger.info(f"\nLoading dataset from: {config['data']['dataset_path']}")
    problems = load_dataset(
        config['data']['dataset_path'],
        config['data']['num_problems']
    )
    logger.info(f"Loaded {len(problems)} problems")

    # Verify initial stratification
    from collections import Counter
    difficulties = [p['difficulty'] for p in problems]
    counts = Counter(difficulties)
    logger.info("\nInitial dataset distribution:")
    for difficulty in sorted(counts.keys()):
        count = counts[difficulty]
        percentage = 100 * count / len(problems)
        logger.info(f"  {difficulty}: {count} ({percentage:.1f}%)")

    # Stratified train/test split
    logger.info(f"\nSplitting into train ({config['data']['train_split']:.0%}) / test ({config['data']['test_split']:.0%})...")
    train_problems, test_problems = stratified_split(
        problems,
        train_ratio=config['data']['train_split'],
        random_seed=config['data']['random_seed']
    )

    # Verify stratification
    verify_stratification(train_problems, "train")
    verify_stratification(test_problems, "test")

    # Initialize extractor
    logger.info(f"\nInitializing CODI model from: {config['model']['checkpoint_path']}")
    device = get_device(config)
    logger.info(f"Device: {device}")

    extractor = ContinuousThoughtExtractor(
        model_path=config['model']['checkpoint_path'],
        device=str(device)
    )

    # Extract activations for training set
    logger.info("\n" + "="*70)
    logger.info("EXTRACTING TRAINING SET")
    logger.info("="*70)

    train_data = extract_activations(
        train_problems,
        extractor,
        config['data']['layers_to_collect'],
        config['data']['representation'],
        config,
        logger
    )

    # Save training data
    train_output_path = config['output']['train_data_file'].format(
        model=config['model']['name'],
        representation=config['data']['representation']
    )
    save_dataset(train_data, train_output_path, logger)

    # Extract activations for test set
    logger.info("\n" + "="*70)
    logger.info("EXTRACTING TEST SET")
    logger.info("="*70)

    test_data = extract_activations(
        test_problems,
        extractor,
        config['data']['layers_to_collect'],
        config['data']['representation'],
        config,
        logger
    )

    # Save test data
    test_output_path = config['output']['test_data_file'].format(
        model=config['model']['name'],
        representation=config['data']['representation']
    )
    save_dataset(test_data, test_output_path, logger)

    # Final summary
    logger.info("\n" + "="*70)
    logger.info("DATA COLLECTION COMPLETE!")
    logger.info("="*70)
    logger.info(f"\nTraining set:")
    logger.info(f"  Problems: {len(train_problems)}")
    logger.info(f"  Samples: {len(train_data['hidden_states'])}")
    logger.info(f"  File: {train_output_path}")

    logger.info(f"\nTest set:")
    logger.info(f"  Problems: {len(test_problems)}")
    logger.info(f"  Samples: {len(test_data['hidden_states'])}")
    logger.info(f"  File: {test_output_path}")

    logger.info(f"\nDataset ready for training!")
    logger.info(f"Next step: python train.py --config {args.config}")


if __name__ == '__main__':
    main()
