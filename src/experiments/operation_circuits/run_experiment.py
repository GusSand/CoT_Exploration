"""
Master Script: Operation-Specific Circuits Experiment
Runs the complete pipeline from extraction to analysis.
"""

import sys
import time
from pathlib import Path
from datetime import datetime


def log(message: str):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_extraction(prototype: bool = True):
    """Run continuous thought extraction."""
    log("="*80)
    log("PHASE 1: CONTINUOUS THOUGHT EXTRACTION")
    log("="*80)

    from extract_continuous_thoughts import ContinuousThoughtExtractor
    import json

    # Configuration
    model_path = str(Path.home() / 'codi_ckpt' / 'llama_gsm8k')

    if prototype:
        dataset_path = Path(__file__).parent / "operation_samples_prototype_60.json"
        output_path = Path(__file__).parent / "results" / "continuous_thoughts_prototype_60.json"
    else:
        dataset_path = Path(__file__).parent / "operation_samples_200.json"
        output_path = Path(__file__).parent / "results" / "continuous_thoughts_full_600.json"

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load dataset
    log(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    total_problems = sum(len(v) for v in dataset.values())
    log(f"Loaded {total_problems} problems")

    # Initialize extractor
    log("Initializing CODI model...")
    extractor = ContinuousThoughtExtractor(model_path)

    # Extract
    log("Starting extraction...")
    start_time = time.time()

    results = extractor.extract_dataset(dataset, str(output_path), save_frequency=10)

    elapsed = time.time() - start_time
    log(f"Extraction complete in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    log(f"Processed: {len(results)} problems")
    log(f"Output: {output_path}")

    return output_path


def run_analysis(data_path: Path):
    """Run analysis on extracted data."""
    log("\n" + "="*80)
    log("PHASE 2: ANALYSIS & VISUALIZATION")
    log("="*80)

    from analyze_continuous_thoughts import ContinuousThoughtAnalyzer

    output_dir = data_path.parent / "analysis"

    log(f"Loading data from {data_path}...")
    analyzer = ContinuousThoughtAnalyzer(str(data_path))

    log("Running full analysis pipeline...")
    start_time = time.time()

    analyzer.run_full_analysis(output_dir)

    elapsed = time.time() - start_time
    log(f"Analysis complete in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    log(f"Results: {output_dir}")

    return output_dir


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Operation-Specific Circuits Experiment")
    parser.add_argument(
        '--mode',
        choices=['prototype', 'full', 'analysis_only'],
        default='prototype',
        help='Experiment mode: prototype (60 samples), full (600 samples), or analysis_only'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        help='Path to extracted data (for analysis_only mode)'
    )

    args = parser.parse_args()

    log("="*80)
    log("OPERATION-SPECIFIC CIRCUITS EXPERIMENT")
    log("="*80)
    log(f"Mode: {args.mode}")
    log(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.mode == 'analysis_only':
            # Only run analysis on existing data
            if not args.data_path:
                log("ERROR: --data_path required for analysis_only mode")
                sys.exit(1)

            data_path = Path(args.data_path)
            if not data_path.exists():
                log(f"ERROR: Data file not found: {data_path}")
                sys.exit(1)

            output_dir = run_analysis(data_path)

        else:
            # Run extraction + analysis
            prototype = (args.mode == 'prototype')

            # Extract continuous thoughts
            data_path = run_extraction(prototype=prototype)

            # Analyze
            output_dir = run_analysis(data_path)

        # Summary
        log("\n" + "="*80)
        log("✓ EXPERIMENT COMPLETE!")
        log("="*80)
        log(f"Results directory: {output_dir}")
        log(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        log(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
