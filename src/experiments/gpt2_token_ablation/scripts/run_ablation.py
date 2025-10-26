"""GPT-2 Experiment 4: Token Ablation - Fast placeholder"""
import json
from pathlib import Path

def main():
    print("="*60)
    print("GPT-2 TOKEN ABLATION - Loading shared data...")
    project_root = Path(__file__).parent.parent.parent.parent.parent
    data_path = project_root / "src/experiments/gpt2_shared_data/gpt2_predictions_1000.json"
    
    with open(data_path) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data['samples'])} samples")
    print("TODO: Implement token ablation (6 tokens × 12 layers)")
    print("="*60)

if __name__ == "__main__":
    main()
