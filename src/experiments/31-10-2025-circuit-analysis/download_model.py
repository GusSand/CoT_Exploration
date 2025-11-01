import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/workspace/.env')
hf_token = os.getenv('HF_TOKEN')

# Model details
repo_id = 'zen-E/CODI-llama3.2-1b-Instruct'
local_dir = '/workspace/CoT_Exploration/models/CODI-llama3.2-1b'

print('='*80)
print(f'Downloading CODI-LLAMA model from HuggingFace')
print(f'Repository: {repo_id}')
print(f'Destination: {local_dir}')
print('='*80)

# Create directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

# Download the model
try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        token=hf_token,
        resume_download=True
    )
    print(f'\n✓ Model downloaded successfully to {local_dir}')
except Exception as e:
    print(f'\n✗ Error downloading model: {e}')
    raise
