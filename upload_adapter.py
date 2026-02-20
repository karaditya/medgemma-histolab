"""
Upload fine-tuned LoRA adapter to HuggingFace Hub.

Run this ONCE before deploying to HF Spaces:

    python upload_adapter.py

It will:
  1. Read your HF token from hf_token.txt or HF_TOKEN env var
  2. Create a public model repo  YOUR_USERNAME/medgemma-histolab-5k
  3. Upload the adapter weights (adapter_config.json + adapter_model.safetensors)
  4. Print the ADAPTER_REPO_ID to paste into your Space secrets

The uploaded adapter is ~35 MB (LoRA r=16 on MedGemma-4B language layers).
"""

import os
import sys
from pathlib import Path

# Path to the exp1b adapter inside this repo (populated during training)
ADAPTER_PATH = "models/exp1b_data_scale_5k/final_model"
REPO_NAME = "medgemma-histolab-5k"


def _get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    token_file = Path("hf_token.txt")
    if token_file.exists():
        return token_file.read_text().strip()
    print("Error: HF token not found.")
    print("  Option A: export HF_TOKEN=hf_xxxx")
    print("  Option B: echo 'hf_xxxx' > hf_token.txt")
    sys.exit(1)


def upload_adapter(adapter_path: str = ADAPTER_PATH, repo_name: str = REPO_NAME):
    from huggingface_hub import HfApi, create_repo

    token = _get_token()
    api = HfApi()

    user = api.whoami(token=token)
    username = user["name"]
    repo_id = f"{username}/{repo_name}"

    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print(f"Error: Adapter not found at: {adapter_path}")
        print("Train the model first with:")
        print("  python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml")
        sys.exit(1)

    print(f"Creating HF Hub repo: {repo_id}")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)

    print(f"Uploading adapter from {adapter_path} …")
    # Upload only adapter weights + config (skip merged_model — too large for Hub)
    files_to_upload = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "base_model_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
    ]

    for fname in files_to_upload:
        fpath = adapter_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                token=token,
            )
            print(f"  ✓ {fname}")
        else:
            print(f"  - {fname} (not found, skipping)")

    print(f"\n✅ Done! Adapter uploaded to:")
    print(f"   https://huggingface.co/{repo_id}")
    print(f"\nAdd this to your HuggingFace Space secrets:")
    print(f"   ADAPTER_REPO_ID = {repo_id}")
    print(f"   HF_TOKEN        = <your token with MedGemma access>")


if __name__ == "__main__":
    upload_adapter()
