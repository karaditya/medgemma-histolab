"""
Upload fine-tuned LoRA adapter + model card to HuggingFace Hub.

Run this ONCE before deploying to HF Spaces:

    python upload_adapter.py

It will:
  1. Read your HF token from hf_token.txt or HF_TOKEN env var
  2. Create a public model repo  YOUR_USERNAME/medgemma-histolab-5k
  3. Upload the adapter weights (adapter_config.json + adapter_model.safetensors)
  4. Generate and upload a model card (README.md)
  5. Print the ADAPTER_REPO_ID to paste into your Space secrets

The uploaded adapter is ~35 MB (LoRA r=16 on MedGemma-4B language layers).
"""

import os
import sys
import tempfile
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


def generate_model_card(repo_id: str) -> str:
    """Generate the README.md model card content."""
    return f"""\
---
license: apache-2.0
base_model: google/medgemma-4b-it
tags:
  - medical
  - histopathology
  - vision-language
  - lora
  - peft
  - pathology
  - cancer-detection
datasets:
  - NCT-CRC-HE-100K
  - PatchCamelyon
  - BACH-ICIAR-2018
language:
  - en
pipeline_tag: image-text-to-text
---

# HistoLab — MedGemma-4B Fine-tuned for Histopathology

A QLoRA fine-tuned adapter on top of [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it),
trained to classify histopathology patches across three benchmark datasets.

## Results

| Dataset | Baseline (zero-shot) | Fine-tuned | Δ |
|---|---|---|---|
| **NCT-CRC-HE-100K** (9-class) | 53.4% | **89.6%** | +36.2 pp |
| **PatchCamelyon** (2-class) | ~60% | **85.2%** | +25.2 pp |
| **BACH ICIAR 2018** (4-class) | ~40% | **68.3%** | +28.3 pp |
| **Overall** | 53.4% | **81.0%** | **+27.6 pp** |

## Model Details

- **Base model:** `google/medgemma-4b-it` (4B parameter multimodal LLM)
- **Fine-tuning method:** QLoRA (4-bit base + bfloat16 adapter merge at inference)
- **LoRA config:** r=16, α=32, dropout=0.05, target=language layers only
- **Training:** 1 epoch, lr=2e-5, batch_size=1 + 4 grad accumulation steps
- **Adapter size:** ~35 MB

## Training Data

| Dataset | Task | Classes | Samples used |
|---|---|---|---|
| [NCT-CRC-HE-100K](https://zenodo.org/records/1214456) | 9-class tissue classification | Adipose, Background, Debris, Lymphocyte, Mucus, Muscle, Normal, Stroma, Tumor | 5 000 |
| [PatchCamelyon](https://github.com/basveeling/pcam) | Binary metastasis detection | Normal, Tumor | 5 000 |
| [BACH ICIAR 2018](https://iciar2018-challenge.grand-challenge.org) | 4-class breast cancer grading | Normal, Benign, InSitu, Invasive | ~400 (full dataset) |

## Usage

This repo contains only the **LoRA adapter weights** (~35 MB). The base model
(`google/medgemma-4b-it`) is downloaded separately and requires a HuggingFace
token with access to the gated MedGemma model.

### Quickest way — Kaggle demo notebook

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/)

The notebook handles everything: GPU setup, model download, adapter loading,
and launches a public Gradio UI.

### Load with the HistoLab app

```bash
git clone https://github.com/karaditya/medgemma-histolab
cd medgemma-histolab
pip install -e .
export HF_TOKEN=hf_...
export ADAPTER_REPO_ID={repo_id}
python app.py
```

### Load manually with PEFT

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch

BASE = "google/medgemma-4b-it"
ADAPTER = "{repo_id}"
TOKEN = "hf_..."   # needs MedGemma access

processor = AutoProcessor.from_pretrained(BASE, token=TOKEN)
base = AutoModelForImageTextToText.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto", token=TOKEN
)
model = PeftModel.from_pretrained(base, ADAPTER, token=TOKEN)
model = model.merge_and_unload()   # optional: bake adapter in for faster inference
```

## Intended Use & Limitations

- **Intended for:** Research, education, and demonstration purposes.
- **Not intended for:** Clinical diagnosis or patient care decisions.
- The model was trained on patch-level images (224×224 px) — whole-slide inference
  requires tiling.
- BACH accuracy (68.3%) is limited by the small dataset size (~400 images total).

## License

The adapter weights are released under **Apache 2.0**.
The base model (`google/medgemma-4b-it`) is subject to
[Google's MedGemma Terms of Use](https://huggingface.co/google/medgemma-4b-it).
"""


def upload_model_card(api, repo_id: str, token: str) -> None:
    """Generate and upload the README.md model card."""
    print("\nUploading model card (README.md) …")
    card_content = generate_model_card(repo_id)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as tmp:
        tmp.write(card_content)
        tmp_path = tmp.name
    try:
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        print("  ✓ README.md")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def upload_adapter(adapter_path: str = ADAPTER_PATH, repo_name: str = REPO_NAME,
                   model_card_only: bool = False):
    from huggingface_hub import HfApi, create_repo

    token = _get_token()
    api = HfApi()

    user = api.whoami(token=token)
    username = user["name"]
    repo_id = f"{username}/{repo_name}"

    print(f"Repo: https://huggingface.co/{repo_id}")
    create_repo(repo_id=repo_id, token=token, exist_ok=True, private=False)

    if not model_card_only:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            print(f"Error: Adapter not found at: {adapter_path}")
            print("Train the model first with:")
            print("  python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml")
            sys.exit(1)

        print(f"\nUploading adapter from {adapter_path} …")
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

    upload_model_card(api, repo_id, token)

    print(f"\n✅ Done! Model page: https://huggingface.co/{repo_id}")
    if not model_card_only:
        print(f"\nAdd this to your HuggingFace Space / Kaggle secrets:")
        print(f"   ADAPTER_REPO_ID = {repo_id}")
        print(f"   HF_TOKEN        = <your token with MedGemma access>")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-card-only", action="store_true",
                        help="Only upload/update the README.md, skip adapter weights")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH,
                        help=f"Path to adapter directory (default: {ADAPTER_PATH})")
    parser.add_argument("--repo-name", default=REPO_NAME,
                        help=f"HF Hub repo name (default: {REPO_NAME})")
    args = parser.parse_args()

    upload_adapter(
        adapter_path=args.adapter_path,
        repo_name=args.repo_name,
        model_card_only=args.model_card_only,
    )
