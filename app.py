"""
HistoLab — MedGemma Fine-Tuned Histopathology Classifier
=========================================================
HuggingFace Spaces entry point.

Fine-tuned MedGemma-4B-IT with QLoRA on 5 000 samples from:
  • NCT-CRC-HE-100K  (9-class colorectal tissue)
  • PatchCamelyon    (2-class metastasis detection)
  • BACH ICIAR 2018  (4-class breast cancer grading)

Results (exp1b — 5K samples/dataset):
  Overall: 81.0%  (+27.6 pp vs 53.4% baseline)
  CRC:     89.6%  |  PCam: 85.2%  |  BACH: 68.3%

Usage (local):
    python app.py

HF Spaces secrets required:
    HF_TOKEN        — token with access to google/medgemma-4b-it (gated)
    ADAPTER_REPO_ID — HF Hub repo containing the LoRA adapter weights
                      (upload once with: python upload_adapter.py)
"""

import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# Set these as HuggingFace Space secrets, or export them locally.
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)

# Repo ID of your uploaded adapter on HuggingFace Hub.
# Run `python upload_adapter.py` once to create this, then paste the repo ID here.
ADAPTER_REPO_ID = os.environ.get(
    "ADAPTER_REPO_ID",
    "YOUR_HF_USERNAME/medgemma-histolab-5k",   # <-- replace before deploying
)

ADAPTER_LOCAL_PATH = "./adapter"


# ── Adapter download ─────────────────────────────────────────────────────────
def download_adapter() -> str | None:
    """
    Download the fine-tuned LoRA adapter from HuggingFace Hub.
    Skips download if files are already present locally.
    Returns the local path, or None if ADAPTER_REPO_ID is not configured.
    """
    adapter_dir = Path(ADAPTER_LOCAL_PATH)

    if (adapter_dir / "adapter_config.json").exists():
        logger.info(f"Adapter already present at {ADAPTER_LOCAL_PATH}")
        return ADAPTER_LOCAL_PATH

    if "YOUR_HF_USERNAME" in ADAPTER_REPO_ID:
        logger.warning(
            "ADAPTER_REPO_ID is not configured — set it as a Space secret "
            "or edit ADAPTER_REPO_ID in app.py. Running baseline model only."
        )
        return None

    try:
        from huggingface_hub import snapshot_download

        logger.info(f"Downloading adapter from {ADAPTER_REPO_ID} …")
        snapshot_download(
            repo_id=ADAPTER_REPO_ID,
            local_dir=ADAPTER_LOCAL_PATH,
            token=HF_TOKEN,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        logger.info("Adapter downloaded successfully.")
        return ADAPTER_LOCAL_PATH

    except Exception as exc:
        logger.error(f"Failed to download adapter: {exc}")
        return None


# ── Model setup ───────────────────────────────────────────────────────────────
adapter_path = download_adapter()

from histolab.medgemma_integration import MedGemmaWrapper, MedGemmaConfig
from histolab.ui.chat_app import HistoLabChatApp

config = MedGemmaConfig(
    model_name="google/medgemma-4b-it",
    model_type="fine-tuned" if adapter_path else "baseline",
    fine_tuned_path=adapter_path,
    hf_token=HF_TOKEN,
    max_new_tokens=512,
)

wrapper = MedGemmaWrapper(config)

app_instance = HistoLabChatApp(
    wrapper=wrapper,
    default_fine_tuned_path=adapter_path or "",
)

# ── Build Gradio interface ────────────────────────────────────────────────────
demo = app_instance.build_interface()

if __name__ == "__main__":
    demo.launch(
        css=getattr(app_instance, "_css", None),
        show_api=False,
    )
