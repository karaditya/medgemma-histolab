# HistoLab — MedGemma Fine-Tuning for Histopathology Classification

<div align="center">

**QLoRA fine-tuning of MedGemma-4B-IT on histopathology patch datasets**
**MedGemma Impact Challenge Submission**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MedGemma](https://img.shields.io/badge/Model-MedGemma--4B--IT-ff69b4.svg)](https://huggingface.co/google/medgemma-4b-it)

</div>

---

## Results (Experiment 1b — 5K samples/dataset)

| Model | CRC (9-class) | PCam (2-class) | BACH (4-class) | **Overall** |
|-------|:---:|:---:|:---:|:---:|
| MedGemma-4B baseline (zero-shot) | 62.0% | 52.8% | 41.4% | 53.4% |
| **Fine-tuned (ours)** | **89.6%** | **85.2%** | **68.3%** | **81.0%** |
| **Improvement** | +27.6 pp | +32.4 pp | +26.9 pp | **+27.6 pp** |

Fine-tuning with 5 000 samples per dataset and 1 epoch of QLoRA training achieves **+27.6 percentage point** overall improvement across three diverse histopathology tasks.

---

## What This Does

HistoLab fine-tunes Google's [MedGemma-4B-IT](https://huggingface.co/google/medgemma-4b-it) vision-language model for histopathology patch classification using QLoRA (4-bit NF4 quantised LoRA, r=16). It provides an interactive Gradio demo comparing baseline vs fine-tuned performance with Chain-of-Thought reasoning.

### Supported Datasets

| Dataset | Task | Classes |
|---------|------|---------|
| **NCT-CRC-HE-100K** | Colorectal tissue type | 9 (Adipose, Background, Debris, Lymphocyte, Mucus, Muscle, Normal, Stroma, Tumor) |
| **PatchCamelyon (PCam)** | Metastasis detection | 2 (Normal, Tumor) |
| **BACH ICIAR 2018** | Breast cancer grading | 4 (Normal, Benign, InSitu, Invasive) |

---

## Repository Structure

```
histolab/                           # Core Python package
├── training.py                     # LoRATrainer — QLoRA fine-tuning loop
├── data_loader.py                  # Dataset loaders for CRC, PCam, BACH
├── medgemma_integration.py         # MedGemmaWrapper for inference
├── preprocessing.py                # Image augmentation & preprocessing
├── ui/chat_app.py                  # Gradio chat interface
└── utils/dataset_labels.py         # Label/class name utilities

finetune_runner.py                  # Main training & evaluation orchestrator
app.py                              # HuggingFace Spaces entry point (Gradio demo)
upload_adapter.py                   # Upload trained adapter to HF Hub

experiments/
├── configs/
│   └── exp1b_data_scale_5k.yaml   # Experiment config (5K samples/dataset)
├── run_experiment.py               # Experiment launcher
└── download_datasets.py            # Automated dataset download

configs/
└── training_config.yaml            # Default training hyperparameters
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA GPU with 16 GB+ VRAM (for QLoRA training; inference works with GPU+CPU split)
- HuggingFace account with MedGemma access — accept the license at [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)

### Setup

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/medgemma-histolab.git
cd medgemma-histolab
pip install -r requirements.txt
pip install -e .
```

### HuggingFace Authentication

```bash
huggingface-cli login
# Or: export HF_TOKEN=hf_xxxx
```

---

## Running the Gradio Demo Locally

```bash
export HF_TOKEN=hf_xxxx
export ADAPTER_REPO_ID=YOUR_HF_USERNAME/medgemma-histolab-5k

python app.py
# Opens at http://localhost:7860
```

Upload any histopathology patch image. Select the dataset type (CRC / PCam / BACH) to set the correct class list. Toggle between baseline and fine-tuned model to compare.

---

## Reproducing the Fine-Tuning

### Step 1 — Download datasets

```bash
python experiments/download_datasets.py
```

Or manually place your datasets at:
```
data/datasets/crc/{ADI,BACK,DEB,LYM,MUC,MUS,NORM,STR,TUM}/*.tif
data/datasets/pcam/{Normal,Tumor}/*.png
data/datasets/bach/{Normal,Benign,InSitu,Invasive}/*.png
```

### Step 2 — Run fine-tuning (Experiment 1b)

```bash
python experiments/run_experiment.py \
    --config experiments/configs/exp1b_data_scale_5k.yaml
```

Trains for 1 epoch with early stopping. Adapter saved to `models/finetuned/exp1b_data_scale_5k/final_model/`.

### Step 3 — Upload adapter for the demo

```bash
python upload_adapter.py
# Prints: ADAPTER_REPO_ID = YOUR_USERNAME/medgemma-histolab-5k
```

---

## Deploying to HuggingFace Spaces

1. Fork / push this repo to GitHub
2. Create a new HF Space (type: Gradio)
3. Link your GitHub repo
4. Add Space secrets:
   - `HF_TOKEN` — your HF token with MedGemma access
   - `ADAPTER_REPO_ID` — output of `upload_adapter.py` (e.g. `yourname/medgemma-histolab-5k`)
5. The Space will install requirements and launch `app.py` automatically

---

## Technical Details

### Fine-Tuning Approach

| Hyperparameter | Value |
|---|---|
| Base model | `google/medgemma-4b-it` |
| Quantisation | 4-bit NF4 (QLoRA) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | LLM layers only (not vision encoder) |
| Adapter size | ~35 MB |
| Training epochs | 1 (with EMA early stopping) |
| Batch size | 1 (grad accum ×4 = effective 4) |
| Learning rate | 2e-5 |
| Samples used | 5 000/dataset (15 000 total) |

### Key Engineering Decisions

1. **Lazy image loading** — Images stored as file paths, loaded per-batch during tokenisation. Reduces peak RAM from ~43 GB to ~1 GB for 15K samples.
2. **Full response label masking** — The entire assistant turn `"The answer is: {class}<end_of_turn>"` is included in the loss, not just the class token. This ensures the model learns the response format and EOS placement.
3. **BF16 merge at eval** — LoRA adapter merged into base in bfloat16 (not 4-bit). 4-bit requantisation destroys the LoRA delta signal, producing empty outputs.
4. **EMA-based early stopping** — Prevents overfitting on small datasets (stops when EMA loss plateaus for 50 steps).
5. **No stain normalisation** — MedGemma's vision encoder (MedSigLIP) was pre-trained on 32.6M raw histopathology patches and already learns stain invariance. Macenko normalisation was found to *hurt* accuracy by creating domain shift.

### Prompt Format

```
User:      Classify this histopathology image. The possible tissue types are:
           Adipose, Background, Debris, Lymphocyte, Mucus, Muscle, Normal, Stroma, Tumor.
           What type of tissue is shown? Answer with only the tissue type name.
Assistant: The answer is: Tumor
```

---

## Citation

If you use this work, please cite:

```bibtex
@misc{histolab2026,
  title  = {HistoLab: QLoRA Fine-Tuning of MedGemma for Histopathology Classification},
  year   = {2026},
  url    = {https://github.com/YOUR_GITHUB_USERNAME/medgemma-histolab}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

The base model [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) is subject to Google's Health AI Developer Foundation terms.
