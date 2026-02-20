# HistoLab Experiments: Improving MedGemma-4B Histopathology Fine-Tuning

## Current Baseline
| Dataset | Accuracy | F1 (macro) |
|---------|----------|------------|
| CRC     | 79.0%    | 78.9%      |
| PCam    | 78.7%    | 78.5%      |
| BACH    | 65.0%    | 58.5%      |
| **Overall** | **74.2%** | **72.0%** |

## Experiment Plan

| # | Experiment | Hypothesis | Key Change | Expected Gain |
|---|-----------|-----------|------------|---------------|
| 1b | Data Scale 5K | H1: Data scale dominates | `samples_per_dataset: 5000` | 5-10pp over baseline |
| 2 | Stain Augmentation | H2: Stain jitter helps | `stain_jitter: true`, wider color jitter | 2-5pp |
| 3 | Dataset Prompts | H5: Richer prompts help | CRC/PCam/BACH-specific prompts | 2-5pp |
| 4 | Vision LoRA | H4: Vision encoder LoRA has diminishing returns | `lora_target_scope: llm_plus_vision` | <3pp |
| 5 | Class Weighting | H6: Weighted training fixes weak classes | Oversample STR/NORM/InSitu | 10+pp per-class |
| 6 | Extended Training | - | `epochs: 2`, `patience: 100` | 1-3pp |
| 7 | Combined Best | - | Stack winning interventions | 5-15pp total |

## Success Criteria
- Overall accuracy >80% (from 74.2%)
- BACH accuracy >75% (from 65%)
- BACH InSitu >30% (from 0%)
- CRC Stroma >70% (from 55.6%)

## Project Structure
```
configs/
  training_config.yaml          # Base training hyperparameters
  exp1b_data_scale_5k.yaml      # Experiment config (submitted)
experiments/
  README.md                     # This file
  download_datasets.py          # Dataset downloader and verifier
  run_experiment.py             # Experiment runner (supports all experiments)
models/
  exp1b_data_scale_5k/          # Adapter weights (git-ignored, use HF Hub)
    final_model/
      adapter_config.json
      adapter_model.safetensors
```

## Quick Start

### 1. Download datasets
```bash
python experiments/download_datasets.py --all --data-dir data/datasets
```

### 2. Run the submitted experiment
```bash
python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml
```

### 3. Run with overrides
```bash
# Skip baseline evaluation (faster iteration)
python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml --no-baseline

# Override output directory
python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml --output-dir /scratch/models
```

### 4. Check results
Results are saved as JSON in `models/{experiment_id}/experiments/`:
```bash
ls models/exp1b_data_scale_5k/experiments/exp*_comparison.json
```

## Experiment Details

### Exp 1b: Data Scale (5K samples/dataset)
Uses 5000 samples per dataset, stratified by class. BACH uses all ~400 available samples
since it is smaller. Compared against the zero-shot MedGemma-4B baseline.

### Exp 2: Stain Augmentation
Enables `stain_jitter` which perturbs H&E stain concentrations independently, simulating
inter-lab variation. Also widens color jitter ranges from (0.85, 1.15) to (0.75, 1.25).
Does NOT use stain normalization (which was shown to hurt MedGemma).

### Exp 3: Dataset-Specific Prompts
Replaces the generic prompt with anatomically-aware prompts:
- CRC: mentions "colorectal tissue", "20x magnification"
- PCam: mentions "lymph node", "metastatic tumor"
- BACH: mentions "breast tissue biopsy"

### Exp 4: Vision Encoder LoRA
Adds LoRA adapters to SigLIP vision encoder (in addition to LLM). Uses reduced `lora_r=8`
to prevent overfitting from more parameters. Tests whether MedSigLIP's pre-trained features
are already sufficient.

### Exp 5: Class-Weighted Training
Oversamples known weak classes:
- CRC Stroma (55.6% accuracy) → 2x
- CRC Normal (61.1%) → 2x
- BACH InSitu (0%) → 3x

### Exp 6: Extended Training
Tests whether 1 epoch under-trains. Runs 2 epochs with patience=100 early stopping.

### Exp 7: Combined Best
Stacks all improvements that showed >2pp gain.

## Key Design Decisions

1. **No stain normalization**: Macenko SN hurts MedGemma because MedSigLIP was pre-trained
   on raw (non-normalized) histopath data. SN creates a domain shift.

2. **Stain augmentation instead**: Adds variation to teach stain invariance without
   destroying features the model already knows.

3. **bf16 eval, 4-bit train**: QLoRA uses 4-bit for training efficiency, but evaluation
   loads base model in bf16 for lossless adapter merge. Never 4-bit quantize a merged model.

4. **Adapter-only saves**: Never `merge_and_unload()` on 4-bit models. Save adapter
   separately, merge at eval time in bf16.

5. **Same prompt at train and eval**: Critical for VLM fine-tuning. The model must see
   the same prompt format it was trained on.
