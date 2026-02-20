#!/usr/bin/env python3
"""
Experiment Runner for HistoLab Co-Scientist Experiments

Extends the base finetune_runner.py with experiment-specific features:
- Dataset-specific prompts (Exp 3)
- Stain augmentation toggle (Exp 2)
- Vision encoder LoRA (Exp 4)
- Class-weighted oversampling (Exp 5)
- Extended training with early stopping patience (Exp 6)
- All experiments controlled via YAML config

Usage:
    # Run a single experiment
    python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml

    # Run with overrides
    python experiments/run_experiment.py --config configs/exp1b_data_scale_5k.yaml --no-wandb
"""

import os
import sys
import gc
import json
import logging
import random
import copy
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from histolab.data_loader import (
    DATASETS, DatasetLoader, translate_label, translate_class_names, CRC_LABEL_MAP
)
from histolab.preprocessing import HistoAugmentor, balance_multi_dataset

logger = logging.getLogger(__name__)


def _cleanup_gpu():
    """Force GPU memory release between pipeline stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU cleanup: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Experiment-aware configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Extended configuration supporting all experiment variables."""

    # --- Experiment metadata ---
    experiment_id: str = "exp0_baseline"
    experiment_name: str = "baseline"
    description: str = ""

    # --- Dataset settings ---
    datasets: List[str] = field(default_factory=lambda: ["crc", "pcam", "bach"])
    samples_per_dataset: int = 0  # 0 = use all
    data_dir: str = "data/datasets"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Model settings ---
    base_model: str = "google/medgemma-4b-it"
    quantization: int = 4
    use_qlora: bool = True

    # --- LoRA settings ---
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # "llm_only" = only language_model layers; "llm_plus_vision" = all layers
    lora_target_scope: str = "llm_only"

    # --- Training settings ---
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    image_size: int = 384
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    early_stopping_patience: int = 50  # Steps of EMA loss < threshold before stop

    # --- Prompt settings ---
    # "generic" = original prompt; "dataset_specific" = Exp 3 prompts
    prompt_style: str = "generic"

    # --- Augmentation settings (Exp 2) ---
    augment_train: bool = True
    stain_jitter: bool = False
    brightness_range_min: float = 0.85
    brightness_range_max: float = 1.15
    contrast_range_min: float = 0.85
    contrast_range_max: float = 1.15
    saturation_range_min: float = 0.85
    saturation_range_max: float = 1.15
    hue_shift_range: int = 10

    # --- Class weighting (Exp 5) ---
    class_weighting: str = "none"  # "none" or "weighted"
    # Per-dataset oversampling multipliers for weak classes
    # Format: {"dataset/class": multiplier}
    oversample_map: Dict[str, float] = field(default_factory=dict)

    # --- Output settings ---
    output_dir: str = "models/finetuned"
    compare_baseline: bool = True
    save_predictions: bool = True

    # --- WandB ---
    use_wandb: bool = True
    wandb_project: str = "histolab-experiments"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        valid_fields = set(cls.__annotations__.keys())
        parsed = {}

        for key, value in raw.items():
            if key not in valid_fields:
                logger.warning(f"Ignoring unknown config key: {key}")
                continue

            field_type = cls.__annotations__.get(key)

            # Type coercion
            if field_type == int:
                try:
                    parsed[key] = int(value)
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert {key}={value} to int, skipping")
                    continue
            elif field_type == float:
                try:
                    parsed[key] = float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert {key}={value} to float, skipping")
                    continue
            elif field_type == bool:
                if isinstance(value, str):
                    parsed[key] = value.lower() in ("true", "1", "yes", "on")
                else:
                    parsed[key] = bool(value)
            else:
                parsed[key] = value

        config = cls(**parsed)

        # Resolve relative data_dir relative to project root
        data_path = Path(config.data_dir)
        if not data_path.is_absolute():
            config.data_dir = str(PROJECT_ROOT / data_path)

        # Resolve relative output_dir relative to project root
        output_path = Path(config.output_dir)
        if not output_path.is_absolute():
            config.output_dir = str(PROJECT_ROOT / output_path)

        print(f"Loaded experiment config: {config.experiment_id}")
        print(f"  Description: {config.description}")
        for key in sorted(parsed.keys()):
            print(f"  {key}: {parsed[key]}")
        print(f"  data_dir (resolved): {config.data_dir}")
        print(f"  output_dir (resolved): {config.output_dir}")

        return config


# ---------------------------------------------------------------------------
# Dataset-specific prompts (Experiment 3)
# ---------------------------------------------------------------------------

DATASET_SPECIFIC_PROMPTS = {
    "crc": (
        "This is an H&E stained colorectal tissue patch at 20x magnification. "
        "Classify the tissue type. Options: [{class_list}]. "
        "Answer with only the tissue type."
    ),
    "pcam": (
        "This is an H&E stained lymph node tissue section. "
        "Does this patch contain metastatic tumor tissue? "
        "Options: [{class_list}]. "
        "Answer with only Normal or Tumor."
    ),
    "bach": (
        "This is an H&E stained breast tissue biopsy at high magnification. "
        "Classify the tissue as one of: [{class_list}]. "
        "Answer with only the classification."
    ),
}

GENERIC_PROMPT = (
    "Classify this histopathology image. "
    "The possible tissue types are: {class_list}. "
    "What type of tissue is shown? Answer with only the tissue type name."
)


def generate_prompt(
    dataset_name: str,
    class_names: List[str],
    prompt_style: str = "generic",
) -> str:
    """Generate the classification prompt for a dataset.

    Args:
        dataset_name: Which dataset (crc, pcam, bach)
        class_names: Raw class names (will be translated)
        prompt_style: "generic" or "dataset_specific"

    Returns:
        Prompt string (question only, no answer)
    """
    translated = translate_class_names(class_names)
    class_list = ", ".join(translated)

    if prompt_style == "dataset_specific" and dataset_name in DATASET_SPECIFIC_PROMPTS:
        template = DATASET_SPECIFIC_PROMPTS[dataset_name]
    else:
        template = GENERIC_PROMPT

    return template.format(class_list=class_list)


# ---------------------------------------------------------------------------
# Augmentation pipeline builder
# ---------------------------------------------------------------------------

def build_augmentor(config: ExperimentConfig) -> Optional[HistoAugmentor]:
    """Build augmentation pipeline from config."""
    if not config.augment_train:
        return None

    return HistoAugmentor(
        geometric=True,
        color_jitter=True,
        stain_jitter=config.stain_jitter,
        brightness_range=(config.brightness_range_min, config.brightness_range_max),
        contrast_range=(config.contrast_range_min, config.contrast_range_max),
        saturation_range=(config.saturation_range_min, config.saturation_range_max),
        hue_shift_range=config.hue_shift_range,
    )


# ---------------------------------------------------------------------------
# Class-weighted oversampling (Experiment 5)
# ---------------------------------------------------------------------------

def apply_class_weighting(
    dataset_splits: Dict[str, Any],
    config: ExperimentConfig,
) -> Tuple[List[Dict], List[Dict]]:
    """Apply class-weighted oversampling to training data.

    If config.class_weighting == "weighted", uses oversample_map to
    multiply weak classes. Falls back to balance_multi_dataset for
    general inter-dataset balancing.

    Returns:
        (balanced_train, balanced_val)
    """
    if config.class_weighting == "none":
        # No weighting — use standard balancing
        return balance_multi_dataset(dataset_splits)

    # Weighted: first do standard balancing, then apply oversampling multipliers
    balanced_train, balanced_val = balance_multi_dataset(dataset_splits)

    if not config.oversample_map:
        return balanced_train, balanced_val

    # Group balanced_train by dataset/class
    groups = defaultdict(list)
    for sample in balanced_train:
        ds = sample.get("dataset", "unknown")
        label = sample.get("label_name", sample.get("label", "unknown"))
        key = f"{ds}/{label}"
        groups[key].append(sample)

    extra_samples = []
    for key, multiplier in config.oversample_map.items():
        if key in groups and multiplier > 1.0:
            original = groups[key]
            n_extra = int(len(original) * (multiplier - 1.0))
            extra_samples.extend(random.choices(original, k=n_extra))
            logger.info(f"Oversampled {key}: +{n_extra} samples (x{multiplier})")

    balanced_train.extend(extra_samples)
    random.shuffle(balanced_train)
    logger.info(f"After class weighting: {len(balanced_train)} train samples")

    return balanced_train, balanced_val


# ---------------------------------------------------------------------------
# LoRA target module configuration (Experiment 4)
# ---------------------------------------------------------------------------

def get_lora_target_regex(scope: str) -> str:
    """Return LoRA target_modules regex based on scope.

    Args:
        scope: "llm_only" or "llm_plus_vision"

    Returns:
        Regex string for PEFT target_modules
    """
    if scope == "llm_plus_vision":
        # Match q/k/v/o_proj in ALL layers (both LLM and vision encoder)
        return r".*\.(q_proj|k_proj|v_proj|o_proj)"
    else:
        # Default: only language model layers (vision encoder frozen)
        return r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)"


# ---------------------------------------------------------------------------
# Data loading (reuses patterns from finetune_runner.py)
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplit:
    """Represents a dataset split."""
    train: List[Dict]
    validation: List[Dict]
    test: List[Dict]
    dataset_name: str
    class_names: List[str]
    num_classes: int


class ExperimentDataLoader:
    """Load datasets with experiment-specific preprocessing."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.base_path = Path(config.data_dir)
        self.augmentor = build_augmentor(config)

    def load_dataset(self, dataset_name: str) -> DatasetSplit:
        """Load a single dataset with stratified splitting."""
        dataset_info = DATASETS.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_path = self.base_path / dataset_name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                f"Run: python experiments/download_datasets.py --datasets {dataset_name}"
            )

        max_samples = self.config.samples_per_dataset if self.config.samples_per_dataset > 0 else None

        # Load all samples from directory structure
        all_samples = self._load_from_directory(dataset_path, dataset_info)

        if not all_samples:
            raise ValueError(f"No samples found in {dataset_path}")

        # Cap samples if needed
        if max_samples and len(all_samples) > max_samples:
            # Stratified cap: proportionally reduce each class
            class_groups = defaultdict(list)
            for s in all_samples:
                label = s.get("label_name", s.get("label", "unknown"))
                class_groups[label].append(s)

            per_class = max(1, max_samples // len(class_groups))
            capped = []
            for label, samples in class_groups.items():
                random.shuffle(samples)
                capped.extend(samples[:per_class])

            random.shuffle(capped)
            all_samples = capped[:max_samples]

        # Stratified split
        class_groups = defaultdict(list)
        for s in all_samples:
            label = s.get("label_name", s.get("label", "unknown"))
            class_groups[label].append(s)

        train_samples, val_samples, test_samples = [], [], []

        for label, samples_in_class in class_groups.items():
            random.shuffle(samples_in_class)
            n = len(samples_in_class)
            n_train = int(n * self.config.train_ratio)
            n_val = int(n * self.config.val_ratio)

            train_samples.extend(samples_in_class[:n_train])
            val_samples.extend(samples_in_class[n_train:n_train + n_val])
            test_samples.extend(samples_in_class[n_train + n_val:])

        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        # Log class distribution
        for split_name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            dist = Counter(s.get("label_name", s.get("label", "?")) for s in split_data)
            dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            print(f"  {dataset_name} {split_name} ({len(split_data)}): {dist_str}")

        return DatasetSplit(
            train=train_samples,
            validation=val_samples,
            test=test_samples,
            dataset_name=dataset_name,
            class_names=dataset_info.classes,
            num_classes=dataset_info.num_classes,
        )

    def _load_from_directory(self, dataset_path: Path, dataset_info) -> List[Dict]:
        """Scan image files from class subdirectories (lazy — stores paths only)."""
        samples = []

        for class_dir in dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if class_name not in dataset_info.classes:
                continue

            img_paths = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                img_paths.extend(class_dir.glob(ext))

            for img_path in img_paths:
                samples.append({
                    "image_path": str(img_path),
                    "label_name": class_name,
                    "dataset": dataset_info.name,
                })

        return samples

    def _estimate_memory(self) -> float:
        """Estimate total memory for loading all images (GB)."""
        # Known sizes per image (bytes, uncompressed RGB)
        IMAGE_SIZES = {
            "crc": 224 * 224 * 3,       # 147KB per image
            "pcam": 96 * 96 * 3,        # 27KB per image
            "bach": 2048 * 1536 * 3,    # 9MB per image
        }
        total_bytes = 0
        for ds_name in self.config.datasets:
            ds_path = self.base_path / ds_name
            if not ds_path.exists():
                continue
            # Count images
            count = 0
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                count += len(list(ds_path.rglob(ext)))
            # Cap if configured
            if self.config.samples_per_dataset > 0:
                count = min(count, self.config.samples_per_dataset)
            per_image = IMAGE_SIZES.get(ds_name, 224 * 224 * 3)
            total_bytes += count * per_image
        return total_bytes / (1024 ** 3)

    def load_all(self) -> Dict[str, DatasetSplit]:
        """Load all configured datasets."""
        print(f"\n{'='*70}")
        print("LOADING DATASETS")
        print(f"{'='*70}")
        print(f"Datasets: {self.config.datasets}")
        print(f"Samples per dataset: {self.config.samples_per_dataset or 'ALL'}")

        # Memory estimate (for reference — images are loaded lazily now)
        est_gb = self._estimate_memory()
        print(f"Dataset size on disk: ~{est_gb:.1f}GB (images loaded lazily, not into RAM)")
        print(f"{'='*70}\n")

        result = {}
        for name in self.config.datasets:
            try:
                result[name] = self.load_dataset(name)
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                print(f"  ERROR: {e}")

        # Summary table
        total_train = sum(len(d.train) for d in result.values())
        total_val = sum(len(d.validation) for d in result.values())
        total_test = sum(len(d.test) for d in result.values())

        print(f"\n{'─'*70}")
        print(f"{'Dataset':<10} {'Classes':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
        print(f"{'─'*70}")
        for name, ds in result.items():
            print(f"{name:<10} {ds.num_classes:>8} {len(ds.train):>8} {len(ds.validation):>8} {len(ds.test):>8}")
        print(f"{'─'*70}")
        print(f"{'TOTAL':<10} {'':>8} {total_train:>8} {total_val:>8} {total_test:>8}")
        print(f"{'─'*70}\n")

        return result


# ---------------------------------------------------------------------------
# Data preparation (with experiment-aware prompts and augmentation)
# ---------------------------------------------------------------------------

class ExperimentPreprocessor:
    """Preprocess samples with experiment-specific prompts and augmentation."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.image_size = config.image_size
        self.augmentor = build_augmentor(config)

    def preprocess_image(self, image: Image.Image, augment: bool = False) -> Image.Image:
        """Resize and optionally augment an image."""
        image = image.resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        if augment and self.augmentor is not None:
            image = self.augmentor(image)
        return image

    def prepare_sample_metadata(
        self,
        sample: Dict,
        dataset_name: str,
        class_names: List[str],
    ) -> Dict:
        """Prepare metadata for a sample WITHOUT loading the image.

        Returns a lightweight dict with image_path, text, label, etc.
        Images are loaded lazily during training/eval.
        """
        label_name = sample.get("label_name", sample.get("label", "unknown"))

        # Translate CRC abbreviations
        label_name = translate_label(label_name)
        translated_classes = translate_class_names(class_names)

        # Generate prompt using experiment's prompt style
        text = generate_prompt(
            dataset_name=dataset_name,
            class_names=class_names,
            prompt_style=self.config.prompt_style,
        )

        label = translated_classes.index(label_name) if label_name in translated_classes else -1

        return {
            "image_path": sample.get("image_path", sample.get("path", "")),
            "text": text,
            "label": label,
            "label_name": label_name,
            "dataset": dataset_name,
        }

    def prepare_all(
        self,
        dataset_splits: Dict[str, DatasetSplit],
    ) -> Tuple[List[Dict], List[Dict], Dict[str, List[Dict]]]:
        """Prepare metadata for train, val, and test data (lazy — no image loading).

        Images are NOT loaded here. Only paths and text/label metadata are prepared.
        Images are loaded lazily during training (via HF Dataset Image feature)
        and during evaluation (one at a time in the eval loop).

        Returns:
            (train_data, val_data, test_data_by_dataset)
        """
        train_data = []
        val_data = []
        test_data = {}

        for name, split in dataset_splits.items():
            # Train: metadata only (augmentation applied later during training)
            for sample in split.train:
                prepared = self.prepare_sample_metadata(
                    sample, name, split.class_names,
                )
                train_data.append(prepared)

            # Val: metadata only
            for sample in split.validation:
                prepared = self.prepare_sample_metadata(
                    sample, name, split.class_names,
                )
                val_data.append(prepared)

            # Test: metadata only
            test_data[name] = [
                self.prepare_sample_metadata(s, name, split.class_names)
                for s in split.test
            ]

        random.shuffle(train_data)
        random.shuffle(val_data)

        print(f"Prepared {len(train_data)} training, {len(val_data)} validation samples (lazy — images not loaded)")
        return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Training (extends LoRATrainer with experiment features)
# ---------------------------------------------------------------------------

def run_training(
    config: ExperimentConfig,
    train_data: List[Dict],
    val_data: List[Dict],
) -> str:
    """Run fine-tuning with experiment-specific LoRA configuration.

    Returns path to saved adapter.
    """
    from histolab.training import TrainingConfig, LoRATrainer

    print(f"\n{'='*70}")
    print(f"TRAINING: {config.experiment_id}")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"LoRA target scope: {config.lora_target_scope}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Epochs: {config.epochs}")
    print(f"Early stopping patience: {config.early_stopping_patience}")
    print(f"{'='*70}\n")

    from datasets import Dataset

    # Create lightweight HF Datasets with image paths (NOT loaded into memory).
    # training.py's preprocess_function handles lazy loading from image_path
    # during .map(), loading only batch_size=8 images at a time.
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    print(f"Created lazy datasets: {len(train_dataset)} train, {len(val_dataset)} val (images NOT in memory)")

    training_config = TrainingConfig(
        model_name=config.base_model,
        model_type="qlora" if config.use_qlora else "lora",
        load_in_4bit=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=float(config.learning_rate),
        output_dir=config.output_dir,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        max_seq_length=config.max_seq_length,
        max_image_size=config.image_size,
    )

    trainer = LoRATrainer(training_config=training_config)

    # --- Augmentation: set image transform on trainer for lazy loading ---
    augmentor = build_augmentor(config)
    if augmentor is not None:
        trainer._image_transform = augmentor
        print(f"Augmentation enabled: stain_jitter={config.stain_jitter}")

    # --- Experiment 4: Vision Encoder LoRA ---
    # We need to override the LoRA target_modules and vision encoder freezing
    # BEFORE calling prepare_model_and_tokenizer, OR monkey-patch after.
    # The cleanest approach: override the config's target_modules regex,
    # and conditionally unfreeze vision encoder after model load.

    # Override target_modules in the config
    target_regex = get_lora_target_regex(config.lora_target_scope)
    # Store original for potential restore
    _original_prepare = trainer.prepare_model_and_tokenizer

    def _patched_prepare(model_path=None):
        """Prepare model with experiment-specific LoRA targets."""
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType

        model_name = model_path or trainer.config.model_name

        # Load HF token (check project root, then CWD, then env)
        hf_token = None
        for token_path in [PROJECT_ROOT / "hf_token.txt", Path("hf_token.txt")]:
            if token_path.exists():
                hf_token = token_path.read_text().strip()
                break
        if hf_token is None:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        # Quantization
        quantization_config = None
        if trainer.config.load_in_4bit or trainer.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=trainer.config.load_in_4bit,
                load_in_8bit=trainer.config.load_in_8bit,
                bnb_4bit_compute_dtype=getattr(torch, trainer.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=trainer.config.bnb_4bit_quant_type,
            )

        # Processor
        trainer.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Device map
        device_map = "auto"
        if torch.cuda.is_available():
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                device_map = {"": 0} if gpu_mem >= 10 else "auto"
            except Exception:
                pass

        # Load model
        load_kwargs = dict(
            quantization_config=quantization_config,
            dtype=getattr(torch, trainer.config.bnb_4bit_compute_dtype),
            device_map=device_map,
            trust_remote_code=True,
        )
        if hf_token:
            load_kwargs["token"] = hf_token

        trainer.model = AutoModelForImageTextToText.from_pretrained(
            model_name, **load_kwargs
        )

        # --- Vision encoder freeze/unfreeze (Exp 4) ---
        if config.lora_target_scope == "llm_only":
            # Standard: freeze vision encoder
            if hasattr(trainer.model, "vision_tower"):
                for param in trainer.model.vision_tower.parameters():
                    param.requires_grad = False
                logger.info("Vision encoder FROZEN (llm_only mode)")
        else:
            # Exp 4: keep vision encoder unfrozen for LoRA
            logger.info("Vision encoder UNFROZEN (llm_plus_vision mode)")

        # Freeze image_newline if present
        if hasattr(trainer.model, "image_newline"):
            for param in trainer.model.image_newline.parameters():
                param.requires_grad = False

        # LoRA with experiment-specific target regex
        if trainer.config.model_type in ["lora", "qlora"]:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=trainer.config.lora_r,
                lora_alpha=trainer.config.lora_alpha,
                lora_dropout=trainer.config.lora_dropout,
                target_modules=target_regex,
                bias="none",
                inference_mode=False,
            )
            trainer.model = get_peft_model(trainer.model, lora_config)
            trainer.model.print_trainable_parameters()

        # Enable input gradients BEFORE gradient checkpointing (critical for QLoRA)
        trainer.model.enable_input_require_grads()
        if trainer.config.gradient_checkpointing:
            trainer.model.gradient_checkpointing_enable()

        trainer.model.train()

        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({trainable/total:.2%})")

    # Replace prepare method
    trainer.prepare_model_and_tokenizer = _patched_prepare

    # Prepare model
    trainer.prepare_model_and_tokenizer()

    # Train
    final_model_path = trainer.train(
        train_dataset,
        val_dataset,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        wandb_run_name=config.wandb_run_name or config.experiment_id,
    )

    print(f"\nTraining completed! Adapter saved to: {final_model_path}")
    return str(final_model_path)


# ---------------------------------------------------------------------------
# Evaluation (reuses finetune_runner.py pattern)
# ---------------------------------------------------------------------------

def run_evaluation(
    config: ExperimentConfig,
    model_path: str,
    test_data: Dict[str, List[Dict]],
    dataset_splits: Dict[str, DatasetSplit],
    eval_label: str = "finetuned",
) -> Dict[str, Dict]:
    """Evaluate a model (baseline or fine-tuned) on test sets.

    Args:
        config: Experiment config
        model_path: Path to adapter directory (fine-tuned) or base model name (baseline)
        test_data: Dict mapping dataset name to list of prepared test samples
        dataset_splits: Original dataset splits (for class names)
        eval_label: "baseline" or "finetuned"

    Returns:
        Dict mapping dataset name to metrics dict
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from sklearn.metrics import accuracy_score, f1_score
    from peft import PeftModel

    print(f"\n{'='*70}")
    print(f"{eval_label.upper()} EVALUATION")
    print(f"{'='*70}")

    model_path = Path(model_path) if eval_label == "finetuned" else None

    # Load model
    if eval_label == "finetuned" and model_path is not None:
        base_config_path = model_path / "base_model_config.json"
        if not base_config_path.exists():
            print(f"ERROR: No base_model_config.json in {model_path}")
            return {}

        with open(base_config_path) as f:
            base_cfg = json.load(f)
        base_model_name = base_cfg["base_model_name"]

        print(f"Loading base model in bfloat16: {base_model_name}")
        print(f"Applying adapter from: {model_path}")

        hf_token = None
        for token_path in [PROJECT_ROOT / "hf_token.txt", Path("hf_token.txt")]:
            if token_path.exists():
                hf_token = token_path.read_text().strip()
                break
        if hf_token is None:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

        load_kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        if hf_token:
            load_kwargs["token"] = hf_token

        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name, **load_kwargs
        )

        # Apply LoRA adapter and merge (lossless in bf16)
        model = PeftModel.from_pretrained(base_model, str(model_path))
        print("Merging adapter into bfloat16 base (lossless)...")
        model = model.merge_and_unload()
        model.eval()
    else:
        # Baseline: load zero-shot model
        from histolab.data_loader import MedGemmaTester

        tester = MedGemmaTester(
            model_path=config.base_model,
            quantization=config.quantization,
        )
        if not tester.load_model():
            print("Failed to load baseline model")
            return {}
        processor = None
        model = None

    results = {}

    for dataset_name, samples in test_data.items():
        if not samples:
            continue

        translated_classes = translate_class_names(
            dataset_splits[dataset_name].class_names
        )

        print(f"\nEvaluating on {dataset_name} ({len(samples)} samples)...")

        predictions = []
        ground_truth = []

        for i, sample in enumerate(tqdm(samples, desc=dataset_name)):
            # Lazy load: image stored as path, load on demand
            image = sample.get("image")
            if image is None or isinstance(image, str):
                image_path = sample.get("image_path", sample.get("image", ""))
                image = Image.open(image_path).convert("RGB")
            true_label = sample["label_name"]

            if eval_label == "baseline":
                # Use MedGemmaTester for baseline
                result = tester.test_zero_shot(image, translated_classes)
                pred_label = result.get("predicted_class", "unknown")
            else:
                # Use experiment's prompt style for fine-tuned eval
                prompt_text = generate_prompt(
                    dataset_name=dataset_name,
                    class_names=dataset_splits[dataset_name].class_names,
                    prompt_style=config.prompt_style,
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=text, images=image, return_tensors="pt"
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

                new_tokens = outputs[0][input_len:]
                response = processor.decode(new_tokens, skip_special_tokens=True).strip()

                # Parse: expect "The answer is: <CLASS>" or just "<CLASS>"
                pred_label = response
                if "The answer is:" in response:
                    pred_label = response.split("The answer is:")[-1].strip()

                # Match to closest known class
                matched = "unknown"
                pred_lower = pred_label.lower()
                for cn in translated_classes:
                    if cn.lower() == pred_lower or cn.lower() in pred_lower:
                        matched = cn
                        break
                pred_label = matched

                if i < 10:
                    print(f"  [{i}] raw={repr(response)}, matched={pred_label}, true={true_label}")

            predictions.append(pred_label)
            ground_truth.append(true_label)

        # Metrics
        accuracy = accuracy_score(ground_truth, predictions)
        f1_macro = f1_score(ground_truth, predictions, average="macro", zero_division=0)

        # Per-class accuracy
        class_correct = Counter()
        class_total = Counter()
        for pred, true in zip(predictions, ground_truth):
            class_total[true] += 1
            if pred == true:
                class_correct[true] += 1

        per_class_acc = {}
        print(f"\n  {dataset_name}: acc={accuracy:.4f}, F1={f1_macro:.4f}")
        for cn in translated_classes:
            total = class_total.get(cn, 0)
            correct = class_correct.get(cn, 0)
            acc = correct / total if total > 0 else 0.0
            per_class_acc[cn] = acc
            print(f"    {cn}: {correct}/{total} = {acc:.2%}")

        num_unknown = predictions.count("unknown")
        print(f"  Unknown predictions: {num_unknown}/{len(predictions)}")

        results[dataset_name] = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "per_class_accuracy": per_class_acc,
            "predictions": list(zip(predictions, ground_truth)),
            "num_samples": len(samples),
        }

    # Cleanup model from GPU to free VRAM for next stage
    print(f"Cleaning up {eval_label} evaluation model from GPU...")
    _cleanup_gpu()

    return results


# ---------------------------------------------------------------------------
# Result comparison and saving
# ---------------------------------------------------------------------------

def compare_and_save(
    config: ExperimentConfig,
    baseline_results: Dict,
    finetuned_results: Dict,
    model_path: str,
):
    """Compare baseline vs fine-tuned and save all results."""
    comparison = {
        "experiment_id": config.experiment_id,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "datasets": config.datasets,
            "samples_per_dataset": config.samples_per_dataset,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_target_scope": config.lora_target_scope,
            "prompt_style": config.prompt_style,
            "stain_jitter": config.stain_jitter,
            "class_weighting": config.class_weighting,
            "augment_train": config.augment_train,
            "early_stopping_patience": config.early_stopping_patience,
            "model_path": model_path,
        },
        "baseline": baseline_results,
        "fine_tuned": finetuned_results,
        "improvement": {},
    }

    print(f"\n{'='*70}")
    print(f"RESULTS: {config.experiment_id}")
    print(f"{'='*70}")

    for ds in config.datasets:
        if ds not in baseline_results or ds not in finetuned_results:
            continue

        bl_acc = baseline_results[ds]["accuracy"]
        ft_acc = finetuned_results[ds]["accuracy"]
        bl_f1 = baseline_results[ds]["f1_macro"]
        ft_f1 = finetuned_results[ds]["f1_macro"]

        comparison["improvement"][ds] = {
            "baseline_accuracy": bl_acc,
            "fine_tuned_accuracy": ft_acc,
            "delta_accuracy": ft_acc - bl_acc,
            "baseline_f1_macro": bl_f1,
            "fine_tuned_f1_macro": ft_f1,
            "delta_f1": ft_f1 - bl_f1,
        }

        print(f"\n  {ds.upper()}:")
        print(f"    Baseline:   acc={bl_acc:.4f}, F1={bl_f1:.4f}")
        print(f"    Fine-tuned: acc={ft_acc:.4f}, F1={ft_f1:.4f}")
        print(f"    Delta:      acc={ft_acc - bl_acc:+.4f}, F1={ft_f1 - bl_f1:+.4f}")

    # Overall
    if baseline_results and finetuned_results:
        avg_bl = np.mean([r["accuracy"] for r in baseline_results.values()])
        avg_ft = np.mean([r["accuracy"] for r in finetuned_results.values()])
        avg_bl_f1 = np.mean([r["f1_macro"] for r in baseline_results.values()])
        avg_ft_f1 = np.mean([r["f1_macro"] for r in finetuned_results.values()])

        comparison["overall"] = {
            "avg_baseline_acc": float(avg_bl),
            "avg_finetuned_acc": float(avg_ft),
            "avg_delta_acc": float(avg_ft - avg_bl),
            "avg_baseline_f1": float(avg_bl_f1),
            "avg_finetuned_f1": float(avg_ft_f1),
            "avg_delta_f1": float(avg_ft_f1 - avg_bl_f1),
        }

        print(f"\n  OVERALL:")
        print(f"    Baseline:   acc={avg_bl:.4f}, F1={avg_bl_f1:.4f}")
        print(f"    Fine-tuned: acc={avg_ft:.4f}, F1={avg_ft_f1:.4f}")
        print(f"    Delta:      acc={avg_ft - avg_bl:+.4f}, F1={avg_ft_f1 - avg_bl_f1:+.4f}")

    # Save JSON
    output_dir = Path(config.output_dir) / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f"{config.experiment_id}_comparison.json"
    with open(result_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to: {result_path}")

    # Log to WandB if enabled
    if config.use_wandb:
        try:
            import wandb
            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=f"{config.experiment_id}_results",
                job_type="evaluation",
                config={
                    "experiment_id": config.experiment_id,
                    "experiment_name": config.experiment_name,
                    **comparison.get("config", {}),
                },
                reinit=True,
            )
            # Log per-dataset metrics
            for ds in config.datasets:
                imp = comparison.get("improvement", {}).get(ds, {})
                if imp:
                    wandb.log({
                        f"{ds}/baseline_acc": imp.get("baseline_accuracy", 0),
                        f"{ds}/finetuned_acc": imp.get("fine_tuned_accuracy", 0),
                        f"{ds}/delta_acc": imp.get("delta_accuracy", 0),
                        f"{ds}/baseline_f1": imp.get("baseline_f1_macro", 0),
                        f"{ds}/finetuned_f1": imp.get("fine_tuned_f1_macro", 0),
                        f"{ds}/delta_f1": imp.get("delta_f1", 0),
                    })
            # Log overall metrics
            overall = comparison.get("overall", {})
            if overall:
                wandb.log({
                    "overall/baseline_acc": overall.get("avg_baseline_acc", 0),
                    "overall/finetuned_acc": overall.get("avg_finetuned_acc", 0),
                    "overall/delta_acc": overall.get("avg_delta_acc", 0),
                    "overall/baseline_f1": overall.get("avg_baseline_f1", 0),
                    "overall/finetuned_f1": overall.get("avg_finetuned_f1", 0),
                    "overall/delta_f1": overall.get("avg_delta_f1", 0),
                })
            # Save result JSON as artifact
            artifact = wandb.Artifact(
                name=f"{config.experiment_id}_results",
                type="experiment_results",
            )
            artifact.add_file(str(result_path))
            run.log_artifact(artifact)
            run.finish()
            print(f"Results logged to WandB: {config.wandb_project}/{config.experiment_id}_results")
        except Exception as e:
            logger.warning(f"Failed to log results to WandB: {e}")

    return comparison


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> Dict:
    """Run the complete experiment pipeline.

    1. Load datasets
    2. Preprocess with experiment-aware prompts and augmentation
    3. Apply class weighting if configured
    4. Run baseline evaluation
    5. Fine-tune with experiment-specific LoRA config
    6. Run fine-tuned evaluation
    7. Compare and save results
    """
    random.seed(42)
    np.random.seed(42)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.experiment_id}")
    print(f"Description: {config.description}")
    print(f"{'='*70}\n")

    try:
        # Step 1: Load datasets
        data_loader = ExperimentDataLoader(config)
        dataset_splits = data_loader.load_all()

        if not dataset_splits:
            print("ERROR: No datasets loaded")
            return {}

        # Step 2: Preprocess
        preprocessor = ExperimentPreprocessor(config)
        train_data, val_data, test_data = preprocessor.prepare_all(dataset_splits)

        # Step 3: Apply class weighting (Exp 5)
        if config.class_weighting == "weighted":
            print("\nApplying class-weighted oversampling...")
            train_data_weighted, val_data_weighted = apply_class_weighting(
                dataset_splits, config
            )
            # Re-prepare metadata for the weighted samples (no image loading)
            weighted_train = []
            for sample in train_data_weighted:
                ds_name = sample.get("dataset", list(dataset_splits.keys())[0])
                if ds_name in dataset_splits:
                    prepared = preprocessor.prepare_sample_metadata(
                        sample, ds_name, dataset_splits[ds_name].class_names,
                    )
                    weighted_train.append(prepared)
            if weighted_train:
                train_data = weighted_train
                random.shuffle(train_data)
            print(f"After weighting: {len(train_data)} training samples")

        # Step 4: Baseline evaluation
        baseline_results = {}
        if config.compare_baseline:
            baseline_results = run_evaluation(
                config, config.base_model, test_data, dataset_splits,
                eval_label="baseline",
            )
            # Free baseline model before loading training model
            _cleanup_gpu()

        # Step 5: Train
        model_path = run_training(config, train_data, val_data)
        # Free training model before loading eval model
        _cleanup_gpu()

        # Step 6: Fine-tuned evaluation
        finetuned_results = run_evaluation(
            config, model_path, test_data, dataset_splits,
            eval_label="finetuned",
        )

        # Step 7: Compare and save
        comparison = compare_and_save(
            config, baseline_results, finetuned_results, model_path
        )

        return comparison

    except Exception as e:
        logger.error(f"Experiment {config.experiment_id} FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Always clean up GPU memory, even on crash
        print("\nFinal GPU cleanup...")
        _cleanup_gpu()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HistoLab Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment YAML config",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline evaluation (faster for iteration)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config = ExperimentConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.no_wandb:
        config.use_wandb = False
    if args.no_baseline:
        config.compare_baseline = False
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir

    # Run
    results = run_experiment(config)

    print(f"\nExperiment {config.experiment_id} completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
