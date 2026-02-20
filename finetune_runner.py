"""
Fine-tuning Orchestrator - HistoLab

Comprehensive fine-tuning pipeline that supports:
- Multi-dataset selection (CRC, PCam, BACH)
- Automatic train/val/test splitting with stratification
- Data preprocessing for MedGemma
- Zero-shot vs Fine-tuned comparison
- Automatic evaluation and metrics calculation
- Cache management to prevent disk filling

Usage:
    # Fine-tune on single dataset
    python finetune_runner.py --datasets crc --epochs 3
    
    # Fine-tune on multiple datasets
    python finetune_runner.py --datasets crc pcam bach --samples-per-dataset 1000
    
    # Fine-tune and compare with baseline
    python finetune_runner.py --datasets crc --compare-baseline
    
    # Full benchmark pipeline
    python finetune_runner.py --datasets crc pcam bach panda --full-benchmark
    
    # Clear cache before running
    python finetune_runner.py --datasets crc --clear-cache
"""

import os
import sys
import json
import logging
import random
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def clear_huggingface_cache(cache_dir: Optional[str] = None) -> int:
    """
    Clear HuggingFace cache to free up disk space.
    
    Args:
        cache_dir: Optional cache directory (defaults to HF_HOME or ~/.cache/huggingface)
        
    Returns:
        Number of bytes freed
    """
    import shutil
    
    if cache_dir is None:
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    
    cache_path = Path(cache_dir) / "hub"
    if not cache_path.exists():
        logger.info("No HuggingFace cache found to clear")
        return 0
    
    try:
        # Get size before deletion
        total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        
        # Remove only model checkpoints (keep datasets if possible)
        models_dir = cache_path / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
        
        logger.info(f"Cleared HuggingFace cache, freed {total_size / (1024**3):.2f} GB")
        return total_size
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return 0


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning orchestration."""
    # Dataset settings
    datasets: List[str] = field(default_factory=lambda: ["crc"])
    samples_per_dataset: int = 0  # 0 = use all available samples
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Model settings
    base_model: str = "google/medgemma-4b-it"
    quantization: int = 4  # 4-bit for training
    use_qlora: bool = True
    
    # LoRA settings
    # r=16/alpha=32 prevents memorization (r=64 caused loss=0 by step 300)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training settings
    # 1 epoch is sufficient — model converges quickly, more epochs risk overfitting
    epochs: int = 1
    batch_size: int = 1
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    image_size: int = 384
    
    # Output settings
    output_dir: str = "models"
    experiment_name: str = "medgemma_finetune"
    
    # Evaluation settings
    compare_baseline: bool = True
    save_predictions: bool = True
    
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "histolab-medgemma-finetune"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "FineTuneConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Filter only valid fields and convert to correct types
        valid_fields = set(cls.__annotations__.keys())
        filtered_config = {}
        
        for key, value in config_dict.items():
            if key not in valid_fields:
                continue
                
            # Convert string values to appropriate types
            field_type = cls.__annotations__.get(key)
            if field_type == int:
                try:
                    filtered_config[key] = int(value)
                except:
                    logger.warning(f"Cannot convert {key}={value} to int, using default")
                    continue
            elif field_type == float:
                try:
                    filtered_config[key] = float(value)
                except:
                    logger.warning(f"Cannot convert {key}={value} to float, using default")
                    continue
            elif field_type == bool:
                # Handle boolean values properly (YAML allows true/false, but strings like "true" need conversion)
                if isinstance(value, str):
                    value = value.lower()
                    filtered_config[key] = value in ("true", "1", "yes", "on")
                else:
                    filtered_config[key] = bool(value)
            else:
                filtered_config[key] = value
        
        # Print parsed config for debugging
        print("Loaded configuration from YAML:")
        for key, value in filtered_config.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        return cls(**filtered_config)
    
    def to_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)


@dataclass
class DatasetSplit:
    """Represents a data split."""
    train: List[Dict]
    validation: List[Dict]
    test: List[Dict]
    dataset_name: str
    class_names: List[str]
    num_classes: int


class DataPreprocessor:
    """
    Preprocess data for MedGemma fine-tuning.
    
    Handles:
    - Image resizing and normalization
    - Text prompt generation
    - Label encoding
    """
    
    def __init__(
        self,
        image_size: int = 384,
        max_seq_length: int = 2048
    ):
        self.image_size = image_size
        self.max_seq_length = max_seq_length
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize and normalize image for MedGemma."""
        # Resize to target size
        image = image.resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )
        return image
    
    def encode_label(self, label_name: str, class_names: List[str]) -> int:
        """Encode label name to integer."""
        if label_name in class_names:
            return class_names.index(label_name)
        return -1
    
    def generate_prompt(self, label_name: str, task_type: str = "classification",
                         class_names: Optional[List[str]] = None) -> str:
        """Generate training prompt (question only — answer is NOT included).

        Class names are translated (e.g. CRC abbreviations → full tissue
        names) so the model sees human-readable labels it already knows.
        """
        from histolab.data_loader import translate_class_names

        if task_type == "classification":
            if class_names:
                translated = translate_class_names(class_names)
                class_list = ", ".join(translated)
                return (f"Classify this histopathology image. "
                        f"The possible tissue types are: {class_list}. "
                        f"What type of tissue is shown? Answer with only the tissue type name.")
            return "Classify this histopathology image. What type of tissue is shown? Answer with only the tissue type name."
        elif task_type == "ordinal_regression":
            return "What is the Gleason grade of this prostate tissue? Answer with only the grade."
        else:
            return "Describe the tissue type shown in this histopathology image."

    def prepare_sample(
        self,
        sample: Dict,
        class_names: List[str],
        task_type: str = "classification"
    ) -> Dict:
        """
        Prepare a single sample for training.

        Returns dict with:
        - image: Preprocessed PIL Image
        - text: Input text prompt
        - label: Integer label
        - label_name: Translated label name (full tissue name for CRC)
        """
        from histolab.data_loader import translate_label, translate_class_names

        image = sample.get("image")
        label_name = sample.get("label_name", sample.get("label", "unknown"))

        # Translate CRC abbreviations to full tissue names
        label_name = translate_label(label_name)
        translated_classes = translate_class_names(class_names)

        if image is not None:
            image = self.preprocess_image(image)

        text = self.generate_prompt(label_name, task_type, class_names=class_names)
        label = self.encode_label(label_name, translated_classes)

        return {
            "image": image,
            "text": text,
            "label": label,
            "label_name": label_name
        }
    
    def prepare_dataset(
        self,
        samples: List[Dict],
        class_names: List[str],
        task_type: str = "classification",
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """Prepare a list of samples for training."""
        if max_samples:
            samples = samples[:max_samples]
        
        return [
            self.prepare_sample(s, class_names, task_type)
            for s in samples
        ]


class MultiDatasetLoader:
    """
    Load and combine multiple datasets for training.
    
    Supports automatic format detection and unified loading.
    """
    
    SUPPORTED_FORMATS = ["directory", "hdf5", "svs"]
    
    def __init__(self, datasets_base_path: str = "data/datasets"):
        """
        Initialize multi-dataset loader.
        
        Args:
            datasets_base_path: Base path for all datasets
        """
        self.base_path = Path(datasets_base_path)
        self._dataset_info_cache = {}
    
    def load_dataset(
        self,
        dataset_name: str,
        max_samples: Optional[int] = None,
        splits: Optional[Dict[str, float]] = None
    ) -> DatasetSplit:
        """
        Load a single dataset with automatic splitting.
        
        Args:
            dataset_name: Name of dataset (crc, pcam, bach, panda, tcga)
            max_samples: Maximum samples to load
            splits: Dict with train/val/test ratios
            
        Returns:
            DatasetSplit with train/val/test samples
        """
        from histolab.data_loader import DATASETS, DatasetLoader
        
        dataset_info = DATASETS.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_path = self.base_path / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}\n"
                f"Please download {dataset_info.display_name} first."
            )
        
        # Load all samples — pass known dataset_info so detection isn't needed
        loader = DatasetLoader(str(dataset_path))
        loader.dataset_info = dataset_info

        # Get all samples (no split)
        all_samples = loader.load_samples(split="train", max_samples=max_samples)
        
        # If no samples found, try different approach
        if not all_samples:
            all_samples = self._load_all_files(dataset_path, dataset_info)
        
        if not all_samples:
            raise ValueError(f"No samples found in {dataset_path}")
        
        # Split data with stratification to preserve class distribution
        splits = splits or {"train": 0.7, "validation": 0.15, "test": 0.15}

        # Group samples by class
        from collections import defaultdict
        class_groups = defaultdict(list)
        for sample in all_samples:
            label = sample.get("label_name", sample.get("label", "unknown"))
            class_groups[label].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        for label, samples_in_class in class_groups.items():
            random.shuffle(samples_in_class)
            n_cls = len(samples_in_class)
            n_train = int(n_cls * splits["train"])
            n_val = int(n_cls * splits["validation"])

            train_samples.extend(samples_in_class[:n_train])
            val_samples.extend(samples_in_class[n_train:n_train + n_val])
            test_samples.extend(samples_in_class[n_train + n_val:])

        # Shuffle each split
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        # --- Class distribution logging ---
        from collections import Counter
        for split_name, split_samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
            dist = Counter(s.get("label_name", s.get("label", "?")) for s in split_samples)
            dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
            print(f"  Class distribution — {split_name} ({len(split_samples)} samples): {dist_str}")
            logger.info(f"Class distribution — {split_name}: {dist_str}")

        # --- Class distribution guard ---
        train_classes = set(s.get("label_name", s.get("label")) for s in train_samples)
        val_classes = set(s.get("label_name", s.get("label")) for s in val_samples)
        test_classes = set(s.get("label_name", s.get("label")) for s in test_samples)

        for split_name, split_classes in [("train", train_classes), ("val", val_classes), ("test", test_classes)]:
            if len(split_classes) < 2:
                raise ValueError(
                    f"FATAL: {split_name} split has only {len(split_classes)} class(es): {split_classes}. "
                    f"This means the model will learn nothing useful. "
                    f"Check your data loader — the max_samples cap may be exhausted by a single class."
                )
            if len(split_classes) < dataset_info.num_classes:
                logger.warning(
                    f"WARNING: {split_name} split has {len(split_classes)}/{dataset_info.num_classes} classes "
                    f"(missing: {set(dataset_info.classes) - split_classes})"
                )

        return DatasetSplit(
            train=train_samples,
            validation=val_samples,
            test=test_samples,
            dataset_name=dataset_name,
            class_names=dataset_info.classes,
            num_classes=dataset_info.num_classes
        )
    
    def _load_all_files(self, dataset_path: Path, dataset_info) -> List[Dict]:
        """Load all files from directory structure."""
        samples = []
        
        # Try loading from class subdirectories
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
                try:
                    image = Image.open(img_path).convert("RGB")
                    samples.append({
                        "image": image,
                        "label_name": class_name,
                        "path": str(img_path)
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")
        
        return samples
    
    def load_basket(
        self,
        dataset_names: List[str],
        samples_per_dataset: Optional[int] = None,
        splits: Optional[Dict[str, float]] = None
    ) -> Dict[str, DatasetSplit]:
        """
        Load multiple datasets as a basket.
        
        Args:
            dataset_names: List of dataset names to load
            samples_per_dataset: Max samples per dataset
            splits: Train/val/test split ratios
            
        Returns:
            Dict mapping dataset name to DatasetSplit
        """
        splits = splits or {"train": 0.7, "validation": 0.15, "test": 0.15}
        
        result = {}
        for name in dataset_names:
            try:
                result[name] = self.load_dataset(
                    name,
                    max_samples=samples_per_dataset,
                    splits=splits
                )
                total_loaded = (len(result[name].train) +
                                len(result[name].validation) +
                                len(result[name].test))
                logger.info(f"Loaded {name}: {len(result[name].train)} train, "
                           f"{len(result[name].validation)} val, {len(result[name].test)} test")
                if samples_per_dataset and total_loaded < samples_per_dataset:
                    msg = (f"WARNING: {name} has only {total_loaded} samples "
                           f"(requested {samples_per_dataset}) — using all available")
                    logger.warning(msg)
                    print(f"  {msg}")
            except Exception as e:
                logger.error(f"Failed to load {name}: {e}")
                print(f"  ERROR loading {name}: {e}")

        return result


class FineTuneRunner:
    """
    Main orchestrator for fine-tuning MedGemma on multiple datasets.
    
    Provides:
    - Multi-dataset training
    - Automatic evaluation
    - Baseline comparison
    - Result serialization
    """
    
    def __init__(self, config: Optional[FineTuneConfig] = None):
        """
        Initialize the fine-tuning runner.
        
        Args:
            config: Fine-tune configuration
        """
        self.config = config or FineTuneConfig()
        self.preprocessor = DataPreprocessor(
            image_size=self.config.image_size,
            max_seq_length=self.config.max_seq_length
        )
        
        self.datasets = {}
        self.baseline_results = {}
        self.finetuned_results = {}
        self.training_history = {}
    
    def load_datasets(self) -> Dict[str, DatasetSplit]:
        """Load all configured datasets."""
        print(f"\n{'='*70}")
        print("LOADING DATASETS")
        print(f"{'='*70}")
        print(f"Datasets: {self.config.datasets}")
        print(f"Samples per dataset: {self.config.samples_per_dataset}")
        print(f"{'='*70}\n")
        
        loader = MultiDatasetLoader()
        self.datasets = loader.load_basket(
            self.config.datasets,
            samples_per_dataset=self.config.samples_per_dataset
        )
        
        # Print per-dataset summary table
        total_train = sum(len(d.train) for d in self.datasets.values())
        total_val = sum(len(d.validation) for d in self.datasets.values())
        total_test = sum(len(d.test) for d in self.datasets.values())

        print(f"\n{'─'*70}")
        print(f"{'Dataset':<10} {'Classes':>8} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
        print(f"{'─'*70}")
        for name, ds in self.datasets.items():
            n_train = len(ds.train)
            n_val = len(ds.validation)
            n_test = len(ds.test)
            n_total = n_train + n_val + n_test
            print(f"{name:<10} {ds.num_classes:>8} {n_train:>8} {n_val:>8} {n_test:>8} {n_total:>8}")
        print(f"{'─'*70}")
        print(f"{'TOTAL':<10} {'':>8} {total_train:>8} {total_val:>8} {total_test:>8} "
              f"{total_train + total_val + total_test:>8}")
        print(f"{'─'*70}")

        return self.datasets
    
    def prepare_training_data(self) -> Tuple[List, List]:
        """
        Prepare combined training and validation data from all datasets.
        
        Returns:
            Tuple of (train_data, val_data)
        """
        train_data = []
        val_data = []
        
        for name, split in self.datasets.items():
            # Get class names for this dataset
            class_names = split.class_names
            
            # Prepare training samples
            for sample in split.train:
                prepared = self.preprocessor.prepare_sample(
                    sample, class_names, "classification"
                )
                prepared["dataset"] = name
                train_data.append(prepared)
            
            # Prepare validation samples
            for sample in split.validation:
                prepared = self.preprocessor.prepare_sample(
                    sample, class_names, "classification"
                )
                prepared["dataset"] = name
                val_data.append(prepared)
        
        # Shuffle
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        print(f"\nPrepared {len(train_data)} training samples")
        print(f"Prepared {len(val_data)} validation samples")
        
        return train_data, val_data
    
    def prepare_test_data(self) -> Dict[str, List[Dict]]:
        """Prepare test data for evaluation."""
        test_data = {}
        
        for name, split in self.datasets.items():
            class_names = split.class_names
            
            prepared = [
                self.preprocessor.prepare_sample(s, class_names, "classification")
                for s in split.test
            ]
            test_data[name] = prepared
        
        return test_data
    
    def run_baseline_evaluation(self) -> Dict[str, Dict]:
        """
        Run baseline (zero-shot) evaluation on all test sets.
        
        Returns:
            Dict mapping dataset name to metrics
        """
        from histolab.data_loader import MedGemmaTester
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        
        print(f"\n{'='*70}")
        print("BASELINE (ZERO-SHOT) EVALUATION")
        print(f"{'='*70}")
        
        tester = MedGemmaTester(
            model_path=self.config.base_model,
            quantization=self.config.quantization
        )
        
        if not tester.load_model():
            print("Failed to load baseline model")
            return {}
        
        test_data = self.prepare_test_data()
        results = {}
        
        # Initialize wandb for baseline evaluation if enabled
        if self.config.use_wandb:
            try:
                import wandb
                # Check if wandb is already initialized
                if wandb.run is None:
                    run_name = f"baseline-{'-'.join(self.config.datasets)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    wandb.init(
                        project=self.config.wandb_project,
                        entity=self.config.wandb_entity,
                        name=run_name,
                        config={
                            "model": "baseline",
                            "datasets": self.config.datasets,
                            "samples_per_dataset": self.config.samples_per_dataset,
                            "quantization": self.config.quantization
                        }
                    )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb for baseline evaluation: {e}")
        
        from histolab.data_loader import translate_class_names

        for dataset_name, samples in test_data.items():
            if not samples:
                continue

            # Use translated class names so MedGemma sees full tissue names
            translated_classes = translate_class_names(
                self.datasets[dataset_name].class_names
            )

            print(f"\nEvaluating on {dataset_name} ({len(samples)} samples)...")

            predictions = []
            ground_truth = []

            for i, sample in enumerate(tqdm(samples, desc=dataset_name)):
                image = sample["image"]
                # label_name is already translated by prepare_sample()
                true_label = sample["label_name"]

                result = tester.test_zero_shot(
                    image,
                    translated_classes
                )

                pred_label = result.get("predicted_class", "unknown")
                predictions.append(pred_label)
                ground_truth.append(true_label)
            
            # Calculate metrics
            accuracy = accuracy_score(ground_truth, predictions)
            f1_macro = f1_score(ground_truth, predictions, average="macro", zero_division=0)
            
            print(f"\n{dataset_name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 (macro): {f1_macro:.4f}")
            
            results[dataset_name] = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "predictions": list(zip(predictions, ground_truth)),
                "num_samples": len(samples)
            }
            
            # Log metrics to wandb
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        f"baseline/{dataset_name}/accuracy": accuracy,
                        f"baseline/{dataset_name}/f1_macro": f1_macro,
                        f"baseline/{dataset_name}/num_samples": len(samples)
                    })
                except Exception as e:
                    print(f"Warning: Failed to log baseline metrics to wandb: {e}")
        
        # Finish baseline wandb run if it was started here
        if self.config.use_wandb:
            try:
                import wandb
                if wandb.run and wandb.run.name and "baseline" in wandb.run.name.lower():
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
        
        self.baseline_results = results
        return results
    
    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict]
    ) -> str:
        """
        Run fine-tuning training.
        
        Args:
            train_data: List of training samples
            val_data: List of validation samples
            
        Returns:
            Path to saved model
        """
        from histolab.training import TrainingConfig, LoRATrainer
        from transformers import TrainingArguments, Trainer
        from datasets import Dataset
        import torch
        
        print(f"\n{'='*70}")
        print("FINE-TUNING MEDGEMMA")
        print(f"{'='*70}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"{'='*70}\n")
        
        # Create HuggingFace dataset
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Debug: Check learning rate type and value
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Learning rate type: {type(self.config.learning_rate)}")
        
        # Ensure learning rate is a float
        if isinstance(self.config.learning_rate, str):
            try:
                learning_rate = float(self.config.learning_rate)
                print(f"Converted learning rate to float: {learning_rate}")
            except:
                print("Warning: Invalid learning rate, using default 2e-5")
                learning_rate = 2e-5
        else:
            learning_rate = float(self.config.learning_rate)
        
        # Create trainer
        training_config = TrainingConfig(
            model_name=self.config.base_model,
            model_type="qlora" if self.config.use_qlora else "lora",
            load_in_4bit=True,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=learning_rate,
            output_dir=self.config.output_dir,
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        trainer = LoRATrainer(training_config=training_config, output_dir=self.config.output_dir)

        # Prepare model
        trainer.prepare_model_and_tokenizer()
        
        # Train with wandb integration
        final_model_path = trainer.train(
            train_dataset, 
            val_dataset,
            use_wandb=self.config.use_wandb,
            wandb_project=self.config.wandb_project,
            wandb_entity=self.config.wandb_entity,
            wandb_run_name=self.config.wandb_run_name
        )
        
        print(f"\n✓ Training completed!")
        print(f"Model saved to: {final_model_path}")
        
        return str(final_model_path)
    
    def run_finetuned_evaluation(self, model_path: str) -> Dict[str, Dict]:
        """
        Run evaluation with fine-tuned model.

        Uses the same prompt format as training so the model generates in
        the expected format ("The answer is: <class_name>").

        Args:
            model_path: Path to fine-tuned (merged) model

        Returns:
            Dict mapping dataset name to metrics
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from sklearn.metrics import accuracy_score, f1_score
        import torch

        print(f"\n{'='*70}")
        print("FINE-TUNED MODEL EVALUATION")
        print(f"{'='*70}")

        # Load fine-tuned model.
        # Strategy: load base model in bfloat16 (NOT 4-bit), apply LoRA adapter,
        # then merge_and_unload() LOSSLESSLY.  Merging is only destructive on
        # 4-bit weights (lossy requantization).  In bfloat16 the merge is exact.
        # This also avoids PeftModelForCausalLM.generate() issues with
        # vision-language models (pixel_values not forwarded to vision encoder).
        try:
            base_config_path = Path(model_path) / "base_model_config.json"

            if base_config_path.exists():
                from peft import PeftModel

                with open(base_config_path) as f:
                    base_cfg = json.load(f)
                base_model_name = base_cfg["base_model_name"]
                print(f"Loading base model in bfloat16 (for lossless merge): {base_model_name}")
                print(f"Loading LoRA adapter from: {model_path}")

                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

                # Load base in bfloat16 — NO 4-bit quantization so merge is lossless
                load_kwargs = dict(
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )

                # Read HF token
                hf_token = None
                token_file = Path("hf_token.txt")
                if token_file.exists():
                    hf_token = token_file.read_text().strip()
                if hf_token is None:
                    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
                if hf_token:
                    load_kwargs["token"] = hf_token

                base_model = AutoModelForImageTextToText.from_pretrained(
                    base_model_name, **load_kwargs
                )

                # Apply LoRA adapter on top of bfloat16 base
                model = PeftModel.from_pretrained(base_model, model_path)

                # Merge adapter into base weights (lossless in bfloat16) and
                # discard the PeftModel wrapper so generate() uses the native
                # VLM code path (proper pixel_values handling).
                print("Merging LoRA adapter into bfloat16 base (lossless)...")
                model = model.merge_and_unload()
                model.eval()
                print("Merged model ready for evaluation")
            else:
                # --- Fallback: old merged-model format (backward compatibility) ---
                print("No base_model_config.json found — loading as merged model (legacy)")
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

                load_kwargs = dict(
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
                model.eval()
        except Exception as e:
            print(f"Failed to load fine-tuned model: {e}")
            import traceback
            traceback.print_exc()
            return {}

        test_data = self.prepare_test_data()
        results = {}

        # Initialize wandb for fine-tuned evaluation if enabled
        if self.config.use_wandb:
            try:
                import wandb
                if wandb.run is None:
                    run_name = f"finetuned-{'-'.join(self.config.datasets)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    wandb.init(
                        project=self.config.wandb_project,
                        entity=self.config.wandb_entity,
                        name=run_name,
                        config={
                            "model": "fine-tuned",
                            "datasets": self.config.datasets,
                            "samples_per_dataset": self.config.samples_per_dataset,
                            "epochs": self.config.epochs,
                            "learning_rate": self.config.learning_rate,
                            "lora_r": self.config.lora_r,
                            "lora_alpha": self.config.lora_alpha,
                            "quantization": self.config.quantization
                        }
                    )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb for fine-tuned evaluation: {e}")

        from histolab.data_loader import translate_class_names

        for dataset_name, samples in test_data.items():
            if not samples:
                continue

            # Use translated class names (full tissue names for CRC)
            class_names = translate_class_names(
                self.datasets[dataset_name].class_names
            )
            print(f"\nEvaluating on {dataset_name} ({len(samples)} samples)...")

            predictions = []
            ground_truth = []

            for i, sample in enumerate(tqdm(samples, desc=dataset_name)):
                image = sample["image"]
                # label_name is already translated by prepare_sample()
                true_label = sample["label_name"]

                # Use the SAME prompt format as training (with translated class names)
                class_list = ", ".join(class_names)
                prompt_text = (f"Classify this histopathology image. "
                               f"The possible tissue types are: {class_list}. "
                               f"What type of tissue is shown? Answer with only the tissue type name.")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt_text}
                        ]
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = processor(
                    text=text,
                    images=image,
                    return_tensors="pt"
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    outputs = model.generate(**inputs, max_new_tokens=64, do_sample=False)

                new_tokens = outputs[0][input_len:]
                num_new_tokens = len(new_tokens)
                response = processor.decode(new_tokens, skip_special_tokens=True).strip()

                # Parse prediction: expect "The answer is: <CLASS>" or just "<CLASS>"
                pred_label = response
                if "The answer is:" in response:
                    pred_label = response.split("The answer is:")[-1].strip()

                # Match to closest known class name (case-insensitive)
                matched = "unknown"
                pred_lower = pred_label.lower()
                for cn in class_names:
                    if cn.lower() == pred_lower or cn.lower() in pred_lower:
                        matched = cn
                        break

                # Diagnostic logging for first 10 samples
                if i < 10:
                    # Also decode WITHOUT skip_special_tokens for debugging
                    raw_with_special = processor.decode(new_tokens, skip_special_tokens=False).strip()
                    print(
                        f"  [EVAL SAMPLE {i}] "
                        f"raw_response={repr(response)}, "
                        f"raw_with_special={repr(raw_with_special[:100])}, "
                        f"new_tokens={num_new_tokens}, "
                        f"true={true_label}, "
                        f"matched={matched}"
                    )
                    # For the very first sample, also log raw token IDs
                    if i == 0:
                        print(f"  [EVAL TOKEN IDS] first 20 generated: {new_tokens[:20].tolist()}")

                predictions.append(matched)
                ground_truth.append(true_label)

            # Summary: count unknowns
            num_unknown = predictions.count("unknown")
            print(
                f"\n  [EVAL SUMMARY] {dataset_name}: "
                f"{num_unknown}/{len(predictions)} predictions are 'unknown' "
                f"({num_unknown/len(predictions)*100:.1f}%)"
            )

            # Calculate metrics
            accuracy = accuracy_score(ground_truth, predictions)
            f1_macro = f1_score(ground_truth, predictions, average="macro", zero_division=0)

            # Per-class accuracy
            from collections import Counter
            class_correct = Counter()
            class_total = Counter()
            for pred, true in zip(predictions, ground_truth):
                class_total[true] += 1
                if pred == true:
                    class_correct[true] += 1

            print(f"\n{dataset_name} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 (macro): {f1_macro:.4f}")
            print(f"  Per-class accuracy:")
            per_class_acc = {}
            for cn in class_names:
                total = class_total.get(cn, 0)
                correct = class_correct.get(cn, 0)
                acc = correct / total if total > 0 else 0.0
                per_class_acc[cn] = acc
                print(f"    {cn}: {correct}/{total} = {acc:.2%}")

            # Confusion summary: what did the model predict for each true class?
            pred_dist_by_true = {}
            for pred, true in zip(predictions, ground_truth):
                if true not in pred_dist_by_true:
                    pred_dist_by_true[true] = Counter()
                pred_dist_by_true[true][pred] += 1
            print(f"  Confusion summary (true -> predicted):")
            for cn in class_names:
                if cn in pred_dist_by_true:
                    dist_str = ", ".join(f"{k}={v}" for k, v in pred_dist_by_true[cn].most_common())
                    print(f"    {cn} -> {dist_str}")

            results[dataset_name] = {
                "accuracy": accuracy,
                "f1_macro": f1_macro,
                "per_class_accuracy": per_class_acc,
                "predictions": list(zip(predictions, ground_truth)),
                "num_samples": len(samples)
            }

            # Log metrics to wandb
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        f"finetuned/{dataset_name}/accuracy": accuracy,
                        f"finetuned/{dataset_name}/f1_macro": f1_macro,
                        f"finetuned/{dataset_name}/num_samples": len(samples)
                    })
                except Exception as e:
                    print(f"Warning: Failed to log fine-tuned metrics to wandb: {e}")

        # Log comparison metrics if baseline exists
        if self.config.use_wandb and self.baseline_results:
            try:
                import wandb
                for dataset_name in results:
                    if dataset_name in self.baseline_results:
                        baseline_acc = self.baseline_results[dataset_name]["accuracy"]
                        finetuned_acc = results[dataset_name]["accuracy"]
                        baseline_f1 = self.baseline_results[dataset_name]["f1_macro"]
                        finetuned_f1 = results[dataset_name]["f1_macro"]

                        wandb.log({
                            f"comparison/{dataset_name}/accuracy_improvement": finetuned_acc - baseline_acc,
                            f"comparison/{dataset_name}/f1_improvement": finetuned_f1 - baseline_f1,
                            f"comparison/{dataset_name}/accuracy_ratio": finetuned_acc / baseline_acc if baseline_acc > 0 else 0
                        })
            except Exception as e:
                print(f"Warning: Failed to log comparison metrics to wandb: {e}")

        # Finish fine-tuned wandb run if it was started here
        if self.config.use_wandb:
            try:
                import wandb
                if wandb.run and wandb.run.name and "finetuned" in wandb.run.name.lower():
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")

        self.finetuned_results = results
        return results
    
    def compare_results(self) -> Dict[str, Any]:
        """
        Compare baseline vs fine-tuned results.
        
        Returns:
            Dict with comparison metrics
        """
        print(f"\n{'='*70}")
        print("BASELINE vs FINE-TUNED COMPARISON")
        print(f"{'='*70}")
        
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "datasets": self.config.datasets,
            "baseline": self.baseline_results,
            "fine_tuned": self.finetuned_results,
            "improvement": {}
        }
        
        # Initialize wandb for comparison if enabled
        if self.config.use_wandb:
            try:
                import wandb
                # Check if wandb is already initialized
                if wandb.run is None:
                    run_name = f"comparison-{'-'.join(self.config.datasets)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                    wandb.init(
                        project=self.config.wandb_project,
                        entity=self.config.wandb_entity,
                        name=run_name,
                        config={
                            "datasets": self.config.datasets,
                            "samples_per_dataset": self.config.samples_per_dataset,
                            "epochs": self.config.epochs,
                            "learning_rate": self.config.learning_rate
                        }
                    )
            except Exception as e:
                print(f"Warning: Failed to initialize wandb for comparison: {e}")
        
        for dataset in self.config.datasets:
            if dataset not in self.baseline_results or dataset not in self.finetuned_results:
                continue
            
            baseline_acc = self.baseline_results[dataset]["accuracy"]
            finetuned_acc = self.finetuned_results[dataset]["accuracy"]
            baseline_f1 = self.baseline_results[dataset]["f1_macro"]
            finetuned_f1 = self.finetuned_results[dataset]["f1_macro"]
            
            if baseline_acc > 0:
                improvement_acc = (finetuned_acc - baseline_acc) / baseline_acc * 100
            else:
                improvement_acc = 0.0
                
            if baseline_f1 > 0:
                improvement_f1 = (finetuned_f1 - baseline_f1) / baseline_f1 * 100
            else:
                improvement_f1 = 0.0
            
            comparison["improvement"][dataset] = {
                "baseline_accuracy": baseline_acc,
                "fine_tuned_accuracy": finetuned_acc,
                "absolute_improvement_accuracy": finetuned_acc - baseline_acc,
                "percentage_improvement_accuracy": improvement_acc,
                "baseline_f1_macro": baseline_f1,
                "fine_tuned_f1_macro": finetuned_f1,
                "absolute_improvement_f1": finetuned_f1 - baseline_f1,
                "percentage_improvement_f1": improvement_f1
            }
            
            print(f"\n{dataset.upper()}:")
            print(f"  Baseline:   {baseline_acc:.4f}")
            print(f"  Fine-tuned: {finetuned_acc:.4f}")
            print(f"  Improvement: {improvement_acc:+.2f}%")
            
            # Log comparison metrics to wandb
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        f"comparison/{dataset}/baseline_accuracy": baseline_acc,
                        f"comparison/{dataset}/finetuned_accuracy": finetuned_acc,
                        f"comparison/{dataset}/accuracy_improvement_absolute": finetuned_acc - baseline_acc,
                        f"comparison/{dataset}/accuracy_improvement_percentage": improvement_acc,
                        f"comparison/{dataset}/baseline_f1_macro": baseline_f1,
                        f"comparison/{dataset}/finetuned_f1_macro": finetuned_f1,
                        f"comparison/{dataset}/f1_improvement_absolute": finetuned_f1 - baseline_f1,
                        f"comparison/{dataset}/f1_improvement_percentage": improvement_f1
                    })
                except Exception as e:
                    print(f"Warning: Failed to log comparison metrics to wandb: {e}")
        
        # Overall improvement
        baseline_accs = [r["accuracy"] for r in self.baseline_results.values()]
        finetuned_accs = [r["accuracy"] for r in self.finetuned_results.values()]
        baseline_f1s = [r["f1_macro"] for r in self.baseline_results.values()]
        finetuned_f1s = [r["f1_macro"] for r in self.finetuned_results.values()]
        
        if baseline_accs and finetuned_accs:
            avg_baseline = sum(baseline_accs) / len(baseline_accs)
            avg_finetuned = sum(finetuned_accs) / len(finetuned_accs)
            avg_baseline_f1 = sum(baseline_f1s) / len(baseline_f1s)
            avg_finetuned_f1 = sum(finetuned_f1s) / len(finetuned_f1s)
            
            comparison["overall"] = {
                "avg_baseline": avg_baseline,
                "avg_fine_tuned": avg_finetuned,
                "avg_improvement": avg_finetuned - avg_baseline,
                "avg_baseline_f1": avg_baseline_f1,
                "avg_fine_tuned_f1": avg_finetuned_f1,
                "avg_improvement_f1": avg_finetuned_f1 - avg_baseline_f1
            }
            
            print(f"\nOVERALL:")
            print(f"  Average Baseline:   {avg_baseline:.4f}")
            print(f"  Average Fine-tuned: {avg_finetuned:.4f}")
            print(f"  Average Improvement: {avg_finetuned - avg_baseline:+.4f}")
            
            # Log overall comparison to wandb
            if self.config.use_wandb:
                try:
                    import wandb
                    wandb.log({
                        "comparison/overall/baseline_accuracy": avg_baseline,
                        "comparison/overall/finetuned_accuracy": avg_finetuned,
                        "comparison/overall/accuracy_improvement": avg_finetuned - avg_baseline,
                        "comparison/overall/baseline_f1_macro": avg_baseline_f1,
                        "comparison/overall/finetuned_f1_macro": avg_finetuned_f1,
                        "comparison/overall/f1_improvement": avg_finetuned_f1 - avg_baseline_f1
                    })
                except Exception as e:
                    print(f"Warning: Failed to log overall comparison metrics to wandb: {e}")
        
        # Finish comparison wandb run if it was started here
        if self.config.use_wandb:
            try:
                import wandb
                if wandb.run and wandb.run.name and "comparison" in wandb.run.name.lower():
                    wandb.finish()
            except Exception as e:
                print(f"Warning: Failed to finish wandb run: {e}")
        
        return comparison
    
    def save_results(self, comparison: Dict[str, Any], model_path: str):
        """Save all results to files."""
        output_dir = Path(self.config.output_dir) / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = self.config.experiment_name or f"experiment_{timestamp}"
        
        # Save comparison
        comparison_path = output_dir / f"{exp_name}_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save training config
        config_path = output_dir / f"{exp_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "datasets": self.config.datasets,
                "samples_per_dataset": self.config.samples_per_dataset,
                "epochs": self.config.epochs,
                "learning_rate": self.config.learning_rate,
                "model": self.config.base_model,
                "model_path": model_path
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"  Comparison: {comparison_path}")
        print(f"  Config: {config_path}")
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete fine-tuning pipeline.
        
        1. Load datasets
        2. Run baseline evaluation
        3. Fine-tune model
        4. Run fine-tuned evaluation
        5. Compare results
        
        Returns:
            Comparison results dictionary
        """
        # Step 1: Load datasets
        self.load_datasets()
        
        # Step 2: Prepare data
        train_data, val_data = self.prepare_training_data()
        
        # Step 3: Baseline evaluation (if requested)
        if self.config.compare_baseline:
            self.run_baseline_evaluation()
        
        # Step 4: Train
        model_path = self.train(train_data, val_data)
        
        # Step 5: Fine-tuned evaluation
        self.run_finetuned_evaluation(model_path)
        
        # Step 6: Compare results
        if self.config.compare_baseline:
            comparison = self.compare_results()
        else:
            comparison = {
                "fine_tuned": self.finetuned_results,
                "model_path": model_path
            }
        
        # Step 7: Save results
        if self.config.save_predictions:
            self.save_results(comparison, model_path)
        
        return comparison


def run_finetune(
    datasets: List[str],
    samples_per_dataset: int = 0,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    compare_baseline: bool = True,
    output_dir: str = "models",
    use_wandb: bool = True,
    wandb_project: str = "histolab-medgemma-finetune",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run fine-tuning.
    
    Args:
        datasets: List of dataset names to fine-tune on
        samples_per_dataset: Max samples per dataset
        epochs: Number of training epochs
        learning_rate: Learning rate
        compare_baseline: Whether to compare with baseline
        output_dir: Output directory for model
        use_wandb: Whether to use wandb for tracking
        wandb_project: Wandb project name
        wandb_entity: Wandb entity (username or team)
        wandb_run_name: Custom wandb run name
        
    Returns:
        Comparison results dictionary
    """
    config = FineTuneConfig(
        datasets=datasets,
        samples_per_dataset=samples_per_dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        compare_baseline=compare_baseline,
        output_dir=output_dir,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name
    )
    
    runner = FineTuneRunner(config)
    return runner.run_full_pipeline()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MedGemma Fine-tuning Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        default=["crc"],
        choices=["crc", "pcam", "bach", "panda", "tcga"],
        help="Datasets to fine-tune on"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=0,
        help="Max samples per dataset (0 = use all)"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="Training epochs"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size"
    )
    parser.add_argument(
        "--compare-baseline", "-c",
        action="store_true",
        default=True,
        help="Compare with baseline"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="models/finetuned",
        help="Output directory"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        default=False,
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="histolab-medgemma-finetune",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (username or team)"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Custom wandb run name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        help="Clear HuggingFace cache before running"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory"
    )
    
    args = parser.parse_args()
    
    # Clear cache if requested
    if args.clear_cache:
        print("\nClearing HuggingFace cache...")
        freed = clear_huggingface_cache(args.cache_dir)
        print(f"Cleared {freed / (1024**3):.2f} GB")
    
    # Load configuration
    if args.config:
        config = FineTuneConfig.from_yaml(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        config = FineTuneConfig(
            datasets=args.datasets,
            samples_per_dataset=args.samples,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            compare_baseline=args.compare_baseline,
            output_dir=args.output_dir,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name
        )
    
    # Run fine-tuning
    runner = FineTuneRunner(config)
    results = runner.run_full_pipeline()
    
    print("\n✓ Fine-tuning pipeline completed!")
