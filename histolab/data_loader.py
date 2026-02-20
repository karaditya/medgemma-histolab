"""
Dataset Loader Module - HistoLocal

Provides download utilities and data loading for all benchmark datasets:
- NCT-CRC-HE-100K (CRC)
- PatchCamelyon (PCam)
- BACH (Breast Cancer Histology)
- PANDA (Prostate Cancer)
- TCGA-BRCA (Breast Invasive Carcinoma)

Also provides utilities for testing MedGemma baseline on sample images.
"""

import os
import logging
import zipfile
import tarfile
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CRC label translation: abbreviations → full English tissue names
# MedGemma was pre-trained on histopathology with full tissue names, so using
# abbreviations (ADI, STR, TUM…) tanks zero-shot accuracy.  Translating at
# prompt/label time gives the model a fair shot and makes fine-tuning gains
# scientifically honest (domain adaptation, not abbreviation memorization).
# ---------------------------------------------------------------------------

CRC_LABEL_MAP = {
    "ADI": "Adipose",
    "BACK": "Background",
    "DEB": "Debris",
    "LYM": "Lymphocyte",
    "MUC": "Mucus",
    "MUS": "Muscle",
    "NORM": "Normal",
    "STR": "Stroma",
    "TUM": "Tumor",
}


def translate_label(label: str) -> str:
    """Translate a CRC abbreviation to its full tissue name.

    Non-CRC labels (e.g. PCam's "Normal"/"Tumor") pass through unchanged.
    """
    return CRC_LABEL_MAP.get(label, label)


def translate_class_names(class_names: List[str]) -> List[str]:
    """Translate a list of class names, mapping CRC abbreviations to full names.

    Non-CRC names pass through unchanged, so this is safe to call on any dataset.
    """
    return [CRC_LABEL_MAP.get(cn, cn) for cn in class_names]


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    display_name: str
    url: str
    size_gb: float
    num_samples: int
    num_classes: int
    classes: List[str]
    task_type: str
    format: str  # "directory", "hdf5", "csv", "kaggle"
    download_instructions: str


# Dataset configurations
DATASETS = {
    "crc": DatasetInfo(
        name="crc",
        display_name="NCT-CRC-HE-100K",
        url="https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip",
        size_gb=2.5,
        num_samples=107180,
        num_classes=9,
        classes=["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
        task_type="multiclass_classification",
        format="directory",
        download_instructions="""
# Option 1: Direct Download (from Zenodo)
wget https://zenodo.org/records/1214456/files/NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip

# Option 2: Kaggle
# Download from: https://www.kaggle.com/datasets/gpreda/nct-crc-he-100k-subset
kaggle datasets download -d gpreda/nct-crc-he-100k-subset
"""
    ),
    
    "pcam": DatasetInfo(
        name="pcam",
        display_name="PatchCamelyon",
        url="https://github.com/basveeling/pcam/raw/master/pcam_v1.zip",
        size_gb=8.0,
        num_samples=327680,
        num_classes=2,
        classes=["Normal", "Tumor"],
        task_type="binary_classification",
        format="directory",
        download_instructions="""
# Step 1: Download raw data
kaggle datasets download -d andrewmvd/metastatic-tissue-classification-patchcamelyon

# Step 2: Convert to directory format
python scripts/prepare_pcam.py --input-dir data/datasets/pcam_raw --output-dir data/datasets/pcam

# Or use the automated script:
bash scripts/download_datasets.sh
"""
    ),
    
    "bach": DatasetInfo(
        name="bach",
        display_name="BACH (ICIAR 2018)",
        url="https://doi.org/10.23728/FRB-ICIAR.2018.1045",  # Placeholder
        size_gb=3.5,
        num_samples=400,
        num_classes=4,
        classes=["Normal", "Benign", "InSitu", "Invasive"],
        task_type="multiclass_classification",
        format="directory",
        download_instructions="""
# Option 1: Kaggle
kaggle datasets download -d truthisneverlinear/bach-breast-cancer-histology-images

# Option 2: Grand Challenge (requires registration)
# https://iciar2018-challenge.grand-challenge.org/Dataset/

# Or use the automated script:
bash scripts/download_datasets.sh

# Expected directory structure: bach/{Normal,Benign,InSitu,Invasive}/
"""
    ),
    
    "panda": DatasetInfo(
        name="panda",
        display_name="PANDA (Prostate Cancer Grade Assessment)",
        url="https://zenodo.org/records/6618231/files/prostate-cancer-grade-assessment.zip",
        size_gb=65.0,
        num_samples=10662,
        num_classes=4,
        classes=["Gleason_6", "Gleason_7", "Gleason_8", "Gleason_9"],
        task_type="ordinal_regression",
        format="directory",
        download_instructions="""
# Option 1: Zenodo (Recommended - ~65GB)
wget https://zenodo.org/records/6618231/files/prostate-cancer-grade-assessment.zip
unzip prostate-cancer-grade-assessment.zip

# Option 2: Kaggle (requires authentication)
# kaggle competitions download -d prostate-cancer-grade-assessment
# Note: Kaggle requires accepting competition rules

# Option 3: TCIA (The Cancer Imaging Archive)
# https://wiki.cancerimagingarchive.net/display/Public/PANDA+Challenge
"""
    ),
    
    "tcga": DatasetInfo(
        name="tcga",
        display_name="TCGA-BRCA (Breast Invasive Carcinoma)",
        url="https://portal.gdc.cancer.gov/",
        size_gb="100+",
        num_samples=1000,  # WSI-level samples
        num_classes=4,
        classes=["Normal", "DCIS", "IDC", "ILC"],
        task_type="multiclass_classification",
        format="svs",  # Whole slide images
        download_instructions="""
# Option 1: NCI GDC Portal (Recommended)
# 1. Go to: https://portal.gdc.cancer.gov/
# 2. Select "TCGA-BRCA" project
# 3. Select "Slide" data type
# 4. Download using GDC Data Transfer Tool

# Install GDC tool:
wget https://gdc.cancer.gov/system/files/authenticated%20file/ 
gdc-client download -m manifest.txt

# Option 2: TCIA (Cancer Imaging Archive)
# https://www.cancerimagingarchive.net/collections/
# Search for "TCGA Breast Invasive Carcinoma"

# Option 3: Pre-extracted patches (Kaggle)
# kaggle datasets download -d histolab/tcga-brca-patches
"""
    )
}


def list_datasets() -> Dict[str, DatasetInfo]:
    """List all available datasets."""
    return DATASETS


def get_dataset_info(dataset_name: str) -> Optional[DatasetInfo]:
    """Get information about a specific dataset."""
    return DATASETS.get(dataset_name)


def download_dataset(
    dataset_name: str,
    output_dir: str = "data/datasets",
    method: str = "wget"
) -> bool:
    """
    Download a dataset using various methods.
    
    Args:
        dataset_name: Name of dataset (crc, pcam, bach, panda, tcga)
        output_dir: Output directory for downloaded data
        download_method: "wget", "kaggle", or "manual"
        
    Returns:
        True if download was successful
    """
    dataset = DATASETS.get(dataset_name)
    if dataset is None:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"DOWNLOADING: {dataset.display_name}")
    print(f"{'='*60}")
    print(f"Size: ~{dataset.size_gb} GB")
    print(f"Samples: {dataset.num_samples}")
    print(f"Classes: {dataset.classes}")
    print(f"\n{dataset.download_instructions}")
    print(f"{'='*60}\n")
    
    return True


class DatasetLoader:
    """
    Load and prepare datasets for training and evaluation.
    
    Supports all benchmark datasets with automatic format detection.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.dataset_info = None
        self.samples = []
    
    def detect_dataset(self) -> Optional[DatasetInfo]:
        """Detect dataset type from directory structure."""
        if not self.dataset_path.exists():
            logger.error(f"Dataset path does not exist: {self.dataset_path}")
            return None

        # Check for common dataset structures
        contents = list(self.dataset_path.iterdir())

        # Check subdirectory names against ALL known datasets (not just CRC)
        class_dirs = [c for c in contents if c.is_dir()]
        if class_dirs:
            dir_names = {c.name for c in class_dirs}
            # Try matching against each dataset's known classes
            for _, ds_info in DATASETS.items():
                if ds_info.format == "directory" and set(ds_info.classes) & dir_names:
                    self.dataset_info = ds_info
                    return self.dataset_info

        # Check for HDF5 files (PCam raw format)
        hdf5_files = list(self.dataset_path.glob("*.h5")) + list(self.dataset_path.glob("*.hdf5"))
        if hdf5_files:
            self.dataset_info = DATASETS["pcam"]
            return self.dataset_info

        # Check for SVS files (TCGA/PANDA WSI)
        wsi_files = list(self.dataset_path.glob("*.svs"))
        if wsi_files:
            # Could be TCGA or PANDA
            if len(wsi_files) > 100:
                self.dataset_info = DATASETS["tcga"]
            else:
                self.dataset_info = DATASETS["panda"]
            return self.dataset_info

        logger.warning("Could not auto-detect dataset format")
        return None
    
    def load_samples(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        transform: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Load samples from the dataset.
        
        Args:
            split: Data split ("train", "validation", "test")
            max_samples: Maximum number of samples to load
            transform: Optional transform function for images
            
        Returns:
            List of sample dictionaries with 'image' and 'label'
        """
        if self.dataset_info is None:
            self.detect_dataset()
        
        if self.dataset_info is None:
            logger.error("Dataset not detected")
            return []
        
        samples = []
        
        if self.dataset_info.format == "directory":
            samples = self._load_from_directory(split, max_samples, transform)
        elif self.dataset_info.format == "hdf5":
            samples = self._load_from_hdf5(split, max_samples, transform)
        elif self.dataset_info.format == "svs":
            samples = self._load_from_wsi(split, max_samples, transform)
        
        logger.info(f"Loaded {len(samples)} samples from {split} split")
        return samples
    
    def _load_from_directory(
        self,
        split: str,
        max_samples: Optional[int],
        transform: Optional[callable]
    ) -> List[Dict[str, Any]]:
        """Load samples from directory structure with balanced per-class sampling."""
        import random as _random

        # Assuming train/test subdirectories
        split_path = self.dataset_path / split
        if not split_path.exists():
            split_path = self.dataset_path

        # First pass: enumerate all valid class directories and their image paths
        class_image_paths: Dict[str, List[Path]] = {}
        for class_dir in sorted(split_path.iterdir()):  # sorted for determinism
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            if self.dataset_info and class_name not in self.dataset_info.classes:
                continue

            img_paths = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]:
                img_paths.extend(class_dir.glob(ext))
            if img_paths:
                class_image_paths[class_name] = img_paths

        if not class_image_paths:
            return []

        num_classes = len(class_image_paths)

        # Calculate per-class budget (balanced sampling)
        if max_samples:
            per_class = max_samples // num_classes
            remainder = max_samples % num_classes
        else:
            per_class = None  # no limit
            remainder = 0

        samples = []
        for i, (class_name, img_paths) in enumerate(sorted(class_image_paths.items())):
            class_idx = (self.dataset_info.classes.index(class_name)
                         if self.dataset_info and class_name in self.dataset_info.classes
                         else -1)

            # Shuffle so we get a random subset, not just the first N by filesystem order
            img_paths = list(img_paths)
            _random.shuffle(img_paths)

            # Per-class limit: distribute remainder to first `remainder` classes
            if per_class is not None:
                class_limit = per_class + (1 if i < remainder else 0)
                img_paths = img_paths[:class_limit]

            for img_path in img_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    if transform:
                        image = transform(image)
                    samples.append({
                        "image": image,
                        "label": class_idx,
                        "label_name": class_name,
                        "path": str(img_path)
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {img_path}: {e}")

        if max_samples and len(samples) < max_samples:
            logger.warning(
                f"Requested {max_samples} samples but only {len(samples)} available "
                f"(using all available)"
            )
            print(f"  WARNING: Requested {max_samples} samples but only {len(samples)} available — using all")

        logger.info(f"Loaded {len(samples)} samples across {num_classes} classes "
                     f"(per-class: {per_class}{'+ remainder' if remainder else ''})")
        return samples
    
    def _load_from_hdf5(
        self,
        split: str,
        max_samples: Optional[int],
        transform: Optional[callable]
    ) -> List[Dict[str, Any]]:
        """Load samples from HDF5 files (PCam format)."""
        try:
            import h5py
        except ImportError:
            logger.error("h5py not installed. Install with: pip install h5py")
            return []
        
        samples = []
        
        # Determine which HDF5 file based on split
        file_mapping = {
            "train": "train.h5",
            "validation": "valid.h5",
            "test": "test.h5"
        }
        
        hdf5_file = self.dataset_path / file_mapping.get(split, "train.h5")
        if not hdf5_file.exists():
            hdf5_file = self.dataset_path / "camelyon16_patch_level_trainset.h5"
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                images = f['x'][:]
                labels = f['y'][:]
                
                for i in range(min(len(images), max_samples or len(images))):
                    image = Image.fromarray(images[i])
                    if transform:
                        image = transform(image)
                    
                    samples.append({
                        "image": image,
                        "label": int(labels[i]),
                        "label_name": "Tumor" if labels[i] == 1 else "Normal",
                        "path": f"{hdf5_file}:{i}"
                    })
                    
        except Exception as e:
            logger.error(f"Failed to load HDF5 file: {e}")
        
        return samples
    
    def _load_from_wsi(
        self,
        split: str,
        max_samples: Optional[int],
        transform: Optional[callable]
    ) -> List[Dict[str, Any]]:
        """Load samples from WSI files (TCGA/PANDA). Not available in this build."""
        raise NotImplementedError(
            "WSI loading not available in this build. "
            "Use directory-based datasets (CRC, PCam, BACH) instead."
        )


class MedGemmaTester:
    """
    Test MedGemma baseline on sample images and compare with ground truth.
    
    Provides utilities for:
    - Zero-shot classification testing
    - Comparison with ground truth labels
    - Accuracy/f1 metric calculation
    """
    
    def __init__(self, model_path: str = "google/medgemma-4b-it", quantization: int = 4):
        """
        Initialize MedGemma tester.
        
        Args:
            model_path: Model to test
            quantization: Quantization level (0, 4, 8)
        """
        self.model_path = model_path
        self.quantization = quantization
        self.model = None
        self.processor = None
    
    def load_model(self) -> bool:
        """Load the MedGemma model."""
        from histolab.medgemma_integration import MedGemmaWrapper, MedGemmaConfig
        
        config = MedGemmaConfig(
            model_name=self.model_path,
            quantization=self.quantization
        )
        
        self.model = MedGemmaWrapper(config)
        
        try:
            self.model.load()
            self.processor = self.model.processor
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def test_zero_shot(
        self,
        image: Image.Image,
        class_names: List[str],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test zero-shot classification on a single image.
        
        Args:
            image: PIL Image to classify
            class_names: List of class names
            prompt_template: Optional custom prompt template
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        # Create a direct classification prompt that lists all classes
        prompt = (
            f"Classify this histopathology image. The possible tissue types are: {', '.join(class_names)}. "
            "What tissue type is shown in this image? Answer with only the class name from the list."
        )
        
        result = self.model.analyze_patch(
            image=image,
            prompt=prompt
        )
        
        response_text = result.get("raw_response", "").lower()
        
        # Find the class name that best matches the response
        best_class = "unknown"
        best_score = 0.0
        
        for class_name in class_names:
            class_name_lower = class_name.lower()
            
            # Check if class name is mentioned in response
            if class_name_lower in response_text:
                # Calculate confidence based on match strength
                score = 0.5
                # Boost score for exact matches or stronger indicators
                if response_text.strip() == class_name_lower:
                    score = 0.95
                elif f"{class_name_lower}" in response_text.split():
                    score = 0.85
                else:
                    score = 0.7
                
                if score > best_score:
                    best_score = score
                    best_class = class_name
        
        # Fallback to parser's tissue type if no direct match
        if best_class == "unknown":
            tissue_type = result.get("tissue_type", "unknown")
            if tissue_type in class_names:
                best_class = tissue_type
                best_score = result.get("confidence", 0.5)
        
        return {
            "predictions": {class_name: {"response": result.get("raw_response", ""), "confidence": 0.5} for class_name in class_names},
            "predicted_class": best_class,
            "confidence": best_score
        }
    
    def test_dataset(
        self,
        samples: List[Dict[str, Any]],
        class_names: List[str],
        prompt_template: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Test model on a dataset and calculate metrics.
        
        Args:
            samples: List of samples with 'image' and 'label'
            class_names: List of class names
            prompt_template: Optional custom prompt template
            
        Returns:
            Dictionary with metrics and predictions
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "Model not loaded"}
        
        predictions = []
        ground_truth = []
        correct = 0
        
        print(f"\nTesting on {len(samples)} samples...")
        
        for i, sample in enumerate(samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(samples)}")
            
            image = sample["image"]
            true_label = sample.get("label_name", sample.get("label", -1))
            true_idx = sample.get("label", -1)
            
            result = self.test_zero_shot(image, class_names, prompt_template)
            
            if "error" in result:
                continue
            
            pred_class = result["predicted_class"]
            pred_idx = class_names.index(pred_class) if pred_class in class_names else -1
            
            predictions.append(pred_class)
            ground_truth.append(true_label)
            
            if pred_class == true_label:
                correct += 1
        
        # Calculate metrics
        accuracy = correct / len(predictions) if predictions else 0
        
        # Calculate per-class accuracy
        class_correct = {}
        class_total = {}
        
        for pred, true in zip(predictions, ground_truth):
            if true not in class_total:
                class_total[true] = 0
                class_correct[true] = 0
            
            class_total[true] += 1
            if pred == true:
                class_correct[true] = class_correct.get(true, 0) + 1
        
        class_accuracy = {
            cls: class_correct.get(cls, 0) / class_total.get(cls, 1)
            for cls in class_total.keys()
        }
        
        results = {
            "total_samples": len(predictions),
            "correct": correct,
            "accuracy": accuracy,
            "class_accuracy": class_accuracy,
            "predictions": list(zip(predictions, ground_truth)),
            "class_names": class_names
        }
        
        print(f"\nResults:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Correct: {correct}/{len(predictions)}")
        print(f"  Per-class accuracy:")
        for cls, acc in class_accuracy.items():
            print(f"    {cls}: {acc:.2%}")
        
        return results
    
    def test_single_image(
        self,
        image_path: str,
        ground_truth: str,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Test on a single image and compare with ground truth.
        
        Args:
            image_path: Path to image
            ground_truth: True label
            class_names: List of class names
            
        Returns:
            Test result dictionary
        """
        image = Image.open(image_path).convert("RGB")
        
        result = self.test_zero_shot(image, class_names)
        
        is_correct = result.get("predicted_class") == ground_truth
        
        return {
            "image_path": image_path,
            "ground_truth": ground_truth,
            "predicted_class": result.get("predicted_class"),
            "confidence": result.get("confidence"),
            "all_predictions": result.get("predictions"),
            "is_correct": is_correct
        }


def test_medgemma_interactive(
    model_path: str = "google/medgemma-4b-it",
    dataset_name: str = "crc"
):
    """
    Interactive testing of MedGemma on sample images.
    
    Args:
        model_path: Model to test
        dataset_name: Dataset to use for testing
    """
    print("\n" + "="*60)
    print("MEDGEMMA INTERACTIVE TESTING")
    print("="*60)
    
    # Get dataset info
    dataset_info = DATASETS.get(dataset_name)
    if dataset_info is None:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    print(f"\nDataset: {dataset_info.display_name}")
    print(f"Classes: {dataset_info.classes}")
    
    # Load dataset
    dataset_path = f"data/datasets/{dataset_name}"
    loader = DatasetLoader(dataset_path)
    samples = loader.load_samples(split="test", max_samples=20)
    
    if not samples:
        print(f"\nNo samples found in {dataset_path}")
        print("Please download the dataset first:")
        print(dataset_info.download_instructions)
        return
    
    print(f"\nLoaded {len(samples)} samples")
    
    # Test MedGemma
    tester = MedGemmaTester(model_path=model_path)
    
    if not tester.load_model():
        print("\nFailed to load model. Make sure you have internet access to download the model.")
        return
    
    # Run tests
    results = tester.test_dataset(samples, dataset_info.classes)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"\nThis shows how well the baseline MedGemma model performs")
    print(f"on {dataset_info.display_name} without any fine-tuning.")


def create_sample_test_script():
    """
    Create a standalone test script for quick evaluation.
    
    Returns:
        Path to generated script
    """
    script_content = '''#!/usr/bin/env python3
"""
Quick Test Script for MedGemma Baseline Evaluation

Usage:
    python test_medgemma.py --dataset crc --samples 50
    python test_medgemma.py --image path/to/image.png --classes "Normal Tumor"
"""

import argparse
import sys
from pathlib import Path
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Test MedGemma on sample images")
    parser.add_argument("--dataset", "-d", choices=["crc", "pcam", "bach", "panda", "tcga"],
                        help="Dataset name")
    parser.add_argument("--samples", "-s", type=int, default=20,
                        help="Number of samples to test")
    parser.add_argument("--image", "-i", type=str,
                        help="Single image to test")
    parser.add_argument("--classes", "-c", type=str, nargs="+",
                        help="Class names for classification")
    parser.add_argument("--model", "-m", type=str, default="google/medgemma-4b-it",
                        help="Model path")
    
    args = parser.parse_args()
    
    if args.image:
        # Test single image
        from histolab.data_loader import MedGemmaTester
        
        tester = MedGemmaTester(model_path=args.model)
        if not tester.load_model():
            print("Failed to load model")
            sys.exit(1)
        
        image = Image.open(args.image).convert("RGB")
        class_names = args.classes or ["Normal", "Tumor"]
        
        result = tester.test_zero_shot(image, class_names)
        
        print(f"\\nImage: {args.image}")
        print(f"Ground Truth: {args.classes}")
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        for cls, pred in result["predictions"].items():
            print(f"  {cls}: {pred['confidence']:.2%}")
    
    elif args.dataset:
        from histolab.data_loader import test_medgemma_interactive, DATASETS
        
        dataset_info = DATASETS.get(args.dataset)
        if dataset_info:
            test_medgemma_interactive(
                model_path=args.model,
                dataset_name=args.dataset
            )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
'''
    
    script_path = Path("test_medgemma.py")
    script_path.write_text(script_content)
    
    return script_path


def download_all_datasets(output_dir: str = "data/datasets"):
    """
    Print download instructions for all datasets.
    
    Args:
        output_dir: Base output directory
    """
    print("\n" + "="*70)
    print("DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    
    for name, dataset in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"DATASET: {dataset.display_name} ({name})")
        print(f"{'='*70}")
        print(f"Size: ~{dataset.size_gb} GB")
        print(f"Samples: {dataset.num_samples}")
        print(f"Classes ({dataset.num_classes}): {dataset.classes}")
        print(f"Format: {dataset.format}")
        print(f"\nDownload Instructions:")
        print(dataset.download_instructions)
        print(f"\nExpected location: {output_dir}/{name}/")
    
    print("\n" + "="*70)


# Convenience functions
def get_dataloader(dataset_path: str) -> DatasetLoader:
    """Create a DatasetLoader for the specified path."""
    return DatasetLoader(dataset_path)


def get_tester(model_path: str = "google/medgemma-4b-it") -> MedGemmaTester:
    """Create a MedGemmaTester for model evaluation."""
    return MedGemmaTester(model_path=model_path)
