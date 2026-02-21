"""
Dataset Label Mapping Utility

Provides functionality to map image file paths to their true labels
based on the dataset folder structure.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Base dataset paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "datasets"

# CRC folder names → human-readable full names
CRC_LABEL_MAP = {
    "ADI":  "Adipose",
    "BACK": "Background",
    "DEB":  "Debris",
    "LYM":  "Lymphocyte",
    "MUC":  "Mucus",
    "MUS":  "Muscle",
    "NORM": "Normal",
    "STR":  "Stroma",
    "TUM":  "Tumor",
}

# Dataset configurations — classes list uses the actual folder names on disk.
# Call expand_label() to get the human-readable name for display.
DATASET_CONFIG = {
    "bach": {
        "path": DATA_DIR / "bach",
        "classes": ["Benign", "InSitu", "Invasive", "Normal"],
        "extensions": [".tif", ".png", ".jpg"],
    },
    "crc": {
        "path": DATA_DIR / "crc",
        "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
        "extensions": [".tif", ".png", ".jpg"],
    },
    "pcam": {
        "path": DATA_DIR / "pcam",
        "classes": ["Normal", "Tumor"],
        "extensions": [".png", ".jpg", ".tif"],
    }
}


def expand_label(label: str) -> str:
    """Return the human-readable label, translating CRC abbreviations to full names."""
    return CRC_LABEL_MAP.get(label, label)


def get_true_label(file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the true label and dataset name from a file path.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        Tuple of (dataset_name, true_label) or (None, None) if not found
    """
    path = Path(file_path)
    filename = path.name
    filename_lower = filename.lower()  # For case-insensitive matching
    parent_name = path.parent.name
    
    # First, check if the filename itself starts with a class name (for custom uploads)
    # This handles cases like "TUM_image.tif", "tumor_000020.png", "Invasive_case1.tif"
    # We check PCAM first since it has longer class names that could be prefixes of others
    # (e.g., "Tumor" vs "TUM")
    dataset_order = ["pcam", "bach", "crc"]  # Check PCAM first for better matching
    for dataset_name in dataset_order:
        if dataset_name not in DATASET_CONFIG:
            continue
        config = DATASET_CONFIG[dataset_name]
        for class_name in config["classes"]:
            # Case-insensitive matching - only if filename starts exactly with class name
            # and is followed by underscore, dash, or is the full name
            class_lower = class_name.lower()
            if filename_lower.startswith(class_lower):
                # Check if it's a proper match (followed by separator or end of string)
                remainder = filename_lower[len(class_lower):]
                if remainder == "" or remainder.startswith("_") or remainder.startswith("-") or remainder.startswith("."):
                    return dataset_name, expand_label(class_name)
    
    # Check each dataset for full path matching
    for dataset_name, config in DATASET_CONFIG.items():
        dataset_path = config["path"]
        
        # Check if file is within this dataset
        try:
            relative_path = path.relative_to(dataset_path)
            parts = relative_path.parts
            
            if len(parts) >= 2:
                # The first part should be the class folder
                class_folder = parts[0]
                
                if class_folder in config["classes"]:
                    return dataset_name, expand_label(class_folder)
                    
        except ValueError:
            # Not in this dataset
            continue
    
    # Also check by parent folder name matching class names (case-insensitive)
    for dataset_name, config in DATASET_CONFIG.items():
        parent_name_lower = parent_name.lower()
        for class_name in config["classes"]:
            if parent_name_lower == class_name.lower():
                return dataset_name, expand_label(class_name)
    
    return None, None


def _glob_extensions(directory: Path, extensions: list) -> list:
    """Glob a directory for multiple file extensions."""
    images = []
    for ext in extensions:
        images.extend(directory.glob(f"*{ext}"))
    return images


def get_all_dataset_images() -> Dict[str, Dict[str, list]]:
    """
    Get all available images from all datasets organized by dataset and class.

    Returns:
        Dictionary mapping dataset_name -> class_name -> list of image paths
    """
    all_images = {}

    for dataset_name, config in DATASET_CONFIG.items():
        dataset_path = config["path"]
        all_images[dataset_name] = {}

        for class_name in config["classes"]:
            class_path = dataset_path / class_name
            if class_path.exists():
                images = _glob_extensions(class_path, config["extensions"])
                all_images[dataset_name][class_name] = [str(img) for img in images]
            else:
                all_images[dataset_name][class_name] = []

    return all_images


def get_dataset_stats() -> Dict[str, Dict[str, int]]:
    """
    Get statistics about each dataset.
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    all_images = get_all_dataset_images()
    
    for dataset_name, classes in all_images.items():
        total = sum(len(images) for images in classes.values())
        stats[dataset_name] = {
            "total": total,
            "classes": {cls: len(imgs) for cls, imgs in classes.items()}
        }
    
    return stats


def get_random_image_from_dataset(dataset: str, class_name: Optional[str] = None) -> Optional[str]:
    """
    Get a random image from a specific dataset and optionally filtered by class.
    
    Args:
        dataset: Dataset name (bach, crc, pcam)
        class_name: Optional class name to filter by
        
    Returns:
        Path to random image or None if not found
    """
    import random
    
    if dataset not in DATASET_CONFIG:
        return None
    
    config = DATASET_CONFIG[dataset]
    dataset_path = config["path"]
    
    if class_name and class_name in config["classes"]:
        class_path = dataset_path / class_name
        if class_path.exists():
            images = _glob_extensions(class_path, config["extensions"])
            if images:
                return str(random.choice(images))
    else:
        # Pick a random class first, then a random image from it
        # This ensures all classes are equally represented
        available_classes = []
        for cls in config["classes"]:
            class_path = dataset_path / cls
            if class_path.exists() and _glob_extensions(class_path, config["extensions"]):
                available_classes.append(cls)
        if available_classes:
            chosen_class = random.choice(available_classes)
            images = _glob_extensions(dataset_path / chosen_class, config["extensions"])
            return str(random.choice(images))
    
    return None


def is_valid_image_path(file_path: str) -> bool:
    """
    Check if a file path points to a valid image from the datasets.
    
    Args:
        file_path: Path to check
        
    Returns:
        True if valid dataset image, False otherwise
    """
    dataset, label = get_true_label(file_path)
    return dataset is not None
