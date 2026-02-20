"""
HistoLab - Histopathology Image Classification with MedGemma

Fine-tuning pipeline for MedGemma on histopathology datasets
(CRC, PCam, BACH) with LoRA/QLoRA support, evaluation, and
a Chat UI for interactive analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

__version__ = "1.0.0"
__author__ = "HistoLab Team"

# Core imports
from .config import Config, get_config

# Training & Benchmark
from .training import (
    LoRATrainer,
    TrainingConfig,
    DatasetConfig,
    BenchmarkEvaluator,
    BenchmarkResult,
    run_lora_finetuning,
    run_benchmark_comparison
)

# Data Loading
from .data_loader import (
    DatasetLoader,
    DatasetInfo,
    MedGemmaTester,
    DATASETS,
    list_datasets,
    get_dataset_info,
    download_dataset,
    get_dataloader,
    get_tester,
    download_all_datasets,
    test_medgemma_interactive
)

# MedGemma Integration
from .medgemma_integration import (
    MedGemmaWrapper,
    MedGemmaConfig,
    MedGemmaManager,
    create_medgemma_wrapper
)

__all__ = [
    "__version__",

    # Config
    "Config",
    "get_config",

    # Training & Benchmark
    "LoRATrainer",
    "TrainingConfig",
    "DatasetConfig",
    "BenchmarkEvaluator",
    "BenchmarkResult",
    "run_lora_finetuning",
    "run_benchmark_comparison",

    # Data Loading
    "DatasetLoader",
    "DatasetInfo",
    "MedGemmaTester",
    "DATASETS",
    "list_datasets",
    "get_dataset_info",
    "download_dataset",
    "get_dataloader",
    "get_tester",
    "download_all_datasets",
    "test_medgemma_interactive",

    # MedGemma
    "MedGemmaWrapper",
    "MedGemmaConfig",
    "MedGemmaManager",
    "create_medgemma_wrapper",
]


def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available."""
    dependencies = {}

    for name in ["torch", "transformers", "peft", "bitsandbytes", "gradio", "scipy"]:
        try:
            __import__(name)
            dependencies[name] = True
        except ImportError:
            dependencies[name] = False

    try:
        from PIL import Image
        dependencies["Pillow"] = True
    except ImportError:
        dependencies["Pillow"] = False

    return dependencies


def get_system_info() -> Dict[str, Any]:
    """Get system and GPU information."""
    import platform
    import torch

    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 2) if torch.cuda.is_available() else 0,
    }

    return info
