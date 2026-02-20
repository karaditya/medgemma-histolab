"""
Configuration management for HistoLab application.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for MedGemma model."""
    model_name: str = "google/medgemma-4b-it"
    model_type: str = "baseline"  # "baseline" or "fine-tuned"
    fine_tuned_path: Optional[str] = None
    quantization: int = 4  # 4-bit, 8-bit, or 0 for full precision
    device: str = "auto"  # "auto", "cuda", "cpu", "mps"
    max_new_tokens: int = 512
    temperature: float = 0.1
    torch_dtype: str = "bfloat16"


@dataclass
class PreprocessingConfig:
    """Configuration for WSI preprocessing."""
    patch_size: int = 512
    patch_overlap: int = 0
    target_size: int = 1024  # Resize large WSIs for preview
    tile_size: int = 256  # Size for MedGemma input
    max_patches: int = 100  # Limit for MVP
    save_patches: bool = True
    patches_dir: str = "data/patches"
    quality_threshold: float = 0.7  # For filtering blurry patches


@dataclass
class AnalysisConfig:
    """Configuration for analysis settings."""
    cancer_threshold: float = 0.5
    grade_weights: dict = field(default_factory=lambda: {
        "low_grade": 0.3,
        "intermediate_grade": 0.5,
        "high_grade": 0.8
    })
    batch_size: int = 4
    aggregate_method: str = "max"  # "max", "mean", "weighted"
    confidence_threshold: float = 0.7


@dataclass
class UIConfig:
    """Configuration for UI settings."""
    theme: str = "default"
    language: str = "en"
    show_confidence: bool = True
    show_heatmaps: bool = True
    enable_annotations: bool = True
    max_image_size: int = 2048
    enable_educational_mode: bool = False


@dataclass
class ExportConfig:
    """Configuration for export settings."""
    export_dir: str = "data/exports"
    include_visualizations: bool = True
    include_text_report: bool = True
    pdf_quality: str = "high"
    anonymize_patient_data: bool = True


@dataclass
class PrivacyConfig:
    """Configuration for privacy and security."""
    offline_mode: bool = True
    encrypt_cache: bool = True
    cache_dir: str = "data/cache"
    telemetry: bool = False
    log_level: str = "WARNING"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = "logs"
    log_level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """Load configuration from JSON or YAML file."""
        path = Path(config_path)
        if path.suffix == ".json":
            return cls._from_json(path)
        elif path.suffix in [".yaml", ".yml"]:
            return cls._from_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    @classmethod
    def _from_json(cls, path: Path) -> "Config":
        """Load from JSON file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def _from_yaml(cls, path: Path) -> "Config":
        """Load from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to file."""
        import json
        path = Path(config_path)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate model config
        if self.model.quantization not in [0, 4, 8]:
            issues.append("Quantization must be 0, 4, or 8 bits")
        
        if self.preprocessing.patch_size < 64:
            issues.append("Patch size too small (minimum 64)")
        
        if self.analysis.cancer_threshold < 0 or self.analysis.cancer_threshold > 1:
            issues.append("Cancer threshold must be between 0 and 1")
        
        return issues


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration, optionally from file."""
    if config_path and Path(config_path).exists():
        return Config.from_file(config_path)
    return Config()


# Environment variable overrides
def apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    if os.getenv("HISTOLAB_MODEL"):
        config.model.model_name = os.getenv("HISTOLAB_MODEL")
    if os.getenv("HISTOLAB_QUANTIZATION"):
        config.model.quantization = int(os.getenv("HISTOLAB_QUANTIZATION"))
    if os.getenv("HISTOLAB_DEVICE"):
        config.model.device = os.getenv("HISTOLAB_DEVICE")
    if os.getenv("HISTOLAB_OFFLINE"):
        config.privacy.offline_mode = os.getenv("HISTOLAB_OFFLINE").lower() == "true"
    
    return config
