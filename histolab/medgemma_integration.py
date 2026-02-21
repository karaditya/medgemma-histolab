"""
MedGemma Integration Module

Handles loading and inference for MedGemma models
with support for both baseline and fine-tuned versions.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np

# Minimum RAM (bytes) required to attempt loading MedGemma-4B
_MIN_RAM_BYTES = 8 * 1024 ** 3   # 8 GiB


def _available_cpu_ram_bytes() -> int:
    """Return available system RAM in bytes, using psutil if available."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    # Fallback: read /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0  # unknown — don't restrict


def _check_ram_or_warn(label: str = "") -> None:
    """Log a clear warning if available RAM is below the minimum for MedGemma-4B."""
    available = _available_cpu_ram_bytes()
    if available and available < _MIN_RAM_BYTES:
        avail_gib = available / 1024 ** 3
        logger = logging.getLogger(__name__)
        logger.warning(
            f"[{label}] Only {avail_gib:.1f} GiB RAM available. "
            f"MedGemma-4B needs ~8 GiB minimum (bfloat16 weights alone). "
            f"This will likely OOM. Consider running on a machine with ≥10 GiB RAM "
            f"or on a GPU machine (Kaggle, Colab, cluster)."
        )
        print(
            f"\n⚠️  WARNING: Only {avail_gib:.1f} GiB RAM available — "
            f"MedGemma-4B requires ~8 GiB minimum. Expect OOM.\n"
        )

logger = logging.getLogger(__name__)


@dataclass
class MedGemmaConfig:
    """Configuration for MedGemma model."""
    model_name: str = "google/medgemma-4b-it"
    model_type: str = "baseline"  # "baseline" or "fine-tuned"
    fine_tuned_path: Optional[str] = None
    quantization: int = 4  # 4-bit, 8-bit, or 0 for full precision
    device: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.1
    torch_dtype: str = "bfloat16"
    hf_token: Optional[str] = None
    use_fast: bool = True  # Use fast processor for faster image preprocessing


class MedGemmaWrapper:
    """
    Wrapper for MedGemma model inference.
    
    Supports both baseline and fine-tuned models with
    quantization for efficient local deployment.
    """
    
    def __init__(self, config: Optional[MedGemmaConfig] = None):
        """
        Initialize MedGemma wrapper.
        
        Args:
            config: Optional configuration object
        """
        self.config = config or MedGemmaConfig()
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self) -> bool:
        """
        Load the MedGemma model.
        
        Returns:
            True if loaded successfully
            
        Raises:
            RuntimeError: If model fails to load
        """
        if self._loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            
            model_path = self._get_model_path()
            logger.info(f"Loading MedGemma from: {model_path}")

            # --- Fine-tuned model with LoRA adapter: dedicated loading path ---
            # Uses bfloat16 base + adapter merge (lossless) to avoid:
            # 1. PeftModel.generate() not forwarding pixel_values to vision encoder
            # 2. merge_and_unload() on 4-bit models destroying learned weights
            if self.config.model_type == "fine-tuned" and self.config.fine_tuned_path:
                adapter_path = Path(self.config.fine_tuned_path)
                if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
                    return self._load_finetuned_model(adapter_path)

            # --- GPU-adaptive strategy selection ---
            # Detect GPU capabilities and choose quantization strategy
            # that actually works for the available hardware.
            has_gpu = torch.cuda.is_available()
            gpu_mem_gib = 0.0
            bf16_supported = False

            if has_gpu:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                gpu_mem_gib = total_mem / (1024 ** 3)
                bf16_supported = torch.cuda.is_bf16_supported()
                logger.info(
                    f"GPU detected: {gpu_mem_gib:.1f} GiB VRAM, "
                    f"bf16 {'supported' if bf16_supported else 'not supported'}"
                )

            model_dtype = None
            device_map = None
            max_memory = None

            _check_ram_or_warn("baseline")
            if has_gpu:
                # --- GPU available: bfloat16 with GPU+CPU split ---
                # Same strategy as the fine-tuned path: bf16, device_map="auto",
                # reserve headroom for vision encoder activations + KV cache.
                model_dtype = torch.bfloat16 if bf16_supported else torch.float16
                device_map = "auto"
                reserve = int(2.5 * 1024 ** 3)  # 2.5 GiB for inference activations
                model_budget = max(int(total_mem - reserve), int(total_mem // 4))
                cpu_ram = _available_cpu_ram_bytes()
                cpu_budget = f"{int(cpu_ram * 0.8 / 1024**3)}GiB" if cpu_ram else "8GiB"
                max_memory = {0: model_budget, "cpu": cpu_budget}
                logger.info(
                    f"Strategy: bf16 with GPU+CPU split, "
                    f"GPU budget {model_budget / 1024**3:.1f} GiB, CPU budget {cpu_budget}"
                )
            else:
                # --- No GPU: CPU-only, full precision ---
                model_dtype = torch.float32
                device_map = "cpu"
                logger.info("Strategy: CPU-only, float32 (no usable GPU)")

            hf_token = self._get_hf_token()
            
            # Load processor with fast tokenizer support
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                token=hf_token,
                use_fast=self.config.use_fast
            )

            # Load model with the selected strategy
            load_kwargs = dict(
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=model_dtype,
                token=hf_token
            )
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                **load_kwargs,
            )
            
            self._loaded = True
            logger.info("MedGemma loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def _get_model_path(self) -> str:
        """Get the base model path based on configuration."""
        # For fine-tuned, we still load the base model first, then add adapter
        # The adapter is loaded separately via PeftModel
        return self.config.model_name

    def _load_finetuned_model(self, adapter_path: Path) -> bool:
        """
        Load a fine-tuned model.

        Uses a cached merged model with 4-bit quantization for fast inference.
        On first load: bf16 base + adapter merge → save merged → reload in 4-bit.
        On subsequent loads: load merged directly in 4-bit (~2.5GB VRAM).
        """
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch

        # Read base model name from adapter metadata
        base_config_path = adapter_path / "base_model_config.json"
        if base_config_path.exists():
            with open(base_config_path) as f:
                base_cfg = json.load(f)
            base_model_name = base_cfg["base_model_name"]
        else:
            base_model_name = self.config.model_name

        # Load HF token
        hf_token = self._get_hf_token()

        # Check for cached merged model
        merged_path = adapter_path / "merged_model"
        if not merged_path.exists():
            self._create_merged_model(adapter_path, base_model_name, hf_token, merged_path)

        # Load merged model with GPU+CPU split.
        # Use bf16 if GPU supports it natively (Ampere+), otherwise fp16 (T4, etc.)
        # 4-bit quantization destroys the LoRA signal → empty outputs.
        _check_ram_or_warn("fine-tuned")

        bf16_native = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model_dtype = torch.bfloat16 if bf16_native else torch.float16
        logger.info(
            f"Loading merged model in {model_dtype} from: {merged_path} "
            f"(bf16 native: {bf16_native})"
        )

        self.processor = AutoProcessor.from_pretrained(
            str(merged_path),
            trust_remote_code=True,
            token=hf_token,
            use_fast=self.config.use_fast
        )

        load_kwargs = dict(
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=model_dtype,
            token=hf_token,
        )

        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            # Reserve 2.5 GiB for vision encoder activations + KV cache.
            # accelerate oversubscribes by ~40%, so be aggressive.
            model_budget = max(int(total_mem - 2.5 * 1024**3), int(total_mem // 4))
            cpu_ram = _available_cpu_ram_bytes()
            cpu_budget = f"{int(cpu_ram * 0.8 / 1024**3)}GiB" if cpu_ram else "8GiB"
            load_kwargs["max_memory"] = {0: model_budget, "cpu": cpu_budget}
            logger.info(
                f"GPU model budget: {model_budget / 1024**3:.1f} GiB "
                f"(total {total_mem / 1024**3:.1f} GiB, reserving 2.5 GiB for activations)"
            )

        self.model = AutoModelForImageTextToText.from_pretrained(
            str(merged_path),
            **load_kwargs,
        )
        self.model.eval()

        self._loaded = True
        logger.info(f"Fine-tuned model loaded successfully ({model_dtype}, GPU+CPU split)")
        return True

    def _create_merged_model(self, adapter_path: Path, base_model_name: str,
                             hf_token: str, merged_path: Path) -> None:
        """One-time: load bf16 base, merge LoRA adapter, save merged model to disk."""
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        import torch

        logger.info("First load — creating merged model cache (one-time operation)...")
        logger.info(f"  Base model: {base_model_name}")
        logger.info(f"  Adapter: {adapter_path}")
        logger.info(f"  Output: {merged_path}")

        # Load base in bf16 for lossless merge
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            device_map="cpu",  # CPU to avoid VRAM pressure during merge
            torch_dtype=torch.bfloat16,
            token=hf_token
        )

        # Apply and merge adapter
        logger.info("Applying LoRA adapter...")
        base_model = PeftModel.from_pretrained(base_model, str(adapter_path))
        logger.info("Merging adapter (lossless in bfloat16)...")
        base_model = base_model.merge_and_unload()

        # Save merged model + processor
        logger.info(f"Saving merged model to {merged_path}...")
        base_model.save_pretrained(str(merged_path))

        # Load processor: prefer adapter dir (has chat_template.jinja from training),
        # fall back to base model on HF Hub if adapter is incomplete.
        try:
            processor = AutoProcessor.from_pretrained(
                str(adapter_path),
                trust_remote_code=True,
                token=hf_token,
            )
            # Verify the processor has a chat template — critical for correct prompting
            has_template = getattr(processor.tokenizer, 'chat_template', None) is not None
            if not has_template:
                logger.warning("Adapter processor missing chat_template, loading from base model")
                processor = AutoProcessor.from_pretrained(
                    base_model_name,
                    trust_remote_code=True,
                    token=hf_token,
                )
        except Exception as e:
            logger.warning(f"Failed to load processor from adapter ({e}), using base model")
            processor = AutoProcessor.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                token=hf_token,
            )
        processor.save_pretrained(str(merged_path))

        # Free memory
        del base_model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Merged model saved successfully")

    def _get_hf_token(self) -> Optional[str]:
        """Load HF token from config, file, or environment."""
        hf_token = self.config.hf_token
        if hf_token is None:
            token_file = Path("hf_token.txt")
            if token_file.exists():
                try:
                    hf_token = token_file.read_text().strip()
                except Exception:
                    pass
            if hf_token is None:
                hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        return hf_token

    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            self.model.cpu()  # Move to CPU first to free GPU memory
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False

        # Force garbage collection THEN clear CUDA cache
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

        logger.info("Model unloaded")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def analyze_patch(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        response_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a single image patch.

        Args:
            image: PIL Image to analyze
            prompt: Custom prompt (optional)
            context: Patient context/clinical notes (optional)
            max_new_tokens: Override max tokens for this call (optional)
            response_prefix: Force the model response to start with this text.
                Useful for fine-tuned models whose LoRA dominates: setting a
                prefix like "Step 1:" prevents the model from falling back
                to the training pattern ("The answer is: X").

        Returns:
            Dictionary with analysis results
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Build the prompt - optimized for NCT-CRC-HE-100K dataset
        default_prompt = (
            "Analyze this histopathology image patch from the NCT-CRC-HE-100K dataset. "
            "The possible tissue types are: ADI (adipose), BACK (background), DEB (debris), "
            "LYM (lymphocyte), MUC (mucus), MUS (muscle), NORM (normal), STR (stroma), TUM (tumor). "
            "Provide: "
            "1. Whether this patch shows cancerous tissue (yes/no). "
            "2. Cancer probability (0-100%). "
            "3. Tissue type (must be one of the 9 possible types listed above). "
            "4. Notable features (e.g., nuclear pleomorphism, mitoses)."
        )
        
        full_prompt = prompt or default_prompt
        if context:
            full_prompt = f"Patient context: {context}\n\n{full_prompt}"
        
        try:
            import torch

            # Step 1: Build chat text with image placeholder
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": full_prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Append response prefix so the model continues from it
            # instead of falling into the LoRA-trained "The answer is:" pattern
            if response_prefix:
                text += response_prefix

            logger.info(f"[MedGemma INPUT] Prompt text:\n{text}")

            # Step 2: Process text + image together to get
            # input_ids, attention_mask, AND pixel_values.
            # Get the actual device from the model to ensure inputs go to the same device.
            import torch
            
            # Get model's actual device - handles quantized models correctly
            try:
                model_device = next(self.model.parameters()).device
            except Exception:
                # Fallback for quantized models that don't have parameters()
                model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            device = str(model_device)
            
            inputs = self.processor(
                text=text,
                images=image,
                return_tensors="pt"
            ).to(device)

            logger.info(
                f"[MedGemma INPUT] Tensor keys: {list(inputs.keys())}, "
                f"input_ids shape: {inputs['input_ids'].shape}, "
                f"device: {device}"
            )
            if 'pixel_values' in inputs:
                logger.info(f"[MedGemma INPUT] pixel_values shape: {inputs['pixel_values'].shape}")
            else:
                logger.warning("No pixel_values in inputs!")

            input_len = inputs["input_ids"].shape[-1]

            # Free fragmented GPU memory before the forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate — greedy decoding, wrapped in inference_mode
            # to avoid any autograd overhead
            gen_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=gen_tokens,
                    do_sample=False,
                )

            logger.info(f"[MedGemma OUTPUT] Total tokens: {outputs[0].shape[0]}, prompt tokens: {input_len}, new tokens: {outputs[0].shape[0] - input_len}")

            # Decode only the NEW tokens (exclude the prompt)
            response = self.processor.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )

            logger.info(f"[MedGemma OUTPUT] Model response:\n{response}")
            
            # Extract structured information
            logger.debug(f"Raw MedGemma response: {repr(response)}")
            result = self._parse_response(response)
            result["raw_response"] = response
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "error": str(e),
                "cancer_probability": 0.0,
                "is_cancerous": False,
                "confidence": 0.1,
                "raw_response": "",
                "tissue_type": "unknown"
            }
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response into structured format.
        
        Args:
            response: Raw model response text
            
        Returns:
            Parsed result dictionary
        """
        result = {
            "cancer_probability": 0.0,
            "is_cancerous": False,
            "tissue_type": "unknown",
            "features": [],
            "grade": None,
            "confidence": 0.5
        }
        
        response_lower = response.lower()
        
        # Detect cancer probability
        if "yes" in response_lower and "cancer" in response_lower:
            result["is_cancerous"] = True
            result["cancer_probability"] = 0.7
        
        if "no" in response_lower and "cancer" in response_lower:
            result["is_cancerous"] = False
            result["cancer_probability"] = 0.2
        
        # Extract probability from text
        import re
        prob_match = re.search(r'(\d+)%?', response)
        if prob_match:
            prob = int(prob_match.group(1))
            result["cancer_probability"] = prob / 100.0
        
        # Extract tissue type - specific to NCT-CRC-HE-100K dataset
        # First check for exact matches or clear references
        tissue_map = {
            "adi": "ADI",
            "adipose": "ADI",
            "fat": "ADI",
            "back": "BACK",
            "background": "BACK",
            "deb": "DEB",
            "debris": "DEB",
            "lym": "LYM",
            "lymphocyte": "LYM",
            "lymphoid": "LYM",
            "muc": "MUC",
            "mucus": "MUC",
            "mus": "MUS",
            "muscle": "MUS",
            "norm": "NORM",
            "normal": "NORM",
            "str": "STR",
            "stroma": "STR",
            "stromal": "STR",
            "tum": "TUM",
            "tumor": "TUM",
            "cancer": "TUM"
        }
        
        # First check for exact matches or strong indicators
        best_tissue = "unknown"
        best_confidence = 0.5
        
        for keyword, tissue in tissue_map.items():
            if keyword in response_lower:
                # Check for stronger indicators or multiple occurrences for higher confidence
                occurrences = response_lower.count(keyword)
                if occurrences > 1 or len(keyword) > 3:  # More confident if longer keyword or multiple hits
                    confidence = 0.9
                else:
                    confidence = 0.8
                
                # Keep the tissue type with highest confidence
                if confidence > best_confidence:
                    best_tissue = tissue
                    best_confidence = confidence
        
        # Special handling for MedGemma's response formats
        if best_tissue == "unknown":
            # Handle cases like "Histopathology Image of ADI Tissue"
            for tissue_type in ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]:
                if tissue_type in response:
                    best_tissue = tissue_type
                    best_confidence = 0.85
                    break
            
            # Handle cases like "Muscular Tissue (MUS)" 
            if best_tissue == "unknown":
                for tissue_type in ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]:
                    if f"({tissue_type})" in response:
                        best_tissue = tissue_type
                        best_confidence = 0.95
                        break
        
        result["tissue_type"] = best_tissue
        result["confidence"] = best_confidence
        
        # If tissue type still unknown, try more aggressive parsing
        if result["tissue_type"] == "unknown":
            # Check for any possible tissue type references
            possible_types = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
            for tissue_type in possible_types:
                if tissue_type.lower() in response_lower:
                    result["tissue_type"] = tissue_type
                    result["confidence"] = 0.7
                    break
        
        # Special handling for cases where model mentions "ADI" but in complex context
        if "adi" in response_lower and result["tissue_type"] == "unknown":
            result["tissue_type"] = "ADI"
            result["confidence"] = 0.6
        
        # Debug info
        logger.debug(f"Parsed tissue type: {result['tissue_type']} (confidence: {result['confidence']:.2f}) from response: {repr(response_lower)}")
        
        # Extract features
        feature_keywords = [
            "nuclear pleomorphism",
            "mitoses",
            "necrosis",
            "hyperchromasia",
            "atypia",
            "inflammation",
            "fibrosis"
        ]
        
        for feature in feature_keywords:
            if feature in response_lower:
                result["features"].append(feature)
        
        # Detect grade
        if "high grade" in response_lower or "poorly differentiated" in response_lower:
            result["grade"] = "high_grade"
        elif "low grade" in response_lower or "well differentiated" in response_lower:
            result["grade"] = "low_grade"
        elif "intermediate" in response_lower or "moderately" in response_lower:
            result["grade"] = "intermediate_grade"
        
        return result
    
    def analyze_with_text(
        self,
        image: Image.Image,
        text: str,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze image combined with text report.
        
        Args:
            image: PIL Image
            text: Text report or clinical notes
            prompt: Custom prompt
            
        Returns:
            Analysis result dictionary
        """
        context = f"Text Report: {text}"
        return self.analyze_patch(image, prompt, context)
    
    def batch_analyze(
        self,
        images: List[Image.Image],
        prompt: Optional[str] = None,
        context: Optional[str] = None,
        batch_size: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batches.
        
        Args:
            images: List of PIL Images
            prompt: Custom prompt
            context: Patient context
            batch_size: Number of images per batch
            
        Returns:
            List of analysis results
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1}")
            
            for image in batch:
                result = self.analyze_patch(image, prompt, context)
                results.append(result)
        
        return results


class MedGemmaManager:
    """
    Manager for multiple MedGemma model instances.
    
    Supports switching between baseline and fine-tuned models.
    """
    
    def __init__(self):
        self.models: Dict[str, MedGemmaWrapper] = {}
        self.current_model: Optional[str] = None
    
    def get_model(
        self,
        model_type: str = "baseline",
        config: Optional[MedGemmaConfig] = None
    ) -> MedGemmaWrapper:
        """
        Get or create a model instance.
        
        Args:
            model_type: "baseline" or "fine-tuned"
            config: Optional configuration
            
        Returns:
            MedGemmaWrapper instance
        """
        if model_type not in self.models:
            self.models[model_type] = MedGemmaWrapper(config)
        
        self.current_model = model_type
        return self.models[model_type]
    
    def switch_model(self, model_type: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_type: Model type to switch to
            
        Returns:
            True if successful
        """
        if model_type not in self.models:
            logger.error(f"Model {model_type} not loaded")
            return False
        
        self.current_model = model_type
        logger.info(f"Switched to {model_type} model")
        return True
    
    def unload_all(self) -> None:
        """Unload all models."""
        for wrapper in self.models.values():
            wrapper.unload()
        self.models.clear()
        self.current_model = None
    
    def get_current_model(self) -> Optional[MedGemmaWrapper]:
        """Get the currently active model."""
        if self.current_model and self.current_model in self.models:
            return self.models[self.current_model]
        return None


def create_medgemma_wrapper(
    model_name: str = "google/medgemma-4b-it",
    model_type: str = "baseline",
    quantization: int = 4,
    device: str = "auto"
) -> MedGemmaWrapper:
    """
    Factory function to create MedGemma wrapper.
    
    Args:
        model_name: Model identifier or path
        model_type: "baseline" or "fine-tuned"
        quantization: Bit quantization (0, 4, or 8)
        device: Device to run on ("auto", "cuda", "cpu")
        
    Returns:
        Configured MedGemmaWrapper instance
    """
    config = MedGemmaConfig(
        model_name=model_name,
        model_type=model_type,
        quantization=quantization,
        device=device
    )
    
    wrapper = MedGemmaWrapper(config)
    return wrapper
