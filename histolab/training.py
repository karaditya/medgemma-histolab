"""
Fine-tuning Module - HistoLocal

Implements LoRA (Low-Rank Adaptation) and QLoRA for efficient fine-tuning
of MedGemma models on histopathology datasets.

This module enables:
- LoRA/QLoRA fine-tuning with 4-bit quantization
- Zero-shot vs Fine-tuned comparison benchmarks
- Support for datasets: TCGA-BRCA, PANDA, PCam, BACH, NCT-CRC-HE-100K

Reference: https://github.com/microsoft/LoRA
Reference: https://github.com/artidoro/qlora
"""

import os
import logging
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LoRA/QLoRA fine-tuning."""
    # Model settings
    model_name: str = "google/medgemma-4b-it"
    model_type: str = "qlora"  # "lora", "qlora", "full"
    
    # Quantization settings (for QLoRA)
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # LoRA settings
    lora_r: int = 64  # LoRA rank
    lora_alpha: int = 128  # LoRA alpha
    lora_dropout: float = 0.05  # LoRA dropout
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Training settings
    output_dir: str = "models/finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    max_image_size: int = 384
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Evaluation settings
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    
    # Misc
    seed: int = 42
    fp16: bool = False
    bf16: bool = True


@dataclass
class DatasetConfig:
    """Configuration for training dataset."""
    name: str  # Dataset name (tcga, panda, pcam, bach, crc)
    path: str  # Path to dataset
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"
    label_column: str = "label"
    image_column: str = "image_path"
    text_column: Optional[str] = None  # For multimodal datasets
    
    # Dataset-specific settings
    num_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ["negative", "positive"])
    task_type: str = "classification"  # "classification", "generation", "vqa"


@dataclass
class BenchmarkResult:
    """Result from a benchmark evaluation."""
    model_type: str  # "baseline" or "fine-tuned"
    dataset_name: str
    metrics: Dict[str, float]
    predictions: List[Dict[str, Any]]
    ground_truth: List[Any]
    inference_time: float
    memory_used_gb: float
    timestamp: str


class LoRATrainer:
    """
    LoRA/QLoRA trainer for MedGemma models.
    
    Implements efficient fine-tuning using:
    - 4-bit quantization via bitsandbytes
    - Low-rank adaptation for parameter-efficient training
    - Flash Attention for memory efficiency
    """
    
    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        output_dir: str = "models/finetuned"
    ):
        """
        Initialize the LoRA trainer.
        
        Args:
            training_config: Training configuration
            output_dir: Directory to save checkpoints
        """
        self.config = training_config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.trainer = None
        
        # Track best metrics
        self.best_metrics = {}
    
    def prepare_model_and_tokenizer(self, model_path: Optional[str] = None):
        """
        Prepare model and tokenizer with quantization settings.
        
        Args:
            model_path: Optional path to local model or custom model name
        """
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        
        model_name = model_path or self.config.model_name
        
        logger.info(f"Loading model: {model_name}")
        
        # Load Hugging Face token
        hf_token = None
        token_file = Path("hf_token.txt")
        if token_file.exists():
            try:
                hf_token = token_file.read_text().strip()
                logger.info("Loaded HF token from hf_token.txt")
            except Exception as e:
                logger.warning(f"Failed to read HF token from file: {e}")
        # Also try environment variable
        if hf_token is None:
            hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                logger.info("Loaded HF token from environment variable")
        
        # Configure quantization
        quantization_config = None
        if self.config.load_in_4bit or self.config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type
            )
        
        # Load processor - use default size from model config
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Clear GPU cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Determine device map based on available GPU memory
        device_map = "auto"
        if torch.cuda.is_available():
            try:
                # Check available GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"GPU memory available: {gpu_mem:.1f} GB")
                
                # If we have enough GPU memory (>= 10GB), load all on GPU
                if gpu_mem >= 10:
                    device_map = {"": 0}  # Load entire model on GPU 0
                else:
                    # For smaller GPUs, use auto device map
                    device_map = "auto"
            except Exception as e:
                logger.warning(f"Failed to check GPU memory: {e}")
                device_map = "auto"
        else:
            device_map = "cpu"
        
        logger.info(f"Using device map: {device_map}")
        
        # Load model with memory-efficient settings
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            device_map=device_map,
            trust_remote_code=True,
            token=hf_token
        )
        
        # Freeze vision encoder - only fine-tune the LLM part
        # This significantly reduces memory usage
        if hasattr(self.model, 'vision_tower'):
            logger.info("Freezing vision encoder...")
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
        
        # Also freeze any other non-LLM components
        if hasattr(self.model, 'image_newline'):
            for param in self.model.image_newline.parameters():
                param.requires_grad = False
        
        # Configure LoRA only on the language model (NOT the vision tower).
        # PEFT matches target_modules as regex against full module names.
        # Using a regex that requires "language_model" in the path prevents
        # LoRA adapters from being added to the vision encoder.
        if self.config.model_type in ["lora", "qlora"]:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=".*language_model.*\\.(q_proj|k_proj|v_proj|o_proj)",
                bias="none",
                inference_mode=False
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Enable input gradients BEFORE gradient checkpointing.
        # With QLoRA the quantized base model doesn't preserve input gradients,
        # so gradient checkpointing would disconnect the computation graph.
        self.model.enable_input_require_grads()

        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Ensure model is in training mode
        self.model.train()
        
        # Log trainable parameter summary (gradients are None until first backward pass — that's normal)
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())

        logger.info(f"Total parameters: {total_count:,}")
        logger.info(f"Trainable parameters: {trainable_count:,} ({trainable_count/total_count:.2%})")
        
        logger.info("Model prepared for training")
    
    def create_datasets(
        self,
        dataset_config: DatasetConfig,
        train_transforms: Optional[callable] = None,
        val_transforms: Optional[callable] = None
    ):
        """
        Create training and validation datasets.
        
        Args:
            dataset_config: Dataset configuration
            train_transforms: Optional training transforms
            val_transforms: Optional validation transforms
        """
        # This is a placeholder - actual implementation depends on dataset format
        # For TCGA/PANDA: Load from directory structure
        # For PCam/BACH: Load from HDF5 or image files
        
        logger.info(f"Creating datasets for: {dataset_config.name}")
        
        # Import datasets library
        try:
            from datasets import Dataset, DatasetDict, Image
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            raise
        
        # Create mock datasets for demonstration
        # In practice, this would load from actual dataset files
        train_dataset = Dataset.from_dict({
            "image": [],
            "label": [],
            "text": []
        })
        val_dataset = Dataset.from_dict({
            "image": [],
            "label": [],
            "text": []
        })
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset
        })
    
    def train(
        self,
        train_dataset,
        eval_dataset: Optional[Any] = None,
        resume_from_checkpoint: Optional[str] = None,
        use_wandb: bool = True,
        wandb_project: str = "histolab-medgemma-finetune",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None
    ):
        """
        Run training.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Optional checkpoint path to resume from
            use_wandb: Whether to use Weights & Biases for tracking
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team)
            wandb_run_name: Custom W&B run name
        """
        from transformers import TrainingArguments, Trainer
        from transformers import DataCollatorForSeq2Seq
        import torch
        
        logger.info("Starting training...")
        
        # Create output directory
        run_name = f"{self.config.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if wandb_run_name:
            run_name = wandb_run_name
        run_dir = self.output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb if enabled
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=run_name,
                    config={
                        "model_name": self.config.model_name,
                        "model_type": self.config.model_type,
                        "lora_r": self.config.lora_r,
                        "lora_alpha": self.config.lora_alpha,
                        "lora_dropout": self.config.lora_dropout,
                        "num_train_epochs": self.config.num_train_epochs,
                        "learning_rate": self.config.learning_rate,
                        "per_device_train_batch_size": self.config.per_device_train_batch_size,
                        "per_device_eval_batch_size": self.config.per_device_eval_batch_size,
                        "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                        "warmup_ratio": self.config.warmup_ratio,
                        "weight_decay": self.config.weight_decay,
                        "max_seq_length": self.config.max_seq_length,
                        "max_image_size": self.config.max_image_size,
                    }
                )
                logger.info(f"Initialized wandb run: {run_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
                use_wandb = False
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=str(run_dir),
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            eval_strategy=self.config.eval_strategy if eval_dataset else "no",
            save_strategy=self.config.save_strategy,
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            report_to="wandb" if use_wandb else "none",
            seed=self.config.seed,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            run_name=run_name
        )
        
        # Preprocess datasets - handle batched processing correctly
        def preprocess_function(examples):
            """Preprocess function to tokenize text and prepare images."""
            from PIL import Image as PILImage

            # Handle batched data - examples["text"] and examples["image"] are lists
            texts = examples["text"]
            # Support both PIL images (legacy) and file paths (lazy loading)
            raw_images = examples.get("image", examples.get("image_path", [None] * len(texts)))
            # Optional image transform (e.g., augmentation) set via trainer._image_transform
            image_transform = getattr(self, '_image_transform', None)
            images = []
            for img in raw_images:
                if isinstance(img, str):
                    # Lazy loading: img is a file path
                    img = PILImage.open(img).convert("RGB")
                if image_transform is not None and img is not None:
                    img = image_transform(img)
                images.append(img)
            label_names = examples.get("label_name", [None] * len(texts))

            # Build messages for each example in the batch
            all_inputs = []
            for idx, (text, image) in enumerate(zip(texts, images)):
                # Get the class name from label_name (e.g., "ADI", "BACK", etc.)
                class_name = label_names[idx] if label_names[idx] is not None else "unknown"

                # Build messages: user asks question, assistant answers with class name
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": f"The answer is: {class_name}"}
                        ]
                    }
                ]

                # Apply chat template
                formatted_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False  # Include assistant response for training
                )

                # Process text and images
                inputs = self.processor(
                    text=formatted_text,
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.config.max_seq_length
                )

                # Squeeze out the batch dimension since we're processing one by one
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}

                # Label the ENTIRE assistant response so the model learns:
                # 1. "The answer is:" prefix (response format)
                # 2. The class name
                # 3. <end_of_turn> stop token
                # Without this, the model only learns 1 token and can't generate properly.
                input_ids = inputs["input_ids"]
                seq_length = input_ids.shape[0]

                # Create labels filled with -100 (ignore index for cross-entropy)
                label_ids = torch.full((seq_length,), -100, dtype=torch.long)

                # Find "The answer is:" in the tokenized sequence
                answer_prefix = "The answer is:"
                answer_prefix_ids = self.processor.tokenizer.encode(answer_prefix, add_special_tokens=False)
                pad_token_id = self.processor.tokenizer.pad_token_id

                input_ids_list = input_ids.tolist()
                for i in range(len(input_ids_list) - len(answer_prefix_ids) + 1):
                    if input_ids_list[i:i+len(answer_prefix_ids)] == answer_prefix_ids:
                        # Label everything from "The answer is:" to end of content (before padding)
                        for j in range(i, seq_length):
                            tid = input_ids_list[j]
                            if pad_token_id is not None and tid == pad_token_id:
                                break
                            label_ids[j] = tid
                        break

                inputs["labels"] = label_ids
                all_inputs.append(inputs)

            # Stack all inputs into a single batch
            batch = {}
            for key in all_inputs[0].keys():
                batch[key] = torch.stack([f[key] for f in all_inputs])

            return batch
        
        # Process datasets in smaller batches to avoid memory issues
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            batch_size=8,
            remove_columns=train_dataset.column_names
        )
        
        # --- Label masking validation ---
        # Inspect a few samples to confirm labels are set correctly
        num_check = min(3, len(tokenized_train))
        for idx in range(num_check):
            sample = tokenized_train[idx]
            label_tensor = torch.tensor(sample["labels"]) if not isinstance(sample["labels"], torch.Tensor) else sample["labels"]
            non_ignored = (label_tensor != -100).sum().item()
            if non_ignored == 0:
                logger.warning(
                    f"[LABEL CHECK] Train sample {idx}: ALL labels are -100 — "
                    "model will learn nothing! Check that 'The answer is:' pattern "
                    "is found in the tokenized input."
                )
            else:
                target_ids = label_tensor[label_tensor != -100].tolist()
                decoded = self.processor.tokenizer.decode(target_ids)
                logger.info(
                    f"[LABEL CHECK] Train sample {idx}: {non_ignored} target tokens, "
                    f"decoded label = {repr(decoded)}"
                )

        tokenized_val = None
        if eval_dataset:
            tokenized_val = eval_dataset.map(
                preprocess_function,
                batched=True,
                batch_size=8,
                remove_columns=eval_dataset.column_names
            )

            # Validate a few val samples too
            num_check_val = min(3, len(tokenized_val))
            for idx in range(num_check_val):
                sample = tokenized_val[idx]
                label_tensor = torch.tensor(sample["labels"]) if not isinstance(sample["labels"], torch.Tensor) else sample["labels"]
                non_ignored = (label_tensor != -100).sum().item()
                if non_ignored == 0:
                    logger.warning(
                        f"[LABEL CHECK] Val sample {idx}: ALL labels are -100"
                    )
                else:
                    target_ids = label_tensor[label_tensor != -100].tolist()
                    decoded = self.processor.tokenizer.decode(target_ids)
                    logger.info(
                        f"[LABEL CHECK] Val sample {idx}: {non_ignored} target tokens, "
                        f"decoded label = {repr(decoded)}"
                    )
        
        # Custom data collator for multi-modal inputs
        def data_collator(features):
            """Custom collator to handle multi-modal inputs."""
            import torch
            
            if len(features) == 0:
                return {}
            
            # Stack all features - all values should now be tensors from preprocess_function
            batch = {}
            
            for key in features[0].keys():
                values = []
                for f in features:
                    value = f[key]
                    if isinstance(value, list):
                        # Convert lists to tensors
                        values.append(torch.tensor(value))
                    elif isinstance(value, np.ndarray):
                        values.append(torch.tensor(value))
                    else:
                        values.append(value)
                
                # Stack values
                if key == "pixel_values":
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.stack(values)
            
            return batch
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator
        )
        
        # Explicitly set model to training mode and verify gradients
        self.model.train()
        
        # Note: param.grad is None before first backward pass — that's normal, not a bug.
        # Log model parameter information to wandb
        if use_wandb:
            try:
                import wandb
                # Log total parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                non_trainable_params = total_params - trainable_params
                
                wandb.log({
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                    "model/non_trainable_parameters": non_trainable_params,
                    "model/trainable_ratio": trainable_params / total_params
                })
                
                logger.info(f"Total parameters: {total_params:,}")
                logger.info(f"Trainable parameters: {trainable_params:,}")
                logger.info(f"Non-trainable parameters: {non_trainable_params:,}")
                logger.info(f"Trainable ratio: {trainable_params / total_params:.2%}")
            except Exception as e:
                logger.warning(f"Failed to log model parameters to wandb: {e}")
        
        # Train
        logger.info("Starting training loop...")
        
        # Use custom training loop to ensure gradients flow properly
        from torch.nn.functional import cross_entropy
        from tqdm import tqdm
        
        # Set model to training mode
        self.model.train()
        
        # Create dataloaders
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            tokenized_train,
            batch_size=training_args.per_device_train_batch_size,
            collate_fn=data_collator,
            shuffle=True
        )

        val_loader = None
        if tokenized_val is not None:
            val_loader = DataLoader(
                tokenized_val,
                batch_size=training_args.per_device_eval_batch_size,
                collate_fn=data_collator,
                shuffle=False
            )
        
        # Setup optimizer
        from transformers import get_scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay
        )
        
        num_training_steps = len(train_loader) * training_args.num_train_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * training_args.warmup_ratio),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        loss_ema = None  # Exponential moving average of loss
        ema_alpha = 0.1  # EMA smoothing factor
        early_stop_threshold = 0.001  # Stop when EMA loss below this
        early_stop_patience = 50  # Consecutive low-loss steps before stopping
        low_loss_counter = 0
        stopped_early = False

        for epoch in range(training_args.num_train_epochs):
            if stopped_early:
                break
            logger.info(f"Epoch {epoch + 1}/{training_args.num_train_epochs}")

            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Move batch to device
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Check if outputs have logits
                if hasattr(outputs, 'logits') and outputs.logits is not None:
                    logits = outputs.logits
                    labels = batch.get("labels")
                    
                    if labels is not None:
                        # Shift labels for causal LM: labels[:, :-1] vs logits[:, :-1]
                        # Actually, Gemma3's shift_labels does this internally
                        # Let's compute loss manually to ensure gradients
                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                        
                        # Reshape for loss computation: (batch*seq, vocab)
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()
                        
                        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    else:
                        loss = outputs.loss
                else:
                    loss = outputs.loss
                
                # Check loss has gradient — fail fast instead of silently training nothing
                if loss.grad_fn is None:
                    raise RuntimeError(
                        f"Loss at step {step} has no grad_fn — the computation graph is disconnected. "
                        "Check that enable_input_require_grads() is called before gradient_checkpointing_enable()."
                    )
                
                # Backward pass
                loss = loss / training_args.gradient_accumulation_steps
                loss.backward()
                
                # Gradient accumulation
                if (step + 1) % training_args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), training_args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    # Clear memory periodically to avoid fragmentation
                    if torch.cuda.is_available() and (step + 1) % 100 == 0:
                        torch.cuda.empty_cache()
                
                step_loss = loss.item() * training_args.gradient_accumulation_steps
                epoch_loss += step_loss
                num_batches += 1

                # Update exponential moving average
                if loss_ema is None:
                    loss_ema = step_loss
                else:
                    loss_ema = ema_alpha * step_loss + (1 - ema_alpha) * loss_ema

                # Early stopping: if loss stays near zero, stop to prevent overfitting
                if loss_ema is not None and loss_ema < early_stop_threshold:
                    low_loss_counter += 1
                    if low_loss_counter >= early_stop_patience:
                        logger.info(
                            f"[EARLY STOP] EMA loss ({loss_ema:.6f}) below {early_stop_threshold} "
                            f"for {early_stop_patience} consecutive steps at step {step + 1}. "
                            f"Stopping to prevent overfitting."
                        )
                        stopped_early = True
                        break
                else:
                    low_loss_counter = 0

                # Log every logging_steps
                if (step + 1) % training_args.logging_steps == 0:
                    avg_loss = epoch_loss / num_batches if num_batches > 0 else step_loss
                    logger.info(
                        f"Step {step + 1}/{num_training_steps}, "
                        f"Loss(step): {step_loss:.4f}, "
                        f"Loss(ema): {loss_ema:.4f}, "
                        f"Loss(avg): {avg_loss:.4f}"
                    )

                    if use_wandb:
                        try:
                            import wandb
                            wandb.log({
                                "train/loss_avg": avg_loss,
                                "train/loss_step": step_loss,
                                "train/loss_ema": loss_ema,
                                "train/epoch": epoch + (step + 1) / len(train_loader),
                                "train/learning_rate": lr_scheduler.get_last_lr()[0]
                            })
                        except:
                            pass
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

            # --- Validation loop ---
            val_loss_avg = None
            if val_loader is not None:
                # Free training memory before validation (logits.float() needs ~2 GB)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model.eval()
                val_loss_total = 0.0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in tqdm(val_loader, desc="Validation"):
                        val_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                        val_outputs = self.model(**val_batch)
                        if hasattr(val_outputs, 'logits') and val_outputs.logits is not None:
                            val_labels = val_batch.get("labels")
                            if val_labels is not None:
                                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                                shift_logits = val_outputs.logits[..., :-1, :].contiguous()
                                shift_labels = val_labels[..., 1:].contiguous()
                                v_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                            else:
                                v_loss = val_outputs.loss
                        else:
                            v_loss = val_outputs.loss
                        val_loss_total += v_loss.item()
                        val_batches += 1
                val_loss_avg = val_loss_total / val_batches if val_batches > 0 else 0
                logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss_avg:.4f}")
                self.model.train()

            if use_wandb:
                try:
                    import wandb
                    log_dict = {
                        "train/epoch_loss": avg_epoch_loss,
                        "train/epoch": epoch + 1
                    }
                    if val_loss_avg is not None:
                        log_dict["val/epoch_loss"] = val_loss_avg
                    wandb.log(log_dict)
                except:
                    pass

        logger.info("Training completed successfully!")

        # Save final model — save LoRA adapter separately (NOT merge_and_unload).
        # merge_and_unload() on a 4-bit quantized model causes lossy
        # dequantize→add→requantize round-trip that destroys learned weights.
        final_model_path = run_dir / "final_model"
        final_model_path.mkdir(parents=True, exist_ok=True)

        if self.config.model_type in ["lora", "qlora"]:
            logger.info("Saving LoRA adapter (without merging into quantized base)...")
            # PeftModel.save_pretrained saves only the adapter weights
            self.model.save_pretrained(str(final_model_path))
            logger.info("LoRA adapter saved")

            # Write metadata so eval knows which base model to load
            base_model_config = {
                "base_model_name": self.config.model_name,
                "model_type": self.config.model_type,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "quantization": "4bit" if self.config.load_in_4bit else ("8bit" if self.config.load_in_8bit else "none"),
            }
            config_path = final_model_path / "base_model_config.json"
            with open(config_path, "w") as f:
                json.dump(base_model_config, f, indent=2)
            logger.info(f"Base model config written to {config_path}")
        else:
            self.model.save_pretrained(str(final_model_path))

        self.processor.save_pretrained(str(final_model_path))
        
        # Clean up GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("GPU memory cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up GPU memory: {e}")
        
        # Finish wandb run
        if use_wandb:
            try:
                import wandb
                wandb.finish()
                logger.info("Wandb run finished")
            except Exception as e:
                logger.warning(f"Failed to finish wandb run: {e}")
        
        logger.info(f"Training completed. Model saved to: {final_model_path}")
        
        return final_model_path
    
    def save_checkpoint(self, output_path: str):
        """Save a training checkpoint."""
        if self.model is not None:
            self.model.save_pretrained(output_path)
        if self.processor is not None:
            self.processor.save_pretrained(output_path)
        logger.info(f"Checkpoint saved to: {output_path}")


class BenchmarkEvaluator:
    """
    Evaluator for comparing Zero-Shot vs Fine-tuned MedGemma performance.
    
    Supports multiple benchmark datasets:
    - NCT-CRC-HE-100K (9-class tissue classification)
    - PCam (Binary metastasis detection)
    - BACH (4-class breast cancer grading)
    - PANDA (Prostate cancer Gleason grading)
    """
    
    SUPPORTED_DATASETS = {
        "crc": {
            "name": "NCT-CRC-HE-100K",
            "classes": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
            "task": "multiclass_classification",
            "metrics": ["accuracy", "macro_f1", "confusion_matrix"]
        },
        "pcam": {
            "name": "PatchCamelyon",
            "classes": ["Normal", "Tumor"],
            "task": "binary_classification",
            "metrics": ["accuracy", "auc", "f1", "precision", "recall"]
        },
        "bach": {
            "name": "BACH",
            "classes": ["Normal", "Benign", "InSitu", "Invasive"],
            "task": "multiclass_classification",
            "metrics": ["accuracy", "macro_f1", "cohen_kappa"]
        },
        "panda": {
            "name": "PANDA",
            "classes": ["Gleason_6", "Gleason_7", "Gleason_8", "Gleason_9"],
            "task": "ordinal_regression",
            "metrics": ["quadratic_weighted_kappa", "accuracy", "mae"]
        },
        "tcga": {
            "name": "TCGA-BRCA",
            "classes": ["Normal", "DCIS", "IDC", "ILC"],
            "task": "multiclass_classification",
            "metrics": ["accuracy", "macro_f1", "confusion_matrix"]
        }
    }
    
    def __init__(
        self,
        baseline_model_path: Optional[str] = None,
        finetuned_model_path: Optional[str] = None,
        quantization: int = 4,
        device: str = "auto"
    ):
        """
        Initialize the benchmark evaluator.
        
        Args:
            baseline_model_path: Path to baseline (zero-shot) model
            finetuned_model_path: Path to fine-tuned model
            quantization: Quantization bits (0, 4, 8)
            device: Device to run on
        """
        self.baseline_model_path = baseline_model_path or "google/medgemma-4b-it"
        self.finetuned_model_path = finetuned_model_path
        self.quantization = quantization
        self.device = device
        
        self.baseline_model = None
        self.finetuned_model = None
        self.processor = None
        
        self.results = {}
    
    def load_model(self, model_path: str, is_finetuned: bool = False):
        """
        Load a model for evaluation.
        
        Args:
            model_path: Path to model
            is_finetuned: Whether this is a fine-tuned model
            
        Returns:
            Loaded model wrapper
        """
        from .medgemma_integration import MedGemmaWrapper, MedGemmaConfig
        
        config = MedGemmaConfig(
            model_name=model_path,
            model_type="fine-tuned" if is_finetuned else "baseline",
            quantization=self.quantization,
            device=self.device
        )
        
        wrapper = MedGemmaWrapper(config)
        wrapper.load()
        
        return wrapper
    
    def evaluate_zero_shot(
        self,
        dataset_name: str,
        dataset_path: str,
        prompts: Optional[List[str]] = None
    ) -> BenchmarkResult:
        """
        Evaluate model in zero-shot mode.
        
        Args:
            dataset_name: Name of dataset (crc, pcam, bach, panda, tcga)
            dataset_path: Path to dataset
            prompts: Custom prompts for zero-shot classification
            
        Returns:
            BenchmarkResult with metrics
        """
        logger.info(f"Running zero-shot evaluation on {dataset_name}")
        
        # Load model
        if self.baseline_model is None:
            self.baseline_model = self.load_model(self.baseline_model_path, is_finetuned=False)
            self.processor = self.baseline_model.processor
        
        # Get dataset info
        dataset_info = self.SUPPORTED_DATASETS.get(dataset_name)
        if dataset_info is None:
            raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(self.SUPPORTED_DATASETS.keys())}")
        
        # Create prompts
        if prompts is None:
            prompts = self._create_zero_shot_prompts(dataset_name, dataset_info["classes"])
        
        # Run evaluation (placeholder - actual implementation would load dataset)
        metrics, predictions, ground_truth, inference_time, memory = self._run_evaluation(
            self.baseline_model,
            dataset_name,
            dataset_path,
            prompts
        )
        
        result = BenchmarkResult(
            model_type="baseline",
            dataset_name=dataset_info["name"],
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            inference_time=inference_time,
            memory_used_gb=memory,
            timestamp=datetime.now().isoformat()
        )
        
        self.results["zero_shot"] = result
        return result
    
    def evaluate_finetuned(
        self,
        dataset_name: str,
        dataset_path: str
    ) -> BenchmarkResult:
        """
        Evaluate fine-tuned model.
        
        Args:
            dataset_name: Name of dataset
            dataset_path: Path to dataset
            
        Returns:
            BenchmarkResult with metrics
        """
        if self.finetuned_model_path is None:
            raise ValueError("No fine-tuned model path provided")
        
        logger.info(f"Running fine-tuned evaluation on {dataset_name}")
        
        # Load model
        if self.finetuned_model is None:
            self.finetuned_model = self.load_model(
                self.finetuned_model_path,
                is_finetuned=True
            )
        
        dataset_info = self.SUPPORTED_DATASETS.get(dataset_name)
        
        # Run evaluation
        metrics, predictions, ground_truth, inference_time, memory = self._run_evaluation(
            self.finetuned_model,
            dataset_name,
            dataset_path,
            None  # No prompts needed for fine-tuned classifier
        )
        
        result = BenchmarkResult(
            model_type="fine-tuned",
            dataset_name=dataset_info["name"] if dataset_info else dataset_name,
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            inference_time=inference_time,
            memory_used_gb=memory,
            timestamp=datetime.now().isoformat()
        )
        
        self.results["fine_tuned"] = result
        return result
    
    def _create_zero_shot_prompts(
        self,
        dataset_name: str,
        classes: List[str]
    ) -> Dict[str, str]:
        """Create zero-shot prompts for different datasets."""
        prompts = {}
        
        if dataset_name == "crc":
            for cls in classes:
                prompts[cls] = f"A histopathology image of {cls} tissue."
        elif dataset_name == "pcam":
            prompts["Normal"] = "A histopathology image of normal lymph node tissue."
            prompts["Tumor"] = "A histopathology image containing metastatic breast cancer cells."
        elif dataset_name == "bach":
            prompts["Normal"] = "A histopathology image of normal breast tissue."
            prompts["Benign"] = "A histopathology image of benign breast lesion."
            prompts["InSitu"] = "A histopathology image of ductal carcinoma in situ."
            prompts["Invasive"] = "A histopathology image of invasive breast carcinoma."
        elif dataset_name == "panda":
            for cls in classes:
                grade = cls.split("_")[1]
                prompts[cls] = f"A histopathology image of prostate cancer with Gleason grade {grade}."
        elif dataset_name == "tcga":
            prompts["Normal"] = "A histopathology image of normal breast tissue."
            prompts["DCIS"] = "A histopathology image of ductal carcinoma in situ."
            prompts["IDC"] = "A histopathology image of invasive ductal carcinoma."
            prompts["ILC"] = "A histopathology image of invasive lobular carcinoma."
        else:
            # Generic prompts
            for cls in classes:
                prompts[cls] = f"A histopathology image showing {cls}."
        
        return prompts
    
    def _run_evaluation(
        self,
        model_wrapper,
        dataset_name: str,
        dataset_path: str,
        prompts: Optional[Dict[str, str]] = None
    ) -> Tuple[Dict[str, float], List[Dict], List, float, float]:
        """
        Run evaluation on a dataset.
        
        Returns:
            Tuple of (metrics, predictions, ground_truth, inference_time, memory_gb)
        """
        import time
        import torch
        
        # Placeholder for actual evaluation
        # In practice, this would:
        # 1. Load dataset from dataset_path
        # 2. For each sample, run inference
        # 3. Calculate metrics
        
        metrics = {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }
        
        predictions = []
        ground_truth = []
        inference_time = 0.0
        
        # Get memory usage
        memory_gb = 0.0
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return metrics, predictions, ground_truth, inference_time, memory_gb
    
    def compare_results(self) -> Dict[str, Any]:
        """
        Compare zero-shot and fine-tuned results.
        
        Returns:
            Dictionary with comparison metrics
        """
        if "zero_shot" not in self.results or "fine_tuned" not in self.results:
            raise ValueError("Both zero-shot and fine-tuned evaluations must be run first")
        
        zr = self.results["zero_shot"]
        fr = self.results["fine_tuned"]
        
        comparison = {
            "dataset": zr.dataset_name,
            "timestamp": datetime.now().isoformat(),
            "zero_shot": {
                "model_type": zr.model_type,
                "metrics": zr.metrics,
                "inference_time": zr.inference_time,
                "memory_gb": zr.memory_used_gb
            },
            "fine_tuned": {
                "model_type": fr.model_type,
                "metrics": fr.metrics,
                "inference_time": fr.inference_time,
                "memory_gb": fr.memory_used_gb
            },
            "improvement": {}
        }
        
        # Calculate improvement for each metric
        for metric in zr.metrics:
            if metric in fr.metrics:
                z_val = zr.metrics[metric]
                f_val = fr.metrics[metric]
                
                if z_val > 0:
                    pct_improvement = ((f_val - z_val) / z_val) * 100
                else:
                    pct_improvement = float('inf') if f_val > 0 else 0.0
                
                comparison["improvement"][metric] = {
                    "zero_shot": z_val,
                    "fine_tuned": f_val,
                    "absolute_change": f_val - z_val,
                    "percentage_improvement": pct_improvement
                }
        
        return comparison
    
    def run_full_benchmark(
        self,
        dataset_name: str,
        dataset_path: str,
        finetuned_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete benchmark comparison.
        
        Args:
            dataset_name: Name of dataset
            dataset_path: Path to dataset
            finetuned_path: Optional path to fine-tuned model
            
        Returns:
            Complete benchmark results
        """
        logger.info(f"Running full benchmark on {dataset_name}")
        
        # Zero-shot evaluation
        zr = self.evaluate_zero_shot(dataset_name, dataset_path)
        
        # Fine-tuned evaluation (if model available)
        if finetuned_path:
            self.finetuned_model_path = finetuned_path
            fr = self.evaluate_finetuned(dataset_name, dataset_path)
            comparison = self.compare_results()
        else:
            comparison = {"zero_shot": zr.metrics, "note": "No fine-tuned model provided"}
        
        return comparison
    
    def generate_report(self, output_path: str) -> str:
        """
        Generate a benchmark report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to generated report
        """
        from .utils.report import format_report_text
        
        if not self.results:
            raise ValueError("No benchmark results to report")
        
        report = []
        report.append("=" * 60)
        report.append("HISTOLOCAL BENCHMARK REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for eval_type, result in self.results.items():
            report.append(f"--- {eval_type.upper()} ---")
            report.append(f"Dataset: {result.dataset_name}")
            report.append(f"Model: {result.model_type}")
            report.append(f"Inference Time: {result.inference_time:.2f}s")
            report.append(f"Memory: {result.memory_used_gb:.2f} GB")
            report.append("Metrics:")
            for metric, value in result.metrics.items():
                report.append(f"  {metric}: {value:.4f}")
            report.append("")
        
        if "zero_shot" in self.results and "fine_tuned" in self.results:
            comparison = self.compare_results()
            report.append("--- COMPARISON ---")
            report.append(f"Dataset: {comparison['dataset']}")
            report.append("")
            report.append("Metric Improvements:")
            for metric, data in comparison["improvement"].items():
                report.append(f"  {metric}:")
                report.append(f"    Zero-Shot: {data['zero_shot']:.4f}")
                report.append(f"    Fine-Tuned: {data['fine_tuned']:.4f}")
                report.append(f"    Improvement: {data['percentage_improvement']:.2f}%")
        
        report_text = "\n".join(report)
        
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Benchmark report saved to: {output_path}")
        return output_path


def run_lora_finetuning(
    dataset_path: str,
    output_dir: str = "models/finetuned",
    model_name: str = "google/medgemma-4b-it",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 1,
    use_qlora: bool = True
) -> str:
    """
    Convenience function to run LoRA fine-tuning.
    
    Args:
        dataset_path: Path to training dataset
        output_dir: Output directory for checkpoints
        model_name: Base model to fine-tune
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        use_qlora: Use QLoRA (4-bit quantization)
        
    Returns:
        Path to saved model
    """
    # Configure training
    training_config = TrainingConfig(
        model_name=model_name,
        model_type="qlora" if use_qlora else "lora",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir
    )
    
    # Create trainer
    trainer = LoRATrainer(training_config=training_config, output_dir=output_dir)
    
    # Prepare model
    trainer.prepare_model_and_tokenizer()
    
    # Create dataset (placeholder - actual implementation depends on format)
    dataset_config = DatasetConfig(
        name="custom",
        path=dataset_path
    )
    datasets = trainer.create_datasets(dataset_config)
    
    # Train
    final_model_path = trainer.train(
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation")
    )
    
    return str(final_model_path)


def run_benchmark_comparison(
    dataset_name: str,
    dataset_path: str,
    baseline_model: str = "google/medgemma-4b-it",
    finetuned_model: Optional[str] = None,
    quantization: int = 4
) -> Dict[str, Any]:
    """
    Convenience function to run benchmark comparison.
    
    Args:
        dataset_name: Name of dataset (crc, pcam, bach, panda, tcga)
        dataset_path: Path to dataset
        baseline_model: Baseline model path
        finetuned_model: Optional fine-tuned model path
        quantization: Quantization bits
        
    Returns:
        Benchmark comparison results
    """
    evaluator = BenchmarkEvaluator(
        baseline_model_path=baseline_model,
        finetuned_model_path=finetuned_model,
        quantization=quantization
    )
    
    return evaluator.run_full_benchmark(dataset_name, dataset_path, finetuned_model)
