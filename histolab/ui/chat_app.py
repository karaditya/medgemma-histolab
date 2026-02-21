"""
Chat UI Gradio Application for HistoLab

Interactive chat interface for histopathology image analysis with:
- Image upload from any of the three datasets (BACH, CRC, PCAM)
- Chat with MedGemma model
- Class prediction with Chain of Thought
- True label display
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image
import time

from ..medgemma_integration import MedGemmaWrapper, MedGemmaConfig
from ..utils.dataset_labels import (
    get_true_label, 
    get_all_dataset_images, 
    get_dataset_stats,
    is_valid_image_path,
    DATASET_CONFIG
)

logger = logging.getLogger(__name__)


class HistoLabChatApp:
    """
    Chat-based Gradio application for histopathology analysis.
    
    Provides conversational interface for image analysis with
    Chain of Thought reasoning and true label verification.
    """
    
    # Chain of Thought prompt template - includes all possible dataset classes
    COT_PROMPT = """Please analyze this histopathology image carefully and provide your reasoning step by step.

First, observe the image carefully and describe what you see at the cellular and tissue level.

Then, based on your observations, explain what pathological features you can identify and what diagnostic significance they have.

Finally, provide your classification with confidence.

Format your response as follows:
CHAIN_OF_THOUGHT:
[Your detailed step-by-step reasoning here]

CLASSIFICATION:
[Your predicted class]

CONFIDENCE:
[Your confidence level as a percentage]

Please think through this carefully before answering."""

    # Dataset-specific class labels — full human-readable names used in prompts and display
    DATASET_CLASSES = {
        "bach": ["Normal", "Benign", "InSitu", "Invasive"],
        "crc":  ["Adipose", "Background", "Debris", "Lymphocyte",
                 "Mucus", "Muscle", "Normal", "Stroma", "Tumor"],
        "pcam": ["Normal", "Tumor"],
    }

    def _get_dataset_specific_prompt(self, dataset_name: str = None) -> str:
        """Get a prompt with dataset-specific class labels - matching training format."""
        if dataset_name and dataset_name.lower() in self.DATASET_CLASSES:
            classes = self.DATASET_CLASSES[dataset_name.lower()]
            classes_str = ", ".join(classes)
            
            return f"""Classify this histopathology image. The possible tissue types are: {classes_str}. What type of tissue is shown? Answer with only the tissue type name."""
        else:
            # Generic prompt for unknown dataset/custom images
            return f"""Classify this histopathology image. What type of tissue is shown? Answer with only the tissue type name."""
    
    # Class prediction prompt - includes all possible dataset classes
    CLASS_PROMPT = """Analyze this histopathology image and provide classification.

Choose ONE class from these options:
- For BACH dataset: Normal, Benign, InSitu, Invasive
- For CRC (NCT-CRC-HE) dataset: ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
- For PCAM dataset: Normal, Tumor

Respond in this format:
CLASS: [predicted class - use exact spelling from options above]
CONFIDENCE: [percentage]
REASONING: [brief explanation]"""
    
    def __init__(self, wrapper: Optional[MedGemmaWrapper] = None, default_fine_tuned_path: str = None):
        """
        Initialize the chat application.

        Args:
            wrapper: Optional pre-configured MedGemmaWrapper instance
            default_fine_tuned_path: Default path to fine-tuned model
        """
        self.wrapper = wrapper or MedGemmaWrapper(MedGemmaConfig())
        self.current_image_path: Optional[str] = None
        self.current_image: Optional[Image.Image] = None
        self.current_dataset: Optional[str] = None  # Track current dataset
        self.chat_history: List[Dict[str, str]] = []
        self.model_loaded = False
        self._loaded_model_key = None  # (model_type, model_path) of currently loaded model

        # Default configuration
        self.default_fine_tuned_path = default_fine_tuned_path or "models/exp1b_data_scale_5k/final_model"
        self.default_temperature = 0.1
        self.default_max_tokens = 512
        self.default_top_p = 0.9
        self.default_top_k = 50
        
    def load_model_if_needed(self, model_type: str = "fine-tuned", fine_tuned_path: str = None,
                         temperature: float = None, max_new_tokens: int = None) -> bool:
        """Load the model if not already loaded."""
        if self.model_loaded:
            if temperature is not None:
                self.wrapper.config.temperature = temperature
            if max_new_tokens is not None:
                self.wrapper.config.max_new_tokens = max_new_tokens
            return True

        try:
            # Unload previous model so all loaded flags reset
            self.wrapper.unload()

            self.wrapper.config.model_type = model_type

            if model_type == "fine-tuned":
                path = fine_tuned_path or self.default_fine_tuned_path
                self.wrapper.config.fine_tuned_path = path

            self.wrapper.config.temperature = temperature if temperature is not None else self.default_temperature
            self.wrapper.config.max_new_tokens = max_new_tokens if max_new_tokens is not None else self.default_max_tokens

            success = self.wrapper.load()
            self.model_loaded = success
            if success:
                ft_path = fine_tuned_path or self.default_fine_tuned_path
                self._loaded_model_key = (model_type, ft_path if model_type == "fine-tuned" else self.wrapper.config.model_name)
            return success
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an uploaded image and extract metadata.
        
        Args:
            image_path: Path to the uploaded image
            
        Returns:
            Dictionary with image info including true label
        """
        try:
            # Get true label from path
            dataset_name, true_label = get_true_label(image_path)
            
            # Load image for display
            img = Image.open(image_path)
            
            # Convert to RGB if needed (handles .tif and other formats)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize for display if too large (histopathology images can be 6000x6000+)
            max_display_size = 1024
            if img.width > max_display_size or img.height > max_display_size:
                ratio = min(max_display_size / img.width, max_display_size / img.height)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            width, height = img.size
            
            # Build result
            result = {
                "image": img,
                "image_path": image_path,
                "file_name": Path(image_path).name,
                "dataset": dataset_name or "Unknown",
                "true_label": true_label or "Unknown",
                "image_size": f"{width} x {height}",
                "is_valid_dataset_image": dataset_name is not None
            }
            
            self.current_image_path = image_path
            self.current_image = img
            self.current_dataset = dataset_name  # Track the dataset
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {
                "image": None,
                "image_path": image_path,
                "file_name": Path(image_path).name,
                "dataset": "Error",
                "true_label": "Error",
                "image_size": "Error",
                "is_valid_dataset_image": False,
                "error": str(e)
            }
    
    def query_model(self, message: str, image: Image.Image, model_type: str, use_cot: bool = True) -> Tuple[str, str]:
        """
        Query the model with an image and message.

        For fine-tuned models, uses a two-pass approach:
          Pass 1: Classification with training prompt (accurate)
          Pass 2: Reasoning prompt asking the model to explain the classification

        Args:
            message: User message/prompt
            image: PIL Image to analyze
            model_type: Model type (baseline or fine-tuned)
            use_cot: Whether to use Chain of Thought prompting (default True for reasoning)

        Returns:
            Tuple of (response, full_output)
        """
        if image is None:
            return "Please upload an image first.", ""

        # Ensure model is loaded
        if not self.load_model_if_needed(model_type):
            return "Failed to load model. Please check configuration.", ""

        try:
            if model_type == "fine-tuned":
                return self._two_pass_query(image, message)
            elif use_cot:
                # Baseline model: use CoT prompt for step-by-step reasoning
                dataset_key = self.current_dataset.lower() if self.current_dataset else ""
                if dataset_key and dataset_key in self.DATASET_CLASSES:
                    classes_list = self.DATASET_CLASSES[dataset_key]
                else:
                    classes_list = ["Normal", "Tumor", "ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

                cot_prompt = self.COT_PROMPT + "\n\nIMPORTANT - The possible tissue types are: " + ", ".join(classes_list)
                if message:
                    prompt = f"{cot_prompt}\n\nUser's question: {message}"
                else:
                    prompt = cot_prompt
            else:
                prompt = message if message else self.CLASS_PROMPT

            # Get model response using analyze_patch
            result = self.wrapper.analyze_patch(
                image=image,
                prompt=prompt
            )

            response = result.get("raw_response", "No response from model")
            return response, response

        except Exception as e:
            logger.error(f"Error querying model: {e}")
            return f"Error: {str(e)}", f"Error: {str(e)}"

    def _classify_image(self, image: 'Image.Image') -> tuple:
        """
        Pass 1: Run classification with the training-format prompt.
        Returns (predicted_class_name, confidence_str).
        confidence_str is empty string if the model doesn't provide one.
        """
        import torch

        classify_prompt = self._get_dataset_specific_prompt(self.current_dataset)
        classify_result = self.wrapper.analyze_patch(
            image=image,
            prompt=classify_prompt,
            max_new_tokens=32
        )
        raw_classification = classify_result.get("raw_response", "")

        # Parse "The answer is: X"
        pred_class = raw_classification.strip()
        if "The answer is:" in raw_classification:
            pred_class = raw_classification.split("The answer is:")[-1].strip()
        pred_class = pred_class.strip().rstrip('.')

        # Match to known class name
        dataset_key = (self.current_dataset or "").lower()
        if dataset_key in self.DATASET_CLASSES:
            classes = self.DATASET_CLASSES[dataset_key]
            pred_lower = pred_class.lower()
            for cn in classes:
                if cn.lower() == pred_lower or cn.lower() in pred_lower:
                    pred_class = cn
                    break

        # Free GPU memory before Pass 2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pred_class, ""

    # Prefix injected into the model's response to prevent the fine-tuned
    # LoRA pattern ("The answer is: X" → EOS) from dominating Pass 2.
    _REASONING_PREFIX = (
        "I'll describe the key visual features in this histopathology image.\n\n"
        "**Cellular morphology:** "
    )

    def _explain_classification(self, image: 'Image.Image', pred_class: str, user_message: str = "") -> tuple:
        """
        Pass 2: Generate reasoning explanation for the classification.
        Uses response_prefix to bypass the LoRA classification pattern.
        Returns (reasoning_text, confidence_str).
        """
        reasoning_prompt = self._build_reasoning_prompt(pred_class, user_message)
        reasoning_result = self.wrapper.analyze_patch(
            image=image,
            prompt=reasoning_prompt,
            response_prefix=self._REASONING_PREFIX
        )
        raw = reasoning_result.get("raw_response", "").strip()
        full_reasoning = self._REASONING_PREFIX + raw

        # Extract confidence percentage if the model included one (e.g. "Confidence: 87%")
        confidence_str = ""
        import re
        match = re.search(r'[Cc]onfidence[:\s]+(\d{1,3})\s*%', full_reasoning)
        if match:
            confidence_str = f"{match.group(1)}%"

        return full_reasoning, confidence_str

    def _format_classification_header(self, pred_class: str, confidence: str = "") -> str:
        """Format the classification header with dataset info and optional confidence."""
        from ..utils.dataset_labels import expand_label
        dataset = self.current_dataset or "unknown"
        dataset_key = (self.current_dataset or "").lower()
        display_class = expand_label(pred_class)
        classes_str = ""
        if dataset_key in self.DATASET_CLASSES:
            classes_str = ", ".join(self.DATASET_CLASSES[dataset_key])

        conf_part = f" — Confidence: **{confidence}**" if confidence else ""
        header = f"**Classification: {display_class}**{conf_part}\nDataset: {dataset}"
        if classes_str:
            header += f"\nPossible classes: {classes_str}"
        return header

    def _two_pass_query(self, image: 'Image.Image', user_message: str = "") -> Tuple[str, str]:
        """Non-streaming two-pass query (used by baseline path)."""
        pred_class, _ = self._classify_image(image)
        reasoning, confidence = self._explain_classification(image, pred_class, user_message)

        header = self._format_classification_header(pred_class, confidence)
        response = f"{header}\n\n---\n\n**Reasoning:**\n{reasoning}"
        return response, response

    def _build_reasoning_prompt(self, predicted_class: str, user_message: str = "") -> str:
        """Build the Pass 2 reasoning prompt, seeded with the classification result."""
        dataset_key = (self.current_dataset or "").lower()
        if dataset_key in self.DATASET_CLASSES:
            classes = self.DATASET_CLASSES[dataset_key]
            classes_str = ", ".join(classes)
        else:
            classes_str = "various tissue types"

        prompt = (
            f"This histopathology image has been classified as {predicted_class}. "
            f"The possible tissue classes are: {classes_str}.\n\n"
            f"Explain step by step what visual features in this image support "
            f"the classification of {predicted_class}. Describe the cellular "
            f"morphology, tissue architecture, and any distinctive patterns you observe.\n\n"
            f"End your response with a line in this exact format:\n"
            f"Confidence: X% (where X is your confidence that the classification is correct)"
        )

        if user_message:
            prompt += f"\n\nAlso address the user's question: {user_message}"

        return prompt
    
    def predict_class(self, image: Image.Image, model_type: str) -> Dict[str, str]:
        """
        Get class prediction from the model.
        
        Args:
            image: PIL Image to analyze
            model_type: Model type (baseline or fine-tuned)
            
        Returns:
            Dictionary with prediction results
        """
        if image is None:
            return {
                "prediction": "No image",
                "confidence": "0%",
                "reasoning": "Please upload an image first",
                "full_response": "No image provided"
            }
        
        # Ensure model is loaded
        if not self.load_model_if_needed(model_type):
            return {
                "prediction": "Error",
                "confidence": "0%",
                "reasoning": "Failed to load model",
                "full_response": "Model load failed"
            }
        
        try:
            # Use class prediction prompt
            result = self.wrapper.analyze_patch(
                image=image,
                prompt=self.CLASS_PROMPT
            )
            
            response = result.get("raw_response", "")
            
            # Parse response
            prediction = "Unknown"
            confidence = "Unknown"
            reasoning = "Could not parse response"
            
            lines = response.split('\n')
            for line in lines:
                if line.startswith("CLASS:"):
                    prediction = line.replace("CLASS:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence = line.replace("CONFIDENCE:", "").strip()
                elif line.startswith("REASONING:"):
                    reasoning = line.replace("REASONING:", "").strip()
            
            # Also try to get from result dict
            if prediction == "Unknown":
                prediction = result.get("tissue_type", "Unknown")
            if confidence == "Unknown":
                confidence = f"{result.get('confidence', 0) * 100:.0f}%"
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "reasoning": reasoning,
                "full_response": response
            }
            
        except Exception as e:
            logger.error(f"Error predicting class: {e}")
            return {
                "prediction": "Error",
                "confidence": "0%",
                "reasoning": str(e),
                "full_response": f"Error: {str(e)}"
            }
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio chat interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        # Store CSS for launch method (Gradio 6.0 compatibility)
        self._css = """
            #chat-container {height: 500px; overflow-y: auto;}
            #image-info {background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 15px;}
            #cot-output {background-color: #fffacd; padding: 15px; border-radius: 10px; margin-top: 10px; border-left: 4px solid #ffd700;}
            #prediction-box {background-color: #f0fff0; padding: 15px; border-radius: 10px; margin-top: 10px; border-left: 4px solid #32cd32;}
            #true-label {background-color: #ffe4e1; padding: 15px; border-radius: 10px; margin-top: 10px; border-left: 4px solid #ff6b6b;}
            .main-header {
                text-align: center;
                margin-bottom: 20px;
                padding: 24px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                color: white;
                border-radius: 16px;
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
                position: relative;
                overflow: hidden;
            }
            
            .main-header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
                animation: shimmer 3s infinite;
            }
            
            @keyframes shimmer {
                0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
                100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
            }
            #dataset-stats {font-size: 12px; color: #666;}
            
            /* Chat interface styling */
            .gradio-chatbot {
                border: 2px solid #e0e7ff;
                border-radius: 16px;
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
                box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.15), 0 4px 8px -2px rgba(0, 0, 0, 0.1);
                position: relative;
                overflow: hidden;
            }
            
            .gradio-chatbot::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%);
                animation: gradientShift 3s ease infinite;
                z-index: 1;
            }
            
            @keyframes gradientShift {
                0%, 100% { opacity: 0.8; }
                50% { opacity: 1; }
            }
            
            /* Chat container subtle pattern overlay */
            .gradio-chatbot::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-image: 
                    radial-gradient(circle at 25% 25%, rgba(14, 165, 233, 0.05) 0%, transparent 50%),
                    radial-gradient(circle at 75% 75%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
                z-index: 0;
                pointer-events: none;
            }
            
            .gradio-chatbot .user-message {
                background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
                color: white;
                border-radius: 20px;
                padding: 14px 20px;
                margin: 10px 0;
                max-width: 85%;
                align-self: flex-end;
                box-shadow: 0 4px 12px rgba(14, 165, 233, 0.4);
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .gradio-chatbot .user-message:hover {
                background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #a78bfa 100%);
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(14, 165, 233, 0.5);
            }
            
            .gradio-chatbot .bot-message {
                background: linear-gradient(135deg, #334155 0%, #475569 100%);
                color: #f1f5f9;
                border-radius: 20px;
                padding: 14px 20px;
                margin: 10px 0;
                max-width: 85%;
                align-self: flex-start;
                border: 1px solid #475569;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
                position: relative;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .gradio-chatbot .bot-message:hover {
                background: linear-gradient(135deg, #475569 0%, #64748b 100%);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                transform: translateY(-1px);
            }
            
            .gradio-chatbot .message-content {
                line-height: 1.6;
                position: relative;
                z-index: 1;
            }
            
            /* Animated typing indicator */
            .typing-indicator {
                display: inline-flex;
                align-items: center;
                padding: 12px 20px;
            }
            
            .typing-indicator span {
                height: 10px;
                width: 10px;
                margin: 0 3px;
                background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
                border-radius: 50%;
                display: inline-block;
                animation: typingAnimation 1.4s infinite ease-in-out;
                box-shadow: 0 0 8px rgba(14, 165, 233, 0.5);
            }
            
            .typing-indicator span:nth-child(1) {
                animation-delay: -0.32s;
            }
            
            .typing-indicator span:nth-child(2) {
                animation-delay: -0.16s;
            }
            
            @keyframes typingAnimation {
                0%, 80%, 100% {
                    transform: scale(0.8);
                    opacity: 0.5;
                }
                40% {
                    transform: scale(1);
                    opacity: 1;
                }
            }
            
            /* Gradient text for bot responses */
            .bot-message .gradient-text {
                background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                font-weight: 600;
            }
            
            /* Chat input styling */
            .gradio-textbox input {
                border: 2px solid #e0e7ff;
                border-radius: 25px;
                padding: 14px 20px;
                font-size: 14px;
                transition: all 0.3s ease;
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
            }
            
            .gradio-textbox input:focus {
                border-color: #6366f1;
                box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1), inset 0 2px 4px rgba(0, 0, 0, 0.06);
                outline: none;
                transform: translateY(-1px);
            }
            
            /* Send button styling */
            .gradio-button-primary {
                background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%) !important;
                border: none !important;
                border-radius: 25px !important;
                padding: 14px 24px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
                position: relative;
                overflow: hidden;
            }
            
            .gradio-button-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4) !important;
            }
            
            .gradio-button-primary:active {
                transform: translateY(0);
            }
            
            /* Clear button styling */
            .gradio-button-secondary {
                border: 2px solid #e0e7ff !important;
                border-radius: 25px !important;
                padding: 12px 20px !important;
                background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%) !important;
                color: #6366f1 !important;
                font-weight: 500 !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            
            .gradio-button-secondary:hover {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
                border-color: #6366f1 !important;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            }
            
            /* Model settings styling */
            .gradio-radio, .gradio-dropdown, .gradio-slider {
                border: 1px solid #e0e7ff;
                border-radius: 8px;
                padding: 8px;
                background: white;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            }
            
            /* True label display styling */
            #true-label-display {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 16px;
                border-radius: 12px;
                margin-bottom: 15px;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            }
            
            /* Image preview container */
            .gradio-image {
                border: 2px solid #e0e7ff;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }
        """
        
        with gr.Blocks(
            title="HistoLab Chat - MedGemma Assistant"
        ) as app:
            self._build(app)
        
        return app
    
    def _build(self, app: gr.Blocks):
        """Build the interface components."""
        
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>🔬 HistoLab Chat</h1>
                <p>Chat with MedGemma about histopathology images</p>
                <p style="font-size: 12px;">Upload an image or select from datasets (BACH, CRC, PCAM)</p>
            </div>
        """)
        
        with gr.Row():
            # Left column - Image upload and display
            with gr.Column(scale=1):
                gr.HTML("<h3>📤 Image</h3>")
                gr.HTML("<p style='font-size:12px; color:#666;'>Upload or select from datasets</p>")
                
                # True label display - appears above image when loaded
                true_label_display = gr.HTML(label="Image Info", elem_id="true-label-display")
                
                # Single Image component for both upload and display
                image_display = gr.Image(
                    label="Upload / Preview",
                    type="pil",
                    height=400,
                    interactive=True  # Allows both upload and display
                )
                
                # Dataset quick-load buttons
                gr.HTML("<p style='font-size:12px;'><b>Load from dataset:</b></p>")
                with gr.Row():
                    bach_btn = gr.Button("📁 BACH", variant="secondary", size="sm")
                    crc_btn = gr.Button("📁 CRC", variant="secondary", size="sm")
                    pcam_btn = gr.Button("📁 PCAM", variant="secondary", size="sm")

                # Local file path input — preserves full path for true label extraction
                with gr.Row():
                    file_path_input = gr.Textbox(
                        label="Or paste local file path",
                        placeholder="/path/to/data/datasets/crc/TUM/image.tif",
                        scale=4
                    )
                    load_path_btn = gr.Button("Load", variant="secondary", size="sm", scale=1)
            
            # Right column - Chat interface with compact settings above
            with gr.Column(scale=2):
                # Compact model settings bar
                with gr.Row():
                    model_type = gr.Radio(
                        ["baseline", "fine-tuned"],
                        value="fine-tuned",
                        label="Model",
                        scale=1
                    )
                    
                    # Get available fine-tuned models
                    models_dir = Path("models")
                    available_models = []
                    if models_dir.exists():
                        for d in models_dir.iterdir():
                            if d.is_dir() and (d / "final_model").exists():
                                available_models.append(str(d / "final_model"))
                    
                    # Use default if no models found
                    if not available_models:
                        available_models = [self.default_fine_tuned_path]
                    
                    fine_tuned_path = gr.Dropdown(
                        choices=available_models,
                        value=self.default_fine_tuned_path,
                        label="Model Path",
                        scale=2,
                        allow_custom_value=True,
                        visible=True
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=self.default_temperature,
                        step=0.1,
                        label="Temp",
                        scale=1
                    )
                    
                    max_tokens = gr.Slider(
                        minimum=64,
                        maximum=2048,
                        value=1024,  # Increased default for complete responses
                        step=64,
                        label="MaxTok",
                        scale=1
                    )
                    
                    model_status = gr.Textbox(label="Status", interactive=False, lines=1, scale=2)
                
                # Baseline model version selector (hidden by default, shown when baseline selected)
                baseline_version = gr.Dropdown(
                    choices=[
                        "google/medgemma-4b-it",
                        "google/medgemma-2b-it",
                        "google/medgemma-4b-multimodal"
                    ],
                    value="google/medgemma-4b-it",
                    label="Baseline Version",
                    visible=False  # Hidden by default (fine-tuned selected)
                )
                
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    type="messages",
                    elem_classes=["gradio-chatbot"]
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask about the image (e.g., 'What do you see?', 'Is this cancerous?', 'Explain your reasoning')",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Clear chat button
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat", variant="secondary")
                    gr.HTML("<div style='flex-grow:1;'></div>")
                    gr.HTML("<p style='font-size:11px; color:#888;'>💡 Tip: Leave message empty and click Send to auto-analyze the image</p>")
        
        # Event handlers
        self._setup_handlers(
            app, image_display, true_label_display,
            file_path_input, load_path_btn,
            model_type, fine_tuned_path, baseline_version, temperature, max_tokens,
            model_status,
            chatbot, msg_input, send_btn, clear_btn,
            bach_btn, crc_btn, pcam_btn
        )

    def _setup_handlers(
        self,
        app: gr.Blocks,
        image_display: gr.Image,
        true_label_display: gr.HTML,
        file_path_input: gr.Textbox,
        load_path_btn: gr.Button,
        model_type: gr.Radio,
        fine_tuned_path,
        baseline_version,
        temperature,
        max_tokens,
        model_status: gr.Textbox,
        chatbot: gr.Chatbot,
        msg_input: gr.Textbox,
        send_btn: gr.Button,
        clear_btn: gr.Button,
        bach_btn: gr.Button,
        crc_btn: gr.Button,
        pcam_btn: gr.Button
    ):
        """Setup all event handlers."""

        # --- Load image from local file path (preserves full path for label extraction) ---
        def load_from_path(file_path_str):
            """Load an image from a local file path — the full path is preserved
            so get_true_label can read the parent directory (e.g. .../TUM/img.tif)."""
            if not file_path_str or not file_path_str.strip():
                return None, ""
            file_path_str = file_path_str.strip()
            p = Path(file_path_str)
            if not p.exists():
                return None, f"<div style='background-color:red;color:white;padding:12px;border-radius:8px;'>File not found: {file_path_str}</div>"
            return _load_local_image(str(p))

        load_path_btn.click(
            load_from_path,
            inputs=[file_path_input],
            outputs=[image_display, true_label_display]
        )
        file_path_input.submit(
            load_from_path,
            inputs=[file_path_input],
            outputs=[image_display, true_label_display]
        )

        def _load_local_image(image_path: str):
            """Shared helper: load image from a real local path, extract true label."""
            try:
                info = self.process_image(image_path)
                display_image = info["image"]
                ds = info["dataset"] or "Unknown"
                label = info["true_label"] or "Unknown"

                if label != "Unknown" and ds != "Unknown":
                    label_html = f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 16px; border-radius: 12px; margin-bottom: 15px; font-size: 14px; font-weight: 500; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);'>
                        <strong>📋 True Label:</strong> {label}<br>
                        <strong>📂 Dataset:</strong> {ds}
                    </div>
                    """
                else:
                    label_html = f"""
                    <div style='background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); color: white; padding: 16px; border-radius: 12px; margin-bottom: 15px; font-size: 14px; font-weight: 500; box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);'>
                        <strong>📋 True Label:</strong> {label}<br>
                        <strong>📂 Dataset:</strong> {ds}<br>
                        <em style='font-size:12px;'>Could not detect label from path. Expected structure: .../dataset/CLASS/image.ext</em>
                    </div>
                    """
                return display_image, label_html
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                return None, f"<div style='background-color:red;color:white;padding:12px;'>Error: {e}</div>"

        # --- Dataset buttons (BACH, CRC, PCAM) ---
        def get_bach_image():
            from ..utils.dataset_labels import get_random_image_from_dataset
            image_path = get_random_image_from_dataset("bach")
            if image_path is None:
                return None, "<p>No images found in BACH dataset</p>"
            return _load_local_image(image_path)

        def get_crc_image():
            from ..utils.dataset_labels import get_random_image_from_dataset
            image_path = get_random_image_from_dataset("crc")
            if image_path is None:
                return None, "<p>No images found in CRC dataset</p>"
            return _load_local_image(image_path)

        def get_pcam_image():
            from ..utils.dataset_labels import get_random_image_from_dataset
            image_path = get_random_image_from_dataset("pcam")
            if image_path is None:
                return None, "<p>No images found in PCAM dataset</p>"
            return _load_local_image(image_path)

        bach_btn.click(
            get_bach_image,
            outputs=[image_display, true_label_display]
        )

        crc_btn.click(
            get_crc_image,
            outputs=[image_display, true_label_display]
        )

        pcam_btn.click(
            get_pcam_image,
            outputs=[image_display, true_label_display]
        )
            
        # Handle model loading (simplified - no top_p/top_k)
        def load_model(m_type, ft_path, b_version, temp, max_tok):
            # Store generation parameters
            self.default_temperature = temp
            self.default_max_tokens = max_tok

            # Configure model based on type
            if m_type == "baseline":
                self.wrapper.config.model_name = b_version
                self.wrapper.config.model_name = b_version
            else:
                self.default_fine_tuned_path = ft_path

            # Force reload with new settings
            self.model_loaded = False

            success = self.load_model_if_needed(
                model_type=m_type,
                fine_tuned_path=ft_path,
                temperature=temp,
                max_new_tokens=max_tok
            )
            
            if success:
                return f"✅ Model loaded successfully!\n\nModel: {m_type}\nPath: {ft_path if m_type == 'fine-tuned' else b_version}\nTemperature: {temp}\nMax Tokens: {max_tok}"
            else:
                return "❌ Failed to load model"
        
        # Show/hide baseline version dropdown based on model type selection
        def update_baseline_visibility(m_type):
            # Show baseline version dropdown only when baseline is selected
            return {
                baseline_version: gr.update(visible=(m_type == "baseline")),
                fine_tuned_path: gr.update(visible=(m_type == "fine-tuned"))
            }
        
        model_type.change(
            update_baseline_visibility,
            inputs=[model_type],
            outputs=[baseline_version, fine_tuned_path]
        )
        
        # Streaming handler: yields intermediate results so the UI updates
        # after classification (fast) and again after reasoning (slow).
        def handle_send(message, history, image, m_type, ft_path, b_version, temp, max_tok):
            user_msg = message.strip() if message else ""

            if image is None:
                display_msg = user_msg or "(no message)"
                yield ("", history + [
                    {"role": "user", "content": display_msg},
                    {"role": "assistant", "content": "Please upload an image first."}
                ], "")
                return

            # Auto-load model
            self.default_fine_tuned_path = ft_path
            if m_type == "baseline":
                self.wrapper.config.model_name = b_version
            new_key = (m_type, ft_path if m_type == "fine-tuned" else b_version)
            if self._loaded_model_key != new_key:
                self.model_loaded = False
            if not self.load_model_if_needed(m_type, ft_path, temp, max_tok):
                yield ("", history + [
                    {"role": "user", "content": user_msg or "Analyze"},
                    {"role": "assistant", "content": "Failed to load model."}
                ], "❌ Failed to load model")
                return

            model_info = f"✅ {m_type}: {ft_path if m_type == 'fine-tuned' else b_version}"

            if m_type == "fine-tuned":
                display_msg = user_msg or "Analyze this image"

                # --- Yield 1: classification (no confidence yet) ---
                pred_class, _ = self._classify_image(image)
                header_interim = self._format_classification_header(pred_class)
                yield ("", history + [
                    {"role": "user", "content": display_msg},
                    {"role": "assistant", "content": f"{header_interim}\n\n<div class='typing-indicator'><span></span><span></span><span></span></div>"}
                ], model_info)

                # --- Yield 2: reasoning + confidence ---
                reasoning, confidence = self._explain_classification(image, pred_class, user_msg)
                header_final = self._format_classification_header(pred_class, confidence)
                final = f"{header_final}\n\n---\n\n**Reasoning:**\n{reasoning}"
                yield ("", history + [
                    {"role": "user", "content": display_msg},
                    {"role": "assistant", "content": final}
                ], model_info)
            else:
                # Baseline: single pass (no streaming split)
                display_msg = user_msg or "Please analyze this image with step-by-step reasoning"
                response, _ = self.query_model(display_msg, image, m_type, use_cot=True)
                yield ("", history + [
                    {"role": "user", "content": display_msg},
                    {"role": "assistant", "content": response}
                ], model_info)

        send_btn.click(
            handle_send,
            inputs=[msg_input, chatbot, image_display, model_type, fine_tuned_path, baseline_version, temperature, max_tokens],
            outputs=[msg_input, chatbot, model_status]
        )

        msg_input.submit(
            handle_send,
            inputs=[msg_input, chatbot, image_display, model_type, fine_tuned_path, baseline_version, temperature, max_tokens],
            outputs=[msg_input, chatbot, model_status]
        )
        
        # Clear chat
        def clear_chat():
            self.chat_history = []
            return []
        
        clear_btn.click(clear_chat, outputs=[chatbot])


def create_chat_app(wrapper: Optional[MedGemmaWrapper] = None) -> gr.Blocks:
    """
    Create and configure the chat application.

    Args:
        wrapper: Optional pre-configured MedGemmaWrapper

    Returns:
        Gradio Blocks interface
    """
    app_instance = HistoLabChatApp(wrapper)
    return app_instance.build_interface()


def launch_chat_app(
    wrapper: Optional[MedGemmaWrapper] = None,
    share: bool = False,
    server_port: int = 7860
):
    """
    Launch the chat application.

    Args:
        wrapper: Optional pre-configured MedGemmaWrapper
        share: Whether to create a public link
        server_port: Port to run the server on
    """
    app = create_chat_app(wrapper)
    app.launch(share=share, server_port=server_port, css=app._css)


if __name__ == "__main__":
    launch_chat_app()
