"""
HistoLab Setup Configuration

A local, offline AI-powered histopathology assistant for cancer diagnosis.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="histolab",
    version="1.0.0",
    author="HistoLab Team",
    description="Histopathology Assistant for Cancer Diagnosis using MedGemma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HistoLab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Medical Science",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core ML/AI
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        
        # Image Processing
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scipy>=1.11.0",
        "scikit-image>=0.21.0",
        
        # WSI Support (Python bindings - system package required separately)
        "openslide-python>=1.3.0",
        
        # UI
        "gradio>=4.0.0",
        
        # Report Generation
        "reportlab>=4.0.0",
        
        # Configuration
        "PyYAML>=6.0",
        
        # Scientific Computing
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        
        # Dataset Handling
        "datasets>=2.14.0",
        "huggingface_hub>=0.19.0",
        
        # Evaluation
        "scikit-learn>=1.3.0",
        
        # Utilities
        "tqdm>=4.66.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "benchmark": [
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "histolab=histolab.data_loader:main",
        ],
    },
    include_package_data=True,
    package_data={
        "histolab": ["data/**/*"],
    },
    zip_safe=False,
)
