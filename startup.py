#!/usr/bin/env python3
"""
Startup script to download models only when needed
This helps reduce the Docker image size
"""

import os
import sys
import subprocess
from pathlib import Path

def download_models():
    """Download models only if they don't exist"""
    print("Checking for required models...")
    
    # Create cache directories
    cache_dir = Path("/tmp/models")
    cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables for model caching
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    
    # Download Whisper model (smallest base model)
    whisper_model = "openai/whisper-base"
    print(f"Downloading {whisper_model}...")
    try:
        import whisper
        whisper.load_model("base")
        print("✅ Whisper model downloaded")
    except Exception as e:
        print(f"❌ Whisper download failed: {e}")
    
    # Download DistilBERT model
    qa_model = "distilbert-base-cased-distilled-squad"
    print(f"Downloading {qa_model}...")
    try:
        from transformers import pipeline
        pipeline("question-answering", model=qa_model)
        print("✅ QA model downloaded")
    except Exception as e:
        print(f"❌ QA model download failed: {e}")
    
    print("Model download complete!")

if __name__ == "__main__":
    download_models()
    # Start the main application
    subprocess.run(["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"])
