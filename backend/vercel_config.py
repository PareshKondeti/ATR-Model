"""
Vercel-specific configuration and optimizations
"""

import os
import tempfile
from pathlib import Path

def setup_vercel_environment():
    """Configure environment for Vercel deployment"""
    
    # Set up temporary directories for model caching
    temp_dir = Path(tempfile.gettempdir())
    
    # Model cache directories
    transformers_cache = temp_dir / "transformers_cache"
    torch_cache = temp_dir / "torch_cache"
    huggingface_cache = temp_dir / "huggingface_cache"
    
    # Create directories
    transformers_cache.mkdir(exist_ok=True)
    torch_cache.mkdir(exist_ok=True)
    huggingface_cache.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["HF_HOME"] = str(huggingface_cache)
    
    # Vercel-specific optimizations
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    os.environ["OMP_NUM_THREADS"] = "1"  # Limit CPU threads for serverless
    
    print(f"✅ Vercel environment configured")
    print(f"   Transformers cache: {transformers_cache}")
    print(f"   Torch cache: {torch_cache}")
    print(f"   HuggingFace cache: {huggingface_cache}")

def get_vercel_optimized_models():
    """Return model configurations optimized for Vercel"""
    return {
        "whisper_model": "base",  # Smaller model for faster loading
        "qa_model": "distilbert-base-cased-distilled-squad",  # Efficient QA model
        "tts_model": None  # Disable TTS for Vercel (file system limitations)
    }
