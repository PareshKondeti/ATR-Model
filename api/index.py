"""
Vercel serverless function entry point
This file is required for Vercel to recognize the Python app
"""

import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import and run the FastAPI app
from backend.main import app

# Vercel expects the app to be available as 'app'
# This is already defined in backend/main.py
