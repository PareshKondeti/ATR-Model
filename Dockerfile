# Use Python 3.9 slim image to reduce size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models
RUN mkdir -p training_data/whisper_data training_data/tts_data

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
