# RunPod Serverless Endpoint for DeepSeek-OCR-2
# Extracts INDICATIONS AND USAGE from FDA pharmaceutical labels

# Base image with CUDA and Python
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_USE_V1=0 \
    MODEL_PATH=deepseek-ai/DeepSeek-OCR-2 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install flash-attention (requires GPU for compilation - will be built on first run if needed)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || echo "Flash attention will be installed at runtime"

# Copy application code
COPY config.py .
COPY extraction.py .
COPY handler.py .
COPY process/ ./process/
COPY deepencoderv2/ ./deepencoderv2/
COPY deepseek_ocr2.py .

# Pre-download the model (optional - comment out if you want to use RunPod volumes)
# This bakes the model into the image for faster cold starts
# RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True)"

# Create cache directories
RUN mkdir -p /app/hf_cache

# Set the entrypoint
CMD ["python", "-u", "handler.py"]
