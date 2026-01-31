# RunPod Serverless Endpoint for DeepSeek-OCR-2
# Extracts INDICATIONS AND USAGE from FDA pharmaceutical labels

# Base image: vLLM v0.8.5 (required for DeepSeek-OCR-2 multimodal APIs)
# This version has PromptUpdate, BaseMultiModalProcessor, etc.
FROM vllm/vllm-openai:v0.8.5

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_USE_V1=0 \
    VLLM_ATTENTION_BACKEND=XFORMERS \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MODEL_PATH=deepseek-ai/DeepSeek-OCR-2 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# CRITICAL: Force install exact transformers/tokenizers versions FIRST
# The base vLLM image has newer versions that break DeepSeek-OCR-2 tokenizer
# Error without this: "TokenizersBackend has no attribute all_special_tokens_extended"
RUN pip install --no-cache-dir --force-reinstall \
    transformers==4.46.3 \
    tokenizers==0.20.3

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Verify correct versions installed
RUN python3 -c "import transformers; print(f'transformers: {transformers.__version__}')" && \
    python3 -c "import tokenizers; print(f'tokenizers: {tokenizers.__version__}')"

# Install flash-attention (requires GPU for compilation - will be built on first run if needed)
# Flash attention is already included in the base image

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

# Set the entrypoint (OVERRIDE base image entrypoint)
ENTRYPOINT ["python3", "-u", "handler.py"]
