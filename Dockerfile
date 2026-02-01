# RunPod Serverless Endpoint for DeepSeek-OCR-2
# Base image: CUDA 11.8 Devel (includes nvcc for compiling flash-attn)
# Selected for clean environment and explicit Python 3.12 support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VLLM_USE_V1=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MODEL_PATH=deepseek-ai/DeepSeek-OCR-2 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# Install system dependencies
# Includes dependencies for Python/pip and potential build tools
# Install system dependencies
# Includes dependencies for Python/pip and potential build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# Add Python PPA
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.12
RUN apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Upgrade pip and build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CUDA 11.8 compatible)
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install vLLM from wheel (v0.8.5 for CUDA 11.8)
RUN wget https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl && \
    pip install --no-cache-dir vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl && \
    rm vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl

# CRITICAL: Force install exact transformers/tokenizers versions to avoid conflicts
RUN pip install --no-cache-dir --force-reinstall \
    transformers==4.46.3 \
    tokenizers==0.20.3

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Flash Attention (compile from source - requires nvcc)
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

# Create cache directories
RUN mkdir -p /app/hf_cache

# Copy application code
COPY config.py .
COPY extraction.py .
COPY handler.py .
COPY test_model.py .
COPY start.sh .
COPY process/ ./process/
COPY deepencoderv2/ ./deepencoderv2/
COPY deepseek_ocr2.py .

RUN chmod +x /app/start.sh

# Entrypoint
ENTRYPOINT ["/bin/bash", "/app/start.sh"]
