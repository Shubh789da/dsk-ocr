# RunPod Direct Testing Guide

Test your DeepSeek-OCR-2 code directly on a RunPod GPU pod before building Docker.

## Why Test Directly First?

1. **Faster iteration** - No waiting for Docker builds
2. **Easier debugging** - See errors immediately, fix in real-time
3. **Cost effective** - Fix issues before committing to production image
4. **Interactive** - Can test with different inputs on the fly

---

## Step 1: Deploy a GPU Pod on RunPod

### 1.1 Go to RunPod Console
- Visit [https://www.runpod.io/console/pods](https://www.runpod.io/console/pods)
- Click **"+ Deploy"** or **"New Pod"**

### 1.2 Select GPU
Choose one of these (H100 recommended for production):
- **RTX 4090** (24GB) - ~$0.44/hr - Minimum viable, limited concurrency
- **A100 40GB** - ~$1.00/hr - Good balance
- **H100** (80GB) - Best for production with higher concurrency

### 1.3 Select Template
Choose: **vllm/vllm-openai:v0.8.5** as custom image (RECOMMENDED)
- This has the exact vLLM version needed
- All CUDA dependencies pre-configured

Or use **RunPod Pytorch 2.4.0** and install vLLM manually.

### 1.4 Configure Pod
- **Container Disk**: 50GB (for model downloads)
- **Volume Disk**: 100GB (optional, for persistent storage)
- Enable **SSH** access
- Click **Deploy**

---

## Step 2: Connect to Your Pod

### 2.1 Get SSH Command
Once pod is running, click on it and find the SSH command:
```
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_rsa
```

Or use **Web Terminal** (click "Connect" → "Start Web Terminal")

### 2.2 Verify GPU
```bash
nvidia-smi
```
You should see your GPU (RTX 4090, H100, etc.)

---

## Step 3: Set Up Environment

### 3.1 Check vLLM Version
```bash
# If using vllm-openai:v0.8.5 image, verify:
python3.12 -c "import vllm; print(vllm.__version__)"
# Should print: 0.8.5
```

### 3.2 Install vLLM v0.8.5 (if not using vllm image)
```bash
pip3.12 install vllm==0.8.5

# Verify version
python3.12 -c "import vllm; print(vllm.__version__)"
```

### 3.3 Install Other Dependencies
```bash
pip3.12 install runpod transformers tokenizers Pillow PyMuPDF torchvision einops addict easydict numpy requests tqdm
```

### 3.4 Clone/Upload Your Code

**Option A: Clone from GitHub**
```bash
cd /workspace
git clone https://github.com/Shubh789da/dsk-ocr.git
cd dsk-ocr/dsk-ep
```

**Option B: Upload via SCP**
```bash
# From your local machine
scp -P <port> -r d:/CT_FDA/drug_history/deepseek-OCR/dsk-ep root@<pod-ip>:/workspace/
```

**Option C: Copy-paste files manually**
Use the web terminal to create files with `nano` or `vim`.

---

## Step 4: Test the Code

### 4.1 Navigate to Project
```bash
cd /workspace/dsk-ep
ls -la
```

### 4.2 Run the Test Script (RECOMMENDED)
```bash
# This auto-detects GPU and configures memory settings
python3.12 test_model.py
```

Expected output for H100:
```
============================================================
DeepSeek-OCR-2 Test Script
============================================================

PyTorch: 2.6.0+cu124
CUDA available: True
GPU: NVIDIA H100 80GB HBM3
VRAM: 85.0 GB

[AUTO-CONFIG] High-VRAM GPU detected, using moderate settings

VLLM_USE_V1: 0
VLLM_ATTENTION_BACKEND: XFORMERS
PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True

Importing vLLM...
vLLM version: 0.8.5
...
Configuration:
  MODEL_PATH: deepseek-ai/DeepSeek-OCR-2
  GPU_MEMORY_UTILIZATION: 0.5
  MAX_MODEL_LEN: 4096
  MAX_NUM_SEQS: 4

============================================================
Loading model (this may take a few minutes)...
============================================================
...
[SUCCESS] Model loaded successfully!
```

### 4.3 Test with a Sample PDF
```bash
# Upload a test PDF to the pod first:
# scp -P <port> /path/to/test.pdf root@<pod-ip>:/workspace/dsk-ep/

# Run test with PDF
python3.12 test_model.py /workspace/dsk-ep/test.pdf
```

---

## Step 5: Memory Configuration Reference

### Auto-Configured Settings by GPU

| GPU | VRAM | gpu_memory_utilization | max_model_len | max_num_seqs |
|-----|------|------------------------|---------------|--------------|
| RTX 4090 | 24GB | 0.40 | 2048 | 1 |
| A100 40GB | 40GB | 0.45 | 4096 | 2 |
| H100 80GB | 80GB | 0.50 | 4096 | 4 |

### Manual Override (if needed)
```bash
# For very conservative settings (if still OOM):
export GPU_MEMORY_UTILIZATION=0.35
export MAX_MODEL_LEN=1024
export MAX_NUM_SEQS=1

python3.12 test_model.py
```

### Why These Settings Matter

1. **gpu_memory_utilization** (0.40-0.50):
   - vLLM pre-allocates this % of VRAM for KV cache
   - DeepSeek-OCR-2 needs ~6-7GB for model + 2-4GB for vision encoder activations
   - Setting too high leaves no room for vision encoder → OOM during profile_run

2. **max_model_len** (2048-4096):
   - Maximum sequence length
   - Directly affects KV cache size
   - Lower = less memory, but limits output length

3. **max_num_seqs** (1-8):
   - Maximum concurrent sequences
   - Each sequence needs KV cache space
   - Lower = less memory, but limits throughput

4. **limit_mm_per_prompt={"image": 1}**:
   - Limits multimodal inputs during profiling
   - CRITICAL for preventing OOM during vLLM's profile run

---

## Step 6: Debug Common Issues

### Issue: CUDA out of memory during model loading
This is the most common issue! The script now auto-configures, but if still failing:

```bash
# Try ultra-conservative settings
export GPU_MEMORY_UTILIZATION=0.30
export MAX_MODEL_LEN=1024
export MAX_NUM_SEQS=1

# Check for other GPU processes
nvidia-smi

# Kill any zombie processes
pkill -f python

python3.12 test_model.py
```

### Issue: OOM during profile_run phase
The log shows:
```
INFO ... Model loading took 6.3336 GiB
[ERROR] CUDA out of memory. Tried to allocate 32.00 MiB...
```

This means KV cache pre-allocation + vision encoder = too much.
**Solution**: Lower `gpu_memory_utilization` to leave room for vision encoder.

### Issue: Module not found
```bash
# Check what's installed
pip3.12 list | grep -i <module_name>

# Install missing module
pip3.12 install <module_name>
```

### Issue: V1 engine being used (causes OOM)
Check the log for:
```
Initializing a V1 LLM engine  # BAD
```
Should show:
```
Initializing a V0 LLM engine  # GOOD
```

Fix: Ensure `VLLM_USE_V1=0` is set BEFORE any vLLM imports.

### Issue: Model download fails
```bash
# Set HuggingFace token for gated models
export HF_TOKEN=your_token_here

# Or login
huggingface-cli login
```

---

## Step 7: Once Working, Build Docker Image

### 7.1 On the Same Pod (Recommended)
```bash
# Install Docker (if not available)
curl -fsSL https://get.docker.com | sh

# Navigate to project
cd /workspace/dsk-ep

# Build image
docker build -t shubh0078/dsk-ocr-endpoint:v1.3 .

# Login to Docker Hub
docker login

# Push image
docker push shubh0078/dsk-ocr-endpoint:v1.3
```

### 7.2 Update Dockerfile with Working Versions
After testing, note the exact versions that worked:
```bash
pip3.12 freeze > working_versions.txt
cat working_versions.txt
```

---

## Step 8: Deploy to Serverless

Once Docker image is pushed:

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Docker Image**
4. Enter: `docker.io/shubh0078/dsk-ocr-endpoint:v1.3`
5. Configure:
   - GPU: H100 80GB (recommended) or RTX 4090
   - Min Workers: 0
   - Max Workers: 3
   - Idle Timeout: 60s
6. Deploy!

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Check Python packages
pip3.12 list

# Check vLLM version
python3.12 -c "import vllm; print(vllm.__version__)"

# Test imports
python3.12 -c "from deepseek_ocr2 import DeepseekOCR2ForCausalLM; print('OK')"

# Run test script (auto-configures memory)
python3.12 test_model.py

# Run with PDF
python3.12 test_model.py test.pdf

# Monitor GPU usage (live)
watch -n 1 nvidia-smi

# Check disk space
df -h

# Download test PDF from URL
wget -O test.pdf "https://example.com/label.pdf"

# Manual memory override
export GPU_MEMORY_UTILIZATION=0.35
export MAX_MODEL_LEN=1024
python3.12 test_model.py
```

---

## Memory Troubleshooting Flowchart

```
OOM Error?
    │
    ├─→ During "profile_run" or "KV cache"?
    │       │
    │       └─→ Lower GPU_MEMORY_UTILIZATION (try 0.35)
    │           Lower MAX_MODEL_LEN (try 1024)
    │           Lower MAX_NUM_SEQS (try 1)
    │
    ├─→ During model weight loading?
    │       │
    │       └─→ GPU too small (need 20GB+ minimum)
    │           Check nvidia-smi for other processes
    │
    └─→ During inference?
            │
            └─→ Image too large? Reduce DPI
                Batch too big? Process fewer pages
```

---

## Files to Upload/Create

Make sure these files exist in `/workspace/dsk-ep/`:
```
dsk-ep/
├── handler.py
├── test_model.py
├── extraction.py
├── config.py
├── deepseek_ocr2.py
├── process/
│   ├── __init__.py
│   ├── image_process.py
│   └── ngram_norepeat.py
└── deepencoderv2/
    ├── __init__.py
    ├── build_linear.py
    ├── qwen2_d2e.py
    └── sam_vary_sdpa.py
```
