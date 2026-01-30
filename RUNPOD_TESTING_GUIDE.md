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
Choose one of these (RTX 4090 recommended for testing):
- **RTX 4090** (24GB) - ~$0.44/hr - Best value for testing
- **A100 40GB** - ~$1.00/hr - More VRAM if needed
- **L40S** (48GB) - ~$0.80/hr - Good alternative

### 1.3 Select Template
Choose: **RunPod Pytorch 2.4.0** or **RunPod Pytorch 2.1**
- This gives you CUDA, PyTorch pre-installed
- Or use **vllm/vllm-openai:v0.8.5** as custom image

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
You should see your GPU (RTX 4090, A100, etc.)

---

## Step 3: Set Up Environment

### 3.1 Install vLLM v0.8.5 (Required Version)
```bash
# Install vLLM 0.8.5 specifically
pip install vllm==0.8.5

# Verify version
python -c "import vllm; print(vllm.__version__)"
# Should print: 0.8.5
```

### 3.2 Install Other Dependencies
```bash
pip install runpod transformers tokenizers Pillow PyMuPDF torchvision einops addict easydict numpy requests tqdm
```

### 3.3 Clone/Upload Your Code

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

### 4.2 Test Imports First
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

import vllm
print(f'vLLM: {vllm.__version__}')

from vllm.multimodal.processing import BaseMultiModalProcessor, PromptUpdate
print('vLLM multimodal imports OK')
"
```

### 4.3 Test Model Loading
```bash
python -c "
from vllm import LLM
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM

# Register model
ModelRegistry.register_model('DeepseekOCR2ForCausalLM', DeepseekOCR2ForCausalLM)

print('Loading model...')
llm = LLM(
    model='deepseek-ai/DeepSeek-OCR-2',
    hf_overrides={'architectures': ['DeepseekOCR2ForCausalLM']},
    trust_remote_code=True,
    max_model_len=4096,
    gpu_memory_utilization=0.9
)
print('Model loaded successfully!')
"
```

### 4.4 Test with a Sample PDF
Upload a test PDF to the pod:
```bash
# From local machine
scp -P <port> /path/to/test.pdf root@<pod-ip>:/workspace/dsk-ep/

# On the pod, run test
cd /workspace/dsk-ep
python handler.py /workspace/dsk-ep/test.pdf
```

### 4.5 Test Full Handler (Interactive)
```bash
python -c "
import base64
from handler import process_pdf

# Read a test PDF
with open('/workspace/dsk-ep/test.pdf', 'rb') as f:
    pdf_bytes = f.read()

print(f'PDF size: {len(pdf_bytes)} bytes')
print('Processing...')

result = process_pdf(pdf_bytes)

print('=' * 60)
print('INDICATIONS AND USAGE:')
print('=' * 60)
print(result['indications_and_usage'][:2000])
print(f'\\nPages: {result[\"page_count\"]}')
"
```

---

## Step 5: Debug Common Issues

### Issue: Module not found
```bash
# Check what's installed
pip list | grep -i <module_name>

# Install missing module
pip install <module_name>
```

### Issue: CUDA out of memory
```bash
# Reduce GPU memory usage
export GPU_MEMORY_UTILIZATION=0.8

# Or reduce max_model_len
export MAX_MODEL_LEN=4096
```

### Issue: Model download fails
```bash
# Set HuggingFace token for gated models
export HF_TOKEN=your_token_here

# Or login
huggingface-cli login
```

### Issue: vLLM version mismatch
```bash
# Check version
pip show vllm

# Force reinstall correct version
pip uninstall vllm -y
pip install vllm==0.8.5
```

---

## Step 6: Once Working, Build Docker Image

### 6.1 On the Same Pod (Recommended)
```bash
# Install Docker (if not available)
curl -fsSL https://get.docker.com | sh

# Navigate to project
cd /workspace/dsk-ep

# Build image
docker build -t shubh0078/dsk-ocr-endpoint:v1.2 .

# Login to Docker Hub
docker login

# Push image
docker push shubh0078/dsk-ocr-endpoint:v1.2
```

### 6.2 Update Dockerfile with Working Versions
After testing, note the exact versions that worked:
```bash
pip freeze > working_versions.txt
cat working_versions.txt
```

Then update `requirements.txt` with pinned versions.

---

## Step 7: Deploy to Serverless

Once Docker image is pushed:

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Select **Docker Image**
4. Enter: `docker.io/shubh0078/dsk-ocr-endpoint:v1.2`
5. Configure:
   - GPU: RTX 4090
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
pip list

# Check vLLM version
python -c "import vllm; print(vllm.__version__)"

# Test imports
python -c "from deepseek_ocr2 import DeepseekOCR2ForCausalLM; print('OK')"

# Run handler with test PDF
python handler.py test.pdf

# Monitor GPU usage (live)
watch -n 1 nvidia-smi

# Check disk space
df -h

# Download test PDF from URL
wget -O test.pdf "https://example.com/label.pdf"
```

---

## Cost Estimate

| Activity | GPU | Hourly Cost | Typical Duration |
|----------|-----|-------------|------------------|
| Testing/debugging | RTX 4090 | $0.44/hr | 1-2 hours |
| Docker build | RTX 4090 | $0.44/hr | 10-30 min |
| **Total** | | | **~$1-2** |

**Tip**: Stop your pod when not using it to save costs!

---

## Files to Upload/Create

Make sure these files exist in `/workspace/dsk-ep/`:
```
dsk-ep/
├── handler.py
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
