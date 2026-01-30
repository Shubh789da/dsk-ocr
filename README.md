# DeepSeek-OCR-2 RunPod Serverless Endpoint

Extract **INDICATIONS AND USAGE** from FDA pharmaceutical label PDFs using DeepSeek-OCR-2 with vLLM on RunPod Serverless.

## Project Structure

```
dsk-ep/
├── handler.py           # RunPod serverless handler
├── extraction.py        # Indication/usage extraction logic
├── config.py            # Configuration settings
├── deepseek_ocr2.py     # vLLM model definition
├── Dockerfile           # Docker image build file
├── requirements.txt     # Python dependencies
├── process/             # Image processing modules
│   ├── __init__.py
│   ├── image_process.py
│   └── ngram_norepeat.py
└── deepencoderv2/       # Vision encoder modules
    ├── __init__.py
    ├── build_linear.py
    ├── qwen2_d2e.py
    └── sam_vary_sdpa.py
```

## Quick Start

### Option 1: Use Pre-built RunPod vLLM Worker (Simplest)

RunPod provides pre-built vLLM workers. However, DeepSeek-OCR-2 requires custom model code, so you'll need Option 2.

### Option 2: Build Custom Docker Image

#### Step 1: Build Docker Image on CPU Machine

You can build the Docker image on any machine (including CPU-only). The GPU is only needed at runtime.

```bash
# On your local machine or a CPU pod
cd dsk-ep

# Build the image (specify linux/amd64 platform for RunPod)
docker build --platform linux/amd64 -t your-dockerhub-username/dsk-ocr-endpoint:v1.0 .

# Push to Docker Hub
docker login
docker push your-dockerhub-username/dsk-ocr-endpoint:v1.0
```

#### Step 2: Build on RunPod CPU Pod (Alternative)

If you don't have Docker locally, use a RunPod CPU pod:

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Deploy a **CPU Pod** (cheapest option for building)
3. SSH into the pod
4. Clone your repo and build:

```bash
# On RunPod CPU pod
git clone <your-repo-url>
cd dsk-ep

# Install Docker if not available
apt-get update && apt-get install -y docker.io

# Build image
docker build -t your-dockerhub-username/dsk-ocr-endpoint:v1.0 .

# Push to Docker Hub
docker login
docker push your-dockerhub-username/dsk-ocr-endpoint:v1.0
```

#### Step 3: Create RunPod Serverless Endpoint

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Click **"New Endpoint"**
3. Under **Custom Source**, select **"Docker Image"**
4. Enter your Docker image URL: `docker.io/your-username/dsk-ocr-endpoint:v1.0`
5. Configure settings:
   - **GPU Type**: RTX 4090 (24GB) or A100 (40GB/80GB)
   - **Min Workers**: 0 (scale to zero when idle)
   - **Max Workers**: Based on your needs
   - **Idle Timeout**: 30-60 seconds
   - **Execution Timeout**: 600 seconds (10 min for large PDFs)

6. Add **Environment Variables** (optional):
   ```
   MODEL_PATH=deepseek-ai/DeepSeek-OCR-2
   GPU_MEMORY_UTILIZATION=0.9
   MAX_CONCURRENCY=50
   ```

7. Click **Deploy**

## API Usage

### Endpoint URL
```
https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync
```

### Request Format

**With Base64 PDF:**
```json
{
    "input": {
        "pdf_base64": "<base64-encoded-pdf>",
        "return_full_text": false
    }
}
```

**With PDF URL:**
```json
{
    "input": {
        "pdf_url": "https://example.com/label.pdf",
        "return_full_text": false
    }
}
```

### Response Format
```json
{
    "output": {
        "indications_and_usage": "KEYTRUDA is indicated for...",
        "page_count": 180
    }
}
```

### Python Client Example

```python
import requests
import base64

RUNPOD_API_KEY = "your-api-key"
ENDPOINT_ID = "your-endpoint-id"

# Read PDF file
with open("keytruda.pdf", "rb") as f:
    pdf_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "input": {
            "pdf_base64": pdf_base64,
            "return_full_text": False
        }
    },
    timeout=600
)

result = response.json()
print(result["output"]["indications_and_usage"])
```

### Async Request (for large PDFs)

```python
import requests
import time

# Submit job
response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
    headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    json={"input": {"pdf_url": "https://example.com/large-label.pdf"}}
)
job_id = response.json()["id"]

# Poll for results
while True:
    status = requests.get(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}",
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    ).json()

    if status["status"] == "COMPLETED":
        print(status["output"]["indications_and_usage"])
        break
    elif status["status"] == "FAILED":
        print("Error:", status.get("error"))
        break

    time.sleep(5)
```

## Configuration Options

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MODEL_PATH` | `deepseek-ai/DeepSeek-OCR-2` | HuggingFace model path |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction to use |
| `MAX_CONCURRENCY` | `50` | Max concurrent page processing |
| `MAX_MODEL_LEN` | `8192` | Maximum context length |
| `CROP_MODE` | `True` | Enable dynamic image cropping |
| `NUM_WORKERS` | `32` | Image preprocessing workers |

## Performance Expectations

Based on RTX 4090 (24GB):
- **180-page PDF**: ~2 minutes
- **Throughput**: ~1.5 pages/second
- **Cold start**: ~30-60 seconds (first request after idle)

## Baking Model into Docker Image

For faster cold starts, bake the model into the image:

```dockerfile
# Add to Dockerfile before CMD
RUN python -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-OCR-2', local_dir='/app/model')
"
ENV MODEL_PATH=/app/model
```

**Note**: This increases image size significantly (~15GB).

## Using RunPod Network Volumes

For persistent model storage without baking into image:

1. Create a Network Volume in RunPod console
2. Attach volume to endpoint
3. Set environment variable: `MODEL_PATH=/runpod-volume/models/DeepSeek-OCR-2`
4. Pre-download model to volume (one-time):

```python
from huggingface_hub import snapshot_download
snapshot_download("deepseek-ai/DeepSeek-OCR-2", local_dir="/runpod-volume/models/DeepSeek-OCR-2")
```

## Troubleshooting

### Out of Memory Errors
- Reduce `GPU_MEMORY_UTILIZATION` to 0.85
- Reduce `MAX_CONCURRENCY` to 25
- Use larger GPU (A100)

### Slow Cold Starts
- Bake model into Docker image
- Use Network Volume with cached model
- Increase Min Workers to 1

### Flash Attention Issues
If flash-attn fails to install, the model will fall back to standard attention (slower but functional).

## Cost Estimation

RunPod Serverless pricing (as of 2025):
- RTX 4090: ~$0.00044/second
- A100 40GB: ~$0.00097/second

For a 180-page PDF (~2 min processing):
- RTX 4090: ~$0.053
- A100: ~$0.116

## License

This project uses DeepSeek-OCR-2 which is subject to DeepSeek's license terms.

## Support

For issues with:
- This endpoint code: Open an issue in this repository
- RunPod platform: [RunPod Documentation](https://docs.runpod.io)
- DeepSeek-OCR-2 model: [DeepSeek GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
