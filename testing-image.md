# Testing DeepSeek-OCR-2 on RunPod (Code Verification)

> [!IMPORTANT]
> **Why this approach?**
> Standard RunPod instances do not support running "Docker-in-Docker" securely by default (hence the `dockerd` errors). Instead of testing the *container image*, we will test the **code and model** directly in the pod. This verifies that the GPU, dependencies, and inference logic work correctly.

## 1. Provision a Standard GPU Pod

1.  Go to **[RunPod Console > Pods](https://www.runpod.io/console/pods)**.
2.  Deploy a **RunPod PyTorch 2.X** pod with a GPU (e.g., RTX 3090/4090).
3.  Connect via **Web Terminal** or SSH.

## 2. Setup the Environment

Since we are testing the code directly, we need to clone your repo and install dependencies.

```bash
# 1. Clone your repository
git clone https://github.com/Shubh789da/dsk-ocr.git
cd dsk-ocr/dsk-ep

# 2. Install dependencies
# (The PyTorch pod has torch installed, but we need other requirements)
pip install -r requirements.txt
```

## 3. Prepare a Test PDF

Download a sample PDF to test the extraction.

```bash
wget https://github.com/Starttoaster/Fda_indications_extraction/raw/main/keytruda.pdf -O test.pdf
```

## 4. Run the Test

Execute `handler.py` directly. This will load the model (downloading it if needed) and process the PDF using the GPU.

```bash
python handler.py test.pdf
```

### Supported Output:
1.  **Model Loading**: You should see logs about "Loading DeepSeek-OCR-2...".
2.  **Inference**: It will process the pages (check GPU usage with `nvidia-smi` in another tab if you want).
3.  **Result**: It will print the extracted **INDICATIONS AND USAGE** text.

## 5. What verifies what?

| Component | Verified? | Notes |
| :--- | :--- | :--- |
| **GPU / CUDA** | ✅ Yes | If `handler.py` runs, the GPU is working. |
| **Model Code** | ✅ Yes | Confirms `deepseek_ocr2.py` and vLLM are compatible. |
| **Extraction Logic** | ✅ Yes | Confirms `extraction.py` logic is correct. |
| **Docker Image** | ❌ No | This doesn't test the actual *image build*. However, if the code works here, and your `Dockerfile` just copies this code on top of a similar base image, the image is 99% likely to work. |

## 6. (Optional) Final Verification via Serverless

Once the code test above passes:
1.  Deploy your Docker image (`shubh0078/dsk-ocr-endpoint:v1.0`) as a **Serverless Endpoint**.
2.  Use the "Requests" tab in RunPod to send a test JSON payload.
