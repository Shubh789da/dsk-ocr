# Testing DeepSeek-OCR-2 Docker Image on Google Cloud Platform (GCP)

Yes! GCP VMs are **full virtual machines**, so they support Docker perfectly (unlike RunPod's container-based pods). This is the most reliable way to test your Docker image exactly as it was built.

We will use a **Deep Learning VM** image, which comes with Nvidia Drivers, Docker, and the Nvidia Container Toolkit pre-installed.

## 1. Create a Deep Learning VM

1.  Go to the **[Google Cloud Console](https://console.cloud.google.com/)**.
2.  Navigate to **Computer Engine** > **VM Instances**.
3.  Click **Create Instance**.
4.  **Name**: `ocr-test-vm`
5.  **Region**: Select a region with GPU availability (e.g., `us-central1` or `us-east1`).
6.  **Machine Configuration**:
    *   **Series**: `N1` or `G2` (depending on GPU choice).
    *   **GPU**: Click "Change" to add a GPU.
        *   *Recommendation*: **NVIDIA T4** (Cheapest, good for functional testing) or **L4** (Modern, faster).
        *   (Note: DeepSeek-OCR-2 runs best on Ampere+ like A100, but T4 is fine for verifying the *image works*).
7.  **Boot Disk**:
    *   Click "Switch Image".
    *   **OS**: **Deep Learning on Linux**.
    *   **Version**: **Deep Learning VM with CUDA 11.8 M114** (or newer).
        *   *Tip*: Using this image saves you 30 minutes of installing drivers!
8.  **Firewall**: Check "Allow HTTP traffic" and "Allow HTTPS traffic".
9.  Click **Create**.

## 2. Connect to the VM

1.  Once the VM is running, click the **SSH** button next to it in the console.
2.  This opens a browser-based terminal.

## 3. Verify Docker and GPU

Run these commands to confirm everything is ready:

```bash
# Check simple Docker command
docker --version

# Check Nvidia drivers
nvidia-smi
```

## 4. Pull Your Image

Authenticating properly for a clean pull:

```bash
docker pull shubh0078/dsk-ocr-endpoint:v1.0
```

## 5. Run the Test

Now you can run the standard `docker run` command. Note that GCP VMs map GPUs slightly differently, but `--gpus all` usually works if the toolkit is installed (which it is on DL Cloud images).

1.  **Download Test PDF**:
    ```bash
    wget https://github.com/Starttoaster/Fda_indications_extraction/raw/main/keytruda.pdf -O test.pdf
    ```

2.  **Run Extraction**:
    ```bash
    docker run --rm --gpus all \
      -v $(pwd)/test.pdf:/app/test.pdf \
      shubh0078/dsk-ocr-endpoint:v1.0 \
      python handler.py test.pdf
    ```

### Troubleshooting GCP Quotas
*   **"Quota Exceeded"**: If you can't create a GPU VM, you might need to request a quota increase for "NVIDIA T4 GPUs" in your region.
*   **Alternative**: If getting a GPU is hard, you can test the *container logic* on a CPU-only VM (it will crash on model load, but will verify imports and env vars).
    *   To do this, remove `--gpus all`.

## 6. (Optional) Testing Serverless Behavior

To simulate the HTTP server:

```bash
# Run in background mapping port 8000
docker run --rm --gpus all -p 8000:8000 shubh0078/dsk-ocr-endpoint:v1.0
```

Then send a request from the same VM:

```bash
# In a new SSH window
curl -X POST http://localhost:8000/runsync \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
        "pdf_url": "https://github.com/Starttoaster/Fda_indications_extraction/raw/main/keytruda.pdf"
    }
  }'
```
