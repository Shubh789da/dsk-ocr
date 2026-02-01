"""
RunPod Serverless Handler for DeepSeek-OCR-2
Extracts INDICATIONS AND USAGE from FDA pharmaceutical label PDFs
"""

# CRITICAL: Set environment variables FIRST, before ANY imports
import os
os.environ['VLLM_USE_V1'] = '0'  # Disable V1 engine (causes OOM with multimodal)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevent fork warning in serverless
# Don't force XFORMERS - let vLLM auto-detect the best backend for the GPU
# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Help with memory fragmentation

import io
import re
import base64
import tempfile
import requests

import torch
import fitz  # PyMuPDF
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Now import vLLM (after env vars are set)
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Import custom model and processors
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor
from extraction import extract_indications_and_usage_fda, clean_ocr_artifacts

# Import runpod AFTER vllm to avoid any conflicts
import runpod

# Import boto3 for S3 uploads
import boto3
from botocore.exceptions import ClientError

# Configuration with auto-detection
MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2')
CROP_MODE = os.getenv('CROP_MODE', 'True').lower() == 'true'

# Auto-configure memory settings based on GPU
if torch.cuda.is_available():
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Detected GPU: {torch.cuda.get_device_name(0)} with {total_vram_gb:.1f}GB VRAM")

    # OPTIMIZED SETTINGS - Tested for batch processing performance
    if total_vram_gb >= 70:  # H100/A100 80GB
        DEFAULT_GPU_MEM = 0.90  # Maximize for batch throughput
        DEFAULT_MAX_LEN = 5192
        DEFAULT_MAX_SEQS = 80  # High concurrency for H100
        DEFAULT_WORKERS = 80
    elif total_vram_gb >= 40:  # A100 40GB
        DEFAULT_GPU_MEM = 0.85
        DEFAULT_MAX_LEN = 5192
        DEFAULT_MAX_SEQS = 36
        DEFAULT_WORKERS = 36
    elif total_vram_gb >= 20:  # RTX 4090/3090 - TESTED SETTINGS
        DEFAULT_GPU_MEM = 0.90  # Tested: works well
        DEFAULT_MAX_LEN = 4096
        DEFAULT_MAX_SEQS = 20  # Tested: 1.025s/page, 58.6 pages/min
        DEFAULT_WORKERS = 20
    else:
        raise RuntimeError(f"GPU has only {total_vram_gb:.1f}GB VRAM. Minimum 20GB required.")
else:
    raise RuntimeError("No CUDA GPU available!")

# Allow environment variable overrides
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', str(DEFAULT_GPU_MEM)))
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', str(DEFAULT_MAX_LEN)))
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', str(DEFAULT_MAX_SEQS)))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', str(DEFAULT_WORKERS)))

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Register custom model
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# Initialize model globally (load once, use many times)
print("=" * 60)
print("Loading DeepSeek-OCR-2 model...")
print("=" * 60)
print(f"Configuration:")
print(f"  GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
print(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
print(f"  MAX_CONCURRENCY: {MAX_CONCURRENCY}")
print(f"  VLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'not set')}")

llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    # Memory optimization settings - CRITICAL
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,  # Leave room for vision encoder
    max_model_len=MAX_MODEL_LEN,  # Reduced to lower KV cache memory
    max_num_seqs=MAX_CONCURRENCY,  # Limit concurrent sequences
    # Multimodal settings - CRITICAL for OOM prevention
    limit_mm_per_prompt={"image": 1},  # Limit images per prompt during profiling
    # Other settings
    block_size=16,  # Smaller block size = less memory waste
    enforce_eager=True,  # Use eager mode for stability
    trust_remote_code=True,
    swap_space=0,
    tensor_parallel_size=1,
    disable_mm_preprocessor_cache=True,
    enable_prefix_caching=False,  # Disable for OCR (from official docs)
)
print("Model loaded successfully!")
print("=" * 60)

# Logits processors
logits_processors = [
    NoRepeatNGramLogitsProcessor(
        ngram_size=20,
        window_size=50,
        whitelist_token_ids={128821, 128822}
    )
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_MODEL_LEN,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)


def pdf_to_images(pdf_bytes, dpi=108):
    """Convert PDF bytes to PIL images (reduced DPI for memory efficiency)"""
    images = []
    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        images.append(img)

    pdf_document.close()
    return images


def process_single_image(image):
    """Prepare single image for model input"""
    return {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": DeepseekOCR2Processor().tokenize_with_images(
                images=[image],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
        },
    }


def clean_ocr_output(text):
    """Clean OCR output by removing detection markers"""
    import re

    # Remove ref/det tags
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)

    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            text = text.replace(match[0], '')
        else:
            text = text.replace(match[0], '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    # Remove end of sentence marker
    text = text.replace('<｜end▁of▁sentence｜>', '')

    # Clean up multiple newlines
    text = re.sub(r'\n\n\n+', '\n\n', text)

    return text.strip()


def process_pdf(pdf_bytes):
    """
    Process PDF and extract INDICATIONS AND USAGE

    Args:
        pdf_bytes: Raw PDF file bytes

    Returns:
        dict with full_text, indications_and_usage, page_count
    """
    # Debug: Check PDF page count before conversion
    if os.getenv('DEBUG_OCR', '').lower() == 'true':
        import fitz
        temp_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"\n[DEBUG] PDF contains {len(temp_doc)} pages")
        temp_doc.close()

    # Convert PDF to images
    images = pdf_to_images(pdf_bytes)

    if os.getenv('DEBUG_OCR', '').lower() == 'true':
        print(f"[DEBUG] Converted {len(images)} images from PDF")

    # Prepare batch inputs
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        batch_inputs = list(executor.map(process_single_image, images))

    # Run OCR inference
    outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

    # Collect and clean output
    full_text_parts = []
    for output in outputs_list:
        content = output.outputs[0].text
        content = clean_ocr_output(content)
        if content:
            full_text_parts.append(content)

    # Combine all pages
    full_text = '\n\n--- Page Break ---\n\n'.join(full_text_parts)

    # Clean OCR artifacts
    full_text = clean_ocr_artifacts(full_text)

    # Debug output if enabled
    if os.getenv('DEBUG_OCR', '').lower() == 'true':
        print(f"\n[DEBUG] Full OCR text length: {len(full_text)} chars")
        print(f"[DEBUG] First 1000 chars of OCR output:")
        print("-" * 40)
        print(full_text[:1000])
        print("-" * 40)

        # Check for section header (not just the word "INDICATIONS")
        # Look for the actual section header patterns
        section_patterns = [
            r'---\s*INDICATIONS\s+AND\s+USAGE\s*---',  # FDA highlights format
            r'##\s*---?\s*INDICATIONS\s+AND\s+USAGE',  # Markdown with optional dashes
            r'\n\s*1\s+INDICATIONS\s+AND\s+USAGE',     # Numbered section
            r'\n\s*INDICATIONS\s+AND\s+USAGE\s*\n',    # Standalone header
        ]

        found_section = False
        for pattern in section_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                idx = match.start()
                print(f"\n[DEBUG] Found section header matching pattern: {pattern}")
                print(f"[DEBUG] Location: character {idx}")
                print(f"[DEBUG] Context (50 chars before, 500 after):")
                print("-" * 40)
                print(full_text[max(0, idx-50):idx+500])
                print("-" * 40)
                found_section = True
                break

        if not found_section:
            upper_text = full_text.upper()
            if 'INDICATIONS' in upper_text:
                idx = upper_text.find('INDICATIONS')
                print(f"\n[DEBUG] WARNING: No section header found, but 'INDICATIONS' appears at char {idx}")
                print(f"[DEBUG] This might be in a table or reference, not the actual section.")
                print(f"[DEBUG] Context:")
                print("-" * 40)
                print(full_text[max(0, idx-50):idx+300])
                print("-" * 40)
            else:
                print("\n[DEBUG] WARNING: 'INDICATIONS' not found anywhere in OCR output!")
                print("[DEBUG] This explains why extraction returns empty.")

    # Extract INDICATIONS AND USAGE section
    indications = extract_indications_and_usage_fda(full_text)

    if os.getenv('DEBUG_OCR', '').lower() == 'true':
        print(f"\n[DEBUG] Extraction result: {len(indications)} chars")
        if indications:
            print(f"[DEBUG] Extracted text preview:")
            print("-" * 40)
            print(indications[:500])
            print("-" * 40)

    return {
        "full_text": full_text,
        "indications_and_usage": indications,
        "page_count": len(images)
    }


def upload_to_s3(content, filename, content_type="text/markdown"):
    """
    Upload content to S3 bucket.
    
    Args:
        content: String content to upload
        filename: Name of the file (e.g., 'job123_full_text.md')
        content_type: MIME type of the content
    
    Returns:
        S3 URL if successful, None if failed
    """
    bucket_name = os.getenv('S3_BUCKET_NAME', 'deepseek-ocr-sy')
    region = os.getenv('AWS_DEFAULT_REGION', 'ap-south-2')
    
    # Check if AWS credentials are available
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if not aws_access_key or not aws_secret_key:
        print("[S3] Warning: AWS credentials not configured, skipping S3 upload")
        return None
    
    try:
        s3_client = boto3.client(
            's3',
            region_name=region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=filename,
            Body=content.encode('utf-8'),
            ContentType=content_type
        )
        
        s3_url = f"s3://{bucket_name}/{filename}"
        print(f"[S3] Uploaded: {s3_url}")
        return s3_url
        
    except ClientError as e:
        print(f"[S3] Error uploading to S3: {e}")
        return None
    except Exception as e:
        print(f"[S3] Unexpected error: {e}")
        return None

def handler(job):
    """
    RunPod Serverless Handler

    Input format:
    {
        "input": {
            "pdf_base64": "<base64 encoded PDF>",
            # OR
            "pdf_url": "<URL to PDF file>",

            # Optional
            "return_full_text": false,  # If true, also return full OCR text
            "save_to_s3": false,        # If true, save outputs to S3 bucket
            "filename_prefix": "doc"    # Optional prefix for S3 filenames
        }
    }

    Output format:
    {
        "indications_and_usage": "<extracted text>",
        "page_count": 180,
        "full_text": "<optional, if return_full_text=true>",
        "s3_full_text_url": "<optional, if save_to_s3=true>",
        "s3_indications_url": "<optional, if save_to_s3=true>"
    }
    """
    job_input = job["input"]
    job_id = job.get("id", "unknown")

    # Get PDF data
    pdf_bytes = None

    if "pdf_base64" in job_input:
        # Decode base64 PDF
        try:
            pdf_bytes = base64.b64decode(job_input["pdf_base64"])
        except Exception as e:
            return {"error": f"Failed to decode base64 PDF: {str(e)}"}

    elif "pdf_url" in job_input:
        # Download PDF from URL
        try:
            response = requests.get(job_input["pdf_url"], timeout=120)
            response.raise_for_status()
            pdf_bytes = response.content
        except Exception as e:
            return {"error": f"Failed to download PDF: {str(e)}"}

    else:
        return {"error": "Must provide either 'pdf_base64' or 'pdf_url' in input"}

    # Process PDF
    try:
        result = process_pdf(pdf_bytes)
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}

    # Prepare response
    response = {
        "indications_and_usage": result["indications_and_usage"],
        "page_count": result["page_count"]
    }

    # Include full text if requested
    if job_input.get("return_full_text", False):
        response["full_text"] = result["full_text"]

    # Upload to S3 if requested
    if job_input.get("save_to_s3", False):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = job_input.get("filename_prefix", job_id)
        
        # Upload full OCR text
        full_text_filename = f"{prefix}_{timestamp}_full_text.md"
        full_text_url = upload_to_s3(result["full_text"], full_text_filename)
        if full_text_url:
            response["s3_full_text_url"] = full_text_url
        
        # Upload indications section
        if result["indications_and_usage"]:
            indications_filename = f"{prefix}_{timestamp}_indications.md"
            indications_url = upload_to_s3(result["indications_and_usage"], indications_filename)
            if indications_url:
                response["s3_indications_url"] = indications_url

    return response


# For local testing
if __name__ == "__main__":
    # Test with a sample PDF (file path or URL)
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]

        # Check if it's a URL or local file
        if input_path.lower().startswith(('http://', 'https://')):
            print(f"Downloading PDF from URL: {input_path}")
            response = requests.get(input_path, timeout=120)
            response.raise_for_status()
            pdf_bytes = response.content
            print(f"Downloaded {len(pdf_bytes):,} bytes")
        else:
            print(f"Reading local file: {input_path}")
            with open(input_path, "rb") as f:
                pdf_bytes = f.read()

        result = process_pdf(pdf_bytes)
        print("=" * 80)
        print("INDICATIONS AND USAGE:")
        print("=" * 80)
        if result["indications_and_usage"]:
            print(result["indications_and_usage"][:2000])
            print(f"\n... (Total: {len(result['indications_and_usage'])} chars)")
        else:
            print("(No content extracted)")
        print(f"\nProcessed {result['page_count']} pages")
    else:
        # Start RunPod serverless
        runpod.serverless.start({"handler": handler})
