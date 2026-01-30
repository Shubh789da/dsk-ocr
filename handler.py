"""
RunPod Serverless Handler for DeepSeek-OCR-2
Extracts INDICATIONS AND USAGE from FDA pharmaceutical label PDFs
"""

import os
import io
import base64
import tempfile
import runpod

# Set environment variables before importing vLLM
os.environ['VLLM_USE_V1'] = '0'

import torch
import fitz  # PyMuPDF
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Import custom model and processors
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor
from extraction import extract_indications_and_usage_fda, clean_ocr_artifacts

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2')
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '50'))
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '32'))
CROP_MODE = os.getenv('CROP_MODE', 'True').lower() == 'true'
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.9'))
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', '8192'))

PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'

# Register custom model
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# Initialize model globally (load once, use many times)
print("Loading DeepSeek-OCR-2 model...")
llm = LLM(
    model=MODEL_PATH,
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    block_size=256,
    enforce_eager=False,
    trust_remote_code=True,
    max_model_len=MAX_MODEL_LEN,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    disable_mm_preprocessor_cache=True
)
print("Model loaded successfully!")

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


def pdf_to_images(pdf_bytes, dpi=144):
    """Convert PDF bytes to PIL images"""
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
    # Convert PDF to images
    images = pdf_to_images(pdf_bytes)

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

    # Extract INDICATIONS AND USAGE section
    indications = extract_indications_and_usage_fda(full_text)

    return {
        "full_text": full_text,
        "indications_and_usage": indications,
        "page_count": len(images)
    }


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
            "extraction_only": true     # If true, only return indications (default)
        }
    }

    Output format:
    {
        "indications_and_usage": "<extracted text>",
        "page_count": 180,
        "full_text": "<optional, if return_full_text=true>"
    }
    """
    job_input = job["input"]

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
        import requests
        try:
            response = requests.get(job_input["pdf_url"], timeout=60)
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

    return response


# For local testing
if __name__ == "__main__":
    # Test with a sample PDF
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        result = process_pdf(pdf_bytes)
        print("=" * 80)
        print("INDICATIONS AND USAGE:")
        print("=" * 80)
        print(result["indications_and_usage"][:2000])
        print(f"\n... (Total: {len(result['indications_and_usage'])} chars)")
        print(f"\nProcessed {result['page_count']} pages")
    else:
        # Start RunPod serverless
        runpod.serverless.start({"handler": handler})
