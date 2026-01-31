#!/usr/bin/env python3.12
"""
OPTIMIZED test script for DeepSeek-OCR-2 on RunPod
Uses BATCH PROCESSING for faster throughput

Usage:
    python3.12 test_model_fast.py <pdf_path_or_url>
    python3.12 test_model_fast.py <pdf_path_or_url> --no-crop  # Disable cropping for speed
    python3.12 test_model_fast.py <pdf_path_or_url> --low-dpi  # Use 72 DPI for speed
"""

# CRITICAL: Set environment variables FIRST, before ANY imports
import os
os.environ['VLLM_USE_V1'] = '0'  # Disable V1 engine (causes OOM with multimodal)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import time
import re
import torch
from concurrent.futures import ThreadPoolExecutor

print("=" * 60)
print("DeepSeek-OCR-2 FAST Test Script (Batch Processing)")
print("=" * 60)

# Parse arguments
args = sys.argv[1:]
USE_CROP = '--no-crop' not in args
USE_LOW_DPI = '--low-dpi' in args
input_path = [a for a in args if not a.startswith('--')][0] if [a for a in args if not a.startswith('--')] else None

if not input_path:
    print("Usage: python3.12 test_model_fast.py <pdf_path_or_url> [--no-crop] [--low-dpi]")
    sys.exit(1)

DPI = 72 if USE_LOW_DPI else 108
print(f"\nOptions: CROP_MODE={USE_CROP}, DPI={DPI}")

# Check GPU
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram_gb:.1f} GB")

    # Optimized settings for BATCH processing
    if total_vram_gb >= 70:  # H100/A100 80GB
        GPU_MEMORY_UTILIZATION = 0.80  # Use more memory for batching
        MAX_MODEL_LEN = 4096
        MAX_NUM_SEQS = 16  # More concurrent sequences
        NUM_WORKERS = 16
    elif total_vram_gb >= 40:  # A100 40GB
        GPU_MEMORY_UTILIZATION = 0.75
        MAX_MODEL_LEN = 4096
        MAX_NUM_SEQS = 8
        NUM_WORKERS = 8
    elif total_vram_gb >= 20:  # RTX 4090
        GPU_MEMORY_UTILIZATION = 0.70  # Maximize for batch
        MAX_MODEL_LEN = 4096
        MAX_NUM_SEQS = 4  # Batch 4 pages at once
        NUM_WORKERS = 4
    else:
        print(f"[ERROR] GPU has only {total_vram_gb:.1f}GB VRAM.")
        sys.exit(1)
else:
    print("[ERROR] No CUDA GPU available!")
    sys.exit(1)

# Allow environment variable overrides
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', str(GPU_MEMORY_UTILIZATION)))
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', str(MAX_MODEL_LEN)))
MAX_NUM_SEQS = int(os.getenv('MAX_NUM_SEQS', str(MAX_NUM_SEQS)))
MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2')

print(f"\nConfiguration (OPTIMIZED FOR BATCH):")
print(f"  GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
print(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
print(f"  MAX_NUM_SEQS: {MAX_NUM_SEQS} (batch size)")
print(f"  NUM_WORKERS: {NUM_WORKERS}")

# Import vLLM
print("\nImporting vLLM...")
from vllm import LLM, SamplingParams
import vllm
print(f"vLLM version: {vllm.__version__}")

from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

# Register model
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# Load model
print("\n" + "=" * 60)
print("Loading model...")
print("=" * 60)

model_load_start = time.time()

try:
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        limit_mm_per_prompt={"image": 1},
        block_size=16,
        enforce_eager=True,
        trust_remote_code=True,
        swap_space=0,
        tensor_parallel_size=1,
        disable_mm_preprocessor_cache=True,
        enable_prefix_caching=False,
    )
    model_load_time = time.time() - model_load_start
    print(f"\n[SUCCESS] Model loaded in {model_load_time:.1f}s")

except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def clean_ocr_output(text):
    """Clean OCR output"""
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        if '<|ref|>image<|/ref|>' in match[0]:
            text = text.replace(match[0], '')
        else:
            text = text.replace(match[0], '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')
    text = text.replace('<｜end▁of▁sentence｜>', '')
    text = re.sub(r'\n\n\n+', '\n\n', text)
    return text.strip()


# Main processing
print(f"\n" + "=" * 60)
is_url = input_path.lower().startswith(('http://', 'https://'))
if is_url:
    print(f"Downloading: {input_path}")
else:
    print(f"Processing: {input_path}")
print("=" * 60)

import fitz
from PIL import Image
import io
import requests

pdf_start_time = time.time()

try:
    if is_url:
        response = requests.get(input_path, timeout=120)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pdf_filename = input_path.split('/')[-1] or "downloaded.pdf"
    else:
        pdf_doc = fitz.open(input_path)
        pdf_filename = os.path.basename(input_path)

    total_pages = len(pdf_doc)
    print(f"Total pages: {total_pages}")

except Exception as e:
    print(f"[ERROR] Failed to open PDF: {e}")
    sys.exit(1)

# Prepare processor
PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
processor = DeepseekOCR2Processor()

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

# STEP 1: Convert ALL pages to images (parallel)
print(f"\n[1/3] Converting {total_pages} pages to images (DPI={DPI})...")
convert_start = time.time()

zoom = DPI / 72.0
matrix = fitz.Matrix(zoom, zoom)

images = []
for page_num in range(total_pages):
    page = pdf_doc[page_num]
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    images.append(img)

pdf_doc.close()
convert_time = time.time() - convert_start
print(f"    Converted in {convert_time:.2f}s ({convert_time/total_pages:.3f}s per page)")

# STEP 2: Prepare ALL inputs (parallel)
print(f"\n[2/3] Preparing {total_pages} inputs (CROP_MODE={USE_CROP})...")
prep_start = time.time()

def prepare_input(img):
    return {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                images=[img],
                bos=True,
                eos=True,
                cropping=USE_CROP
            )
        },
    }

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    batch_inputs = list(executor.map(prepare_input, images))

prep_time = time.time() - prep_start
print(f"    Prepared in {prep_time:.2f}s ({prep_time/total_pages:.3f}s per page)")

# STEP 3: BATCH inference (all at once!)
print(f"\n[3/3] Running BATCH inference on {total_pages} pages...")
print(f"    (Processing up to {MAX_NUM_SEQS} pages concurrently)")
inference_start = time.time()

# This is the key: process ALL pages in one call
outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

inference_time = time.time() - inference_start
print(f"    Inference completed in {inference_time:.2f}s ({inference_time/total_pages:.3f}s per page)")

# Collect outputs
all_outputs = []
for output in outputs_list:
    text = output.outputs[0].text
    text = clean_ocr_output(text)
    all_outputs.append(text)

total_pdf_time = time.time() - pdf_start_time

# Summary
print("\n" + "=" * 60)
print("PERFORMANCE SUMMARY")
print("=" * 60)
combined_output = "\n\n--- PAGE BREAK ---\n\n".join(all_outputs)
print(f"Total pages: {total_pages}")
print(f"Total output: {len(combined_output)} chars")
print(f"\n--- TIMING BREAKDOWN ---")
print(f"  PDF conversion:  {convert_time:6.2f}s ({convert_time/total_pages:.3f}s/page)")
print(f"  Input prep:      {prep_time:6.2f}s ({prep_time/total_pages:.3f}s/page)")
print(f"  Inference:       {inference_time:6.2f}s ({inference_time/total_pages:.3f}s/page)")
print(f"  ---------------------------------")
print(f"  Total:           {total_pdf_time:6.2f}s ({total_pdf_time/total_pages:.3f}s/page)")
print(f"\n--- THROUGHPUT ---")
print(f"  Pages per second: {total_pages/total_pdf_time:.2f}")
print(f"  Pages per minute: {60*total_pages/total_pdf_time:.1f}")

# Compare with sequential
seq_estimate = 8.3 * total_pages  # ~8.3s per page sequential
speedup = seq_estimate / total_pdf_time
print(f"\n--- SPEEDUP ---")
print(f"  Estimated sequential: {seq_estimate:.1f}s")
print(f"  Batch processing:     {total_pdf_time:.1f}s")
print(f"  Speedup: {speedup:.1f}x faster")

# Save output
output_file = pdf_filename.replace('.pdf', '_fast_ocr.md')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"# OCR Output for {pdf_filename}\n\n")
    f.write(f"- Pages: {total_pages}\n")
    f.write(f"- Processing time: {total_pdf_time:.1f}s\n")
    f.write(f"- Per page: {total_pdf_time/total_pages:.2f}s\n\n---\n\n")
    for i, page_output in enumerate(all_outputs):
        f.write(f"## Page {i + 1}\n\n{page_output}\n\n---\n\n")
print(f"\nSaved to: {output_file}")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
