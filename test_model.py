#!/usr/bin/env python3.12
"""
Simple test script for DeepSeek-OCR-2 on RunPod
Tests model loading and basic inference

Usage:
    python3.12 test_model.py
    python3.12 test_model.py /path/to/test.pdf
"""

# CRITICAL: Set environment variables FIRST, before ANY imports
import os
os.environ['VLLM_USE_V1'] = '0'  # Disable V1 engine (causes OOM with multimodal)
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'  # Use xformers backend

import sys
import torch

print("=" * 60)
print("DeepSeek-OCR-2 Test Script")
print("=" * 60)

# Check GPU
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check env vars
print(f"\nVLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'NOT SET')}")
print(f"VLLM_ATTENTION_BACKEND: {os.environ.get('VLLM_ATTENTION_BACKEND', 'NOT SET')}")

# Now import vLLM
print("\nImporting vLLM...")
from vllm import LLM, SamplingParams
import vllm
print(f"vLLM version: {vllm.__version__}")

from vllm.model_executor.models.registry import ModelRegistry

# Import custom model
print("Importing custom model...")
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

# Register model
print("Registering model...")
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# Configuration
MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2')
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', '0.85'))
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', '8192'))

print(f"\nConfiguration:")
print(f"  MODEL_PATH: {MODEL_PATH}")
print(f"  GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
print(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")

# Load model
print("\n" + "=" * 60)
print("Loading model (this may take a few minutes)...")
print("=" * 60)

try:
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        block_size=256,
        enforce_eager=True,  # Use eager mode for stability
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
        swap_space=0,
        max_num_seqs=10,  # Reduced for testing
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_mm_preprocessor_cache=True,
        enable_prefix_caching=False,  # Disable for OCR
    )
    print("\n[SUCCESS] Model loaded successfully!")

except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test inference with a simple image if PDF provided
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
    print(f"\n" + "=" * 60)
    print(f"Testing with PDF: {pdf_path}")
    print("=" * 60)

    import fitz
    from PIL import Image
    import io

    # Convert first page to image
    pdf_doc = fitz.open(pdf_path)
    page = pdf_doc[0]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    pdf_doc.close()

    print(f"Image size: {img.size}")

    # Prepare input
    PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
    CROP_MODE = True

    processor = DeepseekOCR2Processor()
    batch_input = {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                images=[img],
                bos=True,
                eos=True,
                cropping=CROP_MODE
            )
        },
    }

    # Sampling params
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=20,
            window_size=50,
            whitelist_token_ids={128821, 128822}
        )
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4096,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

    print("Running inference on first page...")
    outputs = llm.generate([batch_input], sampling_params=sampling_params)

    print("\n" + "=" * 60)
    print("OUTPUT (first 1000 chars):")
    print("=" * 60)
    output_text = outputs[0].outputs[0].text
    print(output_text[:1000])
    print(f"\n... (Total: {len(output_text)} chars)")

else:
    print("\n[INFO] No PDF provided. Model loaded successfully!")
    print("       Run with: python3.12 test_model.py /path/to/test.pdf")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
