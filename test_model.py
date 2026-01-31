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
# Help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import torch

print("=" * 60)
print("DeepSeek-OCR-2 Test Script")
print("=" * 60)

# Check GPU and auto-configure memory settings
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {total_vram_gb:.1f} GB")

    # Auto-configure based on GPU memory
    # DeepSeek-OCR-2 needs:
    #   - ~6-7GB for model weights (bfloat16)
    #   - ~2-4GB for vision encoder activations during forward pass
    #   - KV cache (variable, controlled by gpu_memory_utilization)

    if total_vram_gb >= 70:  # H100/A100 80GB
        GPU_MEMORY_UTILIZATION = 0.50  # Use only 50% for KV cache, leave room for vision
        MAX_MODEL_LEN = 4096
        MAX_NUM_SEQS = 4
        print(f"\n[AUTO-CONFIG] High-VRAM GPU detected, using moderate settings")
    elif total_vram_gb >= 40:  # A100 40GB / A6000
        GPU_MEMORY_UTILIZATION = 0.45
        MAX_MODEL_LEN = 4096
        MAX_NUM_SEQS = 2
        print(f"\n[AUTO-CONFIG] Medium-VRAM GPU detected")
    elif total_vram_gb >= 20:  # RTX 4090 / 3090
        GPU_MEMORY_UTILIZATION = 0.40  # Very conservative for 24GB cards
        MAX_MODEL_LEN = 2048
        MAX_NUM_SEQS = 1
        print(f"\n[AUTO-CONFIG] Consumer GPU detected, using conservative settings")
    else:
        print(f"\n[ERROR] GPU has only {total_vram_gb:.1f}GB VRAM.")
        print("        DeepSeek-OCR-2 requires at least 20GB VRAM.")
        sys.exit(1)
else:
    print("\n[ERROR] No CUDA GPU available!")
    sys.exit(1)

# Allow environment variable overrides
GPU_MEMORY_UTILIZATION = float(os.getenv('GPU_MEMORY_UTILIZATION', str(GPU_MEMORY_UTILIZATION)))
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', str(MAX_MODEL_LEN)))
MAX_NUM_SEQS = int(os.getenv('MAX_NUM_SEQS', str(MAX_NUM_SEQS)))
MODEL_PATH = os.getenv('MODEL_PATH', 'deepseek-ai/DeepSeek-OCR-2')

# Check env vars
print(f"\nVLLM_USE_V1: {os.environ.get('VLLM_USE_V1', 'NOT SET')}")
print(f"VLLM_ATTENTION_BACKEND: {os.environ.get('VLLM_ATTENTION_BACKEND', 'NOT SET')}")
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'NOT SET')}")

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

print(f"\nConfiguration:")
print(f"  MODEL_PATH: {MODEL_PATH}")
print(f"  GPU_MEMORY_UTILIZATION: {GPU_MEMORY_UTILIZATION}")
print(f"  MAX_MODEL_LEN: {MAX_MODEL_LEN}")
print(f"  MAX_NUM_SEQS: {MAX_NUM_SEQS}")

# Load model
print("\n" + "=" * 60)
print("Loading model (this may take a few minutes)...")
print("=" * 60)

try:
    llm = LLM(
        model=MODEL_PATH,
        hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
        # Memory optimization settings
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,  # CRITICAL: Leave room for vision encoder
        max_model_len=MAX_MODEL_LEN,  # Reduced to lower KV cache memory
        max_num_seqs=MAX_NUM_SEQS,  # Limit concurrent sequences
        # Multimodal settings - CRITICAL for OOM prevention
        limit_mm_per_prompt={"image": 1},  # Limit images per prompt during profiling
        # Other settings
        block_size=16,  # Smaller block size = less memory waste
        enforce_eager=True,  # Use eager mode for stability
        trust_remote_code=True,
        swap_space=0,
        tensor_parallel_size=1,
        disable_mm_preprocessor_cache=True,
        enable_prefix_caching=False,  # Disable for OCR
    )
    print("\n[SUCCESS] Model loaded successfully!")

except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()

    # Provide helpful suggestions
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING SUGGESTIONS:")
    print("=" * 60)
    print("1. Try lower GPU_MEMORY_UTILIZATION:")
    print("   export GPU_MEMORY_UTILIZATION=0.35")
    print("\n2. Try smaller MAX_MODEL_LEN:")
    print("   export MAX_MODEL_LEN=1024")
    print("\n3. Check if other processes are using GPU memory:")
    print("   nvidia-smi")
    print("\n4. Try clearing GPU cache before running:")
    print("   torch.cuda.empty_cache()")
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
    # Use lower DPI to reduce memory usage
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 108 DPI instead of 144
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
        max_tokens=min(4096, MAX_MODEL_LEN),
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
