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
# Don't force XFORMERS - let vLLM auto-detect the best backend
# os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'
# Help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import time
import re
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
        GPU_MEMORY_UTILIZATION = 0.35  # Conservative: leave room for vision encoder (4GB)
        MAX_MODEL_LEN = 2048
        MAX_NUM_SEQS = 2
        print(f"\n[AUTO-CONFIG] High-VRAM GPU detected, using conservative settings")
    elif total_vram_gb >= 40:  # A100 40GB / A6000
        GPU_MEMORY_UTILIZATION = 0.35
        MAX_MODEL_LEN = 2048
        MAX_NUM_SEQS = 1
        print(f"\n[AUTO-CONFIG] Medium-VRAM GPU detected")
    elif total_vram_gb >= 20:  # RTX 4090 / 3090
        GPU_MEMORY_UTILIZATION = 0.30  # Very conservative for 24GB cards
        MAX_MODEL_LEN = 2048
        MAX_NUM_SEQS = 1
        print(f"\n[AUTO-CONFIG] Consumer GPU detected, using very conservative settings")
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
print(f"VLLM_ATTENTION_BACKEND: {os.environ.get('VLLM_ATTENTION_BACKEND', 'auto-detect')}")
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

model_load_start = time.time()

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
    model_load_time = time.time() - model_load_start
    print(f"\n[SUCCESS] Model loaded successfully in {model_load_time:.1f} seconds!")

except Exception as e:
    print(f"\n[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()

    # Provide helpful suggestions
    print("\n" + "=" * 60)
    print("TROUBLESHOOTING SUGGESTIONS:")
    print("=" * 60)
    print("1. Try lower GPU_MEMORY_UTILIZATION:")
    print("   export GPU_MEMORY_UTILIZATION=0.25")
    print("\n2. Try smaller MAX_MODEL_LEN:")
    print("   export MAX_MODEL_LEN=512")
    print("\n3. Check if other processes are using GPU memory:")
    print("   nvidia-smi")
    print("\n4. Try clearing GPU cache before running:")
    print("   torch.cuda.empty_cache()")
    print("\n5. If xFormers error, it will auto-detect FLASH_ATTN or other backend")
    sys.exit(1)


def clean_ocr_output(text):
    """Clean OCR output by removing detection markers (same as handler.py)"""
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


# Test inference with a simple image if PDF provided
if len(sys.argv) > 1:
    pdf_path = sys.argv[1]
    print(f"\n" + "=" * 60)
    print(f"Testing with PDF: {pdf_path}")
    print("=" * 60)

    import fitz
    from PIL import Image
    import io

    # Start timing for PDF processing
    pdf_start_time = time.time()

    # Open PDF and get total page count
    pdf_doc = fitz.open(pdf_path)
    total_pages = len(pdf_doc)
    print(f"Total pages in PDF: {total_pages}")

    # Prepare processor and sampling params (reusable for all pages)
    PROMPT = '<image>\n<|grounding|>Convert the document to markdown.'
    CROP_MODE = True
    processor = DeepseekOCR2Processor()

    # Sampling params (same as handler.py)
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
        include_stop_str_in_output=True,  # Same as handler.py
    )

    # Process ALL pages
    all_outputs = []
    page_times = []

    for page_num in range(total_pages):
        page_start = time.time()
        print(f"\n{'='*60}")
        print(f"Processing page {page_num + 1}/{total_pages}...")
        print("=" * 60)

        # Convert page to image (same DPI as handler.py: 108)
        page = pdf_doc[page_num]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))  # 108 DPI
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))

        print(f"Image size: {img.size}")

        # Prepare input for this page (same structure as handler.py)
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

        # Run inference
        inference_start = time.time()
        outputs = llm.generate([batch_input], sampling_params=sampling_params)
        inference_time = time.time() - inference_start

        output_text = outputs[0].outputs[0].text

        # Clean output (same as handler.py)
        output_text = clean_ocr_output(output_text)
        all_outputs.append(output_text)

        page_time = time.time() - page_start
        page_times.append(page_time)

        print(f"\nPage {page_num + 1} OUTPUT (first 500 chars):")
        print("-" * 40)
        print(output_text[:500] if output_text else "[Empty output]")
        print(f"... (Total: {len(output_text)} chars)")
        print(f"Page processing time: {page_time:.2f}s (inference: {inference_time:.2f}s)")

        # Clear some memory between pages
        del img, pixmap, img_data, batch_input, outputs
        torch.cuda.empty_cache()

    pdf_doc.close()

    # Calculate total time
    total_pdf_time = time.time() - pdf_start_time
    avg_page_time = sum(page_times) / len(page_times) if page_times else 0

    # Combine all outputs
    print("\n" + "=" * 60)
    print("FULL DOCUMENT OUTPUT SUMMARY:")
    print("=" * 60)
    combined_output = "\n\n--- PAGE BREAK ---\n\n".join(all_outputs)
    print(f"Total pages processed: {total_pages}")
    print(f"Total combined output: {len(combined_output)} chars")
    print(f"\n--- TIMING ---")
    print(f"Model load time: {model_load_time:.1f} seconds")
    print(f"PDF processing time: {total_pdf_time:.1f} seconds")
    print(f"Average per page: {avg_page_time:.2f} seconds")
    print(f"Pages per minute: {60 / avg_page_time:.1f}" if avg_page_time > 0 else "N/A")
    print(f"Total time: {model_load_time + total_pdf_time:.1f} seconds")

    # Save full output to file
    output_file = pdf_path.replace('.pdf', '_ocr_output.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# OCR Output for {os.path.basename(pdf_path)}\n\n")
        f.write(f"- Pages: {total_pages}\n")
        f.write(f"- Total chars: {len(combined_output)}\n")
        f.write(f"- Processing time: {total_pdf_time:.1f}s\n")
        f.write(f"- Avg per page: {avg_page_time:.2f}s\n\n")
        f.write("---\n\n")
        for i, page_output in enumerate(all_outputs):
            f.write(f"## Page {i + 1}\n\n")
            f.write(page_output)
            f.write("\n\n---\n\n")
    print(f"\nFull output saved to: {output_file}")

else:
    print("\n[INFO] No PDF provided. Model loaded successfully!")
    print(f"       Model load time: {model_load_time:.1f} seconds")
    print("       Run with: python3.12 test_model.py /path/to/test.pdf")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
