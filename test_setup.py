#!/usr/bin/env python3.12
"""
Test script to validate DeepSeek-OCR-2 setup on RunPod
Run this BEFORE building Docker image to catch issues early.

Usage:
    python3.12 test_setup.py              # Test imports and model loading
    python3.12 test_setup.py test.pdf     # Test with actual PDF
"""

import sys
import os

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def print_ok(text):
    print(f"  [OK] {text}")

def print_fail(text):
    print(f"  [FAIL] {text}")

def print_info(text):
    print(f"  [INFO] {text}")


def test_basic_imports():
    """Test basic Python imports"""
    print_header("Testing Basic Imports")

    try:
        import torch
        print_ok(f"torch {torch.__version__}")
        print_info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print_info(f"GPU: {torch.cuda.get_device_name(0)}")
            print_info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError as e:
        print_fail(f"torch: {e}")
        return False

    try:
        import vllm
        print_ok(f"vllm {vllm.__version__}")
        if vllm.__version__ != "0.8.5":
            print_info(f"WARNING: Expected vLLM 0.8.5, got {vllm.__version__}")
    except ImportError as e:
        print_fail(f"vllm: {e}")
        return False

    try:
        from PIL import Image
        print_ok("Pillow")
    except ImportError as e:
        print_fail(f"Pillow: {e}")
        return False

    try:
        import fitz
        print_ok("PyMuPDF (fitz)")
    except ImportError as e:
        print_fail(f"PyMuPDF: {e}")
        return False

    try:
        import transformers
        print_ok(f"transformers {transformers.__version__}")
    except ImportError as e:
        print_fail(f"transformers: {e}")
        return False

    try:
        import einops
        print_ok("einops")
    except ImportError as e:
        print_fail(f"einops: {e}")
        return False

    try:
        import runpod
        print_ok("runpod")
    except ImportError as e:
        print_fail(f"runpod: {e}")
        return False

    return True


def test_vllm_multimodal():
    """Test vLLM multimodal imports (critical for DeepSeek-OCR-2)"""
    print_header("Testing vLLM Multimodal Imports")

    try:
        from vllm.config import VllmConfig
        print_ok("VllmConfig")
    except ImportError as e:
        print_fail(f"VllmConfig: {e}")
        return False

    try:
        from vllm.multimodal.processing import BaseMultiModalProcessor, PromptUpdate
        print_ok("BaseMultiModalProcessor, PromptUpdate")
    except ImportError as e:
        print_fail(f"PromptUpdate: {e}")
        print_info("This usually means wrong vLLM version. Need vLLM 0.8.5")
        return False

    try:
        from vllm.multimodal.inputs import MultiModalDataDict, MultiModalFieldConfig
        print_ok("MultiModalDataDict, MultiModalFieldConfig")
    except ImportError as e:
        print_fail(f"MultiModalInputs: {e}")
        return False

    try:
        from vllm.model_executor.models.registry import ModelRegistry
        print_ok("ModelRegistry")
    except ImportError as e:
        print_fail(f"ModelRegistry: {e}")
        return False

    return True


def test_local_modules():
    """Test local project modules"""
    print_header("Testing Local Modules")

    try:
        from config import IMAGE_SIZE, BASE_SIZE, CROP_MODE, PROMPT
        print_ok(f"config (IMAGE_SIZE={IMAGE_SIZE}, BASE_SIZE={BASE_SIZE})")
    except ImportError as e:
        print_fail(f"config: {e}")
        return False

    try:
        from extraction import extract_indications_and_usage_fda, clean_ocr_artifacts
        print_ok("extraction")
    except ImportError as e:
        print_fail(f"extraction: {e}")
        return False

    try:
        from process.image_process import DeepseekOCR2Processor
        print_ok("process.image_process")
    except ImportError as e:
        print_fail(f"process.image_process: {e}")
        return False

    try:
        from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
        print_ok("process.ngram_norepeat")
    except ImportError as e:
        print_fail(f"process.ngram_norepeat: {e}")
        return False

    try:
        from deepencoderv2.sam_vary_sdpa import build_sam_vit_b
        print_ok("deepencoderv2.sam_vary_sdpa")
    except ImportError as e:
        print_fail(f"deepencoderv2.sam_vary_sdpa: {e}")
        return False

    try:
        from deepencoderv2.qwen2_d2e import build_qwen2_decoder_as_encoder
        print_ok("deepencoderv2.qwen2_d2e")
    except ImportError as e:
        print_fail(f"deepencoderv2.qwen2_d2e: {e}")
        return False

    try:
        from deepseek_ocr2 import DeepseekOCR2ForCausalLM
        print_ok("deepseek_ocr2.DeepseekOCR2ForCausalLM")
    except ImportError as e:
        print_fail(f"deepseek_ocr2: {e}")
        return False

    return True


def test_model_loading():
    """Test actual model loading (takes time, downloads model)"""
    print_header("Testing Model Loading")
    print_info("This will download the model (~6GB) if not cached...")

    try:
        from vllm import LLM
        from vllm.model_executor.models.registry import ModelRegistry
        from deepseek_ocr2 import DeepseekOCR2ForCausalLM

        # Register model
        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)
        print_ok("Model registered")

        # Load model
        print_info("Loading model (this may take a few minutes)...")
        llm = LLM(
            model="deepseek-ai/DeepSeek-OCR-2",
            hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
            trust_remote_code=True,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            enforce_eager=True,  # Faster for testing
        )
        print_ok("Model loaded successfully!")

        return llm

    except Exception as e:
        print_fail(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_pdf_processing(pdf_path, llm=None):
    """Test PDF processing with actual file"""
    print_header(f"Testing PDF Processing: {pdf_path}")

    if not os.path.exists(pdf_path):
        print_fail(f"PDF file not found: {pdf_path}")
        return False

    try:
        # Read PDF
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        print_ok(f"PDF read: {len(pdf_bytes) / 1024 / 1024:.2f} MB")

        # Convert to images
        import fitz
        from PIL import Image
        import io

        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print_ok(f"PDF has {pdf_doc.page_count} pages")

        # Test first page conversion
        page = pdf_doc[0]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        print_ok(f"Page 1 converted to image: {img.size}")

        pdf_doc.close()

        # If model is loaded, test actual OCR
        if llm:
            print_info("Testing OCR inference...")
            from handler import process_pdf
            result = process_pdf(pdf_bytes)
            print_ok(f"OCR complete: {result['page_count']} pages processed")
            print_info(f"Indications length: {len(result['indications_and_usage'])} chars")

            if result['indications_and_usage']:
                print("\n--- INDICATIONS PREVIEW (first 500 chars) ---")
                print(result['indications_and_usage'][:500])
                print("---")

        return True

    except Exception as e:
        print_fail(f"PDF processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print(" DeepSeek-OCR-2 Setup Validation")
    print("=" * 60)

    # Track results
    results = {}

    # Test 1: Basic imports
    results["basic_imports"] = test_basic_imports()

    # Test 2: vLLM multimodal
    results["vllm_multimodal"] = test_vllm_multimodal()

    # Test 3: Local modules
    results["local_modules"] = test_local_modules()

    # Summary so far
    print_header("Import Tests Summary")
    all_passed = all(results.values())
    for test, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test}")

    if not all_passed:
        print("\n[!] Fix import errors before proceeding.")
        print("    Run: pip install <missing_package>")
        sys.exit(1)

    # Test 4: Model loading (optional, takes time)
    print_header("Model Loading Test")
    response = input("Test model loading? This downloads ~6GB model. (y/N): ")

    llm = None
    if response.lower() == 'y':
        llm = test_model_loading()
        results["model_loading"] = llm is not None
    else:
        print_info("Skipped model loading test")
        results["model_loading"] = None

    # Test 5: PDF processing (if PDF provided)
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        results["pdf_processing"] = test_pdf_processing(pdf_path, llm)

    # Final summary
    print_header("Final Summary")
    for test, passed in results.items():
        if passed is None:
            status = "[SKIP]"
        elif passed:
            status = "[PASS]"
        else:
            status = "[FAIL]"
        print(f"  {status} {test}")

    # Recommendation
    print_header("Next Steps")
    if all(v for v in results.values() if v is not None):
        print("  All tests passed! You can now build the Docker image:")
        print("  ")
        print("    docker build -t shubh0078/dsk-ocr-endpoint:v1.2 .")
        print("    docker push shubh0078/dsk-ocr-endpoint:v1.2")
    else:
        print("  Some tests failed. Fix the issues above before building Docker.")


if __name__ == "__main__":
    main()
