#!/usr/bin/env python3.12
"""
Debug script to diagnose handler.py issues:
1. Why only 7 pages processed?
2. Why INDICATIONS AND USAGE extraction returns empty?
"""

# CRITICAL: Set environment variables FIRST
import os
os.environ['VLLM_USE_V1'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import io
import fitz  # PyMuPDF
from PIL import Image

# Check command line args
if len(sys.argv) < 2:
    print("Usage: python3.12 debug_handler.py <pdf_path_or_url>")
    sys.exit(1)

input_path = sys.argv[1]
is_url = input_path.lower().startswith(('http://', 'https://'))

print("=" * 60)
print("DEBUG: Checking PDF page count and OCR output")
print("=" * 60)

# Step 1: Check PDF page count
print("\n[STEP 1] Checking PDF page count...")

try:
    if is_url:
        import requests
        print(f"Downloading: {input_path}")
        response = requests.get(input_path, timeout=120)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    else:
        print(f"Opening: {input_path}")
        pdf_doc = fitz.open(input_path)

    total_pages = len(pdf_doc)
    print(f"✓ PDF has {total_pages} pages")
    pdf_doc.close()

except Exception as e:
    print(f"✗ Failed to open PDF: {e}")
    sys.exit(1)

# Step 2: Check if extraction patterns work
print("\n[STEP 2] Testing extraction patterns on sample text...")

from extraction import extract_indications_and_usage_fda

# Test with known FDA label format
test_texts = [
    # Format 1: --- markers
    """
--- INDICATIONS AND USAGE ---
DRUG X is indicated for:
- Treatment of condition A in adults
- Prevention of condition B in patients with risk factors
--- DOSAGE AND ADMINISTRATION ---
""",
    # Format 2: Numbered section
    """
1 INDICATIONS AND USAGE
DRUG Y is a kinase inhibitor indicated for the treatment of:
- Locally advanced or metastatic cancer

2 DOSAGE AND ADMINISTRATION
""",
    # Format 3: ALL CAPS header
    """
INDICATIONS AND USAGE
Drug Z is indicated for treatment of moderate to severe condition
in adult patients who have had an inadequate response.

DOSAGE AND ADMINISTRATION
""",
]

for i, text in enumerate(test_texts, 1):
    result = extract_indications_and_usage_fda(text)
    print(f"  Pattern test {i}: {'✓ Found' if result else '✗ Empty'} ({len(result)} chars)")

# Step 3: Run OCR on first few pages and check output
print("\n[STEP 3] Running OCR on first 3 pages to check output format...")

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("✗ No CUDA GPU available!")
    sys.exit(1)

# Import vLLM and model components
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
from deepseek_ocr2 import DeepseekOCR2ForCausalLM
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCR2Processor

# Register model
ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)

# Load model with conservative settings
print("\nLoading model...")
llm = LLM(
    model='deepseek-ai/DeepSeek-OCR-2',
    hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    max_num_seqs=10,
    limit_mm_per_prompt={"image": 1},
    block_size=16,
    enforce_eager=True,
    trust_remote_code=True,
    swap_space=0,
    tensor_parallel_size=1,
    disable_mm_preprocessor_cache=True,
    enable_prefix_caching=False,
)
print("✓ Model loaded")

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
    max_tokens=4096,
    logits_processors=logits_processors,
    skip_special_tokens=False,
    include_stop_str_in_output=True,
)

# Open PDF and process first 3 pages
if is_url:
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
else:
    with open(input_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

pages_to_check = min(3, len(pdf_doc))
print(f"\nProcessing {pages_to_check} pages...")

# Convert pages to images
dpi = 108
zoom = dpi / 72.0
matrix = fitz.Matrix(zoom, zoom)

images = []
for page_num in range(pages_to_check):
    page = pdf_doc[page_num]
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    images.append(img)
    print(f"  Page {page_num + 1}: {img.size}")

pdf_doc.close()

# Prepare batch inputs
def prepare_input(img):
    return {
        "prompt": PROMPT,
        "multi_modal_data": {
            "image": processor.tokenize_with_images(
                images=[img],
                bos=True,
                eos=True,
                cropping=True
            )
        },
    }

batch_inputs = [prepare_input(img) for img in images]

# Run OCR
print("\nRunning OCR inference...")
outputs_list = llm.generate(batch_inputs, sampling_params=sampling_params)

# Clean OCR output
import re
def clean_ocr_output(text):
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

# Show raw OCR output
print("\n" + "=" * 60)
print("RAW OCR OUTPUT (first 3 pages)")
print("=" * 60)

full_text_parts = []
for i, output in enumerate(outputs_list):
    text = output.outputs[0].text
    cleaned = clean_ocr_output(text)
    full_text_parts.append(cleaned)

    print(f"\n--- PAGE {i + 1} (first 1000 chars) ---")
    print(cleaned[:1000])
    print(f"... (Total: {len(cleaned)} chars)")

# Combine and test extraction
combined_text = '\n\n--- Page Break ---\n\n'.join(full_text_parts)

print("\n" + "=" * 60)
print("EXTRACTION TEST")
print("=" * 60)

# Check if "INDICATIONS" appears in the text
if "INDICATIONS" in combined_text.upper():
    print("✓ Found 'INDICATIONS' in OCR output")

    # Find where it appears
    upper_text = combined_text.upper()
    idx = upper_text.find("INDICATIONS")
    context = combined_text[max(0, idx-50):idx+500]
    print(f"\nContext around 'INDICATIONS':")
    print("-" * 40)
    print(context)
    print("-" * 40)
else:
    print("✗ 'INDICATIONS' NOT found in OCR output")
    print("  The extraction patterns cannot find what's not there!")
    print("  This means either:")
    print("    1. The INDICATIONS section is on a later page (not in first 3)")
    print("    2. OCR didn't recognize the text correctly")

# Try extraction
result = extract_indications_and_usage_fda(combined_text)
print(f"\nExtraction result: {len(result)} chars")

if result:
    print("\nExtracted text (first 500 chars):")
    print("-" * 40)
    print(result[:500])
else:
    print("\n✗ Extraction returned empty!")
    print("  Possible reasons:")
    print("    1. 'INDICATIONS' section not in first 3 pages")
    print("    2. OCR output format doesn't match regex patterns")
    print("    3. Section text is too short (<100 chars)")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
