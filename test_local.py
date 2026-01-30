"""
Local testing script for the DeepSeek-OCR-2 handler
Run this before deploying to RunPod to verify everything works.

Usage:
    python test_local.py path/to/your/label.pdf
"""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_extraction():
    """Test the extraction module independently"""
    from extraction import extract_indications_and_usage_fda, clean_ocr_artifacts

    # Sample text from a pharmaceutical label
    sample_text = """
    HIGHLIGHTS OF PRESCRIBING INFORMATION

    --- INDICATIONS AND USAGE ---

    KEYTRUDA is a programmed death receptor-1 (PD-1)-blocking antibody indicated for:

    Melanoma
    - Treatment of patients with unresectable or metastatic melanoma.
    - Adjuvant treatment following complete resection of melanoma.

    Non-Small Cell Lung Cancer (NSCLC)
    - Treatment of patients with metastatic NSCLC whose tumors express PD-L1
      (Tumor Proportion Score [TPS] ≥1%) as determined by an FDA-approved test.

    --- DOSAGE AND ADMINISTRATION ---
    """

    cleaned = clean_ocr_artifacts(sample_text)
    indications = extract_indications_and_usage_fda(cleaned)

    print("=" * 60)
    print("EXTRACTION TEST")
    print("=" * 60)
    print("Input text length:", len(sample_text))
    print("Extracted indications length:", len(indications))
    print("\nExtracted content:")
    print(indications[:500] if indications else "No indications found")
    print("=" * 60)

    return len(indications) > 50


def test_handler_mock():
    """Test the handler with mock data"""
    import json

    print("\n" + "=" * 60)
    print("HANDLER MOCK TEST")
    print("=" * 60)

    # Test input validation
    from handler import handler

    # Test missing input
    result = handler({"input": {}})
    print("Missing input test:", "PASS" if "error" in result else "FAIL")

    print("=" * 60)


def test_full_pipeline(pdf_path):
    """Test the full pipeline with a real PDF"""
    print("\n" + "=" * 60)
    print("FULL PIPELINE TEST")
    print("=" * 60)

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        return False

    # Import handler
    from handler import process_pdf

    print(f"Processing: {pdf_path}")
    print("This may take a few minutes...")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = process_pdf(pdf_bytes)

    print("\nResults:")
    print(f"- Pages processed: {result['page_count']}")
    print(f"- Full text length: {len(result['full_text'])} chars")
    print(f"- Indications length: {len(result['indications_and_usage'])} chars")

    print("\nIndications preview:")
    print("-" * 40)
    print(result['indications_and_usage'][:1000])
    print("-" * 40)

    # Save output
    output_path = pdf_path.replace('.pdf', '_indications.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result['indications_and_usage'])
    print(f"\nFull indications saved to: {output_path}")

    return True


if __name__ == "__main__":
    print("DeepSeek-OCR-2 Local Test Suite")
    print("=" * 60)

    # Test 1: Extraction module
    extraction_ok = test_extraction()
    print(f"\nExtraction test: {'PASS' if extraction_ok else 'FAIL'}")

    # Test 2: If PDF provided, test full pipeline
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        pipeline_ok = test_full_pipeline(pdf_path)
        print(f"\nFull pipeline test: {'PASS' if pipeline_ok else 'FAIL'}")
    else:
        print("\nTo test the full pipeline, run:")
        print("  python test_local.py path/to/your/label.pdf")
        print("\nNote: Full pipeline requires GPU with vLLM installed.")
