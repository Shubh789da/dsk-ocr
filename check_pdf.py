#!/usr/bin/env python3.12
"""
Lightweight script to check PDF properties without loading the model.
Use this to verify:
1. How many pages the PDF has
2. Whether INDICATIONS section exists (by text extraction, not OCR)

Usage:
    python3.12 check_pdf.py <pdf_path_or_url>
"""

import sys
import fitz  # PyMuPDF

if len(sys.argv) < 2:
    print("Usage: python3.12 check_pdf.py <pdf_path_or_url>")
    sys.exit(1)

input_path = sys.argv[1]
is_url = input_path.lower().startswith(('http://', 'https://'))

print("=" * 60)
print("PDF Inspection Tool")
print("=" * 60)

# Open PDF
try:
    if is_url:
        import requests
        print(f"\nDownloading: {input_path}")
        response = requests.get(input_path, timeout=120)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        print(f"Downloaded: {len(pdf_bytes):,} bytes")
    else:
        print(f"\nOpening: {input_path}")
        pdf_doc = fitz.open(input_path)

except Exception as e:
    print(f"\n[ERROR] Failed to open PDF: {e}")
    sys.exit(1)

# Basic info
print(f"\n[INFO] PDF Statistics:")
print(f"  Total pages: {len(pdf_doc)}")

# Get metadata
metadata = pdf_doc.metadata
if metadata:
    print(f"  Title: {metadata.get('title', 'N/A')}")
    print(f"  Author: {metadata.get('author', 'N/A')}")

# Extract text from each page to find INDICATIONS
print(f"\n[SEARCH] Looking for 'INDICATIONS' in text...")

all_text = []
indications_found = False
indications_page = None

for page_num in range(len(pdf_doc)):
    page = pdf_doc[page_num]
    text = page.get_text()
    all_text.append(text)

    if not indications_found and 'INDICATIONS' in text.upper():
        indications_found = True
        indications_page = page_num + 1

full_text = '\n'.join(all_text)

if indications_found:
    print(f"  ✓ Found 'INDICATIONS' on page {indications_page}")

    # Find context
    upper_text = full_text.upper()
    idx = upper_text.find('INDICATIONS')
    context = full_text[max(0, idx-20):idx+500]

    print(f"\n[CONTEXT] Text around 'INDICATIONS':")
    print("-" * 40)
    print(context)
    print("-" * 40)
else:
    print("  ✗ 'INDICATIONS' NOT found in any page!")
    print("  This PDF may not contain an INDICATIONS AND USAGE section.")
    print("  Make sure you're using a Full Prescribing Information PDF,")
    print("  not just the Highlights summary.")

# Test extraction
print(f"\n[TEST] Testing extraction patterns...")

from extraction import extract_indications_and_usage_fda

result = extract_indications_and_usage_fda(full_text)

if result:
    print(f"  ✓ Extraction successful: {len(result)} chars")
    print(f"\n[EXTRACTED] First 500 chars:")
    print("-" * 40)
    print(result[:500])
    print("-" * 40)
else:
    print("  ✗ Extraction returned empty!")
    print("  The regex patterns may not match this PDF's format.")
    print("  Try running with DEBUG_OCR=true python3.12 handler.py <pdf>")

total_pages = len(pdf_doc)
pdf_doc.close()

print("\n" + "=" * 60)
print("Inspection complete!")
print("=" * 60)

# Summary
print(f"\nSUMMARY:")
print(f"  Pages: {total_pages}")
print(f"  INDICATIONS found: {'Yes (page ' + str(indications_page) + ')' if indications_found else 'No'}")
print(f"  Extraction result: {len(result)} chars")
