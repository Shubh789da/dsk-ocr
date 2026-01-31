#!/usr/bin/env python3.12
"""
Test extraction patterns against actual OCR output files.
No model loading required - just tests the regex patterns.

Usage:
    python3.12 test_extraction.py <markdown_file.mmd>
"""

import sys
import re

# Import the extraction function
from extraction import extract_indications_and_usage_fda

if len(sys.argv) < 2:
    print("Usage: python3.12 test_extraction.py <markdown_file.mmd>")
    sys.exit(1)

input_file = sys.argv[1]

print("=" * 60)
print("Extraction Pattern Test")
print("=" * 60)

# Read the file
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"\nLoaded: {input_file}")
    print(f"Total length: {len(content)} chars")
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Search for section headers
print("\n[1] Searching for INDICATIONS AND USAGE section headers...")

section_patterns = [
    (r'---\s*INDICATIONS\s+AND\s+USAGE\s*---', 'FDA highlights format'),
    (r'##\s*---?\s*INDICATIONS\s+AND\s+USAGE', 'Markdown ## header'),
    (r'\n\s*1\s+INDICATIONS\s+AND\s+USAGE', 'Numbered section'),
    (r'\n\s*INDICATIONS\s+AND\s+USAGE\s*\n', 'Standalone header'),
]

for pattern, desc in section_patterns:
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        print(f"  ✓ Found: {desc}")
        print(f"    Pattern: {pattern}")
        print(f"    Location: char {match.start()}")
        print(f"    Match: {match.group()[:50]}...")
    else:
        print(f"  ✗ Not found: {desc}")

# Try extraction
print("\n[2] Running extract_indications_and_usage_fda()...")

result = extract_indications_and_usage_fda(content)

if result:
    print(f"  ✓ Extraction successful!")
    print(f"  Length: {len(result)} chars")
    print(f"\n  First 1000 chars of extracted content:")
    print("-" * 60)
    print(result[:1000])
    print("-" * 60)

    # Check if it contains expected content
    if 'indicated' in result.lower():
        print("\n  ✓ Contains 'indicated' - looks like valid content")
    else:
        print("\n  ⚠ Does not contain 'indicated' - may be wrong section")

    # Save extracted content
    output_file = input_file.replace('.mmd', '_indications.txt').replace('.md', '_indications.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(result)
    print(f"\n  Saved extracted content to: {output_file}")

else:
    print("  ✗ Extraction returned empty!")
    print("\n  Debugging: Let's check what patterns match...")

    # Debug each pattern
    extraction_patterns = [
        (r'#{1,3}\s*---\s*INDICATIONS\s+AND\s+USAGE\s*---?\s*\n(.*?)(?=#{1,3}\s*---?\s*DOSAGE|#{1,3}\s+DOSAGE|## DOSAGE|\Z)',
         'Markdown with dashes'),
        (r'---\s*INDICATIONS\s+AND\s+USAGE\s*---\s*\n?(.*?)(?=---\s*DOSAGE|---\s*[A-Z]|DOSAGE AND ADMINISTRATION|\Z)',
         'Plain dashes'),
        (r'##\s+INDICATIONS\s+AND\s+USAGE\s*\n(.*?)(?=##\s+DOSAGE|##\s+[A-Z]{5,}|\Z)',
         'Markdown ## header'),
    ]

    for pattern, desc in extraction_patterns:
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            print(f"\n  Pattern '{desc}' matched!")
            print(f"  Extracted length: {len(extracted)} chars")
            if len(extracted) < 50:
                print(f"  Content: {extracted}")
            else:
                print(f"  First 200 chars: {extracted[:200]}...")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
