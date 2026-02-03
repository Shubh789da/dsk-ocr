
import os
import requests
import fitz  # PyMuPDF
from extraction import extract_indications_and_usage_fda

pdf_urls = [
    "https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/021108s015lbl.pdf",
    "https://www.accessdata.fda.gov/drugsatfda_docs/label/2000/21108lbl.pdf",
    "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/021976s045_202895s020lbl.pdf"
]

def download_pdf(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {output_path} ({len(response.content)} bytes)")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

import time

def test_urls():
    with open("results.log", "w", encoding="utf-8") as log:
        log.write("--- Starting Verification ---\n")
        
        for url in pdf_urls:
            filename = url.split('/')[-1]
            print(f"Processing {filename}...")
            log.write(f"\n--- LOOP: {filename} ---\n")
            
            try:
                if not download_pdf(url, filename):
                    log.write(f"Failed to download {filename}\n")
                    continue
                
                time.sleep(2)
                
                full_text = extract_text_from_pdf(filename)
                log.write(f"  PDF Text Length: {len(full_text)} chars\n")
                log.write(f"  Header Snippet: {repr(full_text[:100])}\n")
                
                if "RENOVA" in full_text:
                    log.write("  Identity: RENOVA detected.\n")
                elif "PREZISTA" in full_text:
                    log.write("  Identity: PREZISTA detected.\n")
                elif "Trizivir" in full_text:
                    log.write("  Identity: TRIZIVIR detected.\n")
                else:
                    log.write("  Identity: Unknown drug.\n")

                result = extract_indications_and_usage_fda(full_text)
                
                if result:
                    log.write(f"  [SUCCESS] Extracted {len(result)} chars\n")
                    cleaned_res = result.replace('\n', ' ').replace('\r', '')
                    log.write(f"  Snippet: {cleaned_res[:200]}...\n")
                    log.write(f"  End Snippet: ...{cleaned_res[-100:]}\n")
                else:
                    log.write("  [FAILURE] No INDICATIONS AND USAGE section found.\n")
            except Exception as e:
                log.write(f"CRASH in loop for {filename}: {e}\n")
        
        log.write("\n--- ALL DONE ---\n")

if __name__ == "__main__":
    test_urls()
