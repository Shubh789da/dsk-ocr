
import re
import os

def test_safe_logic():
    files = [
        "021976s045_202895s020lbl.pdf.txt", # Prezista (FPI + TOC)
        "021108s015lbl.pdf.txt",            # Renova (No FPI)
        "125057s417lbl.pdf.txt"             # Humira (FPI + TOC)
    ]
    
    for fname in files:
        if not os.path.exists(fname):
            continue
            
        print(f"\nScanning {fname}...")
        with open(fname, 'r', encoding='utf-8') as f:
            context = f.read()
            
        # PROPOSED SAFE LOGIC
        fpi_iter = re.finditer(r'(?:^|\n)\s*FULL\s+PRESCRIBING\s+INFORMATION', context, re.IGNORECASE)
        search_start_index = 0
        found = False
        
        for match in fpi_iter:
            # Check next 50 chars for ": CONTENTS"
            snippet = context[match.end():match.end()+50]
            # Normalize to check
            snippet_norm = snippet.upper().replace('\n', ' ')
            
            print(f"  Match at {match.start()}... snippet: {repr(snippet)}")
            
            if ":" in snippet_norm and "CONTENTS" in snippet_norm:
                print("    -> SKIPPING (Contents detected)")
                continue
            
            print("    -> ACCEPTED as FPI Header")
            search_start_index = match.end()
            found = True
            break
            
        if not found:
            print("    -> No valid FPI Header found (Using 0)")
            
        print(f"  Effective Start Index: {search_start_index}")

if __name__ == "__main__":
    test_safe_logic()
