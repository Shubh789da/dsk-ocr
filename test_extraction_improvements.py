
import os
from extraction import extract_indications_and_usage_fda

def test_extraction(filename):
    print(f"\nTesting extraction on: {filename}")
    if not os.path.exists(filename):
        print("File not found.")
        return

    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    extracted = extract_indications_and_usage_fda(content)
    print(f"Original Length: {len(content)}")
    print(f"Extracted Length: {len(extracted)}")
    
    if extracted:
        print("--- Start of Extracted ---")
        print(extracted[:200])
        print("--- End of Extracted ---")
        print(extracted[-200:])
    else:
        print("Nothing extracted.")

if __name__ == "__main__":
    files = [
        "017381s050lbl.pdf.txt",
        "125057s417lbl.pdf.txt"
    ]
    for f in files:
        test_extraction(f)
