
import fitz  # PyMuPDF
import os

def inspect_pdf(pdf_path):
    print(f"--- Inspecting {pdf_path} ---")
    try:
        doc = fitz.open(pdf_path)
        text_content = ""
        # Inspect first 20 pages (labels can be long, but TOC/Highlights usually early)
        for i in range(min(20, len(doc))):
            page = doc.load_page(i)
            text = page.get_text("text")
            text_content += f"\n--- PAGE {i+1} ---\n{text}"
        
        # Save to a file for me to read via view_file
        output_path = pdf_path + ".txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        print(f"Saved text to {output_path}")
        
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")

if __name__ == "__main__":
    pdf_files = [
        "017381s050lbl.pdf",
        "125057s417lbl.pdf"
    ]
    for p in pdf_files:
        if os.path.exists(p):
            inspect_pdf(p)
        else:
            print(f"File not found: {p}")
