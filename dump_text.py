
import fitz
import sys

def dump_pdf(filename):
    print(f"Dumping {filename}...")
    try:
        doc = fitz.open(filename)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        
        out_txt = filename + ".txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved to {out_txt}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dump_pdf(sys.argv[1])
    else:
        # Default debug
        dump_pdf("021108s015lbl.pdf")
        dump_pdf("21108lbl.pdf")
        dump_pdf("021976s045_202895s020lbl.pdf")
