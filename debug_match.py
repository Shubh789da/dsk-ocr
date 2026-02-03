
import re

def debug_context():
    filename = "125057s417lbl.pdf.txt"
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'FULL\s+PRESCRIBING\s+INFORMATION(?!\s*:\s*CONTENTS)'
    match = re.search(pattern, content, re.IGNORECASE)
    
    if match:
        print(f"Match found at: {match.start()}-{match.end()}")
        print(f"Match content: '{match.group()}'")
        # Print surrounding text
        start = max(0, match.start() - 50)
        end = min(len(content), match.end() + 50)
        print(f"Surrounding text: {repr(content[start:end])}")
    else:
        print("No match found.")

if __name__ == "__main__":
    debug_context()
