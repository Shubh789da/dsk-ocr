"""
Extraction methods for pharmaceutical label INDICATIONS AND USAGE
Adapted for RunPod serverless endpoint
"""

import re
from typing import Dict, List


def extract_indications_and_usage_fda(context: str) -> str:
    """
    Extract INDICATIONS AND USAGE section from FDA pharmaceutical labels.
    Handles multiple FDA label formats.

    Args:
        context: Full pharmaceutical label text

    Returns:
        Extracted INDICATIONS AND USAGE section text
    """

    # Try multiple patterns in order of specificity
    patterns = [
        # Pattern 1: FDA highlights format with --- markers
        (r'---\s*INDICATIONS AND USAGE\s*---\s*(.*?)(?=---[A-Z]|DOSAGE AND ADMINISTRATION|FULL PRESCRIBING INFORMATION|\Z)',
         re.DOTALL | re.MULTILINE),

        # Pattern 2: Numbered section format (1 INDICATIONS AND USAGE)
        (r'(?:^|\n)\s*1\s+INDICATIONS AND USAGE\s*(.*?)(?=\n\s*2\s+DOSAGE|\n\s*\d+\s+[A-Z]{5,}|\Z)',
         re.DOTALL | re.MULTILINE),

        # Pattern 3: Numbered with decimal (1.0 INDICATIONS AND USAGE)
        (r'(?:^|\n)\s*1\.0?\s+INDICATIONS AND USAGE\s*(.*?)(?=\n\s*2\.|\Z)',
         re.DOTALL | re.MULTILINE),

        # Pattern 4: ALL CAPS header
        (r'(?:^|\n)\s*INDICATIONS AND USAGE\s*\n(.*?)(?=\n\s*[A-Z\s]{15,}\n|\Z)',
         re.DOTALL | re.MULTILINE),

        # Pattern 5: Generic catch-all
        (r'INDICATIONS?(?:\s+AND)?\s+USAGE?\s*:?\s*(.*?)(?=\n[A-Z\s]{10,}\n|DOSAGE|WARNINGS|\Z)',
         re.DOTALL | re.IGNORECASE),
    ]

    for pattern, flags in patterns:
        match = re.search(pattern, context, flags)
        if match:
            section = match.group(1).strip()
            # Filter out if too short (likely false positive)
            if len(section) > 100:
                return section

    return ""


def parse_indications_list(indications_text: str) -> List[Dict[str, str]]:
    """
    Parse INDICATIONS AND USAGE text into structured list.

    Args:
        indications_text: Raw text of indications section

    Returns:
        List of dicts with {disease, population, approval_type}
    """

    indications = []

    lines = indications_text.split('\n')
    current_disease = None
    current_details = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this is a disease header (all caps or title case, standalone)
        if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+\([A-Z]+\))?$', line):
            # Save previous disease if exists
            if current_disease:
                indications.append({
                    'disease': current_disease,
                    'population': '\n'.join(current_details),
                    'approval_type': _detect_approval_type('\n'.join(current_details))
                })

            current_disease = line
            current_details = []

        # Check for bulleted indication
        elif line.startswith('-') or line.startswith('•'):
            if current_disease:
                current_details.append(line[1:].strip())
            else:
                # Extract disease from the line itself
                disease_match = re.search(r'(?:treatment of|indicated for)(.*?)(?:in|with|whose)', line, re.IGNORECASE)
                if disease_match:
                    disease = disease_match.group(1).strip()
                    indications.append({
                        'disease': disease,
                        'population': line[1:].strip(),
                        'approval_type': _detect_approval_type(line)
                    })

        else:
            # Continuation of current indication
            if current_disease:
                current_details.append(line)

    # Don't forget last disease
    if current_disease and current_details:
        indications.append({
            'disease': current_disease,
            'population': '\n'.join(current_details),
            'approval_type': _detect_approval_type('\n'.join(current_details))
        })

    return indications


def _detect_approval_type(text: str) -> str:
    """Detect if indication has accelerated approval or limitations"""

    # Look for footnote markers
    if any(marker in text for marker in ['¹', '(1)', '^1', '²', '(2)', '^2', '³', '(3)', '^3']):
        return 'Accelerated Approval'

    # Look for "Limitations of Use"
    if 'limitations of use' in text.lower():
        return 'Limited Use'

    return 'Full Approval'


def clean_ocr_artifacts(text: str) -> str:
    """
    Clean common OCR artifacts from pharmaceutical labels.

    Args:
        text: Raw OCR text

    Returns:
        Cleaned text
    """

    # Remove excessive whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    # Fix common OCR errors in pharmaceutical text
    replacements = {
        r'\bO(?=\d)': '0',  # O instead of 0
        r'(?<=\d)O\b': '0',
        r'\bl(?=\d)': '1',  # lowercase l instead of 1
        r'\bmg\s*/\s*mL': 'mg/mL',  # Fix spacing in units
        r'\bm\s+g\b': 'mg',
        r'\bm\s+L\b': 'mL',
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # Fix line breaks in middle of words (common OCR issue)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)

    # Remove \r characters (Windows line endings)
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    return text


def extract_section_robust(context: str, section_name: str) -> str:
    """
    Robust section extraction for pharmaceutical labels.

    Args:
        context: Full label text
        section_name: Section to extract (e.g., 'indications', 'dosage', 'warnings')

    Returns:
        Extracted section text
    """

    section_patterns = {
        'indications': [
            r'---\s*INDICATIONS AND USAGE\s*---\s*(.*?)(?=---|\nDOSAGE|\nFULL PRES|\Z)',
            r'(?:^|\n)\s*1\s+INDICATIONS AND USAGE\s*(.*?)(?=\n\s*2\s|\Z)',
            r'INDICATIONS AND USAGE\s*\n(.*?)(?=\n[A-Z\s]{15,}\n|DOSAGE|\Z)',
        ],
        'dosage': [
            r'---\s*DOSAGE AND ADMINISTRATION\s*---\s*(.*?)(?=---|\Z)',
            r'(?:^|\n)\s*2\s+DOSAGE AND ADMINISTRATION\s*(.*?)(?=\n\s*3\s|\Z)',
            r'DOSAGE AND ADMINISTRATION\s*\n(.*?)(?=\n[A-Z\s]{15,}\n|\Z)',
        ],
        'warnings': [
            r'---\s*WARNINGS AND PRECAUTIONS\s*---\s*(.*?)(?=---|\Z)',
            r'(?:^|\n)\s*5\s+WARNINGS AND PRECAUTIONS\s*(.*?)(?=\n\s*6\s|\Z)',
            r'WARNINGS AND PRECAUTIONS\s*\n(.*?)(?=\n[A-Z\s]{15,}\n|\Z)',
        ],
        'adverse': [
            r'---\s*ADVERSE REACTIONS\s*---\s*(.*?)(?=---|\Z)',
            r'(?:^|\n)\s*6\s+ADVERSE REACTIONS\s*(.*?)(?=\n\s*7\s|\Z)',
            r'ADVERSE REACTIONS\s*\n(.*?)(?=\n[A-Z\s]{15,}\n|\Z)',
        ],
    }

    patterns = section_patterns.get(section_name.lower(), [])

    for pattern in patterns:
        match = re.search(pattern, context, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if match:
            section = match.group(1).strip()
            if len(section) > 50:  # Minimum length check
                return section

    return ""
