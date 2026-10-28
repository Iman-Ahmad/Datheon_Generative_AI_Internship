from pathlib import Path
import re

def clean_text(text: str) -> str:
    """Remove layout noise that confuses the model and wastes time."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue                                    # drop empty lines
        if re.fullmatch(r"(\d{1,3}\s?%\s*)+", line):    # skill-bar percentages like "80% 60%"
            continue
        lines.append(line)
    return "\n".join(lines)

def read_document(path: str) -> str:
    """Return the plain text of a .txt, .md, .pdf or .docx file."""
    p = Path(path)
    suffix = p.suffix.lower()

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if suffix in {".txt", ".md"}:
        text = p.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
    elif suffix == ".docx":
        from docx import Document
        text = "\n".join(paragraph.text for paragraph in Document(str(p)).paragraphs)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .txt, .pdf or .docx")

    text = clean_text(text)

    if not text.strip():
        raise ValueError("No text could be extracted (is it a scanned PDF? OCR is not supported)")
    return text

