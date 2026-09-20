from __future__ import annotations

from pathlib import Path

from .cleaning import clean_noise

# from docx import Document
 


def load_pdf_text(file_path: str | Path) -> str:
    """
    Extract text from PDF using pypdf.
    """
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError(
            "pypdf is required for PDF ingestion. Install: python -m pip install -U pypdf"
        ) from e

    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    reader = PdfReader(str(p))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts)


def load_txt_text(file_path: str | Path, encoding: str = "utf-8") -> str:
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding=encoding, errors="ignore")


def load_docx_text(file_path: str | Path) -> str:
    """
    Extract text from .docx using python-docx.
    """
    try:
        from docx import Document
    except Exception as e:
        raise RuntimeError(
            "python-docx is required for DOCX ingestion. Install: python -m pip install -U python-docx"
        ) from e

    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    doc = Document(str(p))
    paras = [para.text for para in doc.paragraphs if para.text and para.text.strip()]
    return "\n".join(paras)


def load_and_clean(file_path: str | Path) -> str:
    """
    Load supported files and apply clean_noise().
    """
    p = Path(file_path).expanduser().resolve()
    ext = p.suffix.lower()

    if ext == ".pdf":
        raw = load_pdf_text(p)
    elif ext == ".docx":
        raw = load_docx_text(p)
    elif ext in {".txt", ".md", ".markdown", ".yaml", ".yml", ".json",
                 ".sql", ".ddl", ".graphql", ".gql", ".xml", ".xsd"}:
        raw = load_txt_text(p)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    return clean_noise(raw)
