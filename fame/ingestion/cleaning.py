from __future__ import annotations

import re


def remove_reference_section(text: str) -> str:
    """
    Cut off text from common reference-section headings onward.
    """
    reference_patterns = [
        r"\n\s*(?:Acknowledgements|References|Bibliography|Citations|Literature Cited|Works Cited)\s*\n",
        r"\n\s*(?:ACKNOWLEDGEMENTS|REFERENCES|BIBLIOGRAPHY|CITATIONS|LITERATURE CITED|WORKS CITED)\s*\n",
    ]

    for pattern in reference_patterns:
        match = re.search(pattern, text)
        if match:
            return text[: match.start()].strip()
    return text.strip()


def remove_inline_citations(text: str) -> str:
    """
    Remove common inline citations:
      - numeric citations like [1], [2, 5], (3)
      - author-year style like (Smith et al., 2020) (heuristic)
    """
    # numeric citations
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]|\(\d+(?:,\s*\d+)*\)", "", text)

    # author-year (heuristic; may remove some parenthetical content)
    text = re.sub(r"\([^)]*?\d{4}[^)]*?\)", "", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_noise(text: str) -> str:
    """
    Cleans raw extracted text by removing common non-content elements such as:
      - reference section
      - citations
      - header/footer patterns
      - page numbers
      - excess whitespace
    """
    if not isinstance(text, str):
        return ""

    text = remove_reference_section(text)
    text = remove_inline_citations(text)

    flags = re.MULTILINE | re.IGNORECASE

    # Remove common header/footer patterns, e.g., 'Chapter X'
    text = re.sub(r"^(?:Chapter|Section|Appendix)\s+\w+\s*\n?", "", text, flags=flags)

    # Remove page numbers like '- 5 -', 'Page 5 of 12'
    text = re.sub(r"-\s*\d+\s*-", "", text)
    text = re.sub(r"(Page|Pg\.?)\s+\d+\s*(?:of\s+\d+)?", "", text, flags=flags)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=flags)

    # Remove bracket/brace markers
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)
    text = re.sub(r"\{\s*\d+\s*\}", "", text)

    # Normalize whitespace
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^\s*|\s*$", "", text, flags=re.MULTILINE)

    return text.strip()
