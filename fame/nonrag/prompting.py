from __future__ import annotations

import re
from xml.etree import ElementTree as ET
from pathlib import Path
from typing import Dict, Optional

from fame.utils.dirs import build_paths


DEFAULT_SS_NONRAG_PROMPT = """You are an expert Software Product Line (SPL) Engineer.

TASK:
Using ONLY the evidence below, construct a Feature Model for:
ROOT FEATURE: {root_feature}
DOMAIN: {domain}

OUTPUT FORMAT:
Return ONLY the XML feature model (no markdown, no explanation).

EVIDENCE:
{context}
"""


def serialize_high_level_features(data: Optional[Dict[str, str]]) -> str:
    """
    Convert a dict of feature name -> description into an XML block.
    Returns empty string if data is empty or None.
    """
    if not data:
        return ""

    root = ET.Element("high_level_features_guidance")
    for name, description in data.items():
        feature_elem = ET.SubElement(root, "feature", name=str(name), abstract="true")
        desc_elem = ET.SubElement(feature_elem, "description")
        desc_elem.text = str(description or "")

    return ET.tostring(root, encoding="unicode")


def load_ss_nonrag_prompt(prompt_path: Optional[str | Path] = None) -> str:
    """
    If prompt_path is provided, load it.
    Else try repo prompt convention: prompts/fm_extraction_prompt.txt
    Else fallback to DEFAULT_SS_NONRAG_PROMPT.
    """
    if prompt_path:
        p = Path(prompt_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Prompt file not found: {p}")
        return p.read_text(encoding="utf-8")

    paths = build_paths()
    candidate = paths.prompts / "fm_extraction_prompt.txt"
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")

    return DEFAULT_SS_NONRAG_PROMPT


def render_ss_nonrag_prompt(
    *,
    root_feature: str,
    domain: str,
    context: str,
    prompt_template: str,
    extra_placeholders: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> str:
    values: Dict[str, str] = {
        "root_feature": root_feature,
        "domain": domain,
        "context": context,
    }
    if extra_placeholders:
        values.update({k: str(v) for k, v in extra_placeholders.items()})
    return render_prompt_template(prompt_template, values=values, strict=strict)


def render_prompt_template(template: str, *, values: Dict[str, str], strict: bool = True) -> str:
    """
    Replace placeholders in a prompt template using both styles:
      - {{PLACEHOLDER}} (uppercase convention)
      - {placeholder}   (format-style)
    Raises if any placeholders remain (when strict=True).
    """
    # normalize values for flexible matching
    repl: Dict[str, str] = {}
    for k, v in values.items():
        v = "" if v is None else str(v)
        repl[k] = v
        repl[k.upper()] = v
        repl[k.lower()] = v

    def replace_double(m: re.Match) -> str:
        key = m.group(1)
        return repl.get(key, m.group(0))

    def replace_single(m: re.Match) -> str:
        key = m.group(1)
        return repl.get(key, m.group(0))

    # Capture which placeholders exist in the original template BEFORE substitution,
    # so the strict check only flags keys that were declared in the template but
    # not supplied — not patterns inside substituted values (e.g. {id} in OpenAPI paths).
    template_double = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", template))
    template_single = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", template))

    out = re.sub(r"\{\{([A-Z0-9_]+)\}\}", replace_double, template)
    out = re.sub(r"\{([a-zA-Z0-9_]+)\}", replace_single, out)

    if strict:
        remaining_double = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", out))
        remaining_single = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", out))
        missing = (template_double & remaining_double) | (template_single & remaining_single)
        if missing:
            raise ValueError(f"Unreplaced prompt placeholders: {', '.join(sorted(missing))}")

    return out
