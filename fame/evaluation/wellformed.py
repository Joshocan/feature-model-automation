from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import xmlschema
from lxml import etree


NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


@dataclass
class WellFormedResult:
    ok: bool
    errors: List[str]


def validate_feature_model(xml_path: Path | str, xsd_path: Path | str | None = None) -> WellFormedResult:
    xml_path = Path(xml_path)
    errors: List[str] = []

    # Basic text-boundary check
    text = xml_path.read_text(errors="ignore")
    if not text.lstrip().startswith("<?xml"):
        errors.append("Missing or misplaced XML declaration before content")
    if not text.rstrip().endswith("</featureModel>"):
        errors.append("Content found after closing </featureModel> tag")

    # Parse XML
    try:
        tree = etree.parse(str(xml_path))
    except Exception as e:  # pragma: no cover
        errors.append(f"XML parse error: {e}")
        return WellFormedResult(False, errors)

    root = tree.getroot()
    if root.tag != "featureModel":
        errors.append(f"Root tag is '{root.tag}', expected 'featureModel'")

    struct = root.find("struct")
    if struct is None:
        errors.append("Missing <struct> element")

    # Name checks & duplicates
    names = []

    def walk(node):
        name = node.attrib.get("name")
        if name:
            names.append(name)
            if not NAME_RE.match(name):
                errors.append(f"Invalid feature/group name '{name}' (must match {NAME_RE.pattern})")
        for child in node:
            walk(child)

    if struct is not None:
        for child in struct:
            walk(child)

    seen = set()
    for n in names:
        if n in seen:
            errors.append(f"Duplicate feature/group name '{n}'")
        seen.add(n)

    # XSD validation (optional)
    if xsd_path:
        try:
            schema = xmlschema.XMLSchema(str(xsd_path))
            if not schema.is_valid(str(xml_path)):
                for e in schema.iter_errors(str(xml_path)):
                    errors.append(f"XSD validation: {e.message}")
        except Exception as e:  # pragma: no cover
            errors.append(f"XSD schema load/validation failed: {e}")

    return WellFormedResult(len(errors) == 0, errors)

