from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class ConstraintRecord:
    constraint_type: str  # requires | excludes | requires_any | not_implies_any | unknown
    left_features: List[str]
    right_features: List[str]
    evidence_refs: List[str]
    confidence: float | None = None
    raw_expr: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "constraint_type": self.constraint_type,
            "left_features": self.left_features,
            "right_features": self.right_features,
            "evidence_refs": self.evidence_refs,
            "confidence": self.confidence,
            "raw_expr": self.raw_expr,
        }


def _get_text(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()


def _first_child(tag: str, node: ET.Element):
    for ch in node:
        if ch.tag == tag:
            return ch
    return None


def extract_constraints(xml_path: Path | str) -> List[ConstraintRecord]:
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    constraints_el = root.find("constraints")
    if constraints_el is None:
        return []

    # Build feature name -> evidence refs map from descriptions (Trace: [...])
    feature_trace: Dict[str, List[str]] = {}
    struct = root.find("struct")

    def walk(node: ET.Element):
        name = node.attrib.get("name")
        if name:
            desc = node.findtext("description", default="").strip()
            refs = _trace_refs(desc)
            if refs:
                feature_trace[name] = refs
        for ch in node:
            if ch.tag in {"feature", "and", "or", "alt"}:
                walk(ch)

    if struct is not None:
        for ch in struct:
            walk(ch)

    records: List[ConstraintRecord] = []

    for rule in constraints_el.findall("rule"):
        if len(rule) == 0:
            continue
        expr = rule[0]
        ctype = "unknown"
        lefts: List[str] = []
        rights: List[str] = []
        raw_expr = ET.tostring(expr, encoding="unicode")

        # Pattern: Requires(A,B) => <imp><var>A</var><var>B</var></imp>
        if expr.tag == "imp":
            vars_ = expr.findall("var")
            not_ = _first_child("not", expr)
            disj = _first_child("disj", expr)

            # Simple requires
            if len(vars_) == 2:
                l = (vars_[0].text or "").strip()
                r = (vars_[1].text or "").strip()
                if l and r:
                    lefts = [l]
                    rights = [r]
                    ctype = "requires"

            # NOT A -> (B OR C ...)
            elif not_ is not None and disj is not None:
                not_vars = not_.findall("var")
                disj_vars = [ (v.text or "").strip() for v in disj.findall("var") if (v.text or "").strip() ]
                if not_vars and disj_vars:
                    left_raw = (not_vars[0].text or "").strip()
                    if left_raw:
                        lefts = [left_raw]
                        rights = disj_vars
                        ctype = "not_implies_any"

        # Pattern: Excludes(A,B) => <and><var>A</var><not><var>B</var></not></and>
        elif expr.tag == "and":
            vars_ = expr.findall("var")
            not_ = _first_child("not", expr)
            not_var = None
            if not_ is not None:
                not_vars = not_.findall("var")
                if not_vars:
                    not_var = not_vars[0].text or ""
            if len(vars_) == 1 and not_var:
                l = (vars_[0].text or "").strip()
                r = not_var.strip()
                if l and r:
                    lefts = [l]
                    rights = [r]
                    ctype = "excludes"

        # Fallback: store raw text if pattern unknown
        if not lefts and expr.tag == "var":
            v = (expr.text or "").strip()
            if v:
                lefts = [v]
        if ctype == "unknown" and expr.tag:
            rawtext = _get_text(expr)
            if rawtext:
                rights = rights or [rawtext]

        # gather evidence refs from involved features
        refs = []
        for lf in lefts:
            refs.extend(feature_trace.get(lf, []))
        for rf in rights:
            refs.extend(feature_trace.get(rf, []))
        # deduplicate
        refs = list(dict.fromkeys(refs))

        records.append(
            ConstraintRecord(
                constraint_type=ctype,
                left_features=lefts,
                right_features=rights,
                evidence_refs=refs,
                confidence=None,
                raw_expr=raw_expr,
            )
        )

    return records
from .feature_list import _trace_refs
