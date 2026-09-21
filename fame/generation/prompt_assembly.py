"""Prompt assembly with invariant-first block ordering (Phase 5.5, 5.7).

Renders the frozen Jinja template ``prompts/fm_prompt_template.txt`` with two
switches:

* ``metamodel_block``  — off in the ablation arm (H3), on otherwise
* ``previous_model``   — off when N==1 (single-stage), on when there's an
                         accumulated model from step j-1

Refuses to render if any placeholder is unresolved. Emits blocks in a strict
invariant-first order so identical prefixes hit prefix caches on the LLM side.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, StrictUndefined, FileSystemLoader


# Placeholders that MUST be resolved before hitting the model.
_REQUIRED_PLACEHOLDERS = frozenset({
    "DOMAIN", "ROOT_FEATURE", "CONTEXT_DOC_IDS", "CONTEXT",
})
# XSD_METAMODEL required only when metamodel_block=true.
# PREVIOUS_FM_XML required only when previous_model=true.


@dataclass(frozen=True)
class PromptBundle:
    """Rendered prompt + provenance for D9/D10 logging."""
    text: str
    template_hash: str
    metamodel_block: bool
    previous_model: bool
    domain: str
    root_feature: str


def _load_template(template_path: Path) -> Any:
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )
    return env.get_template(template_path.name)


def render_prompt(
    *,
    template_path: Path | str,
    template_hash: str,
    domain: str,
    root_feature: str,
    context_text: str,
    context_doc_ids: List[str],
    metamodel_block: bool,
    metamodel_xsd: Optional[str] = None,
    previous_model: bool,
    previous_fm_xml: Optional[str] = None,
) -> PromptBundle:
    """Render the unified prompt template. Raises on unresolved placeholders."""
    tpath = Path(template_path).expanduser().resolve()
    tmpl = _load_template(tpath)

    if metamodel_block and not metamodel_xsd:
        raise ValueError("metamodel_block=True but metamodel_xsd is empty")
    if previous_model and not previous_fm_xml:
        raise ValueError("previous_model=True but previous_fm_xml is empty")

    ctx: Dict[str, Any] = {
        "DOMAIN":            domain,
        "ROOT_FEATURE":      root_feature,
        "CONTEXT":           context_text,
        "CONTEXT_DOC_IDS":   "\n".join(context_doc_ids),
        "XSD_METAMODEL":     metamodel_xsd or "",
        "PREVIOUS_FM_XML":   previous_fm_xml or "",
        "metamodel_block":   bool(metamodel_block),
        "previous_model":    bool(previous_model),
    }

    text = tmpl.render(**ctx)

    # Post-render sanity check — no {{PLACEHOLDER}} tokens survived.
    survivors = re.findall(r"\{\{\s*([A-Z_]+)\s*\}\}", text)
    if survivors:
        raise ValueError(
            f"unresolved template placeholders after render: {sorted(set(survivors))}"
        )

    return PromptBundle(
        text=text,
        template_hash=template_hash,
        metamodel_block=metamodel_block,
        previous_model=previous_model,
        domain=domain,
        root_feature=root_feature,
    )
