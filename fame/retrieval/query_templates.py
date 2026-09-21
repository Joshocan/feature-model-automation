"""Fixed retrieval sub-queries (frozen in Phase 2.9).

Four sub-queries, identical at every step, in both RAG arms, independent of
generation state. **Never query with the current feature model** — that
confounds H1b with a feedback effect (see brief §1). The strings below must
match ``config/experiment.yaml`` verbatim; the source of truth is that config
file, and this module simply loads and validates it.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

# Repo root — used for the default config lookup.
_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_CFG = _REPO / "config/experiment.yaml"

_DOMAIN_PLACEHOLDER = re.compile(r"\{domain\}", re.IGNORECASE)


def load_sub_queries(config_path: Path | str = _DEFAULT_CFG) -> List[str]:
    """Load the four fixed sub-queries verbatim from ``experiment.yaml``."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    qs = list(cfg["retrieval"]["sub_queries"])
    if len(qs) != 4:
        raise ValueError(
            f"experiment.yaml must define exactly 4 retrieval.sub_queries, got {len(qs)}"
        )
    for q in qs:
        if not q.startswith("search_query: "):
            raise ValueError(
                f"every retrieval sub-query must start with 'search_query: ' — got: {q!r}"
            )
    return qs


def format_sub_query(template: str, *, domain: str) -> str:
    """Substitute ``{domain}`` in a sub-query template.

    The placeholder is the only permitted templating: nothing about
    the feature model may leak into the query.
    """
    return _DOMAIN_PLACEHOLDER.sub(domain, template)
