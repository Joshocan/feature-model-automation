from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_RAG_QUERY_TEMPLATE = """({{ROOT_FEATURE}} AND {{DOMAIN}})
AND (
  approach OR methodology OR method OR framework OR architecture OR design
  OR implementation OR pipeline OR workflow OR algorithm OR technique
  OR system OR tool OR platform OR infrastructure
)"""


@dataclass(frozen=True)
class QueryContext:
    root_feature: str
    domain: str
    extra: Optional[Dict[str, str]] = None


def _clean_token(s: str) -> str:
    """
    Minimal cleanup:
    - strips whitespace/newlines
    - removes braces to avoid placeholder injection
    - collapses spaces
    """
    s = (s or "").strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip()


def build_query(ctx: QueryContext, template: str = DEFAULT_RAG_QUERY_TEMPLATE) -> str:
    """
    Fill {{ROOT_FEATURE}} and {{DOMAIN}} placeholders.
    """
    q = template
    q = q.replace("{{ROOT_FEATURE}}", _clean_token(ctx.root_feature))
    q = q.replace("{{DOMAIN}}", _clean_token(ctx.domain))

    if ctx.extra:
        for k, v in ctx.extra.items():
            q = q.replace(f"{{{{{k}}}}}", _clean_token(v))

    # normalize whitespace but keep readability
    q = re.sub(r"[ \t]+", " ", q).strip()
    return q
