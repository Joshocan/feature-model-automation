from __future__ import annotations

import re
from typing import Optional


class UnresolvedPlaceholdersError(ValueError):
    def __init__(self, placeholders):
        super().__init__(f"Unreplaced prompt placeholders: {', '.join(sorted(placeholders))}")
        self.placeholders = placeholders


def assert_no_placeholders(text: str, *, original_template: Optional[str] = None) -> None:
    """
    Raises UnresolvedPlaceholdersError if declared template placeholders remain unresolved.

    When original_template is supplied, only keys that appeared in the template
    are checked — patterns like {id} that appear inside substituted content
    (e.g. OpenAPI path params) are ignored.

    When original_template is omitted, only double-brace {{PLACEHOLDER}} patterns
    are checked to avoid false positives from single-brace content.
    """
    if original_template is not None:
        declared_double = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", original_template))
        declared_single = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", original_template))
        remaining_double = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))
        remaining_single = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", text))
        missing = (declared_double & remaining_double) | (declared_single & remaining_single)
    else:
        # Without the original template we can only safely check double-brace
        # patterns; single-brace {word} is too common in substituted content.
        missing = set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", text))

    if missing:
        raise UnresolvedPlaceholdersError(missing)
