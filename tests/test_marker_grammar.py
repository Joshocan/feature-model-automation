"""Tests for the frozen Trace provenance grammar."""
from fame.utils.marker_grammar import parse_marker


def test_trace_marker_parses_at_description_end() -> None:
    parsed = parse_marker("Supported concept. Trace: [rep_01, rep_17]")
    assert parsed.marker_found
    assert parsed.doc_ids == ["rep_01", "rep_17"]


def test_legacy_src_marker_is_rejected() -> None:
    assert not parse_marker("Supported concept. [src: rep_01]").marker_found


def test_trailing_text_after_trace_is_rejected() -> None:
    assert not parse_marker("Supported. Trace: [rep_01] extra").marker_found
