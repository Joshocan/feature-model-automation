"""Tests for the deterministic chunker (Phase 4 core)."""
from __future__ import annotations

import pytest

from fame.ingestion.chunking import Chunk, chunk_text
from fame.ingestion.cleaning import strip_front_matter, strip_back_matter, strip_matter


# ─────────────────────────────────────────────────────────────────────────────
# Chunker
# ─────────────────────────────────────────────────────────────────────────────

def _make_text(paragraphs, sep="\n\n"):
    return sep.join(paragraphs)


def test_empty_input_returns_no_chunks():
    assert chunk_text("", doc_id="d") == []
    assert chunk_text("   \n\n  ", doc_id="d") == []


def test_short_input_returns_single_chunk():
    text = "One short paragraph well below the window size."
    chunks = chunk_text(text, doc_id="d", max_chunk_chars=1500, overlap_chars=225)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("One short paragraph")
    assert chunks[0].offsets == [0, len(text)]


def test_offsets_recover_text():
    """text[start:end] must equal chunk.text after stripping."""
    text = _make_text([("Paragraph %d. " % i) * 30 for i in range(6)])
    chunks = chunk_text(text, doc_id="d", max_chunk_chars=400, overlap_chars=60)
    assert len(chunks) > 1
    for c in chunks:
        span = text[c.offsets[0]:c.offsets[1]].strip()
        assert span == c.text


def test_chunk_id_is_deterministic():
    text = _make_text([("Paragraph %d. " % i) * 30 for i in range(4)])
    a = chunk_text(text, doc_id="d", max_chunk_chars=500, overlap_chars=75)
    b = chunk_text(text, doc_id="d", max_chunk_chars=500, overlap_chars=75)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_chunk_id_changes_with_doc_id():
    text = "This paragraph is short but non-empty."
    a = chunk_text(text, doc_id="doc_a")
    b = chunk_text(text, doc_id="doc_b")
    assert a[0].chunk_id != b[0].chunk_id


def test_chunk_id_changes_with_preprocessing_version():
    text = "This paragraph is short but non-empty."
    a = chunk_text(text, doc_id="d", preprocessing_version="v1")
    b = chunk_text(text, doc_id="d", preprocessing_version="v2")
    assert a[0].chunk_id != b[0].chunk_id


def test_overlap_between_consecutive_chunks():
    """Consecutive chunks should share a nontrivial character overlap when the
    text is long enough to require more than one chunk."""
    text = _make_text([("Paragraph %d. " % i) * 40 for i in range(6)])
    chunks = chunk_text(text, doc_id="d", max_chunk_chars=500, overlap_chars=75)
    assert len(chunks) >= 2
    for prev, cur in zip(chunks, chunks[1:]):
        # cur.start should be strictly less than prev.end (i.e. overlaps)
        assert cur.offsets[0] < prev.offsets[1]


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        chunk_text("x", doc_id="d", max_chunk_chars=100, overlap_chars=100)
    with pytest.raises(ValueError):
        chunk_text("x", doc_id="d", max_chunk_chars=0, overlap_chars=0)


def test_prefers_paragraph_boundary():
    """Given a paragraph break in the second half of the window, chunk should
    end at that break, not at the hard cut."""
    prefix = "a" * 300
    tail = " b" * 200
    text = prefix + "\n\n" + tail
    chunks = chunk_text(text, doc_id="d", max_chunk_chars=500, overlap_chars=50)
    # First chunk should end at (or just before) the paragraph break
    assert chunks[0].offsets[1] <= 302   # break is at index 300, "\n\n" spans 300..302


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────

def test_strip_back_matter_cuts_at_references():
    text = "Body text here.\n\nReferences\n[1] Author. Title. Year."
    assert strip_back_matter(text) == "Body text here."


def test_strip_back_matter_no_anchor_is_identity():
    text = "Body text without a back-matter section."
    assert strip_back_matter(text) == text


def test_strip_front_matter_starts_at_abstract():
    text = "Author. Affiliation.\n\nAbstract\nThe abstract begins."
    out = strip_front_matter(text)
    assert out.startswith("Abstract")


def test_strip_matter_composes_both():
    text = "Author\n\nAbstract\nBody.\n\nReferences\nJunk."
    out = strip_matter(text)
    assert out.startswith("Abstract")
    assert "References" not in out
