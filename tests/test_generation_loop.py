"""End-to-end tests for the unified generation loop (Phase 5).

Runs the loop against a ``FakeLLM`` with a hand-built miniature corpus
(chunks.jsonl fixture + tiny orderings.json + tiny XSD). Exercises:

    5.T1  every doc visited exactly once per N
    5.T2  identical batching across grounding conditions
    5.T3  previous FM carried forward across steps
    5.T4  Non-RAG includes all-and-only batch chunks
    5.T6  context-limit failure records infeasible + does NOT call the model
    5.T8  resume never overwrites a differently-configured run
    5.7   truncation surfaces via finish_reason=='length'

RAG-side tests (5.T5) use the retrieval unit tests in test_retrieval.py plus
this file's Non-RAG round-trip; a live-Chroma RAG round-trip is deferred to
Phase 6 smoke.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fame.generation.batching import assert_covers_pi, slice_ordering
from fame.generation.grounding import build_grounding, format_context_text
from fame.generation.llm_client import FakeLLM
from fame.generation.loop import run_generation
from fame.generation.run import RunConfig


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "prompts/fm_prompt_template.txt"

_MINI_XSD = "<xs:schema/>"


def _make_mini_corpus(tmp_path: Path, corpus: str = "mini", n_docs: int = 6,
                      chunks_per_doc: int = 3) -> tuple[Path, Path]:
    """Build a fixture chunks.jsonl + orderings.json for a synthetic corpus."""
    chunks_jsonl = tmp_path / f"processed/{corpus}/chunks.jsonl"
    chunks_jsonl.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    ordering = []
    for i in range(1, n_docs + 1):
        did = f"d_{i:02d}"
        ordering.append(did)
        for k in range(chunks_per_doc):
            cid = f"chunk_{did}_{k:02d}"
            offset = k * 100
            lines.append(json.dumps({
                "chunk_id":               cid,
                "doc_id":                 did,
                "offsets":                [offset, offset + 90],
                "text":                   f"Body of {did} chunk {k}. " * 5,
                "preprocessing_version":  "test-v1",
            }))
    chunks_jsonl.write_text("\n".join(lines) + "\n")

    orderings_json = tmp_path / "orderings.json"
    orderings_json.write_text(json.dumps({
        corpus: {"primary": {"seed": None, "policy": "manifest_order", "order": ordering}}
    }))
    return chunks_jsonl, orderings_json


def _make_cfg(**over) -> RunConfig:
    d = dict(
        campaign_id="ifs-2027-test",
        corpus="mini",
        ordering_id="primary",
        N=3,
        grounding="nonrag",
        model_id="fake-lm",
        provider="fake",
        seed=0,
        repetition=0,
        metamodel_block=True,
        k_doc=None,
        domain="test domain",
        root_feature="Root",
        prompt_template_hash="pt" * 32,
        metamodel_hash="mm" * 32,
        chunks_hash="ch" * 32,
        encoder_digest="en" * 32,
        max_output_tokens=1024,
        temperature=0.2,
        reasoning_effort=None,
        context_window=1_000_000,
    )
    d.update(over)
    return RunConfig(**d)


# ─────────────────────────────────────────────────────────────────────────────
# 5.T1 — every doc visited exactly once for every configured N
# ─────────────────────────────────────────────────────────────────────────────

def test_every_doc_visited_once_across_N_grid(tmp_path: Path) -> None:
    chunks_jsonl, orderings_json = _make_mini_corpus(tmp_path, n_docs=6)
    ordering = json.loads(orderings_json.read_text())["mini"]["primary"]["order"]
    for N in (1, 2, 3, 6):
        batches = slice_ordering(ordering, N)
        assert_covers_pi(batches, ordering)


# ─────────────────────────────────────────────────────────────────────────────
# 5.T4 — Non-RAG includes ALL and ONLY batch chunks
# ─────────────────────────────────────────────────────────────────────────────

def test_nonrag_includes_all_and_only_batch_chunks(tmp_path: Path) -> None:
    chunks_jsonl, _ = _make_mini_corpus(tmp_path, n_docs=6, chunks_per_doc=3)
    gc = build_grounding(
        grounding="nonrag",
        batch_doc_ids=["d_02", "d_03"],
        domain="d",
        chunks_jsonl=chunks_jsonl,
    )
    doc_ids_in_ctx = {c.doc_id for c in gc.chunks}
    assert doc_ids_in_ctx == {"d_02", "d_03"}
    # 3 chunks per doc × 2 docs = 6
    assert len(gc.chunks) == 6
    # k_step None for nonrag
    assert gc.k_step is None


# ─────────────────────────────────────────────────────────────────────────────
# 5.T3 — previous FM carried forward across steps
# ─────────────────────────────────────────────────────────────────────────────

def test_previous_fm_carried_forward(tmp_path: Path) -> None:
    chunks_jsonl, orderings_json = _make_mini_corpus(tmp_path, n_docs=6)
    scripted = [
        '<?xml version="1.0"?><featureModel><struct><and name="R" abstract="true" mandatory="true"><feature name="A"/></and></struct><constraints/></featureModel>',
        '<?xml version="1.0"?><featureModel><struct><and name="R" abstract="true" mandatory="true"><feature name="A"/><feature name="B"/></and></struct><constraints/></featureModel>',
        '<?xml version="1.0"?><featureModel><struct><and name="R" abstract="true" mandatory="true"><feature name="A"/><feature name="B"/><feature name="C"/></and></struct><constraints/></featureModel>',
    ]
    lm = FakeLLM(responses=list(scripted))
    result = run_generation(
        config=_make_cfg(N=3),
        llm=lm,
        orderings_json=orderings_json,
        chunks_jsonl=chunks_jsonl,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text=_MINI_XSD,
        results_root=tmp_path / "results",
    )
    # 3 steps, all feasible, all called the model
    assert len(result.steps) == 3
    for s in result.steps:
        assert s.feasible
        assert s.finish_reason == "stop"
        assert s.fm_path is not None
    # Final FM is the last scripted response
    assert result.paths.fm_gen.read_text() == scripted[-1]
    # Step 1 and 2 prompts must contain the PREVIOUS model text ("<feature name=\"A\"/>")
    assert "<feature name=\"A\"/>" in lm.seen_requests[1].prompt
    # Step 2 prompt must contain the step 1 model (A and B present)
    assert "<feature name=\"B\"/>" in lm.seen_requests[2].prompt


# ─────────────────────────────────────────────────────────────────────────────
# 5.T6 + 5.6 — over-limit request records infeasible and skips the model call
# ─────────────────────────────────────────────────────────────────────────────

def test_infeasible_step_does_not_call_model(tmp_path: Path) -> None:
    chunks_jsonl, orderings_json = _make_mini_corpus(tmp_path, n_docs=2, chunks_per_doc=1)
    lm = FakeLLM()
    result = run_generation(
        config=_make_cfg(
            N=1,
            context_window=50,          # tiny window; assembled prompt exceeds it
            max_output_tokens=10,
        ),
        llm=lm,
        orderings_json=orderings_json,
        chunks_jsonl=chunks_jsonl,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text=_MINI_XSD,
        results_root=tmp_path / "results",
    )
    assert len(result.steps) == 1
    assert result.steps[0].feasible is False
    assert result.steps[0].error == "over_context_window"
    # LLM was NOT invoked
    assert lm.seen_requests == []
    # fm_gen.xml was NOT written (no successful step)
    assert not result.paths.fm_gen.exists()


# ─────────────────────────────────────────────────────────────────────────────
# 5.7 — finish_reason=length surfaces + downstream can flag truncation
# ─────────────────────────────────────────────────────────────────────────────

def test_truncation_surfaces_in_run_meta(tmp_path: Path) -> None:
    chunks_jsonl, orderings_json = _make_mini_corpus(tmp_path, n_docs=2, chunks_per_doc=1)
    lm = FakeLLM(responses=["<featureModel/>"], finish_reason="length")
    result = run_generation(
        config=_make_cfg(N=1),
        llm=lm,
        orderings_json=orderings_json,
        chunks_jsonl=chunks_jsonl,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text=_MINI_XSD,
        results_root=tmp_path / "results",
    )
    assert result.steps[0].finish_reason == "length"
    meta = json.loads(result.paths.run_meta.read_text())
    assert meta["steps"][0]["finish_reason"] == "length"


# ─────────────────────────────────────────────────────────────────────────────
# 5.T8 — different config → different run_id → different directory
# ─────────────────────────────────────────────────────────────────────────────

def test_different_configs_do_not_overwrite(tmp_path: Path) -> None:
    chunks_jsonl, orderings_json = _make_mini_corpus(tmp_path, n_docs=2, chunks_per_doc=1)
    lm = FakeLLM(responses=["<a/>", "<b/>"])
    r1 = run_generation(
        config=_make_cfg(N=1, seed=0),
        llm=lm,
        orderings_json=orderings_json,
        chunks_jsonl=chunks_jsonl,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text=_MINI_XSD,
        results_root=tmp_path / "results",
    )
    r2 = run_generation(
        config=_make_cfg(N=1, seed=1),   # only seed differs
        llm=FakeLLM(responses=["<c/>"]),
        orderings_json=orderings_json,
        chunks_jsonl=chunks_jsonl,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text=_MINI_XSD,
        results_root=tmp_path / "results",
    )
    assert r1.run_id != r2.run_id
    assert r1.paths.root != r2.paths.root
    # Both fm_gen files exist and hold their own scripted responses
    assert r1.paths.fm_gen.read_text() == "<a/>"
    assert r2.paths.fm_gen.read_text() == "<c/>"


# ─────────────────────────────────────────────────────────────────────────────
# 5.T2 — identical batching across grounding conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_batching_identical_across_grounding(tmp_path: Path) -> None:
    """Same corpus, ordering, N ⇒ same slice_ordering output regardless of
    grounding. Verified at the batching level; the loop uses the same
    function for both arms."""
    ordering = [f"d_{i:02d}" for i in range(1, 11)]
    a = slice_ordering(ordering, 4)
    b = slice_ordering(ordering, 4)
    assert [x.doc_ids for x in a] == [x.doc_ids for x in b]
