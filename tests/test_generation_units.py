"""Unit tests for the Phase 5 generation building blocks.

Covers batching, token budget, prompt assembly, LLM client Protocol, persistence,
and run_id derivation. No network. No Ollama. No real Chroma.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fame.generation.batching import Batch, assert_covers_pi, slice_ordering
from fame.generation.llm_client import (
    FakeLLM,
    GenerationLLM,
    GenerationRequest,
    make_client,
)
from fame.generation.persistence import (
    RunPaths,
    atomic_append_jsonl,
    atomic_write_json,
    atomic_write_text,
)
from fame.generation.prompt_assembly import render_prompt
from fame.generation.run import RunConfig
from fame.generation.token_budget import (
    UniversalCounter,
    is_feasible,
    universal_estimate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Batching (5.1, 5.T1)
# ─────────────────────────────────────────────────────────────────────────────

def _ids(n: int, prefix: str = "d") -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(1, n + 1)]


def test_slice_N1_single_batch() -> None:
    ids = _ids(23)
    batches = slice_ordering(ids, 1)
    assert len(batches) == 1
    assert batches[0].doc_ids == ids


def test_slice_Nn_singleton_batches() -> None:
    ids = _ids(5)
    batches = slice_ordering(ids, 5)
    assert [b.doc_ids for b in batches] == [[x] for x in ids]


def test_slice_policy_repair_54_N5() -> None:
    ids = _ids(54, "rep")
    batches = slice_ordering(ids, 5)
    sizes = [len(b.doc_ids) for b in batches]
    # floor(54/5)=10, rem=4 → [11,11,11,11,10]
    assert sizes == [11, 11, 11, 11, 10]
    assert_covers_pi(batches, ids)


def test_slice_policy_repair_54_N10() -> None:
    ids = _ids(54, "rep")
    batches = slice_ordering(ids, 10)
    sizes = [len(b.doc_ids) for b in batches]
    # floor(54/10)=5, rem=4 → [6,6,6,6,5,5,5,5,5,5]
    assert sizes == [6, 6, 6, 6, 5, 5, 5, 5, 5, 5]
    assert_covers_pi(batches, ids)


def test_slice_policy_repair_54_N20() -> None:
    ids = _ids(54, "rep")
    batches = slice_ordering(ids, 20)
    sizes = [len(b.doc_ids) for b in batches]
    # floor(54/20)=2, rem=14 → [3]*14 + [2]*6 (matches brief "3 docs/step")
    assert sizes == [3] * 14 + [2] * 6
    assert sum(sizes) == 54
    assert_covers_pi(batches, ids)


def test_slice_covers_every_doc_once_for_repair_N_grid() -> None:
    ids = _ids(54, "rep")
    for N in (1, 5, 10, 20, 54):
        batches = slice_ordering(ids, N)
        assert_covers_pi(batches, ids)


def test_slice_covers_every_doc_once_for_federation_N_grid() -> None:
    ids = _ids(23, "fed")
    for N in (1, 5, 10, 23):
        batches = slice_ordering(ids, N)
        assert_covers_pi(batches, ids)


def test_slice_rejects_out_of_range_N() -> None:
    ids = _ids(5)
    with pytest.raises(ValueError):
        slice_ordering(ids, 0)
    with pytest.raises(ValueError):
        slice_ordering(ids, 6)   # N > n


def test_slice_rejects_empty_ordering() -> None:
    with pytest.raises(ValueError):
        slice_ordering([], 1)


# ─────────────────────────────────────────────────────────────────────────────
# Token budget (5.6)
# ─────────────────────────────────────────────────────────────────────────────

def test_universal_estimate_char_over_4() -> None:
    assert universal_estimate("") == 0
    assert universal_estimate("abcd") == 1
    assert universal_estimate("abcdefgh") == 2


def test_is_feasible_leaves_room_for_completion_and_margin() -> None:
    """A 1000-char prompt has ~250 assembled tokens. With a 900_000-token
    window and 32_000 output, we're well within budget."""
    ok, assembled = is_feasible(
        "x" * 1000, context_window=1_000_000, max_output_tokens=32_000,
    )
    assert ok
    assert 200 <= assembled <= 300


def test_is_feasible_rejects_oversized_prompt() -> None:
    huge = "x" * (5_000_000)   # ~1.25M tokens
    ok, _ = is_feasible(huge, context_window=1_000_000, max_output_tokens=32_000)
    assert not ok


def test_universal_counter_matches_helper() -> None:
    c = UniversalCounter()
    assert c.count("abcd") == universal_estimate("abcd")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt assembly (5.5, 5.7)
# ─────────────────────────────────────────────────────────────────────────────

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "prompts/fm_prompt_template.txt"


def test_render_prompt_single_stage_no_metamodel() -> None:
    """N=1, ablation arm — no previous_model, no metamodel block."""
    bundle = render_prompt(
        template_path=TEMPLATE,
        template_hash="test-hash",
        domain="model repair",
        root_feature="Repair",
        context_text="[EVIDENCE 1] doc_id=rep_01 chunk_id=abc\nSome text.",
        context_doc_ids=["rep_01"],
        metamodel_block=False,
        metamodel_xsd=None,
        previous_model=False,
        previous_fm_xml=None,
    )
    # No placeholders survived
    assert "{{DOMAIN}}" not in bundle.text
    assert "{{ROOT_FEATURE}}" not in bundle.text
    # Ablation ⇒ METAMODEL SPECIFICATION block excluded
    assert "METAMODEL SPECIFICATION" not in bundle.text
    # Single-stage ⇒ REFINEMENT POLICY block excluded
    assert "REFINEMENT POLICY" not in bundle.text
    # Root feature is present
    assert "Repair" in bundle.text


def test_render_prompt_iterative_with_metamodel() -> None:
    """N>1, guided arm — metamodel_block AND previous_model on."""
    bundle = render_prompt(
        template_path=TEMPLATE,
        template_hash="test-hash",
        domain="model federation",
        root_feature="Federation",
        context_text="[EVIDENCE 1] doc_id=fed_01 chunk_id=abc\nX.",
        context_doc_ids=["fed_01"],
        metamodel_block=True,
        metamodel_xsd="<xs:schema/>",
        previous_model=True,
        previous_fm_xml="<featureModel/>",
    )
    assert "METAMODEL SPECIFICATION" in bundle.text
    assert "REFINEMENT POLICY" in bundle.text
    assert "<xs:schema/>" in bundle.text
    assert "<featureModel/>" in bundle.text


def test_render_prompt_rejects_missing_metamodel_xsd() -> None:
    with pytest.raises(ValueError):
        render_prompt(
            template_path=TEMPLATE, template_hash="h", domain="d", root_feature="r",
            context_text="ctx", context_doc_ids=["d1"],
            metamodel_block=True, metamodel_xsd=None,     # <— missing
            previous_model=False, previous_fm_xml=None,
        )


def test_render_prompt_rejects_missing_previous_fm() -> None:
    with pytest.raises(ValueError):
        render_prompt(
            template_path=TEMPLATE, template_hash="h", domain="d", root_feature="r",
            context_text="ctx", context_doc_ids=["d1"],
            metamodel_block=False, metamodel_xsd=None,
            previous_model=True, previous_fm_xml=None,     # <— missing
        )


# ─────────────────────────────────────────────────────────────────────────────
# LLM client Protocol + FakeLLM
# ─────────────────────────────────────────────────────────────────────────────

def test_fake_llm_is_a_generation_llm() -> None:
    assert isinstance(FakeLLM(), GenerationLLM)


def test_fake_llm_consumes_scripted_responses_in_fifo() -> None:
    lm = FakeLLM(responses=["<fm1/>", "<fm2/>"])
    r1 = lm.generate(GenerationRequest(prompt="a", max_output_tokens=100))
    r2 = lm.generate(GenerationRequest(prompt="b", max_output_tokens=100))
    r3 = lm.generate(GenerationRequest(prompt="c", max_output_tokens=100))  # fallback
    assert r1.text == "<fm1/>"
    assert r2.text == "<fm2/>"
    assert r3.text.startswith("<?xml")   # canonical fallback


def test_fake_llm_records_finish_reason() -> None:
    lm = FakeLLM(finish_reason="length")
    r = lm.generate(GenerationRequest(prompt="p", max_output_tokens=1))
    assert r.finish_reason == "length"


def test_make_client_dispatches_fake() -> None:
    lm = make_client(provider="fake", model_id="fake-lm")
    assert isinstance(lm, FakeLLM)


def test_make_client_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        make_client(provider="claude", model_id="opus")


# ─────────────────────────────────────────────────────────────────────────────
# RunConfig / run_id determinism (5.T8)
# ─────────────────────────────────────────────────────────────────────────────

def _sample_cfg(**over) -> RunConfig:
    d = dict(
        campaign_id="ifs-2027-test",
        corpus="repair",
        ordering_id="primary",
        N=5,
        grounding="rag",
        model_id="fake-lm",
        provider="fake",
        seed=0,
        repetition=0,
        metamodel_block=True,
        k_doc=5,
        domain="model repair",
        root_feature="Repair",
        prompt_template_hash="a" * 64,
        metamodel_hash="b" * 64,
        chunks_hash="c" * 64,
        encoder_digest="d" * 64,
        max_output_tokens=16384,
        temperature=0.2,
        reasoning_effort=None,
        context_window=1_000_000,
    )
    d.update(over)
    return RunConfig(**d)


def test_run_id_is_deterministic() -> None:
    a = _sample_cfg().run_id()
    b = _sample_cfg().run_id()
    assert a == b
    assert len(a) == 16


def test_run_id_changes_with_seed() -> None:
    assert _sample_cfg(seed=0).run_id() != _sample_cfg(seed=1).run_id()


def test_run_id_changes_with_grounding() -> None:
    assert _sample_cfg(grounding="rag").run_id() != _sample_cfg(grounding="nonrag", k_doc=None).run_id()


def test_run_id_changes_with_prompt_hash() -> None:
    a = _sample_cfg(prompt_template_hash="a" * 64).run_id()
    b = _sample_cfg(prompt_template_hash="e" * 64).run_id()
    assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — atomic writes
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_write_text_creates_parents(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    atomic_write_text(target, "hi")
    assert target.read_text() == "hi"
    assert not (target.with_suffix(target.suffix + ".tmp")).exists()


def test_atomic_write_json_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "x.json"
    atomic_write_json(target, {"b": 1, "a": 2})
    # sort_keys=True in the writer
    assert target.read_text().startswith('{\n  "a"')


def test_atomic_append_jsonl_writes_one_line_per_record(tmp_path: Path) -> None:
    p = tmp_path / "log.jsonl"
    atomic_append_jsonl(p, {"i": 1})
    atomic_append_jsonl(p, {"i": 2})
    lines = p.read_text().splitlines()
    assert [json.loads(l)["i"] for l in lines] == [1, 2]


def test_run_paths_layout(tmp_path: Path) -> None:
    paths = RunPaths.for_run(
        results_root=tmp_path,
        campaign_id="c",
        corpus="repair",
        config_hash="h",
        run_id="r",
    )
    assert paths.root == tmp_path / "c" / "repair" / "h" / "r"
    assert paths.fm_gen.name == "fm_gen.xml"
    assert paths.fm_iter_dir.name == "fm_iter"
    assert paths.context_log.name == "context_log.jsonl"
    assert paths.run_meta.name == "run_meta.json"


# ─────────────────────────────────────────────────────────────────────────────
# Structural isolation (5.T7)
# ─────────────────────────────────────────────────────────────────────────────

def test_generation_does_not_import_evaluation() -> None:
    """Generation modules must NOT import from fame.evaluation or ground truth.
    Enforced by inspecting imports of every module in fame.generation."""
    import fame.generation
    import ast

    gen_dir = Path(fame.generation.__file__).parent
    offenders: list[tuple[str, str]] = []
    for py in gen_dir.rglob("*.py"):
        source = py.read_text()
        tree = ast.parse(source, filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("fame.evaluation"):
                    offenders.append((py.name, node.module))
    assert not offenders, f"generation imported evaluation: {offenders}"
