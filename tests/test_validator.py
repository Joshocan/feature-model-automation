"""Tests for the Phase 6.6 run validator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fame.generation import FakeLLM, RunConfig, run_generation
from fame.validation import Severity, validate_run


REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "prompts/fm_prompt_template.txt"
_MINI_XSD = "<xs:schema/>"


def _mini_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    chunks = tmp_path / "processed/mini/chunks.jsonl"
    chunks.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    ordering = []
    for i in range(1, 5):
        did = f"d_{i:02d}"
        ordering.append(did)
        for k in range(2):
            lines.append(json.dumps({
                "chunk_id":              f"c_{did}_{k}",
                "doc_id":                did,
                "offsets":               [k * 100, k * 100 + 90],
                "text":                  f"Body of {did} chunk {k}.",
                "preprocessing_version": "test-v1",
            }))
    chunks.write_text("\n".join(lines) + "\n")
    orderings = tmp_path / "orderings.json"
    orderings.write_text(json.dumps({
        "mini": {"primary": {"seed": None, "policy": "manifest_order", "order": ordering}}
    }))
    return chunks, orderings


def _cfg(**over) -> RunConfig:
    d = dict(
        campaign_id="test", corpus="mini", ordering_id="primary", N=2,
        grounding="nonrag", model_id="fake-lm", provider="fake", seed=0,
        repetition=0, metamodel_block=True, k_doc=None,
        domain="d", root_feature="R",
        prompt_template_hash="a" * 64, metamodel_hash="b" * 64,
        chunks_hash="c" * 64, encoder_digest="d" * 64,
        max_output_tokens=256, temperature=0.2, reasoning_effort=None,
        context_window=1_000_000,
    )
    d.update(over)
    return RunConfig(**d)


def _do_run(tmp_path: Path, **over) -> Path:
    chunks, orderings = _mini_fixtures(tmp_path)
    result = run_generation(
        config=_cfg(**over),
        llm=FakeLLM(responses=[
            '<?xml version="1.0"?><featureModel><struct><and name="R"/></struct><constraints/></featureModel>',
            '<?xml version="1.0"?><featureModel><struct><and name="R"/></struct><constraints/></featureModel>',
        ]),
        orderings_json=orderings,
        chunks_jsonl=chunks,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text=_MINI_XSD,
        results_root=tmp_path / "results",
    )
    return result.paths.root


# ─────────────────────────────────────────────────────────────────────────────
# Happy paths
# ─────────────────────────────────────────────────────────────────────────────

def test_completed_run_passes_when_hashes_not_frozen(tmp_path: Path) -> None:
    """When no protocol.sha256 exists in repo_root, hash checks are skipped."""
    run_root = _do_run(tmp_path)
    report = validate_run(run_root, repo_root=tmp_path)   # empty repo → no hashes
    assert report.complete, [f for f in report.findings]


# ─────────────────────────────────────────────────────────────────────────────
# Presence failures
# ─────────────────────────────────────────────────────────────────────────────

def test_missing_fm_gen_is_error(tmp_path: Path) -> None:
    run_root = _do_run(tmp_path)
    (run_root / "fm_gen.xml").unlink()
    report = validate_run(run_root, repo_root=tmp_path)
    assert not report.complete
    assert any(f.check == "presence:fm_gen.xml" for f in report.findings)


def test_missing_run_meta_is_error(tmp_path: Path) -> None:
    run_root = _do_run(tmp_path)
    (run_root / "run_meta.json").unlink()
    report = validate_run(run_root, repo_root=tmp_path)
    assert not report.complete
    assert any(f.check == "presence:run_meta.json" for f in report.findings)


# ─────────────────────────────────────────────────────────────────────────────
# Consistency failures
# ─────────────────────────────────────────────────────────────────────────────

def test_fm_gen_diff_from_last_iter_is_error(tmp_path: Path) -> None:
    run_root = _do_run(tmp_path)
    (run_root / "fm_gen.xml").write_text("<featureModel>tampered</featureModel>")
    report = validate_run(run_root, repo_root=tmp_path)
    assert not report.complete
    assert any(f.check == "consistency:fm_gen_vs_iter" for f in report.findings)


def test_chunk_doc_id_outside_batch_is_error(tmp_path: Path) -> None:
    run_root = _do_run(tmp_path)
    # Corrupt one context_log line so a chunk claims a doc not in the batch.
    lines = (run_root / "context_log.jsonl").read_text().splitlines()
    log = json.loads(lines[0])
    log["chunk_doc_ids"] = ["d_99"]     # not in batch
    lines[0] = json.dumps(log)
    (run_root / "context_log.jsonl").write_text("\n".join(lines) + "\n")
    report = validate_run(run_root, repo_root=tmp_path)
    assert not report.complete
    assert any("batch_leak" in f.check for f in report.findings)


# ─────────────────────────────────────────────────────────────────────────────
# Hash-pinning failures
# ─────────────────────────────────────────────────────────────────────────────

def test_hash_mismatch_is_error(tmp_path: Path) -> None:
    run_root = _do_run(tmp_path)
    # Fabricate a protocol.sha256 with a different prompt hash.
    frozen = tmp_path / "data/frozen/protocol.sha256"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(
        "deadbeef" * 8 + "  prompts/fm_prompt_template.txt\n"
        "cafef00d" * 8 + "  prompts/feature-model-schema.xsd\n"
    )
    report = validate_run(run_root, repo_root=tmp_path)
    assert not report.complete
    assert any(f.check.startswith("hash:") for f in report.findings)
