"""failure-mode tests for the generation loop.

Covers:
- provider exception surfaces via ``step.error``
- retry-with-backoff triggers on transient exceptions, gives up after max_attempts
- 4xx errors fail fast (no retry)
- resume guard: refuse to overwrite unless force=True
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from fame.generation import FakeLLM, RunAlreadyExists, RunConfig, run_generation
from fame.generation.llm_client import (GenerationResponse, _is_transient,
                                        _with_retry)


REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "prompts/fm_prompt_template.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Retry helper unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_is_transient_connection_error() -> None:
    assert _is_transient(requests.ConnectionError("no route"))


def test_is_transient_timeout() -> None:
    assert _is_transient(requests.Timeout("slow"))


def test_is_transient_recognises_5xx() -> None:
    r = requests.Response()
    r.status_code = 503
    err = requests.HTTPError(response=r)
    assert _is_transient(err)


def test_is_transient_recognises_429() -> None:
    r = requests.Response()
    r.status_code = 429
    err = requests.HTTPError(response=r)
    assert _is_transient(err)


def test_is_transient_rejects_4xx_client_error() -> None:
    r = requests.Response()
    r.status_code = 400
    err = requests.HTTPError(response=r)
    assert not _is_transient(err)


def test_is_transient_rejects_401_auth() -> None:
    r = requests.Response()
    r.status_code = 401
    err = requests.HTTPError(response=r)
    assert not _is_transient(err)


def test_with_retry_succeeds_after_one_transient() -> None:
    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.ConnectionError("first attempt")
        return "ok"

    with patch("fame.generation.llm_client.time.sleep") as sleep_mock:
        result = _with_retry(flaky, max_attempts=3, base_seconds=0.001, cap_seconds=0.01)
    assert result == "ok"
    assert calls["n"] == 2
    sleep_mock.assert_called_once()   # exactly one backoff between attempts


def test_with_retry_gives_up_after_max_attempts() -> None:
    def always_broken() -> None:
        raise requests.ConnectionError("nope")

    with patch("fame.generation.llm_client.time.sleep"):
        with pytest.raises(requests.ConnectionError):
            _with_retry(always_broken, max_attempts=3, base_seconds=0.001)


def test_with_retry_fails_fast_on_non_transient() -> None:
    calls = {"n": 0}

    def hard_fail() -> None:
        calls["n"] += 1
        raise ValueError("bad prompt")

    with pytest.raises(ValueError):
        _with_retry(hard_fail, max_attempts=3)
    assert calls["n"] == 1        # NO retry for non-transient errors


# ─────────────────────────────────────────────────────────────────────────────
# Loop-level failure modes
# ─────────────────────────────────────────────────────────────────────────────

def _mini_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    chunks = tmp_path / "processed/mini/chunks.jsonl"
    chunks.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    ordering = []
    for i in range(1, 3):
        did = f"d_{i:02d}"
        ordering.append(did)
        lines.append(json.dumps({
            "chunk_id": f"c_{did}", "doc_id": did, "offsets": [0, 90],
            "text": f"body of {did}", "preprocessing_version": "test",
        }))
    chunks.write_text("\n".join(lines) + "\n")
    orderings = tmp_path / "orderings.json"
    orderings.write_text(json.dumps({
        "mini": {"primary": {"seed": None, "policy": "manifest_order", "order": ordering}}
    }))
    return chunks, orderings


def _cfg(**over) -> RunConfig:
    d = dict(
        campaign_id="fail-tests", corpus="mini", ordering_id="primary", N=1,
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


class _RaisingLLM:
    provider = "test"
    model_id = "raising"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def generate(self, request):
        raise self._exc


class _SequenceLLM:
    provider = "test"
    model_id = "sequence"

    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.seen_requests = []

    def generate(self, request):
        self.seen_requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return GenerationResponse(
            text=outcome, finish_reason="stop", prompt_tokens=10,
            completion_tokens=5, provider=self.provider, model_id=self.model_id,
        )


def test_provider_exception_captured_in_step_error(tmp_path: Path) -> None:
    """Non-transient errors from generate() must surface in the step record
    and not crash the loop."""
    chunks, orderings = _mini_fixtures(tmp_path)
    result = run_generation(
        config=_cfg(),
        llm=_RaisingLLM(ValueError("bad prompt")),
        orderings_json=orderings,
        chunks_jsonl=chunks,
        prompt_template_path=TEMPLATE,
        metamodel_xsd_text="<xs:schema/>",
        results_root=tmp_path / "results",
    )
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.finish_reason is None
    assert step.error is not None
    assert "ValueError" in step.error


def test_resume_guard_refuses_without_force(tmp_path: Path) -> None:
    """Second run against the same config must raise unless force=True."""
    chunks, orderings = _mini_fixtures(tmp_path)
    cfg = _cfg()
    # First run
    run_generation(
        config=cfg,
        llm=FakeLLM(responses=['<featureModel id="a"/>']),
        orderings_json=orderings, chunks_jsonl=chunks,
        prompt_template_path=TEMPLATE, metamodel_xsd_text="<xs:schema/>",
        results_root=tmp_path / "results",
    )
    # Second run — must refuse
    with pytest.raises(RunAlreadyExists):
        run_generation(
            config=cfg,
            llm=FakeLLM(responses=['<featureModel id="b"/>']),
            orderings_json=orderings, chunks_jsonl=chunks,
            prompt_template_path=TEMPLATE, metamodel_xsd_text="<xs:schema/>",
            results_root=tmp_path / "results",
        )


def test_resume_guard_allows_with_force(tmp_path: Path) -> None:
    chunks, orderings = _mini_fixtures(tmp_path)
    cfg = _cfg()
    run_generation(
        config=cfg,
        llm=FakeLLM(responses=['<featureModel id="a"/>']),
        orderings_json=orderings, chunks_jsonl=chunks,
        prompt_template_path=TEMPLATE, metamodel_xsd_text="<xs:schema/>",
        results_root=tmp_path / "results",
    )
    result = run_generation(
        config=cfg,
        llm=FakeLLM(responses=['<featureModel id="b"/>']),
        orderings_json=orderings, chunks_jsonl=chunks,
        prompt_template_path=TEMPLATE, metamodel_xsd_text="<xs:schema/>",
        results_root=tmp_path / "results",
        force=True,
    )
    assert result.paths.fm_gen.read_text() == '<featureModel id="b"/>'


def test_resume_provider_failure_calls_only_failed_step(tmp_path: Path) -> None:
    chunks, orderings = _mini_fixtures(tmp_path)
    cfg = _cfg(N=2)
    first_xml = '<featureModel><struct><and name="R"/></struct></featureModel>'
    final_xml = '<featureModel><struct><and name="R"><feature name="B"/></and></struct></featureModel>'
    initial = _SequenceLLM([first_xml, requests.ConnectionError("gateway lost")])
    failed = run_generation(
        config=cfg, llm=initial, orderings_json=orderings,
        chunks_jsonl=chunks, prompt_template_path=TEMPLATE,
        metamodel_xsd_text="<xs:schema/>", results_root=tmp_path / "results",
    )
    assert failed.completed is False
    assert failed.steps[-1].terminal_failure == "provider_error"
    assert len(initial.seen_requests) == 2

    resumed_llm = _SequenceLLM([final_xml])
    resumed = run_generation(
        config=cfg, llm=resumed_llm, orderings_json=orderings,
        chunks_jsonl=chunks, prompt_template_path=TEMPLATE,
        metamodel_xsd_text="<xs:schema/>", results_root=tmp_path / "results",
        resume_provider_failure=True,
    )
    assert resumed.completed is True
    assert resumed.invocation_start_step == 1
    assert len(resumed_llm.seen_requests) == 1
    assert first_xml in resumed_llm.seen_requests[0].prompt
    assert [s.step_index for s in resumed.steps] == [0, 1]
    assert resumed.paths.fm_gen.read_text() == final_xml
    context_rows = [json.loads(line) for line in resumed.paths.context_log.read_text().splitlines()]
    assert [row["step_index"] for row in context_rows] == [0, 1]
    meta = json.loads(resumed.paths.run_meta.read_text())
    assert meta["terminal_status"] == "completed"
    assert meta["recovery_history"][0]["resumed_step"] == 1
    assert "gateway lost" in meta["recovery_history"][0]["prior_error"]


def test_resume_rejects_model_output_failure(tmp_path: Path) -> None:
    chunks, orderings = _mini_fixtures(tmp_path)
    cfg = _cfg()
    run_generation(
        config=cfg, llm=FakeLLM(responses=['<featureModel/>'], finish_reason="length"),
        orderings_json=orderings, chunks_jsonl=chunks,
        prompt_template_path=TEMPLATE, metamodel_xsd_text="<xs:schema/>",
        results_root=tmp_path / "results",
    )
    with pytest.raises(ValueError, match="provider_error"):
        run_generation(
            config=cfg, llm=FakeLLM(), orderings_json=orderings,
            chunks_jsonl=chunks, prompt_template_path=TEMPLATE,
            metamodel_xsd_text="<xs:schema/>", results_root=tmp_path / "results",
            resume_provider_failure=True,
        )
