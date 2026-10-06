"""Unified N-step generation loop.

For each step j = 0..N-1:
    1. Build the current batch B_j from π and N (:func:`slice_ordering`)
    2. Assemble grounding context (RAG or Non-RAG)
    3. Render prompt with invariant-first blocks + previous_model switch
    4. Pre-flight token count; if over-limit → mark infeasible, skip
    5. Call the LLM; require finish_reason == "stop"
    6. Persist the response verbatim and check XML integrity
    7. Carry forward only a well-formed <featureModel>; otherwise stop the run

At the end, write fm_gen.xml (final) + run_meta.json (D10).
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .batching import assert_covers_pi, slice_ordering
from .grounding import (GroundingContext, build_grounding,
                        build_grounding_from_record, format_context_text)
from .llm_client import GenerationLLM, GenerationRequest, GenerationResponse
from .persistence import RunPaths, atomic_append_jsonl, atomic_write_json, atomic_write_text
from .prompt_assembly import PromptBundle, render_prompt
from .run import RunConfig
from .token_budget import TokenCounter, is_feasible


def _ollama_think_value(model_id: str, reasoning_effort: Optional[str]) -> Union[bool, str]:
    """Map the experiment effort setting to Ollama's model-specific API value."""
    effort = (reasoning_effort or "none").lower()
    level_only = model_id.lower().startswith(("glm-5.3-flash", "gpt-oss:"))
    if level_only:
        # These families cannot disable reasoning; Ollama expects an effort level.
        return effort if effort in {"low", "medium", "high", "max"} else "low"
    return effort in {"medium", "high", "max"}


@dataclass
class StepRecord:
    step_index: int
    batch_doc_ids: List[str]
    grounding: str
    n_chunks: int
    k_step: Optional[int]
    per_sub_query_k: Optional[List[int]]
    assembled_tokens: int
    feasible: bool
    finish_reason: Optional[str]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    wall_seconds: float
    fm_path: Optional[str]
    error: Optional[str] = None
    raw_response_path: Optional[str] = None
    xml_parseable: Optional[bool] = None
    xml_parse_error: Optional[str] = None
    expected_root: Optional[bool] = None
    carry_forward: bool = False
    terminal_failure: Optional[str] = None
    provider_attempts: int = 0
    retry_events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    run_id: str
    config: RunConfig
    paths: RunPaths
    steps: List[StepRecord] = field(default_factory=list)
    total_wall_seconds: float = 0.0
    completed: bool = False
    invocation_start_step: int = 0


def _read_ordering(orderings_json: Path, corpus: str, ordering_id: str) -> List[str]:
    import json
    data = json.loads(orderings_json.read_text())
    return list(data[corpus][ordering_id]["order"])


class RunAlreadyExists(RuntimeError):
    """Raised when a run directory holds artefacts and ``force`` is False."""


def _xml_integrity(text: str) -> tuple[bool, Optional[str], bool]:
    """Return ``(parseable, error, expected_root)`` for an exact response.

    ``ElementTree.fromstring`` rejects surrounding prose and multiple documents,
    which is intentional: the prompt contract requires one XML document and
    nothing else.  This is an execution-integrity gate, not XSD evaluation.
    """
    try:
        root = ET.fromstring(text)
    except (ET.ParseError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}", False
    return True, None, root.tag == "featureModel"


def run_generation(
    *,
    config: RunConfig,
    llm: GenerationLLM,
    orderings_json: Path | str,
    chunks_jsonl: Path | str,
    prompt_template_path: Path | str,
    metamodel_xsd_text: str,
    retrieval_service: Optional[Any] = None,   # fame.retrieval.RetrievalService
    results_root: Path | str = "results",
    token_counter: Optional[TokenCounter] = None,
    force: bool = False,
    resume_provider_failure: bool = False,
    # Progress callbacks (interactive campaign runner).
    # on_step_start(config, step_index, N, batch_doc_ids)
    # on_step_done(config, step_index, StepRecord)
    on_step_start: Optional[Any] = None,
    on_step_done:  Optional[Any] = None,
) -> RunResult:
    """Execute one full run per :class:`RunConfig`.

    Preconditions
    -------------
    * ``orderings_json`` contains ``config.corpus / config.ordering_id / order``
    * ``chunks_jsonl`` is the D11 store for ``config.corpus``
    * When ``config.grounding == "rag"``, ``retrieval_service`` must be a
      configured :class:`fame.retrieval.RetrievalService` for the same corpus
    """
    orderings_json = Path(orderings_json).expanduser().resolve()
    chunks_jsonl_p = Path(chunks_jsonl).expanduser().resolve()
    template_path  = Path(prompt_template_path).expanduser().resolve()

    ordering = _read_ordering(orderings_json, config.corpus, config.ordering_id)
    batches = slice_ordering(ordering, config.N)
    assert_covers_pi(batches, ordering)

    paths = RunPaths.for_run(
        results_root=results_root,
        campaign_id=config.campaign_id,
        corpus=config.corpus,
        config_hash=config.config_hash(),
        run_id=config.run_id(),
    )

    if force and resume_provider_failure:
        raise ValueError("force and resume_provider_failure are mutually exclusive")

    # 6.V5 resume guard — refuse to overwrite unless explicitly reset or a
    # validated provider-failure checkpoint is being continued.
    existing = []
    if paths.fm_gen.exists():
        existing.append(paths.fm_gen.name)
    if paths.fm_iter_dir.exists() and any(paths.fm_iter_dir.iterdir()):
        existing.append(f"{paths.fm_iter_dir.name}/")
    if paths.context_log.exists() and paths.context_log.stat().st_size > 0:
        existing.append(paths.context_log.name)
    if paths.run_meta.exists():
        existing.append(paths.run_meta.name)
    if existing and not force and not resume_provider_failure:
        raise RunAlreadyExists(
            f"run {config.run_id()} already has artefacts under {paths.root}: "
            f"{existing}. Pass force=True to overwrite."
        )
    if existing and force:
        # Explicit reset — remove all prior artefacts.
        for p in [paths.fm_gen, paths.context_log, paths.run_meta]:
            if p.exists():
                p.unlink()
        if paths.fm_iter_dir.exists():
            for p in paths.fm_iter_dir.iterdir():
                p.unlink()

    paths.ensure_dirs()

    result = RunResult(run_id=config.run_id(), config=config, paths=paths)
    previous_fm_xml: Optional[str] = None
    prior_meta: Dict[str, Any] = {}
    recovery_history: List[Dict[str, Any]] = []
    start_step = 0
    resume_expected_context: Optional[Dict[str, Any]] = None

    if resume_provider_failure:
        if not paths.run_meta.exists():
            raise ValueError("resume requires an existing run_meta.json")
        prior_meta = json.loads(paths.run_meta.read_text())
        if prior_meta.get("config") != config.canonical_dict():
            raise ValueError("resume checkpoint config does not match the selected matrix row")
        if prior_meta.get("completed") or prior_meta.get("terminal_status") != "provider_error":
            raise ValueError("resume is allowed only after terminal_status='provider_error'")
        failed_step = prior_meta.get("failed_step")
        prior_steps = prior_meta.get("steps") or []
        if (not isinstance(failed_step, int) or failed_step <= 0
                or failed_step >= config.N or len(prior_steps) != failed_step + 1):
            raise ValueError("provider-error checkpoint is not a resumable terminal step")
        successful = prior_steps[:failed_step]
        if any(s.get("step_index") != i or not s.get("carry_forward")
               for i, s in enumerate(successful)):
            raise ValueError("resume requires contiguous successful steps from step 0")
        failed = prior_steps[failed_step]
        if failed.get("step_index") != failed_step or failed.get("terminal_failure") != "provider_error":
            raise ValueError("failed checkpoint step is not a provider error")
        for i, step in enumerate(successful):
            xml_path = paths.root / str(step.get("fm_path") or "")
            if not xml_path.is_file():
                raise ValueError(f"resume checkpoint is missing XML for step {i}")
            parseable, _, expected_root = _xml_integrity(xml_path.read_text())
            if not parseable or not expected_root:
                raise ValueError(f"resume checkpoint XML for step {i} is invalid")
        previous_fm_xml = (paths.root / successful[-1]["fm_path"]).read_text()
        result.steps = [StepRecord(**{
            name: step.get(name)
            for name in StepRecord.__dataclass_fields__
        }) for step in successful]
        start_step = failed_step
        result.invocation_start_step = start_step
        recovery_history = list(prior_meta.get("recovery_history") or [])
        recovery_history.append({
            "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            "resumed_step": failed_step,
            "prior_terminal_status": prior_meta.get("terminal_status"),
            "prior_error": failed.get("error"),
        })
        context_rows = [json.loads(line) for line in paths.context_log.read_text().splitlines()
                        if line.strip()]
        if len(context_rows) == failed_step:
            # A prior resume may have stopped after removing the failed row but
            # before making an API call. Recover it from the immutable archive.
            results_root_path = paths.root.parents[3]
            archive_root = (results_root_path / "recovery_archive" / config.campaign_id
                            / config.corpus / config.run_id())
            candidates = sorted(
                archive_root.glob("*/context_log.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for candidate in candidates:
                archived_rows = [json.loads(line) for line in candidate.read_text().splitlines()
                                 if line.strip()]
                if (len(archived_rows) == failed_step + 1
                        and archived_rows[-1].get("step_index") == failed_step):
                    context_rows = archived_rows
                    recovery_history.append({
                        "restored_context_checkpoint": str(candidate),
                    })
                    break
        if len(context_rows) != failed_step + 1 or any(
                row.get("step_index") != i for i, row in enumerate(context_rows[:failed_step])):
            raise ValueError("context log does not contain exactly one failed-step checkpoint")
        resume_expected_context = context_rows[failed_step]
        atomic_write_text(
            paths.context_log,
            "".join(json.dumps(row, sort_keys=True) + "\n"
                    for row in context_rows[:failed_step]),
        )

    # Write config snapshot up front so a crash leaves the intent visible.
    atomic_write_json(paths.root / "run_config.json", config.canonical_dict())

    total_t0 = time.time()
    started_at = prior_meta.get("started_at_utc") or datetime.now(timezone.utc).isoformat()
    for batch in batches[start_step:]:
        step_t0 = time.time()
        if on_step_start:
            try: on_step_start(config, batch.step_index, config.N, batch.doc_ids)
            except Exception: pass
        # ── Grounding
        if (batch.step_index == start_step and resume_expected_context is not None
                and config.grounding == "rag"):
            gc = build_grounding_from_record(
                record=resume_expected_context,
                chunks_jsonl=chunks_jsonl_p,
            )
        else:
            gc = build_grounding(
                grounding=config.grounding,
                batch_doc_ids=batch.doc_ids,
                domain=config.domain,
                chunks_jsonl=chunks_jsonl_p,
                retrieval_service=retrieval_service,
                k_doc=config.k_doc,
            )
        context_text = format_context_text(gc)

        # ── Prompt
        bundle: PromptBundle = render_prompt(
            template_path=template_path,
            template_hash=config.prompt_template_hash,
            domain=config.domain,
            root_feature=config.root_feature,
            context_text=context_text,
            context_doc_ids=batch.doc_ids,
            metamodel_block=config.metamodel_block,
            metamodel_xsd=metamodel_xsd_text if config.metamodel_block else None,
            previous_model=(previous_fm_xml is not None),
            previous_fm_xml=previous_fm_xml,
        )

        # ── Feasibility guard
        feasible, assembled = is_feasible(
            bundle.text,
            context_window=config.context_window,
            max_output_tokens=config.max_output_tokens,
            counter=token_counter,
        )

        record = StepRecord(
            step_index=batch.step_index,
            batch_doc_ids=batch.doc_ids,
            grounding=gc.grounding,
            n_chunks=len(gc.chunks),
            k_step=gc.k_step,
            per_sub_query_k=gc.per_sub_query_k,
            assembled_tokens=assembled,
            feasible=feasible,
            finish_reason=None,
            prompt_tokens=None,
            completion_tokens=None,
            wall_seconds=0.0,
            fm_path=None,
        )

        # ── D9 context_log entry (written even for infeasible)
        context_entry = {
            "run_id":            config.run_id(),
            "step_index":        batch.step_index,
            "N":                 config.N,
            "grounding":         gc.grounding,
            "batch_doc_ids":     batch.doc_ids,
            "chunk_ids":         [c.chunk_id for c in gc.chunks],
            "chunk_doc_ids":     [c.doc_id for c in gc.chunks],
            "retrieval_scores":  [c.distance for c in gc.chunks],
            "sub_query_indices": [c.sub_query_index for c in gc.chunks],
            "k_step":            gc.k_step,
            "per_sub_query_k":   gc.per_sub_query_k,
            "assembled_tokens":  assembled,
            "feasible":          feasible,
        }
        if batch.step_index == start_step and resume_expected_context is not None:
            comparable_keys = (
                "step_index", "batch_doc_ids", "chunk_ids", "chunk_doc_ids",
                "retrieval_scores", "sub_query_indices", "k_step",
                "per_sub_query_k", "assembled_tokens", "feasible",
            )
            changed = [key for key in comparable_keys
                       if context_entry.get(key) != resume_expected_context.get(key)]
            if changed:
                raise ValueError(
                    "resumed step context differs from the recorded failed call: "
                    + ", ".join(changed)
                )
        atomic_append_jsonl(paths.context_log, context_entry)

        if not feasible:
            record.wall_seconds = round(time.time() - step_t0, 2)
            record.error = "over_context_window"
            record.terminal_failure = "over_context_window"
            result.steps.append(record)
            if on_step_done:
                try: on_step_done(config, batch.step_index, record)
                except Exception: pass
            break    # dependent refinement cannot skip a batch

        # ── LLM call.
        # Ollama supports either a boolean think toggle or a model-specific
        # effort level. GLM 5.3 Flash and GPT-OSS require level strings; the
        # other configured families retain the boolean low/off mapping.
        # OpenAI reasoning models get reasoning_effort passed straight through.
        think_flag: Optional[Union[bool, str]]
        if config.provider == "ollama_cloud":
            think_flag = _ollama_think_value(config.model_id, config.reasoning_effort)
        else:
            think_flag = None
        try:
            resp: GenerationResponse = llm.generate(GenerationRequest(
                prompt=bundle.text,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
                think=think_flag,
                seed=config.seed,
            ))
        except Exception as exc:
            record.wall_seconds = round(time.time() - step_t0, 2)
            record.error = f"{type(exc).__name__}: {exc}"
            record.terminal_failure = "provider_error"
            record.provider_attempts = int(getattr(exc, "fame_attempts", 1))
            record.retry_events = list(getattr(exc, "fame_retry_events", []))
            result.steps.append(record)
            if on_step_done:
                try: on_step_done(config, batch.step_index, record)
                except Exception: pass
            break

        # ── Validate before carry-forward.  A failed response is preserved
        # verbatim as .raw.txt but never presented as a valid feature model.
        step_fm_path = paths.fm_iter_dir / f"step_{batch.step_index:02d}.xml"
        raw_path = paths.fm_iter_dir / f"step_{batch.step_index:02d}.raw.txt"
        record.finish_reason   = resp.finish_reason
        record.prompt_tokens   = resp.prompt_tokens
        record.completion_tokens = resp.completion_tokens
        record.wall_seconds    = round(time.time() - step_t0, 2)
        record.provider_attempts = int(resp.raw.get("fame_provider_attempts", 1))
        record.retry_events = list(resp.raw.get("fame_retry_events", []))

        response_present = bool(resp.text and resp.text.strip())
        if response_present:
            parseable, parse_error, expected_root = _xml_integrity(resp.text)
            record.xml_parseable = parseable
            record.xml_parse_error = parse_error
            record.expected_root = expected_root

        if resp.finish_reason == "length":
            record.error = "non_stop_finish_reason:length"
            record.terminal_failure = "truncated_output"
            atomic_write_text(raw_path, resp.text or "")
            record.raw_response_path = str(raw_path.relative_to(paths.root))
        elif not response_present:
            record.error = f"empty_response (finish_reason={resp.finish_reason})"
            record.terminal_failure = "empty_response"
            atomic_write_text(raw_path, resp.text or "")
            record.raw_response_path = str(raw_path.relative_to(paths.root))
        elif resp.finish_reason != "stop":
            record.error = f"non_stop_finish_reason:{resp.finish_reason}"
            record.terminal_failure = (
                "truncated_output" if resp.finish_reason == "length"
                else "non_stop_finish_reason"
            )
            atomic_write_text(raw_path, resp.text)
            record.raw_response_path = str(raw_path.relative_to(paths.root))
        else:
            if not record.xml_parseable:
                record.error = f"malformed_xml:{record.xml_parse_error}"
                record.terminal_failure = "malformed_xml"
                atomic_write_text(raw_path, resp.text)
                record.raw_response_path = str(raw_path.relative_to(paths.root))
            elif not record.expected_root:
                record.error = "unexpected_xml_root"
                record.terminal_failure = "unexpected_xml_root"
                atomic_write_text(raw_path, resp.text)
                record.raw_response_path = str(raw_path.relative_to(paths.root))
            else:
                atomic_write_text(step_fm_path, resp.text)
                record.fm_path = str(step_fm_path.relative_to(paths.root))
                record.carry_forward = True
                previous_fm_xml = resp.text
        result.steps.append(record)
        if on_step_done:
            try: on_step_done(config, batch.step_index, record)
            except Exception: pass
        if not record.carry_forward:
            break

    # ── D7 exists only for a complete run.  A last-valid checkpoint from a
    # partial run remains in fm_iter/ and must not masquerade as the final FM.
    result.completed = (
        len(result.steps) == config.N
        and all(s.carry_forward for s in result.steps)
    )
    if result.completed and previous_fm_xml is not None:
        atomic_write_text(paths.fm_gen, previous_fm_xml)

    total_wall = float(prior_meta.get("total_wall_seconds") or 0.0) + time.time() - total_t0
    result.total_wall_seconds = round(total_wall, 2)
    successful_steps = [s for s in result.steps if s.carry_forward]
    failed = next((s for s in result.steps if not s.carry_forward), None)
    documents_processed = sum(len(s.batch_doc_ids) for s in successful_steps)
    observed_completions = [
        s.completion_tokens for s in result.steps if s.completion_tokens is not None
    ]
    successful_completions = [
        s.completion_tokens for s in successful_steps if s.completion_tokens is not None
    ]
    max_successful_completion = max(successful_completions) if successful_completions else None
    output_headroom = (
        config.max_output_tokens - max_successful_completion
        if max_successful_completion is not None else None
    )

    atomic_write_json(paths.run_meta, {
        "run_id":              config.run_id(),
        "config":              config.canonical_dict(),
        "llm_provider":        llm.provider,
        "llm_model_id":        llm.model_id,
        "execution_lane":      config.extra.get("lane"),
        "started_at_utc":      started_at,
        "ended_at_utc":        datetime.now(timezone.utc).isoformat(),
        "steps":               [_step_to_dict(s) for s in result.steps],
        "total_wall_seconds":  result.total_wall_seconds,
        "completed":           result.completed,
        "terminal_status":     "completed" if result.completed else (
            failed.terminal_failure or failed.error or "incomplete"
            if failed else "incomplete"
        ),
        "failed_step":         failed.step_index if failed else None,
        "last_valid_step":     successful_steps[-1].step_index if successful_steps else None,
        "completed_steps":     len(successful_steps),
        "planned_steps":       config.N,
        "documents_processed": documents_processed,
        "documents_planned":   len(ordering),
        "corpus_fraction_processed": round(documents_processed / len(ordering), 6),
        "max_completion_tokens_observed": max(observed_completions) if observed_completions else None,
        "max_successful_completion_tokens": max_successful_completion,
        "minimum_successful_output_headroom_tokens": output_headroom,
        "minimum_successful_output_headroom_percent": (
            round(100.0 * output_headroom / config.max_output_tokens, 3)
            if output_headroom is not None else None
        ),
        "length_finish_count": sum(s.finish_reason == "length" for s in result.steps),
        "provider_attempts":   sum(s.provider_attempts for s in result.steps),
        "provider_retry_count": sum(max(0, s.provider_attempts - 1) for s in result.steps),
        "provider_retry_events": [
            {"step_index": s.step_index, **event}
            for s in result.steps for event in s.retry_events
        ],
        "recovery_history": recovery_history,
    })

    return result


def _step_to_dict(s: StepRecord) -> Dict[str, Any]:
    return {
        "step_index":         s.step_index,
        "batch_doc_ids":      s.batch_doc_ids,
        "grounding":          s.grounding,
        "n_chunks":           s.n_chunks,
        "k_step":             s.k_step,
        "per_sub_query_k":    s.per_sub_query_k,
        "assembled_tokens":   s.assembled_tokens,
        "feasible":           s.feasible,
        "finish_reason":      s.finish_reason,
        "prompt_tokens":      s.prompt_tokens,
        "completion_tokens":  s.completion_tokens,
        "wall_seconds":       s.wall_seconds,
        "fm_path":            s.fm_path,
        "error":              s.error,
        "raw_response_path":  s.raw_response_path,
        "xml_parseable":      s.xml_parseable,
        "xml_parse_error":    s.xml_parse_error,
        "expected_root":      s.expected_root,
        "carry_forward":      s.carry_forward,
        "terminal_failure":   s.terminal_failure,
        "provider_attempts":  s.provider_attempts,
        "retry_events":       s.retry_events,
    }
