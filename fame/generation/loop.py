"""Unified N-step generation loop (Phase 5, all sub-tasks).

For each step j = 0..N-1:
    1. Build the current batch B_j from π and N (:func:`slice_ordering`)
    2. Assemble grounding context (RAG or Non-RAG)
    3. Render prompt with invariant-first blocks + previous_model switch
    4. Pre-flight token count; if over-limit → mark infeasible, skip
    5. Call the LLM; capture finish_reason
    6. Persist fm_iter/step_<j>.xml + context_log line
    7. Update previous_model with the latest FM

At the end, write fm_gen.xml (final) + run_meta.json (D10).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .batching import assert_covers_pi, slice_ordering
from .grounding import GroundingContext, build_grounding, format_context_text
from .llm_client import GenerationLLM, GenerationRequest, GenerationResponse
from .persistence import RunPaths, atomic_append_jsonl, atomic_write_json, atomic_write_text
from .prompt_assembly import PromptBundle, render_prompt
from .run import RunConfig
from .token_budget import TokenCounter, is_feasible


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


@dataclass
class RunResult:
    run_id: str
    config: RunConfig
    paths: RunPaths
    steps: List[StepRecord] = field(default_factory=list)
    total_wall_seconds: float = 0.0
    completed: bool = False


def _read_ordering(orderings_json: Path, corpus: str, ordering_id: str) -> List[str]:
    import json
    data = json.loads(orderings_json.read_text())
    return list(data[corpus][ordering_id]["order"])


class RunAlreadyExists(RuntimeError):
    """Raised when a run directory holds artefacts and ``force`` is False."""


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

    # 6.V5 resume guard — refuse to overwrite unless force=True.
    existing = []
    if paths.fm_gen.exists():
        existing.append(paths.fm_gen.name)
    if paths.fm_iter_dir.exists() and any(paths.fm_iter_dir.iterdir()):
        existing.append(f"{paths.fm_iter_dir.name}/")
    if paths.context_log.exists() and paths.context_log.stat().st_size > 0:
        existing.append(paths.context_log.name)
    if paths.run_meta.exists():
        existing.append(paths.run_meta.name)
    if existing and not force:
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

    # Write config snapshot up front so a crash leaves the intent visible.
    atomic_write_json(paths.root / "run_config.json", config.canonical_dict())

    result = RunResult(run_id=config.run_id(), config=config, paths=paths)
    previous_fm_xml: Optional[str] = None

    total_t0 = time.time()
    for batch in batches:
        step_t0 = time.time()
        # ── Grounding
        gc: GroundingContext = build_grounding(
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
        atomic_append_jsonl(paths.context_log, {
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
        })

        if not feasible:
            record.wall_seconds = round(time.time() - step_t0, 2)
            record.error = "over_context_window"
            result.steps.append(record)
            continue    # do not call the model

        # ── LLM call
        try:
            resp: GenerationResponse = llm.generate(GenerationRequest(
                prompt=bundle.text,
                max_output_tokens=config.max_output_tokens,
                temperature=config.temperature,
                reasoning_effort=config.reasoning_effort,
                seed=config.seed,
            ))
        except Exception as exc:
            record.wall_seconds = round(time.time() - step_t0, 2)
            record.error = f"{type(exc).__name__}: {exc}"
            result.steps.append(record)
            continue

        # ── Persist step FM (D8)
        step_fm_path = paths.fm_iter_dir / f"step_{batch.step_index:02d}.xml"
        atomic_write_text(step_fm_path, resp.text)
        record.fm_path         = str(step_fm_path.relative_to(paths.root))
        record.finish_reason   = resp.finish_reason
        record.prompt_tokens   = resp.prompt_tokens
        record.completion_tokens = resp.completion_tokens
        record.wall_seconds    = round(time.time() - step_t0, 2)

        previous_fm_xml = resp.text
        result.steps.append(record)

    # ── D7 final FM (from last successful step) + D10 meta
    if previous_fm_xml is not None:
        atomic_write_text(paths.fm_gen, previous_fm_xml)

    total_wall = time.time() - total_t0
    result.total_wall_seconds = round(total_wall, 2)
    result.completed = any(s.finish_reason == "stop" for s in result.steps) or (
        previous_fm_xml is not None
    )

    atomic_write_json(paths.run_meta, {
        "run_id":              config.run_id(),
        "config":              config.canonical_dict(),
        "llm_provider":        llm.provider,
        "llm_model_id":        llm.model_id,
        "steps":               [_step_to_dict(s) for s in result.steps],
        "total_wall_seconds":  result.total_wall_seconds,
        "completed":           result.completed,
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
    }
