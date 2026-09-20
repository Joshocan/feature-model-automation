from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List, Optional, Sequence

from fame.context import ContextBuildConfig, ContextManager, chunks_from_chunks_json
from fame.ingestion.pipeline import ingest_and_prepare
from fame.utils.dirs import build_paths, ensure_for_stage, ensure_dir
from fame.utils.context_budget import compute_max_total_chars, compute_max_chunks
from fame.evaluation import start_timer, elapsed_seconds
from fame.utils.placeholder_check import assert_no_placeholders, UnresolvedPlaceholdersError
from fame.exceptions import PlaceholderError, MissingChunksError

from .llm_ollama_http import OllamaHTTP, assert_ollama_running
from .prompt_utils import build_ss_nonrag_prompt, save_modified_prompt


@dataclass(frozen=True)
class SSNonRagConfig:
    root_feature: str
    domain: str

    # which chunks to use (default: all produced chunks.json)
    chunks_dir: Optional[Path] = None
    chunks_files: Optional[Sequence[Path]] = None

    # context budget
    max_total_chars: int = 140_000
    max_chunks: int = 120
    max_chunk_chars: int = 6_000

    # prompt
    prompt_path: Optional[Path] = None
    xsd_path: Optional[Path] = None
    feature_metamodel_path: Optional[Path] = None
    high_level_features: Optional[Dict[str, str]] = None
    max_depth: Optional[int] = None
    max_retries: int = 1

    # core-metamodel prompt placeholders (used by prompts/core_structure_metamodel_extraction.txt).
    # Both are file paths. Contents are read at prompt-build time and
    # injected as {{DOMAIN_VOCABULARY}} / {{ARTEFACTS_REGISTRY}}. Both
    # resolve to empty string when unset — safe for FM-style prompts
    # that don't declare these placeholders.
    domain_vocabulary_path: Optional[Path] = None
    artefacts_registry_path: Optional[Path] = None

    # output naming
    run_tag: str = "ss-nonrag"
    dataset: str = ""


def _default_chunks_dir(paths) -> Path:
    # your ingestion writes here: processed_data/chunks
    return paths.processed_data / "chunks"


def _list_chunks_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("*.chunks.json"))


def run_ss_nonrag(cfg: SSNonRagConfig, *, llm_client: Optional[object] = None) -> Dict[str, str]:
    """
    SS-NonRAG:
      - builds one big evidence context from chunks.json
      - invokes LLM once
      - saves context + prompt + output
    Returns dict with output paths.
    """
    print(
        """
===========================================================
SS-NONRAG PIPELINE — Single-Stage Non-RAG Feature Modeling
===========================================================
This pipeline constructs a Feature Model using a single, global in-learning
context derived from all available source documents, without retrieval,
ranking, or iterative refinement.

It is designed to serve as a strong Non-RAG baseline and a comparison point
against RAG based pipelines

No vector database or retrieval step is used.
"""
    )
    paths = build_paths()
    dataset_subdir = (cfg.dataset or '').strip()
    ensure_for_stage("ss-nonrag", paths)  # creates NON_SS_* dirs
    ensure_for_stage("preprocess", paths)  # ensures processed_data exists

    # Ensure Ollama server exists (only when using Ollama client)
    if llm_client is None:
        assert_ollama_running()

    # If chunks aren't there yet, optionally ingest raw PDFs
    chunks_dir = cfg.chunks_dir or _default_chunks_dir(paths)
    ensure_dir(chunks_dir)

    files = list(cfg.chunks_files) if cfg.chunks_files else _list_chunks_files(chunks_dir)
    if not files:
        # run ingestion to create chunks.json
        ingest_and_prepare(raw_dir=paths.raw_data, out_dir=chunks_dir)
        files = _list_chunks_files(chunks_dir)

    if not files:
        raise MissingChunksError(str(chunks_dir))

    # Estimate context budget based on model context window (when known)
    model_name_hint = getattr(llm_client, "model", None) or os.getenv("OLLAMA_LLM_MODEL", "")
    model_hint = str(model_name_hint).strip()
    est_chars = compute_max_total_chars(model_hint)
    if est_chars > 0:
        est_chunks = compute_max_chunks(max_total_chars=est_chars)
        cfg = SSNonRagConfig(
            **{**cfg.__dict__, "max_total_chars": est_chars, "max_chunks": est_chunks}
        )
        print(
            f"ℹ️  Context budget adjusted for {model_hint}: "
            f"max_total_chars={est_chars}, max_chunks={est_chunks}"
        )

    # Build one big in-learning context (all sources)
    cm = ContextManager()
    ctx_cfg = ContextBuildConfig(
        max_total_chars=cfg.max_total_chars,
        max_chunks=cfg.max_chunks,
        max_chunk_chars=cfg.max_chunk_chars,
        include_headers=True,
        order="by_page_then_id",
    )

    all_chunks = []
    for f in files:
        all_chunks.extend(chunks_from_chunks_json(f))

    # If chunks exist but are empty, re-run ingestion once to repopulate.
    if not all_chunks:
        ingest_and_prepare(raw_dir=paths.raw_data, out_dir=chunks_dir)
        files = _list_chunks_files(chunks_dir)
        all_chunks = []
        for f in files:
            all_chunks.extend(chunks_from_chunks_json(f))

    if not all_chunks:
        raise RuntimeError(f"Chunks exist but are empty in {chunks_dir}. Check ingestion output.")

    context = cm.add_initial_context(all_chunks, ctx_cfg, title="IN-LEARNING CONTEXT (SS-NONRAG)")

    # Build prompt
    prompt = build_ss_nonrag_prompt(cfg, context=context, paths=paths)
    try:
        assert_no_placeholders(prompt)
    except UnresolvedPlaceholdersError as e:
        raise PlaceholderError(e.placeholders) from e

    # Run LLM once
    if llm_client is None:
        llm = OllamaHTTP()
    else:
        llm = llm_client

    model_name = getattr(llm, "model", "unknown")
    print(f"⏳ SS Non RAG: Running the LLM ({model_name})... this may take a while, please wait....")

    for attempt in range(1, max(1, int(cfg.max_retries)) + 1):
        try:
            t0 = start_timer()
            if llm_client is None:
                fm_xml = llm.generate(prompt, temperature=0.2)
            else:
                fm_xml = llm.generate(prompt)
            llm_duration = elapsed_seconds(t0)
            break
        except Exception as e:
            if attempt < max(1, int(cfg.max_retries)):
                print(f"   -> attempt {attempt}/{cfg.max_retries} failed: {e}. Retrying...")
            else:
                raise

    # Save artifacts
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    model_safe = re.sub(r"[^a-zA-Z0-9]+", "-", str(model_name)).strip("-").lower()
    run_id = f"ss_nonrag_response_{model_safe}_{ts}"

    context_dir = (paths.non_ss_context / dataset_subdir) if dataset_subdir else paths.non_ss_context
    runs_dir = (paths.non_ss_runs / dataset_subdir) if dataset_subdir else paths.non_ss_runs
    reports_dir = (paths.non_ss_reports / dataset_subdir) if dataset_subdir else paths.non_ss_reports
    fm_dir = (paths.non_ss_fm / dataset_subdir) if dataset_subdir else paths.non_ss_fm
    ensure_dir(context_dir)
    ensure_dir(runs_dir)
    ensure_dir(reports_dir)
    ensure_dir(fm_dir)
    context_file = context_dir / f"{run_id}.context.txt"
    prompt_file = runs_dir / f"{run_id}.prompt.txt"
    meta_file = reports_dir / f"{run_id}.meta.json"
    fm_file = fm_dir / f"{run_id}.xml"

    context_file.write_text(context, encoding="utf-8")
    prompt_file.write_text(prompt, encoding="utf-8")
    modified_prompt_file = save_modified_prompt(
        prompt=prompt, model_safe=model_safe, ts=ts, paths=paths, pipeline_type="ss_nonrag"
    )
    fm_file.write_text(fm_xml, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "root_feature": cfg.root_feature,
        "domain": cfg.domain,
        "num_sources": len(files),
        "num_chunks_total": len(all_chunks),
        "context_chars": len(context),
        "llm_host": getattr(llm, "host", getattr(llm, "base_url", "")),
        "llm_model": model_name,
        "llm_duration_seconds": llm_duration,
        "attempts_used": attempt,
        "chunks_files": [str(p) for p in files],
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "context": str(context_file),
        "prompt": str(prompt_file),
        "meta": str(meta_file),
        "fm_xml": str(fm_file),
    }
