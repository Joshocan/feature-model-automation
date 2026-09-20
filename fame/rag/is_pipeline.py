from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from tempfile import NamedTemporaryFile

from fame.utils.dirs import build_paths, ensure_dir, ensure_for_stage
from fame.vectorization.pipeline import default_collection_name
from fame.retrieval.service import RetrievalService
from fame.nonrag.prompting import render_prompt_template, serialize_high_level_features
from fame.nonrag.llm_ollama_http import OllamaHTTP, assert_ollama_running
from fame.utils.placeholder_check import assert_no_placeholders, UnresolvedPlaceholdersError
from fame.exceptions import PlaceholderError, MissingChunksError
from fame.evaluation import start_timer, elapsed_seconds
from fame.evaluation.wellformed import validate_feature_model


@dataclass(frozen=True)
class ISRgfmConfig:
    root_feature: str
    domain: str

    chunks_dir: Optional[Path] = None
    chunks_files: Optional[Sequence[Path]] = None

    initial_prompt_path: Optional[Path] = None
    iter_prompt_path: Optional[Path] = None

    n_results_per_collection: Optional[int] = None
    max_total_results: Optional[int] = None
    max_total_chars: int = 18_000
    max_chunk_chars: int = 2_500

    xsd_path: Optional[Path] = None
    feature_metamodel_path: Optional[Path] = None
    high_level_features: Optional[Dict[str, str]] = None
    temperature: float = 0.2
    max_retries_per_iteration: int = 1
    retry_backoff_base_seconds: float = 2.0
    retry_backoff_cap_seconds: float = 60.0
    inter_iteration_sleep_seconds: float = 1.0

    run_tag: str = "is-rgfm"
    dataset: str = ""
    collection_prefix: str = ""


def _default_chunks_dir(paths) -> Path:
    return paths.processed_data / "chunks"


def _list_chunks_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("*.chunks.json"))


def _validate_xml_text(xml_text: str, xsd_path: Path) -> tuple[bool, List[str]]:
    if not xml_text or not xml_text.strip():
        return False, ["Empty model output"]
    with NamedTemporaryFile("w", suffix=".xml", encoding="utf-8", delete=True) as tmp:
        tmp.write(xml_text)
        tmp.flush()
        result = validate_feature_model(tmp.name, xsd_path=xsd_path)
    return result.ok, result.errors


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _backoff_seconds(attempt: int, base: float, cap: float) -> float:
    delay = max(0.0, float(base)) * (2 ** max(0, attempt - 1))
    return min(delay, max(0.0, float(cap)))

def run_is_rgfm(cfg: ISRgfmConfig, *, llm: Optional[object] = None, retriever: Optional[RetrievalService] = None) -> Dict[str, str]:
    run_wall_clock_start = time.perf_counter()
    paths = build_paths()
    ensure_for_stage("is-rgfm", paths)
    dataset_subdir = (cfg.dataset or '').strip()
    ensure_for_stage("preprocess", paths)

    # LLM
    if llm is None:
        assert_ollama_running()
        llm = OllamaHTTP()
    model_name = getattr(llm, "model", "unknown")

    # Files
    chunks_dir = cfg.chunks_dir or _default_chunks_dir(paths)
    ensure_dir(chunks_dir)
    files = list(cfg.chunks_files) if cfg.chunks_files else _list_chunks_files(chunks_dir)
    if not files:
        raise MissingChunksError(str(chunks_dir))

    retr = retriever or RetrievalService()

    spec_dir = paths.specifications
    xsd_path = cfg.xsd_path or (spec_dir / "feature_model_featureide.xsd")
    xsd_text = xsd_path.read_text(encoding="utf-8") if xsd_path.exists() else ""
    high_level_xml = serialize_high_level_features(cfg.high_level_features)

    previous_xml = ""
    iteration_meta = []
    ts_start = time.strftime("%Y-%m-%dT%H-%M-%S")
    run_id_base = f"is_rgfm_response_{re.sub(r'[^a-zA-Z0-9]+', '-', model_name).strip('-').lower() or 'unknown'}_{ts_start}"
    max_retries = max(1, int(cfg.max_retries_per_iteration))

    for i, f in enumerate(files, start=1):
        iter_wall_clock_start = time.perf_counter()
        base_collection = default_collection_name(f)
        collection = f"{cfg.collection_prefix}{base_collection}" if cfg.collection_prefix else base_collection
        retrieval_start = time.perf_counter()
        res = retr.retrieve(
            root_feature=cfg.root_feature,
            domain=cfg.domain,
            collections=[collection],
            n_results_per_collection=cfg.n_results_per_collection or 6,
            max_total_results=cfg.max_total_results or 12,
        )
        context = retr.to_prompt_evidence(
            res,
            max_total_chars=cfg.max_total_chars,
            max_chunk_chars=cfg.max_chunk_chars,
        )
        retrieval_duration = time.perf_counter() - retrieval_start

        prompt_build_start = time.perf_counter()
        tmpl_text = (
            Path(cfg.initial_prompt_path).read_text(encoding="utf-8")
            if cfg.initial_prompt_path and Path(cfg.initial_prompt_path).exists() and i == 1
            else Path(cfg.iter_prompt_path).read_text(encoding="utf-8")
            if cfg.iter_prompt_path and Path(cfg.iter_prompt_path).exists() and i > 1
            else ""
        )
        if not tmpl_text:
            # fallback to defaults
            default_init = (paths.base_dir / "prompts" / "fm_extraction_prompt.txt")
            default_iter = (paths.base_dir / "prompts" / "fm_iterated_prompt.txt")
            tmpl_path = default_init if i == 1 else default_iter
            tmpl_text = tmpl_path.read_text(encoding="utf-8")

        values = {
            "ROOT_FEATURE": cfg.root_feature,
            "DOMAIN": cfg.domain,
            "CONTEXT": context,
            "PREVIOUS_FM_XML": previous_xml or "(empty)",
            "HIGH_LEVEL_FEATURES": high_level_xml,
            "XSD_METAMODEL": xsd_text,
        }

        prompt = render_prompt_template(tmpl_text, values=values, strict=True)
        try:
            assert_no_placeholders(prompt)
        except UnresolvedPlaceholdersError as e:
            raise PlaceholderError(e.placeholders) from e
        prompt_build_duration = time.perf_counter() - prompt_build_start

        iter_tag = f"{run_id_base}_iter{i:02d}_{f.stem}"
        prompt_dir = (paths.reports / dataset_subdir) if dataset_subdir else paths.reports
        context_dir = (paths.results / "rag" / "is-rgfm" / "context" / dataset_subdir) if dataset_subdir else (paths.results / "rag" / "is-rgfm" / "context")
        runs_dir = (paths.results / "rag" / "is-rgfm" / "runs" / dataset_subdir) if dataset_subdir else (paths.results / "rag" / "is-rgfm" / "runs")
        prompt_file = prompt_dir / f"{iter_tag}.prompt.txt"
        context_file = context_dir / f"{iter_tag}.context.txt"
        xml_file = runs_dir / f"{iter_tag}.xml"
        ensure_dir(prompt_file.parent)
        ensure_dir(context_file.parent)
        ensure_dir(xml_file.parent)

        print(f"⏳ Iteration {i}/{len(files)} — source: {f.name}")
        out_xml = ""
        validation_errors: List[str] = []
        last_err = None
        attempt_durations_seconds: List[float] = []
        validation_duration_seconds = 0.0
        retry_backoff_seconds = 0.0
        inter_iteration_sleep_seconds = 0.0
        iter_duration = 0.0
        for attempt in range(1, max_retries + 1):
            try:
                t0 = start_timer()
                candidate_xml = llm.generate(prompt, temperature=cfg.temperature)
                iter_duration = elapsed_seconds(t0)
                attempt_durations_seconds.append(iter_duration)
                validation_start = time.perf_counter()
                ok, validation_errors = _validate_xml_text(candidate_xml, xsd_path)
                validation_duration_seconds += time.perf_counter() - validation_start
                if not ok:
                    if attempt < max_retries:
                        reason = "; ".join(validation_errors[:2])
                        print(f"   → attempt {attempt}/{max_retries} produced invalid XML: {reason}. Retrying iteration...")
                        continue
                    raise ValueError("; ".join(validation_errors))
                out_xml = candidate_xml
                previous_xml = out_xml
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    if _is_rate_limit_error(e):
                        delay = _backoff_seconds(attempt, cfg.retry_backoff_base_seconds, cfg.retry_backoff_cap_seconds)
                        retry_backoff_seconds += delay
                        print(f"   → attempt {attempt}/{max_retries} failed: {e}. Backing off for {delay:.1f}s before retry...")
                        time.sleep(delay)
                    else:
                        print(f"   → attempt {attempt}/{max_retries} failed: {e}. Retrying iteration...")
                else:
                    raise

        prompt_file.write_text(prompt, encoding="utf-8")
        context_file.write_text(context, encoding="utf-8")
        xml_file.write_text(out_xml, encoding="utf-8")
        iteration_wall_clock_seconds = time.perf_counter() - iter_wall_clock_start

        iteration_meta.append(
            {
                "iter": i,
                "source_chunks": str(f),
                "collection": collection,
                "prompt": str(prompt_file),
                "context": str(context_file),
                "xml": str(xml_file),
                "llm_duration_seconds": iter_duration,
                "attempt_durations_seconds": attempt_durations_seconds,
                "retrieval_duration_seconds": retrieval_duration,
                "prompt_build_duration_seconds": prompt_build_duration,
                "validation_duration_seconds": validation_duration_seconds,
                "retry_backoff_seconds": retry_backoff_seconds,
                "inter_iteration_sleep_seconds": inter_iteration_sleep_seconds,
                "iteration_wall_clock_seconds": iteration_wall_clock_seconds,
                "attempts_used": attempt,
                "xml_valid": True,
            }
        )

        if i < len(files) and cfg.inter_iteration_sleep_seconds > 0:
            sleep_s = float(cfg.inter_iteration_sleep_seconds)
            time.sleep(sleep_s)
            inter_iteration_sleep_seconds += sleep_s
            iteration_meta[-1]["inter_iteration_sleep_seconds"] = inter_iteration_sleep_seconds

    fm_dir = (paths.is_fm / dataset_subdir) if dataset_subdir else paths.is_fm
    reports_dir = (paths.is_reports / dataset_subdir) if dataset_subdir else paths.is_reports
    final_xml_file = fm_dir / f"{run_id_base}.final.xml"
    ensure_dir(final_xml_file.parent)
    ensure_dir(reports_dir)
    final_xml_file.write_text(previous_xml, encoding="utf-8")

    meta_file = reports_dir / f"{run_id_base}.meta.json"
    meta = {
        "run_id": run_id_base,
        "root_feature": cfg.root_feature,
        "domain": cfg.domain,
        "num_sources": len(files),
        "llm_host": getattr(llm, "host", getattr(llm, "base_url", "")),
        "llm_model": model_name,
        "total_llm_duration_seconds": sum(float(m.get("llm_duration_seconds", 0)) for m in iteration_meta),
        "total_retrieval_duration_seconds": sum(float(m.get("retrieval_duration_seconds", 0)) for m in iteration_meta),
        "total_prompt_build_duration_seconds": sum(float(m.get("prompt_build_duration_seconds", 0)) for m in iteration_meta),
        "total_validation_duration_seconds": sum(float(m.get("validation_duration_seconds", 0)) for m in iteration_meta),
        "total_retry_backoff_seconds": sum(float(m.get("retry_backoff_seconds", 0)) for m in iteration_meta),
        "total_inter_iteration_sleep_seconds": sum(float(m.get("inter_iteration_sleep_seconds", 0)) for m in iteration_meta),
        "total_iteration_wall_clock_seconds": sum(float(m.get("iteration_wall_clock_seconds", 0)) for m in iteration_meta),
        "total_wall_clock_seconds": time.perf_counter() - run_wall_clock_start,
        "iterations": iteration_meta,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"SUCCESS: IS-RGFM finished. Final FM: {final_xml_file}")

    return {
        "final_xml": str(final_xml_file),
        "meta": str(meta_file),
        "run_id": run_id_base,
    }
