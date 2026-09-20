from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from fame.utils.dirs import build_paths, ensure_for_stage, ensure_dir
from fame.ingestion.pipeline import ingest_and_prepare
from fame.vectorization.pipeline import index_all_chunks, default_collection_name
from fame.retrieval.service import RetrievalService
from fame.nonrag.llm_ollama_http import OllamaHTTP, assert_ollama_running
from fame.nonrag.prompt_utils import save_modified_prompt
from fame.nonrag.prompting import render_prompt_template, serialize_high_level_features
from fame.evaluation import start_timer, elapsed_seconds
from fame.exceptions import MissingChunksError
from fame.loggers import get_logger
import json as _json
from fame.vectorization.chroma_health import assert_chroma_running


@dataclass(frozen=True)
class SSRGFMConfig:
    root_feature: str
    domain: str

    # ingestion/chunks
    chunks_dir: Optional[Path] = None
    chunks_files: Optional[Sequence[Path]] = None

    # vectorization
    collection_mode: str = "per_source"   # per_source | one_collection
    one_collection_name: str = "fame_all"
    collection_prefix: str = ""
    batch_size: int = 24

    # retrieval
    n_results_per_collection: int = 6
    max_total_results: int = 12
    max_total_chars: int = 18_000
    max_chunk_chars: int = 2_500
    k_strategy: str = "auto"  # auto | fixed

    # prompt + generation
    prompt_path: Optional[Path] = None
    xsd_path: Optional[Path] = None
    feature_metamodel_path: Optional[Path] = None
    high_level_features: Optional[Dict[str, str]] = None
    temperature: float = 0.2
    max_retries: int = 1

    # core-metamodel prompt placeholders (see fame.nonrag.ss_pipeline for docs).
    domain_vocabulary_path: Optional[Path] = None
    artefacts_registry_path: Optional[Path] = None

    # output naming
    run_tag: str = "ss-rgfm"
    dataset: str = ""


def _default_chunks_dir(paths) -> Path:
    return paths.processed_data / "chunks"


def _list_chunks_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("*.chunks.json"))


def _count_total_chunks(files: Sequence[Path]) -> int:
    total = 0
    for f in files:
        try:
            payload = _json.loads(f.read_text(encoding="utf-8"))
            chunks = payload.get("chunks") or payload.get("data") or []
            total += len(chunks)
        except Exception:
            continue
    return total


def _collection_name_for_file(fp: Path, prefix: str = "") -> str:
    base = default_collection_name(fp)
    return f"{prefix}{base}" if prefix else base


def load_ss_rgfm_prompt(prompt_path: Optional[Path], base_dir: Path) -> str:
    """
    Load prompt template for SS-RGFM. Defaults to prompts/fm_extraction_prompt.txt.
    """
    candidate = Path(prompt_path).expanduser() if prompt_path else (base_dir / "prompts" / "fm_extraction_prompt.txt")
    if candidate.exists():
        return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt template not found: {candidate}")


def run_ss_rgfm(
    cfg: SSRGFMConfig,
    *,
    llm: Optional[object] = None,
    retriever: Optional[object] = None,
    skip_vectorize: bool = False,
) -> Dict[str, str]:
    """
    SS-RGFM:
      - ensure Ollama server running (for generation)
      - ensure chunks exist (ingestion if needed)
      - vectorize chunks into Chroma
      - retrieve evidence using default query template
      - build prompt and call LLM once
      - save FM + evidence + prompt + meta
    """
    logger = get_logger("ss_rgfm")
    paths = build_paths()
    ensure_for_stage("ss-rgfm", paths)
    dataset_subdir = (cfg.dataset or '').strip()
    ensure_for_stage("preprocess", paths)
    ensure_for_stage("vectorize", paths)

    # LLM server for generation (Ollama) unless provided
    if llm is None:
        assert_ollama_running()
        llm = OllamaHTTP()

    # Ensure Chroma is reachable before vectorization/retrieval
    assert_chroma_running()

    # Ensure chunks exist
    chunks_dir = cfg.chunks_dir or _default_chunks_dir(paths)
    ensure_dir(chunks_dir)

    files = list(cfg.chunks_files) if cfg.chunks_files else _list_chunks_files(chunks_dir)
    if not files:
        ingest_and_prepare(raw_dir=paths.raw_data, out_dir=chunks_dir)
        files = _list_chunks_files(chunks_dir)

    if not files:
        raise MissingChunksError(str(chunks_dir))

    vec_out = None
    if not skip_vectorize:
        if cfg.collection_mode == "one_collection":
            logger.warning("collection_mode=one_collection not fully supported; using per_source collections instead.")
        vec_out = index_all_chunks(
            chunks_dir=chunks_dir,
            batch_size=cfg.batch_size,
            collection_prefix=cfg.collection_prefix,
        )

    # Determine collections
    if cfg.collection_mode == "one_collection":
        collections = [cfg.one_collection_name]
    else:
        collections = [_collection_name_for_file(f, prefix=cfg.collection_prefix) for f in files]

    retr = retriever or RetrievalService()

    total_chunks = _count_total_chunks(files)
    if cfg.k_strategy == "fixed":
        effective_k = max(1, cfg.n_results_per_collection)
    else:
        effective_k = max(1, total_chunks // 2) if total_chunks else max(1, cfg.n_results_per_collection)

    max_total_results = max(1, min(cfg.max_total_results, effective_k * len(collections))) if total_chunks else cfg.max_total_results

    res = retr.retrieve(
        root_feature=cfg.root_feature,
        domain=cfg.domain,
        collections=collections,
        n_results_per_collection=effective_k,
        max_total_results=max_total_results,
    )
    evidence = retr.to_prompt_evidence(
        res,
        max_total_chars=cfg.max_total_chars,
        max_chunk_chars=cfg.max_chunk_chars,
    )

    tmpl = load_ss_rgfm_prompt(cfg.prompt_path, paths.base_dir)
    spec_dir = paths.specifications
    xsd_path = cfg.xsd_path or (spec_dir / "feature_model_featureide.xsd")
    metamodel_path = cfg.feature_metamodel_path or (spec_dir / "feature_metamodel_specification.txt")

    xsd_text = xsd_path.read_text(encoding="utf-8") if xsd_path.exists() else ""
    metamodel_text = metamodel_path.read_text(encoding="utf-8") if metamodel_path.exists() else ""
    high_level_xml = serialize_high_level_features(cfg.high_level_features)

    # Core-metamodel prompt placeholders. Empty string when unset — safe for
    # FM-style prompts that don't declare {{DOMAIN_VOCABULARY}} /
    # {{ARTEFACTS_REGISTRY}}, and satisfies the strict renderer when they do.
    dv_path = cfg.domain_vocabulary_path
    ar_path = cfg.artefacts_registry_path
    domain_vocabulary_text = (
        Path(dv_path).read_text(encoding="utf-8") if dv_path and Path(dv_path).exists() else ""
    )
    artefacts_registry_text = (
        Path(ar_path).read_text(encoding="utf-8") if ar_path and Path(ar_path).exists() else ""
    )

    prompt = render_prompt_template(
        tmpl,
        values={
            "ROOT_FEATURE": cfg.root_feature,
            "DOMAIN": cfg.domain,
            "CONTEXT": evidence,
            "XSD_METAMODEL": xsd_text,
            "FEATURE_METAMODEL": metamodel_text,
            "HIGH_LEVEL_FEATURES": high_level_xml,
            "DOMAIN_VOCABULARY": domain_vocabulary_text,
            "ARTEFACTS_REGISTRY": artefacts_registry_text,
        },
        strict=True,
    )

    for attempt in range(1, max(1, int(cfg.max_retries)) + 1):
        try:
            t0 = start_timer()
            fm_xml = llm.generate(prompt, temperature=cfg.temperature)
            llm_duration = elapsed_seconds(t0)
            break
        except Exception as e:
            if attempt < max(1, int(cfg.max_retries)):
                print(f"   -> attempt {attempt}/{cfg.max_retries} failed: {e}. Retrying...")
            else:
                raise

    # Persist artifacts
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    model_safe = getattr(llm, "model", "unknown-model").replace(":", "-").replace("/", "-")
    prompt_saved = save_modified_prompt(
        prompt=prompt,
        model_safe=model_safe,
        ts=ts,
        paths=paths,
        pipeline_type="ss_rgfm",
    )
    run_id = f"{cfg.run_tag}_response_{model_safe}_{ts}"

    fm_dir = (paths.ss_fm / dataset_subdir) if dataset_subdir else paths.ss_fm
    reports_dir = (paths.reports / dataset_subdir) if dataset_subdir else paths.reports
    ensure_dir(fm_dir)
    ensure_dir(reports_dir)
    fm_file = fm_dir / f"{run_id}.xml"
    prompt_file = reports_dir / f"{run_id}.prompt.txt"
    evidence_file = reports_dir / f"{run_id}.evidence.txt"
    meta_file = reports_dir / f"{run_id}.meta.json"

    fm_file.write_text(fm_xml, encoding="utf-8")
    prompt_file.write_text(prompt, encoding="utf-8")
    evidence_file.write_text(evidence, encoding="utf-8")

    meta = {
        "run_id": run_id,
        "root_feature": cfg.root_feature,
        "domain": cfg.domain,
        "collection_mode": cfg.collection_mode,
        "collections": collections,
        "vectorization": vec_out,
        "query_used": res.query,
        "num_evidence_chunks": len(res.chunks),
        "k_strategy": cfg.k_strategy,
        "n_results_per_collection_effective": effective_k,
        "llm_host": getattr(llm, "host", getattr(llm, "base_url", "unknown")),
        "llm_model": getattr(llm, "model", "unknown"),
        "llm_duration_seconds": llm_duration,
        "attempts_used": attempt,
        "chunks_dir": str(chunks_dir),
        "prompt_saved": str(prompt_saved),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "ss_rgfm completed",
        extra={
            "run_id": run_id,
            "model": getattr(llm, "model", "unknown"),
            "collections": collections,
            "num_chunks": len(res.chunks),
        },
    )

    return {
        "fm_xml": str(fm_file),
        "prompt": str(prompt_file),
        "evidence": str(evidence_file),
        "meta": str(meta_file),
        "run_id": run_id,
    }
