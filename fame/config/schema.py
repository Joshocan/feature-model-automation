# fame/config/schema.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _as_path(base: Path, p: str) -> Path:
    if not p:
        return base
    q = Path(p).expanduser()
    return q if q.is_absolute() else (base / q).resolve()


@dataclass(frozen=True)
class ProjectCfg:
    name: str
    base_dir: Optional[Path]
    run_tag: str


@dataclass(frozen=True)
class ChromaCfg:
    mode: str
    path: Path
    host: str
    port: int
    startup_timeout_s: int
    force_restart: bool


@dataclass(frozen=True)
class OllamaCfg:
    host: str
    bin: str
    startup_timeout_s: int
    force_restart: bool
    embed_model: str
    llm_model: str


@dataclass(frozen=True)
class LlmJudgeCfg:
    provider: str = "openai"
    model: str = "gpt-4.1"
    base_url: str = ""
    api_key_env: str = "JUDGE_API_KEY"
    api_key_dir: Path = Path("api_keys")
    temperature: float = 0.2
    max_tokens: int = 2048
    model_max_tokens: Dict[str, int] = field(default_factory=dict)
    timeout_s: int = 120


@dataclass(frozen=True)
class DataCfg:
    raw_dir: Path
    processed_dir: Path
    chunks_dir: Path


@dataclass(frozen=True)
class IngestionCfg:
    enabled: bool
    allowed_extensions: List[str]
    skip_if_exists: bool
    chunking: Dict[str, Any]


@dataclass(frozen=True)
class VectorizationCfg:
    enabled: bool
    collection_mode: str
    one_collection_name: str
    prefix: str
    batch_size: int


@dataclass(frozen=True)
class RetrievalCfg:
    default_query_template: str
    n_results_per_collection: int
    max_total_results: int
    max_total_chars: int
    max_chunk_chars: int


@dataclass(frozen=True)
class ContextCfg:
    order: str
    include_headers: bool


@dataclass(frozen=True)
class NonRagSSCfg:
    enabled: bool
    prompt_path: Optional[Path]
    max_total_chars: int
    max_chunks: int
    max_chunk_chars: int
    temperature: float


@dataclass(frozen=True)
class NonRagISCfg:
    enabled: bool
    prompt_path: Optional[Path]
    initial_prompt_path: Optional[Path]
    iter_prompt_path: Optional[Path]
    max_delta_chars: int
    max_delta_chunks: int
    max_delta_chunk_chars: int
    temperature: float
    retry_backoff_base_seconds: float
    retry_backoff_cap_seconds: float
    inter_iteration_sleep_seconds: float


@dataclass(frozen=True)
class PipelinesCfg:
    ss_rgfm: bool
    ms_rgfm: bool
    is_rgfm: bool
    ss_rgfm_prompt_path: Optional[Path]
    is_rgfm_initial_prompt_path: Optional[Path]
    is_rgfm_iter_prompt_path: Optional[Path]
    is_rgfm_retry_backoff_base_seconds: float
    is_rgfm_retry_backoff_cap_seconds: float
    is_rgfm_inter_iteration_sleep_seconds: float
    ss_nonrag: NonRagSSCfg
    is_nonrag: NonRagISCfg


@dataclass(frozen=True)
class OutputsCfg:
    save_prompts: bool
    save_context: bool
    save_meta: bool
    write_latest_pointer: bool
    top_fm: int


@dataclass(frozen=True)
class LoggingCfg:
    level: str
    to_file: bool
    json: bool
    console: bool
    console_level: str
    console_include_exc: bool


@dataclass(frozen=True)
class EvaluationCoverageCfg:
    model_name: str
    similarity_threshold: float
    top_k: int
    feature_weight: float
    parent_weight: float


@dataclass(frozen=True)
class EvaluationCfg:
    coverage: EvaluationCoverageCfg
    ground_truth_xml: Optional[Path]


@dataclass(frozen=True)
class FameConfig:
    project: ProjectCfg
    chroma: ChromaCfg
    ollama: OllamaCfg
    llm_judge: LlmJudgeCfg
    data: DataCfg
    ingestion: IngestionCfg
    vectorization: VectorizationCfg
    retrieval: RetrievalCfg
    context: ContextCfg
    pipelines: PipelinesCfg
    outputs: OutputsCfg
    logging: LoggingCfg
    evaluation: EvaluationCfg


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def parse_config(doc: Dict[str, Any], repo_root: Path) -> FameConfig:
    # Project
    proj = doc.get("project", {})
    base_dir_str = str(proj.get("base_dir", "") or "").strip()
    base_dir = Path(base_dir_str).expanduser().resolve() if base_dir_str else None

    # use repo_root when project.base_dir empty
    base = base_dir or repo_root

    project = ProjectCfg(
        name=str(proj.get("name", "fame-project-tool")),
        base_dir=base_dir,
        run_tag=str(proj.get("run_tag", "dev")),
    )

    # Services
    s = doc.get("services", {})
    chroma = s.get("chroma", {})
    ollama = s.get("ollama", {})

    chroma_cfg = ChromaCfg(
        mode=str(chroma.get("mode", "persistent")).lower(),
        path=_as_path(base, str(chroma.get("path", "data/chroma_db"))),
        host=str(chroma.get("host", "127.0.0.1")),
        port=int(chroma.get("port", 8000)),
        startup_timeout_s=int(chroma.get("startup_timeout_s", 120)),
        force_restart=bool(chroma.get("force_restart", False)),
    )

    ollama_cfg = OllamaCfg(
        host=str(ollama.get("host", "http://127.0.0.1:11434")).rstrip("/"),
        bin=str(ollama.get("bin", "")),
        startup_timeout_s=int(ollama.get("startup_timeout_s", 60)),
        force_restart=bool(ollama.get("force_restart", False)),
        embed_model=str(ollama.get("embed_model", "nomic-embed-text")),
        llm_model=str(ollama.get("llm_model", "llama3.1:8b")),
    )

    judge = s.get("llm_judge", {})
    llm_judge_cfg = LlmJudgeCfg(
        provider=str(judge.get("provider", "openai")).lower(),
        model=str(judge.get("model", "gpt-4.1")),
        base_url=str(judge.get("base_url", "")),
        api_key_env=str(judge.get("api_key_env", "JUDGE_API_KEY")),
        api_key_dir=_as_path(base, str(judge.get("api_key_dir", "api_keys"))),
        temperature=float(judge.get("temperature", 0.2)),
        max_tokens=int(judge.get("max_tokens", 2048)),
        model_max_tokens={str(k): int(v) for k, v in dict(judge.get("model_max_tokens", {})).items()},
        timeout_s=int(judge.get("timeout_s", 120)),
    )

    # Data
    d = doc.get("data", {})
    raw_dir = _as_path(base, str(d.get("raw_dir", "data/raw")))
    processed_dir = _as_path(base, str(d.get("processed_dir", "data/processed/algorithm_1")))
    chunks_subdir = str(d.get("chunks_subdir", "chunks"))
    chunks_dir = (processed_dir / chunks_subdir).resolve()

    data_cfg = DataCfg(raw_dir=raw_dir, processed_dir=processed_dir, chunks_dir=chunks_dir)

    # Ingestion
    ing = doc.get("ingestion", {})
    ingestion_cfg = IngestionCfg(
        enabled=bool(ing.get("enabled", True)),
        allowed_extensions=list(ing.get("allowed_extensions", [".pdf", ".docx", ".txt", ".md"])),
        skip_if_exists=bool(ing.get("skip_if_exists", True)),
        chunking=dict(ing.get("chunking", {})),
    )

    # Vectorization
    vec = doc.get("vectorization", {})
    col = vec.get("collection", {})
    vector_cfg = VectorizationCfg(
        enabled=bool(vec.get("enabled", True)),
        collection_mode=str(col.get("mode", "per_source")),
        one_collection_name=str(col.get("one_collection_name", "fame_all")),
        prefix=str(col.get("prefix", "")),
        batch_size=int(vec.get("batch_size", 24)),
    )

    # Retrieval
    r = doc.get("retrieval", {})
    ef = r.get("evidence_format", {})
    retrieval_cfg = RetrievalCfg(
        default_query_template=str(r.get("default_query_template", "")).strip(),
        n_results_per_collection=int(r.get("n_results_per_collection", 6)),
        max_total_results=int(r.get("max_total_results", 12)),
        max_total_chars=int(ef.get("max_total_chars", 18000)),
        max_chunk_chars=int(ef.get("max_chunk_chars", 2500)),
    )

    # Context
    ctx = doc.get("context", {})
    context_cfg = ContextCfg(
        order=str(ctx.get("order", "by_page_then_id")),
        include_headers=bool(ctx.get("include_headers", True)),
    )

    # Pipelines
    pips = doc.get("pipelines", {})
    ss_rgfm = bool((pips.get("ss_rgfm") or {}).get("enabled", True))
    ms_rgfm = bool((pips.get("ms_rgfm") or {}).get("enabled", True))
    is_rgfm = bool((pips.get("is_rgfm") or {}).get("enabled", True))

    ssnr = pips.get("ss_nonrag", {})
    ssnr_budget = ssnr.get("context_budget", {})
    ss_nonrag_cfg = NonRagSSCfg(
        enabled=bool(ssnr.get("enabled", True)),
        prompt_path=_as_path(base, ssnr.get("prompt_path", "")) if str(ssnr.get("prompt_path", "")).strip() else None,
        max_total_chars=int(ssnr_budget.get("max_total_chars", 140000)),
        max_chunks=int(ssnr_budget.get("max_chunks", 120)),
        max_chunk_chars=int(ssnr_budget.get("max_chunk_chars", 6000)),
        temperature=float(ssnr.get("temperature", 0.2)),
    )

    isnr = pips.get("is_nonrag", {})
    isnr_budget = isnr.get("delta_budget", {})
    isnr_prompts = isnr.get("prompt_paths", {})
    is_nonrag_cfg = NonRagISCfg(
        enabled=bool(isnr.get("enabled", True)),
        prompt_path=_as_path(base, isnr.get("prompt_path", "")) if str(isnr.get("prompt_path", "")).strip() else None,
        initial_prompt_path=_as_path(base, isnr_prompts.get("initial", "")) if str(isnr_prompts.get("initial", "")).strip() else None,
        iter_prompt_path=_as_path(base, isnr_prompts.get("iter", "")) if str(isnr_prompts.get("iter", "")).strip() else None,
        max_delta_chars=int(isnr_budget.get("max_delta_chars", 50000)),
        max_delta_chunks=int(isnr_budget.get("max_delta_chunks", 50)),
        max_delta_chunk_chars=int(isnr_budget.get("max_delta_chunk_chars", 6000)),
        temperature=float(isnr.get("temperature", 0.2)),
        retry_backoff_base_seconds=float(isnr.get("retry_backoff_base_seconds", 2.0)),
        retry_backoff_cap_seconds=float(isnr.get("retry_backoff_cap_seconds", 60.0)),
        inter_iteration_sleep_seconds=float(isnr.get("inter_iteration_sleep_seconds", 1.0)),
    )

    isrgfm = pips.get("is_rgfm", {})
    pipelines_cfg = PipelinesCfg(
        ss_rgfm=ss_rgfm,
        ms_rgfm=ms_rgfm,
        is_rgfm=is_rgfm,
        ss_rgfm_prompt_path=_as_path(base, (pips.get("ss_rgfm") or {}).get("prompt_path", "")) if str((pips.get("ss_rgfm") or {}).get("prompt_path", "")).strip() else None,
        is_rgfm_initial_prompt_path=_as_path(base, isrgfm.get("initial_prompt_path", "")) if str(isrgfm.get("initial_prompt_path", "")).strip() else None,
        is_rgfm_iter_prompt_path=_as_path(base, isrgfm.get("iter_prompt_path", "")) if str(isrgfm.get("iter_prompt_path", "")).strip() else None,
        is_rgfm_retry_backoff_base_seconds=float(isrgfm.get("retry_backoff_base_seconds", 2.0)),
        is_rgfm_retry_backoff_cap_seconds=float(isrgfm.get("retry_backoff_cap_seconds", 60.0)),
        is_rgfm_inter_iteration_sleep_seconds=float(isrgfm.get("inter_iteration_sleep_seconds", 1.0)),
        ss_nonrag=ss_nonrag_cfg,
        is_nonrag=is_nonrag_cfg,
    )

    # Outputs
    out = doc.get("outputs", {})
    outputs_cfg = OutputsCfg(
        save_prompts=bool(out.get("save_prompts", True)),
        save_context=bool(out.get("save_context", True)),
        save_meta=bool(out.get("save_meta", True)),
        write_latest_pointer=bool(out.get("write_latest_pointer", True)),
        top_fm=max(0, int(out.get("top_fm", 3))),
    )

    # Logging
    log = doc.get("logging", {})
    logging_cfg = LoggingCfg(
        level=str(log.get("level", "INFO")).upper(),
        to_file=bool(log.get("to_file", False)),
        json=bool(log.get("json", True)),
        console=bool(log.get("console", True)),
        console_level=str(log.get("console_level", log.get("level", "INFO"))).upper(),
        console_include_exc=bool(log.get("console_include_exc", False)),
    )

    # Evaluation
    eval_doc = doc.get("evaluation", {})
    cov_doc = eval_doc.get("coverage", {})
    eval_cfg = EvaluationCfg(
        coverage=EvaluationCoverageCfg(
            model_name=str(cov_doc.get("model_name", "all-mpnet-base-v2")),
            similarity_threshold=float(cov_doc.get("similarity_threshold", 0.35)),
            top_k=int(cov_doc.get("top_k", 3)),
            feature_weight=float(cov_doc.get("feature_weight", 0.9)),
            parent_weight=float(cov_doc.get("parent_weight", 0.1)),
        ),
        ground_truth_xml=_as_path(base, str(eval_doc.get("ground_truth_xml", "")))
        if str(eval_doc.get("ground_truth_xml", "")).strip()
        else None,
    )

    return FameConfig(
        project=project,
        chroma=chroma_cfg,
        ollama=ollama_cfg,
        llm_judge=llm_judge_cfg,
        data=data_cfg,
        ingestion=ingestion_cfg,
        vectorization=vector_cfg,
        retrieval=retrieval_cfg,
        context=context_cfg,
        pipelines=pipelines_cfg,
        outputs=outputs_cfg,
        logging=logging_cfg,
        evaluation=eval_cfg,
    )
