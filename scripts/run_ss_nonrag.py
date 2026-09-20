#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

from fame.judge import create_judge_client
from fame.config.load import load_config
from fame.evaluation.coverage import CoverageConfig
from fame.evaluation.top_fm import TopFMConfig, rank_top_fms
from fame.exceptions import MissingKeyError, UserMessageError, format_error
from fame.loggers import get_logger, log_exception
from fame.nonrag.ss_pipeline import SSNonRagConfig, run_ss_nonrag, _default_chunks_dir
from fame.nonrag.cli_utils import (
    dataset_choices,
    dataset_defaults,
    load_high_level_features,
    load_key_file,
    prompt_choice,
)
from fame.utils.dirs import build_paths
from fame.utils.web_candidates import emit_web_candidate


def _default_dataset_chunks_dir(dataset: str) -> Path:
    paths = build_paths()
    return (paths.base_dir / "data" / "processed" / dataset / "chunks").resolve()


def _resolve_model_max_tokens(judge_cfg, model: str) -> int:
    return int(getattr(judge_cfg, "model_max_tokens", {}).get(model, judge_cfg.max_tokens))


def _prompt_int(label: str, *, default: int, min_value: int = 1) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
        except ValueError:
            print("Invalid integer. Try again.")
            continue
        if val < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return val


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Single-Stage Non-RAG (SS-NonRAG)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--dataset", choices=dataset_choices(), default="federation", help="Dataset-specific prompt guidance to load.")
    ap.add_argument("--high-level-features-config", default="", help="Optional YAML file overriding dataset high-level features.")
    ap.add_argument("--no-high-level-features", action="store_true", help="Disable high-level feature guidance injection.")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--max-total-chars", type=int, default=int(os.getenv("NONRAG_MAX_CHARS", "140000")))
    ap.add_argument("--max-chunks", type=int, default=int(os.getenv("NONRAG_MAX_CHUNKS", "120")))
    ap.add_argument("--max-chunk-chars", type=int, default=int(os.getenv("NONRAG_MAX_CHUNK_CHARS", "6000")))
    ap.add_argument("--prompt-path", default="", help="Optional prompt file path")
    ap.add_argument("--xsd-path", default="", help="Override XSD path (default: feature_model_featureide.xsd)")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
    ap.add_argument("--run-tag", default=os.getenv("NONRAG_RUN_TAG", "ss-nonrag"))
    ap.add_argument("--verbose", action="store_true", help="Print stage-by-stage progress")
    ap.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    ap.add_argument("--preflight", action="store_true", help="Run fast checks (no prompts, no LLM) and exit.")
    ap.add_argument("--repeats", type=int, default=1, help="How many runs to execute sequentially (default: 1).")
    ap.add_argument("--max-retries", type=int, default=1, help="Retries for a failed single-stage generation (default: 1).")
    ap.add_argument("--gt-path", default="", help="Ground-truth XML used for top_fm ranking")
    ap.add_argument("--top-fm", type=int, default=load_config().outputs.top_fm, help="Top valid FMs to copy per metric (default from fame.yaml)")
    ap.add_argument("--input-dir", default="",
                    help="Per-run raw artefacts directory. Overrides data/raw. "
                         "Also settable via FAME_INPUT_FILES_DIR.")
    ap.add_argument("--output-dir", default="",
                    help="Per-run results root. Overrides <base>/results. "
                         "Also settable via FAME_OUTPUT_DIR.")
    ap.add_argument("--domain-vocabulary", default="",
                    help="Path to a domain-vocabulary YAML file whose content is "
                         "injected into the prompt as {{DOMAIN_VOCABULARY}}. "
                         "Required by prompts/core_structure_metamodel_extraction.txt.")
    ap.add_argument("--artefacts-registry", default="",
                    help="Path to a SourceArtefact registry YAML whose content is "
                         "injected into the prompt as {{ARTEFACTS_REGISTRY}}. "
                         "Required by prompts/core_structure_metamodel_extraction.txt.")
    args = ap.parse_args()

    # Per-run I/O overrides. Set env vars BEFORE any build_paths() call so
    # the whole pipeline (ingest, chunks, prompts, reports, model output)
    # inherits the redirect.
    if args.input_dir:
        os.environ["FAME_INPUT_FILES_DIR"] = str(Path(args.input_dir).expanduser().resolve())
    if args.output_dir:
        os.environ["FAME_OUTPUT_DIR"] = str(Path(args.output_dir).expanduser().resolve())

    if args.preflight:
        paths = build_paths()
        chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else _default_dataset_chunks_dir(args.dataset)

        print("=== SS-NonRAG Preflight ===")
        print(f"Chunks dir   : {chunks_dir}")
        files = sorted(chunks_dir.glob("*.chunks.json"))
        if files:
            print(f"Chunks files : {len(files)}")
        else:
            print("Chunks files : 0 (ingestion will run on first pipeline execution)")

        ollama_host = os.getenv("OLLAMA_LLM_HOST", os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
        try:
            r = requests.get(f"{ollama_host}/api/tags", timeout=3)
            ok = 200 <= r.status_code < 400
            print(f"Ollama host  : {ollama_host} ({'ok' if ok else f'status {r.status_code}'})")
        except Exception as e:
            print(f"Ollama host  : {ollama_host} (unreachable: {e})")

        print("Preflight done. (No LLM call made.)")
        return

    interactive = args.interactive or not (args.root_feature and args.domain)

    llm_client = None
    if interactive:
        mode = prompt_choice("1) Open Source LLM  OR Proprietary LLM", ("Open Source LLM", "Proprietary LLM"))

        if mode == "Open Source LLM":
            model = prompt_choice(
                "Select Open Source LLM model",
                ("gpt-oss:120b-cloud", "glm-4.7:cloud", "deepseek-v3.2:cloud"),
            )
            os.environ["OLLAMA_LLM_MODEL"] = model

            key_path = Path("api_keys/ollama_key.txt")
            key = load_key_file(key_path)
            if key:
                os.environ["OLLAMA_API_KEY_FILE"] = str(key_path)
                os.environ.setdefault("OLLAMA_LLM_HOST", "https://ollama.com")
            else:
                print("WARN:  ollama_key not found. Using local Ollama for LLM.")
                os.environ.setdefault("OLLAMA_LLM_HOST", "http://127.0.0.1:11434")

            os.environ.setdefault("OLLAMA_EMBED_HOST", "http://127.0.0.1:11434")

        else:
            model = prompt_choice(
                "Select Proprietary LLM model",
                ("gpt-4.1", "o3", "claude-opus-4-5", "gemini-3.1-pro-preview", "gemini-2.5-flash"),
            )
            provider_map = {
                "gpt-4.1": ("openai", "OPENAI_API_KEY"),
                "o3": ("openai", "OPENAI_API_KEY"),
                "claude-opus-4-5": ("anthropic", "ANTHROPIC_API_KEY"),
                "gemini-3.1-pro-preview": ("gemini", "GEMINI_API_KEY"),
                "gemini-2.5-flash": ("gemini", "GEMINI_API_KEY"),
            }
            provider, env_var = provider_map[model]
            judge_cfg = load_config().llm_judge
            key_file = judge_cfg.api_key_dir / f"{provider}_key.txt"
            key = load_key_file(key_file)
            if not key:
                raise MissingKeyError(env_var, str(key_file))
            os.environ[env_var] = key

            resolved_max_tokens = _resolve_model_max_tokens(judge_cfg, model)
            print(f"Resolved max tokens for {model}: {resolved_max_tokens}")
            llm_client = create_judge_client(
                provider=provider,
                model=model,
                base_url=judge_cfg.base_url,
                api_key_env=env_var,
                temperature=judge_cfg.temperature,
                max_tokens=resolved_max_tokens,
                timeout_s=judge_cfg.timeout_s,
            )

        dataset_meta = dataset_defaults(args.dataset)
        domain = input(f"Enter domain [{dataset_meta['domain']}]: ").strip() or dataset_meta["domain"]
        root_feature = input(f"Enter root feature [{dataset_meta['root_feature']}]: ").strip() or dataset_meta["root_feature"]
        config_override = Path(args.high_level_features_config).expanduser().resolve() if args.high_level_features_config else None
        feats = None if args.no_high_level_features else load_high_level_features(dataset=args.dataset, config_path=config_override)
        hl = input("Include high-level features? (Y/n): ").strip().lower()
        if hl not in ("n", "no") and feats:
            print("\nHigh-level features (loaded):")
            for k, v in feats.items():
                print(f"- {k}: {v}")
            confirm = input("Use these? (Y/n): ").strip().lower()
            if confirm not in ("n", "no"):
                high_level_features = feats


        args.domain = domain
        args.root_feature = root_feature
        args.high_level_features = high_level_features
        args.repeats = _prompt_int("Number of runs to execute sequentially", default=max(1, args.repeats), min_value=1)
        args.max_retries = _prompt_int("Max retries for a failed generation", default=max(1, args.max_retries), min_value=1)

    if not interactive:
        config_override = Path(args.high_level_features_config).expanduser().resolve() if args.high_level_features_config else None
        high_level_features = None if args.no_high_level_features else load_high_level_features(dataset=args.dataset, config_path=config_override)
    else:
        high_level_features = getattr(args, "high_level_features", None)

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else _default_dataset_chunks_dir(args.dataset)
    prompt_path = Path(args.prompt_path).expanduser().resolve() if args.prompt_path else None

    model_name = getattr(llm_client, "model", None) or os.getenv("OLLAMA_LLM_MODEL", "ollama-default")
    if args.verbose:
        print("\n==================== SS-NONRAG ====================")
        print(f"Root feature   : {args.root_feature}")
        print(f"Domain         : {args.domain}")
        print(f"Model          : {model_name}")
        print(f"Chunks dir     : {chunks_dir or '(default)'}")
        print(f"Max total chars: {args.max_total_chars}")
        print(f"Max chunks     : {args.max_chunks}")
        print(f"Max chunk chars: {args.max_chunk_chars}")
        print(f"Prompt path    : {prompt_path or '(default)'}")
        print(f"Run tag        : {args.run_tag}")
        print(f"Repeats        : {args.repeats}")
        print(f"Max retries    : {cfg.max_retries}")
        print("---------------------------------------------------")
        print("Stage 1: Build configuration")

    cfg = SSNonRagConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        max_total_chars=args.max_total_chars,
        max_chunks=args.max_chunks,
        max_chunk_chars=args.max_chunk_chars,
        prompt_path=prompt_path,
        xsd_path=Path(args.xsd_path).expanduser().resolve() if args.xsd_path else None,
        feature_metamodel_path=Path(args.feature_metamodel_path).expanduser().resolve() if args.feature_metamodel_path else None,
        run_tag=args.run_tag,
        high_level_features=high_level_features,
        max_retries=args.max_retries,
        dataset=args.dataset,
        domain_vocabulary_path=Path(args.domain_vocabulary).expanduser().resolve() if args.domain_vocabulary else None,
        artefacts_registry_path=Path(args.artefacts_registry).expanduser().resolve() if args.artefacts_registry else None,
    )

    if args.verbose:
        print("Stage 2: Execute SS-NonRAG pipeline (may take a while)...")

    results = []
    for i in range(max(1, args.repeats)):
        if args.repeats > 1:
            print(f"\n--- Run {i+1}/{args.repeats} ---")
        out = run_ss_nonrag(cfg, llm_client=llm_client)
        emit_web_candidate(out, i + 1)
        results.append(out)
        print("\nSUCCESS: SS-NonRAG completed")
        print(out)

    if args.repeats > 1:
        print(f"\nCompleted {args.repeats} runs.")

    cfg_yaml = load_config()
    gt_path = Path(args.gt_path).expanduser().resolve() if args.gt_path else cfg_yaml.evaluation.ground_truth_xml
    if args.top_fm > 0 and gt_path:
        paths = build_paths()
        xsd_path = cfg.xsd_path or (paths.specifications / "feature_model_featureide.xsd")
        manifest = rank_top_fms(
            candidates=results,
            pipeline_root=paths.non_ss_fm.parent,
            cfg=TopFMConfig(
                top_n=args.top_fm,
                gt_xml=gt_path,
                xsd_path=xsd_path,
                coverage=CoverageConfig(
                    model_name=cfg_yaml.evaluation.coverage.model_name,
                    similarity_threshold=cfg_yaml.evaluation.coverage.similarity_threshold,
                    top_k=cfg_yaml.evaluation.coverage.top_k,
                    feature_weight=cfg_yaml.evaluation.coverage.feature_weight,
                    parent_weight=cfg_yaml.evaluation.coverage.parent_weight,
                ),
                output_subdir=args.dataset,
            ),
        )
        if manifest:
            print(f"Top-FM ranking written to: {Path(manifest['summary_table']).parent}")
    elif args.top_fm > 0:
        print("Top-FM ranking skipped: no ground-truth XML configured or provided.")


if __name__ == "__main__":
    logger = get_logger("ss_nonrag")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
