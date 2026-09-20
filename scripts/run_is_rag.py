#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fame.config.load import load_config
from fame.evaluation.coverage import CoverageConfig
from fame.evaluation.top_fm import TopFMConfig, rank_top_fms
from fame.rag.is_pipeline import ISRgfmConfig, run_is_rgfm
from fame.loggers import get_logger, log_exception
from fame.exceptions import UserMessageError, MissingKeyError, format_error
from fame.nonrag.cli_utils import (
    dataset_choices,
    dataset_defaults,
    load_high_level_features,
    load_key_file,
    prompt_choice,
)
from fame.judge import create_judge_client
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


def _prompt_float(label: str, *, default: float, min_value: float = 0.0) -> float:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = float(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue
        if val < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        return val


def main() -> None:
    cfg_yaml = load_config()

    ap = argparse.ArgumentParser(description="Run Iterative RAG Generated Feature Modeling (IS-RGFM)")
    ap.add_argument("--root-feature", default="")
    ap.add_argument("--domain", default="")
    ap.add_argument("--dataset", choices=dataset_choices(), default="federation", help="Dataset-specific prompt guidance to load.")
    ap.add_argument("--high-level-features-config", default="", help="Optional YAML file overriding dataset high-level features.")
    ap.add_argument("--no-high-level-features", action="store_true", help="Disable high-level feature guidance injection.")
    ap.add_argument("--chunks-dir", default="", help="Directory containing *.chunks.json (default: processed_data/chunks)")
    ap.add_argument("--n-results-per-collection", type=int, default=6)
    ap.add_argument("--max-total-results", type=int, default=12)
    ap.add_argument("--collection-prefix", default="", help="Optional Chroma collection prefix. Defaults to <dataset>_ when --dataset is used.")
    ap.add_argument("--max-total-chars", type=int, default=18_000)
    ap.add_argument("--max-chunk-chars", type=int, default=2_500)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--xsd-path", default="", help="Override XSD path")
    ap.add_argument("--feature-metamodel-path", default="", help="Override feature metamodel path")
    ap.add_argument("--initial-prompt-path", default="", help="Override initial prompt path")
    ap.add_argument("--iter-prompt-path", default="", help="Override iterative prompt path")
    ap.add_argument("--interactive", action="store_true", help="Run with guided prompts")
    ap.add_argument("--repeats", type=int, default=1, help="How many runs to execute sequentially (default: 1).")
    ap.add_argument("--max-retries", type=int, default=1, help="Retries per iteration/source if it fails (default: 1).")
    ap.add_argument("--retry-backoff-base-seconds", type=float, default=cfg_yaml.pipelines.is_rgfm_retry_backoff_base_seconds, help="Base exponential backoff in seconds for 429/rate-limit retries.")
    ap.add_argument("--retry-backoff-cap-seconds", type=float, default=cfg_yaml.pipelines.is_rgfm_retry_backoff_cap_seconds, help="Maximum exponential backoff in seconds for 429/rate-limit retries.")
    ap.add_argument("--inter-iteration-sleep-seconds", type=float, default=cfg_yaml.pipelines.is_rgfm_inter_iteration_sleep_seconds, help="Fixed sleep in seconds between iterations.")
    ap.add_argument("--gt-path", default="", help="Ground-truth XML used for top_fm ranking")
    ap.add_argument("--top-fm", type=int, default=cfg_yaml.outputs.top_fm, help="Top valid FMs to copy per metric (default from fame.yaml)")
    ap.add_argument("--run-tag", default=os.getenv("FAME_RUN_ID", "is-rgfm"),
                    help="Run identifier passed through to output filenames (default: FAME_RUN_ID env or 'is-rgfm').")
    args = ap.parse_args()

    interactive = args.interactive or not (args.root_feature and args.domain)

    llm_client = None
    high_level_features = None

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
            judge_cfg = cfg_yaml.llm_judge
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
        args.domain = domain
        args.root_feature = root_feature
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

        args.repeats = _prompt_int("Number of runs to execute sequentially", default=max(1, args.repeats), min_value=1)
        args.max_retries = _prompt_int("Max retries per iteration/source", default=max(1, args.max_retries), min_value=1)
        args.retry_backoff_base_seconds = _prompt_float("Base retry backoff seconds", default=max(0.0, args.retry_backoff_base_seconds), min_value=0.0)
        args.retry_backoff_cap_seconds = _prompt_float("Retry backoff cap seconds", default=max(args.retry_backoff_base_seconds, args.retry_backoff_cap_seconds), min_value=0.0)
        args.inter_iteration_sleep_seconds = _prompt_float("Fixed sleep between iterations (seconds)", default=max(0.0, args.inter_iteration_sleep_seconds), min_value=0.0)

    if not interactive:
        config_override = Path(args.high_level_features_config).expanduser().resolve() if args.high_level_features_config else None
        high_level_features = None if args.no_high_level_features else load_high_level_features(dataset=args.dataset, config_path=config_override)

    chunks_dir = Path(args.chunks_dir).expanduser().resolve() if args.chunks_dir else _default_dataset_chunks_dir(args.dataset)
    initial_prompt_path = Path(args.initial_prompt_path).expanduser() if args.initial_prompt_path else cfg_yaml.pipelines.is_rgfm_initial_prompt_path
    iter_prompt_path = Path(args.iter_prompt_path).expanduser() if args.iter_prompt_path else cfg_yaml.pipelines.is_rgfm_iter_prompt_path
    xsd_path = Path(args.xsd_path).expanduser() if args.xsd_path else None
    feature_metamodel_path = Path(args.feature_metamodel_path).expanduser() if args.feature_metamodel_path else None

    collection_prefix = args.collection_prefix or (f"{args.dataset}_" if args.dataset else "")

    cfg = ISRgfmConfig(
        root_feature=args.root_feature,
        domain=args.domain,
        chunks_dir=chunks_dir,
        collection_prefix=collection_prefix,
        initial_prompt_path=initial_prompt_path,
        iter_prompt_path=iter_prompt_path,
        n_results_per_collection=args.n_results_per_collection,
        max_total_results=args.max_total_results,
        max_total_chars=args.max_total_chars,
        max_chunk_chars=args.max_chunk_chars,
        xsd_path=xsd_path,
        feature_metamodel_path=feature_metamodel_path,
        high_level_features=high_level_features,
        temperature=args.temperature,
        max_retries_per_iteration=args.max_retries,
        dataset=args.dataset,
        retry_backoff_base_seconds=args.retry_backoff_base_seconds,
        retry_backoff_cap_seconds=args.retry_backoff_cap_seconds,
        inter_iteration_sleep_seconds=args.inter_iteration_sleep_seconds,
        run_tag=args.run_tag,
    )

    print("\n==================== IS-RGFM ====================")
    print(f"Root feature   : {cfg.root_feature}")
    print(f"Domain         : {cfg.domain}")
    print(f"Model          : {(getattr(llm_client, 'model', None) or os.getenv('OLLAMA_LLM_MODEL', 'ollama-default'))}")
    print(f"Chunk server   : {chunks_dir or 'default processed_data/chunks'}")
    print(f"Collection pref: {collection_prefix or '(none)'}")
    print(f"Iter retries   : {cfg.max_retries_per_iteration}")
    print(f"Retry backoff  : base={cfg.retry_backoff_base_seconds}s cap={cfg.retry_backoff_cap_seconds}s")
    print(f"Iter sleep     : {cfg.inter_iteration_sleep_seconds}s")
    print("-------------------------------------------------")
    print("Stage 1: Build configuration")
    print("Stage 2: Run IS-RGFM pipeline (iterative retrieval)...")

    results = []
    for i in range(max(1, args.repeats)):
        if args.repeats > 1:
            print(f"--- Run {i+1}/{args.repeats} ---")
        out = run_is_rgfm(cfg, llm=llm_client)
        emit_web_candidate(out, i + 1)
        results.append(out)
        print("SUCCESS: IS-RGFM completed")
        for k, v in out.items():
            print(f"{k}: {v}")

    if args.repeats > 1:
        print(f"Completed {args.repeats} runs.")

    gt_path = Path(args.gt_path).expanduser().resolve() if args.gt_path else cfg_yaml.evaluation.ground_truth_xml
    if args.top_fm > 0 and gt_path:
        paths = build_paths()
        xsd_path = cfg.xsd_path or (paths.specifications / "feature_model_featureide.xsd")
        manifest = rank_top_fms(
            candidates=results,
            pipeline_root=paths.is_fm.parent,
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
    logger = get_logger("is_rgfm")
    try:
        main()
    except UserMessageError as e:
        print(f"ERROR: {format_error(e)} (see results/logs/fame.log for details)")
        log_exception(logger, e)
    except Exception as e:
        log_exception(logger, e)
        raise
