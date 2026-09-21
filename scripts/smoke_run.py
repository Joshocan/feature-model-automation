#!/usr/bin/env python3
"""Phase 6 — one end-to-end smoke run through the unified engine.

By default runs with a FakeLLM (deterministic, no network) so we can verify
D7/D8/D9/D10 wiring without cost. Pass ``--provider fake|ollama_cloud|openai``
to switch. Defaults to a tiny federation N=1 non-RAG run for maximum coverage
of the loop with minimum spend.

Usage:
  python scripts/smoke_run.py                                 # Fake, fed N=1 nonrag
  python scripts/smoke_run.py --corpus repair --N 3
  python scripts/smoke_run.py --grounding rag --k-doc 5
  python scripts/smoke_run.py --provider ollama_cloud --model glm-5.3-flash
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.generation import (             # noqa: E402
    RunConfig,
    make_client,
    run_generation,
)
from fame.generation.token_budget import (  # noqa: E402
    OllamaTokenCounter,
    TiktokenCounter,
    UniversalCounter,
)
from fame.retrieval import RetrievalService  # noqa: E402
from fame.validation import validate_run    # noqa: E402


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", default="smoke-2026-09-21")
    p.add_argument("--corpus", choices=["federation", "repair"], default="federation")
    p.add_argument("--ordering", default="primary")
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--grounding", choices=["rag", "nonrag"], default="nonrag")
    p.add_argument("--provider", choices=["fake", "ollama_cloud", "openai"], default="fake")
    p.add_argument("--model", default=None,
                   help="model_id (defaults per provider)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k-doc", type=int, default=None,
                   help="Override RAG k_doc; defaults to experiment.yaml value.")
    p.add_argument("--metamodel-block", action="store_true", default=True,
                   help="Include metamodel prompt block (guided arm).")
    p.add_argument("--no-metamodel-block", dest="metamodel_block", action="store_false",
                   help="Ablation arm — omit metamodel prompt block.")
    p.add_argument("--results-root", default=str(REPO / "results"))
    p.add_argument("--host", default=None,
                   help="Override LLM host (e.g. http://127.0.0.1:11434 for local Ollama).")
    p.add_argument("--validate", action="store_true",
                   help="Run the Phase 6 validator on the produced run directory.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite the run directory if it already exists.")
    return p.parse_args()


def _resolve_model(cfg: dict, provider: str, override: str | None) -> tuple[str, dict]:
    """Pick a model_id from experiment.yaml.generation_models for the given provider."""
    models = cfg["generation_models"]
    for key, m in models.items():
        if m["provider"] == provider and (override is None or m["model_id"] == override):
            return m["model_id"], m
    if override:
        # Unknown to config — allow but warn.
        print(f"WARN: model {override!r} not found in experiment.yaml for provider {provider!r}")
        return override, {"context_window_tokens": 1_000_000, "max_output_tokens": 16_384,
                          "reasoning_effort": None, "temperature": 0.2}
    # Fake fallback
    return "fake-lm", {"context_window_tokens": 1_000_000, "max_output_tokens": 1024,
                       "reasoning_effort": None, "temperature": 0.2}


def _make_token_counter(provider: str, model_id: str):
    if provider == "openai":
        try:
            return TiktokenCounter(model_id=model_id)
        except Exception:
            return UniversalCounter()
    if provider == "ollama_cloud":
        return OllamaTokenCounter(model_id=model_id)
    return UniversalCounter()


def main() -> int:
    args = _cli()
    cfg = yaml.safe_load((REPO / "config/experiment.yaml").read_text())
    corpus_cfg = cfg["corpora"][args.corpus]
    domain_map = {"federation": "model federation", "repair": "model repair"}
    root_map   = {"federation": "Federation",       "repair": "Repair"}

    model_id, m_cfg = _resolve_model(cfg, args.provider, args.model)
    k_doc = args.k_doc if args.k_doc is not None else cfg["retrieval"]["k_doc"]

    prompt_template_path = REPO / cfg["prompt"]["template"]
    metamodel_xsd_path   = REPO / cfg["prompt"]["metamodel_schema"]
    chunks_jsonl         = REPO / f"data/processed/{args.corpus}/chunks.jsonl"
    orderings_json       = REPO / "data/orderings.json"
    chroma_root          = REPO / "data/chroma"

    encoder_pins = (REPO / "data/encoder_versions.txt").read_text()
    encoder_digest_line = next(
        (l for l in encoder_pins.splitlines() if "embedding_ollama_digest:" in l), "")
    encoder_digest = encoder_digest_line.split(":", 1)[1].strip() if encoder_digest_line else ""

    run_cfg = RunConfig(
        campaign_id=args.campaign_id,
        corpus=args.corpus,
        ordering_id=args.ordering,
        N=args.N,
        grounding=args.grounding,
        model_id=model_id,
        provider=args.provider,
        seed=args.seed,
        repetition=0,
        metamodel_block=args.metamodel_block,
        k_doc=(k_doc if args.grounding == "rag" else None),
        domain=domain_map[args.corpus],
        root_feature=root_map[args.corpus],
        prompt_template_hash=_hash_file(prompt_template_path),
        metamodel_hash=_hash_file(metamodel_xsd_path),
        chunks_hash=_hash_file(chunks_jsonl),
        encoder_digest=encoder_digest,
        max_output_tokens=int(m_cfg["max_output_tokens"]),
        temperature=float(m_cfg.get("temperature", 0.2)),
        reasoning_effort=m_cfg.get("reasoning_effort"),
        context_window=int(m_cfg["context_window_tokens"]),
    )

    llm_kwargs = {}
    if args.provider == "ollama_cloud":
        llm_kwargs["api_key_env"] = "OLLAMA_API_KEY"
        llm_kwargs["api_key_file"] = "api_keys/ollama_key.txt"
    if args.host:
        llm_kwargs["host"] = args.host
    llm = make_client(provider=args.provider, model_id=model_id, **llm_kwargs)

    retrieval_service = None
    if args.grounding == "rag":
        retrieval_service = RetrievalService.from_config(
            corpus=args.corpus,
            chroma_root=chroma_root,
            config_path=REPO / "config/experiment.yaml",
        )

    counter = _make_token_counter(args.provider, model_id)

    print(f"===== smoke run: {run_cfg.run_id()} =====")
    print(f"  corpus={args.corpus}  N={args.N}  grounding={args.grounding}")
    print(f"  provider={args.provider}  model_id={model_id}")
    print(f"  root={args.results_root}/{run_cfg.campaign_id}/{args.corpus}/{run_cfg.config_hash()}/{run_cfg.run_id()}")

    result = run_generation(
        config=run_cfg,
        llm=llm,
        orderings_json=orderings_json,
        chunks_jsonl=chunks_jsonl,
        prompt_template_path=prompt_template_path,
        metamodel_xsd_text=metamodel_xsd_path.read_text(),
        retrieval_service=retrieval_service,
        results_root=args.results_root,
        token_counter=counter,
        force=args.force,
    )

    print()
    print(f"===== result =====")
    print(f"  steps:            {len(result.steps)}")
    print(f"  completed:        {result.completed}")
    print(f"  total_wall:       {result.total_wall_seconds}s")
    for s in result.steps:
        print(f"    step {s.step_index}: chunks={s.n_chunks}  assembled_tokens={s.assembled_tokens}  "
              f"feasible={s.feasible}  finish={s.finish_reason}  wall={s.wall_seconds}s")

    print()
    print(f"  fm_gen.xml: {'EXISTS' if result.paths.fm_gen.exists() else 'MISSING'}")
    print(f"  fm_iter/:   {len(list(result.paths.fm_iter_dir.glob('*.xml')))} files")
    print(f"  context_log.jsonl: {result.paths.context_log.stat().st_size if result.paths.context_log.exists() else 0} bytes")
    print(f"  run_meta.json:     {result.paths.run_meta.stat().st_size if result.paths.run_meta.exists() else 0} bytes")

    if args.validate:
        print()
        print(f"===== validator =====")
        report = validate_run(result.paths.root, repo_root=REPO)
        print(f"  complete: {report.complete}  errors: {report.n_errors}  warnings: {report.n_warnings}")
        for f in report.findings:
            print(f"  [{f.severity.value:7s}] {f.check}: {f.detail}")
        return 0 if report.complete else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
