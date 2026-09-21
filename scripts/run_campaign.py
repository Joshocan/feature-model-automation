#!/usr/bin/env python3
"""Phase 7 — main campaign runner (skeleton).

Reads a JSON campaign spec listing the RunConfigs to execute and dispatches
them sequentially through the unified engine. Extended in Phase 7 with the
full arm matrix (guided headline, curve, ablation, order sensitivity, k
sweep).

For now this exposes the plumbing so a Phase 7 spec author can just enumerate
run configs and hit "go". Retries, backoff, budget tracking, and rate-limit
respect land here alongside the spec format.

Usage:
  python scripts/run_campaign.py --spec results/<campaign>/spec.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.generation import RunConfig, make_client, run_generation  # noqa: E402
from fame.generation.token_budget import (  # noqa: E402
    OllamaTokenCounter,
    TiktokenCounter,
    UniversalCounter,
)
from fame.retrieval import RetrievalService  # noqa: E402


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True, type=Path,
                   help="Path to a JSON campaign spec: {campaign_id, runs: [RunConfig-dict, ...]}")
    p.add_argument("--results-root", default=str(REPO / "results"))
    p.add_argument("--dry-run", action="store_true",
                   help="Enumerate + validate configs without invoking any LLM.")
    return p.parse_args()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_client(provider: str, model_id: str) -> Any:
    kwargs: Dict[str, Any] = {}
    if provider == "ollama_cloud":
        kwargs.update(api_key_env="OLLAMA_API_KEY",
                      api_key_file="api_keys/ollama_key.txt")
    return make_client(provider=provider, model_id=model_id, **kwargs)


def _make_counter(provider: str, model_id: str) -> Any:
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
    spec = json.loads(args.spec.read_text())
    cfg = yaml.safe_load((REPO / "config/experiment.yaml").read_text())

    prompt_template_path = REPO / cfg["prompt"]["template"]
    metamodel_xsd_path   = REPO / cfg["prompt"]["metamodel_schema"]
    metamodel_xsd_text   = metamodel_xsd_path.read_text()
    orderings_json       = REPO / "data/orderings.json"
    chroma_root          = REPO / "data/chroma"

    encoder_pins = (REPO / "data/encoder_versions.txt").read_text()
    encoder_digest = ""
    for line in encoder_pins.splitlines():
        if "embedding_ollama_digest:" in line:
            encoder_digest = line.split(":", 1)[1].strip()
            break

    runs: List[Dict[str, Any]] = spec.get("runs", [])
    print(f"campaign {spec.get('campaign_id')}: {len(runs)} runs planned "
          f"({'DRY RUN' if args.dry_run else 'live'})")

    total_t0 = time.time()
    ok = failed = 0
    for i, run_dict in enumerate(runs, start=1):
        corpus = run_dict["corpus"]
        chunks_jsonl = REPO / f"data/processed/{corpus}/chunks.jsonl"

        # Fill in cross-cutting fields not written in the spec
        run_dict.setdefault("prompt_template_hash", _hash_file(prompt_template_path))
        run_dict.setdefault("metamodel_hash",       _hash_file(metamodel_xsd_path))
        run_dict.setdefault("chunks_hash",          _hash_file(chunks_jsonl))
        run_dict.setdefault("encoder_digest",       encoder_digest)
        run_cfg = RunConfig(**run_dict)

        prefix = f"  [{i}/{len(runs)}] {run_cfg.corpus} N={run_cfg.N} " \
                 f"{run_cfg.grounding} {run_cfg.provider}:{run_cfg.model_id} " \
                 f"seed={run_cfg.seed} rep={run_cfg.repetition}"

        if args.dry_run:
            print(f"{prefix}  → {run_cfg.run_id()}")
            ok += 1
            continue

        llm = _make_client(run_cfg.provider, run_cfg.model_id)
        counter = _make_counter(run_cfg.provider, run_cfg.model_id)
        retrieval_service = None
        if run_cfg.grounding == "rag":
            retrieval_service = RetrievalService.from_config(
                corpus=run_cfg.corpus,
                chroma_root=chroma_root,
                config_path=REPO / "config/experiment.yaml",
            )

        t0 = time.time()
        try:
            result = run_generation(
                config=run_cfg,
                llm=llm,
                orderings_json=orderings_json,
                chunks_jsonl=chunks_jsonl,
                prompt_template_path=prompt_template_path,
                metamodel_xsd_text=metamodel_xsd_text,
                retrieval_service=retrieval_service,
                results_root=args.results_root,
                token_counter=counter,
            )
            dt = time.time() - t0
            steps_ok = sum(1 for s in result.steps if s.finish_reason == "stop")
            print(f"{prefix}  → {result.run_id}  steps_ok={steps_ok}/{len(result.steps)}  wall={dt:.1f}s")
            ok += 1
        except Exception as exc:
            print(f"{prefix}  → FAILED  {type(exc).__name__}: {exc}")
            failed += 1

    print()
    print(f"campaign complete: {ok} ok, {failed} failed, "
          f"total {round(time.time() - total_t0, 1)}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
