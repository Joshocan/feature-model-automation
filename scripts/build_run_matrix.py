#!/usr/bin/env python3
"""build the campaign run manifest.

Consumes per-model ``participation`` tables from ``config/experiment.yaml`` so
each model can opt in / out of individual arms (see brief §4 participation
table). Emits a JSON manifest ``campaign.py`` consumes.

Two views:

  --only-enabled  → subset that ``campaign.py`` may execute
                     (drops any model with ``enabled: false``)
  (default)       → full documented plan, including disabled models

The ``--lane`` filter emits disjoint OpenAI and Ollama manifests that can run
in parallel without sharing matrix rows.

Arm structure (from IFS brief §4):

  guided_baseline    baseline_N,          both corpora, RAG + Non-RAG,     20 reps
  guided_headline    headline_N,          both corpora, RAG + Non-RAG,     20 reps
  guided_curve       {repair,federation}_curve_N, matching corpus, RAG+NR, 5  reps
  ablation           ablation_N,          both corpora, RAG only,          20 reps
  order_sensitivity  order_sensitivity_N, Repair only,  RAG only, 2 alts,  5  reps
  k_doc_sweep_kX     k_doc_sweep_N,       Repair only,  RAG only, k value, 3  reps
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]


ORDER_ALT_IDS = ["alt_1", "alt_2"]

REPS_HEADLINE  = 20
REPS_BASELINE  = 20
REPS_CURVE     = 5
REPS_ABLATION  = 20
REPS_ORDER     = 5
REPS_KSWEEP    = 3


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", default="ifs-2027")
    p.add_argument("--only-enabled", action="store_true",
                   help="Drop runs whose model has enabled: false in experiment.yaml.")
    p.add_argument("--lane", default="all", choices=("all", "open_weight", "astra"),
                   help="Emit one disjoint execution lane, or the combined matrix.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config() -> Dict[str, Any]:
    return yaml.safe_load((REPO / "config/experiment.yaml").read_text())


def _hashes() -> Dict[str, str]:
    return {
        "experiment_config_hash": _hash_file(REPO / "config/experiment.yaml"),
        "prompt_template_hash": _hash_file(REPO / "prompts/fm_prompt_template.txt"),
        "metamodel_hash":       _hash_file(REPO / "prompts/feature-model-schema.xsd"),
        "chunks_federation":    _hash_file(REPO / "data/processed/federation/chunks.jsonl"),
        "chunks_repair":        _hash_file(REPO / "data/processed/repair/chunks.jsonl"),
    }


def _encoder_digest() -> str:
    for line in (REPO / "data/encoder_versions.txt").read_text().splitlines():
        if "embedding_ollama_digest:" in line:
            return line.split(":", 1)[1].strip()
    return ""


def make_run_dict(
    *,
    campaign_id: str,
    corpus: str,
    N: int,
    ordering_id: str,
    grounding: str,
    model_key: str,
    model_cfg: Dict[str, Any],
    seed: int,
    repetition: int,
    metamodel_block: bool,
    k_doc: int | None,
    domain: str,
    root_feature: str,
    arm_name: str,
    hashes: Dict[str, str],
    encoder_digest: str,
) -> Dict[str, Any]:
    chunks_key = "chunks_federation" if corpus == "federation" else "chunks_repair"
    return {
        "campaign_id":          campaign_id,
        "corpus":               corpus,
        "ordering_id":          ordering_id,
        "N":                    N,
        "grounding":            grounding,
        "model_id":             model_cfg["model_id"],
        "provider":             model_cfg["provider"],
        "seed":                 seed,
        "repetition":           repetition,
        "metamodel_block":      metamodel_block,
        "k_doc":                (k_doc if grounding == "rag" else None),
        "domain":               domain,
        "root_feature":         root_feature,
        "prompt_template_hash": hashes["prompt_template_hash"],
        "metamodel_hash":       hashes["metamodel_hash"],
        "chunks_hash":          hashes[chunks_key],
        "encoder_digest":       encoder_digest,
        "max_output_tokens":    int(model_cfg["max_output_tokens"]),
        "temperature":          float(model_cfg.get("temperature", 0.2)),
        "reasoning_effort":     model_cfg.get("reasoning_effort"),
        "context_window":       int(model_cfg["context_window_tokens"]),
        "extra":                {"arm": arm_name, "model_key": model_key,
                                 "enabled": bool(model_cfg.get("enabled", True))},
    }


def _validate_retrieval_depth_grid(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Verify each Repair N=10 sweep request fits its batch's chunk supply."""
    chunks_by_doc: collections.Counter[str] = collections.Counter()
    chunks_path = REPO / "data/processed/repair/chunks.jsonl"
    for line in chunks_path.read_text().splitlines():
        if line.strip():
            chunks_by_doc[json.loads(line)["doc_id"]] += 1
    orderings = json.loads((REPO / "data/orderings.json").read_text())
    ordering = orderings["repair"]["primary"]["order"]
    n_steps = 10
    batch_size = (len(ordering) + n_steps - 1) // n_steps
    batches = [ordering[i:i + batch_size] for i in range(0, len(ordering), batch_size)]
    checks: List[Dict[str, Any]] = []
    for k_doc in cfg["retrieval"]["k_doc_grid"]:
        if int(k_doc) <= 0:
            raise ValueError(f"k_doc_grid values must be positive, got {k_doc!r}")
        for step, batch in enumerate(batches):
            k_step = int(k_doc) * len(batch)
            largest_sub_query = (k_step + 3) // 4
            available = sum(chunks_by_doc[d] for d in batch)
            if largest_sub_query > available:
                raise ValueError(
                    f"k_doc={k_doc} exceeds filtered chunk supply at Repair N=10 "
                    f"step={step}: sub-query asks for {largest_sub_query}, available={available}"
                )
        checks.append({"k_doc": int(k_doc), "status": "within_batch_chunk_supply"})
    return checks


def _assert_no_duplicate_generation_rows(runs: List[Dict[str, Any]]) -> None:
    """Reject rows that differ only in bookkeeping and would repeat an API call."""
    seen: Dict[str, Dict[str, Any]] = {}
    ignored = {"campaign_id", "repetition", "extra"}
    for run in runs:
        effective = {k: v for k, v in run.items() if k not in ignored}
        key = json.dumps(effective, sort_keys=True, separators=(",", ":"))
        if key in seen:
            prev = seen[key]
            raise ValueError(
                "duplicate generation rows: "
                f"{prev['extra']['arm']} and {run['extra']['arm']} for "
                f"{run['model_id']} seed={run['seed']}"
            )
        seen[key] = run


def _validate_expected_totals(matrix: Dict[str, Any], cfg: Dict[str, Any], lane: str) -> None:
    expected = cfg["campaign"].get("expected_enabled_matrix", {})
    target = expected if lane == "all" else expected.get("lanes", {}).get(lane, {})
    if not target:
        return
    actual = {"runs": matrix["total_runs"], "calls": matrix["total_calls"]}
    wanted = {
        "runs": int(target.get("runs", target.get("total_runs", -1))),
        "calls": int(target.get("calls", target.get("total_calls", -1))),
    }
    if actual != wanted:
        raise ValueError(f"matrix acceptance totals failed for lane={lane}: {actual} != {wanted}")


def build_matrix(campaign_id: str, only_enabled: bool = False,
                 lane: str = "all") -> Dict[str, Any]:
    cfg = _load_config()
    hashes = _hashes()
    encoder_digest = _encoder_digest()
    models = cfg["generation_models"]
    k_doc_default = int(cfg["retrieval"]["k_doc"])
    k_doc_grid    = list(cfg["retrieval"]["k_doc_grid"])

    domain_map = {"federation": "model federation", "repair": "model repair"}
    root_map   = {"federation": "Federation",       "repair": "Repair"}

    runs: List[Dict[str, Any]] = []
    per_arm:   Dict[str, int]  = {}
    per_model: Dict[str, int]  = {}
    per_model_runs: Dict[str, int] = {}
    per_lane_calls: Dict[str, int] = {}
    per_lane_runs: Dict[str, int] = {}

    def _emit(model_key: str, mcfg: dict, arm: str, corpus: str, N: int,
              ordering: str, grounding: str, reps: int, metamodel: bool,
              k_doc_val: int | None, *, seed_start: int = 0) -> None:
        model_lane = str(mcfg.get("lane", "open_weight"))
        for rep in range(reps):
            seed = seed_start + rep
            run = make_run_dict(
                campaign_id=campaign_id, corpus=corpus, N=N,
                ordering_id=ordering, grounding=grounding,
                model_key=model_key, model_cfg=mcfg,
                seed=seed, repetition=rep,
                metamodel_block=metamodel, k_doc=k_doc_val,
                domain=domain_map[corpus], root_feature=root_map[corpus],
                arm_name=arm, hashes=hashes, encoder_digest=encoder_digest,
            )
            run["extra"]["lane"] = model_lane
            runs.append(run)
        per_arm[arm]      = per_arm.get(arm, 0) + reps * N
        per_model[model_key] = per_model.get(model_key, 0) + reps * N
        per_model_runs[model_key] = per_model_runs.get(model_key, 0) + reps
        per_lane_calls[model_lane] = per_lane_calls.get(model_lane, 0) + reps * N
        per_lane_runs[model_lane] = per_lane_runs.get(model_lane, 0) + reps

    for model_key, mcfg in models.items():
        if only_enabled and not mcfg.get("enabled", True):
            continue
        model_lane = str(mcfg.get("lane", "open_weight"))
        if lane != "all" and model_lane != lane:
            continue
        part = mcfg.get("participation", {})

        # Explicit cells take precedence. They are used for reduced or
        # asymmetric model participation such as the Astra ceiling campaign.
        custom_cells = part.get("custom_cells")
        if custom_cells is not None:
            for cell in custom_cells:
                grounding = str(cell["grounding"])
                _emit(
                    model_key, mcfg, str(cell["arm"]), str(cell["corpus"]),
                    int(cell["N"]), str(cell.get("ordering_id", "primary")),
                    grounding, int(cell["repetitions"]),
                    bool(cell.get("metamodel_block", True)),
                    (int(cell.get("k_doc", k_doc_default)) if grounding == "rag" else None),
                    seed_start=int(cell.get("seed_start", 0)),
                )
            continue

        # Guided baseline (N ∈ baseline_N, both corpora, both groundings)
        for N in part.get("baseline_N", []):
            for grounding in ("rag", "nonrag"):
                _emit(model_key, mcfg, "guided_baseline", "repair",     N, "primary", grounding, REPS_BASELINE, True, k_doc_default)
                _emit(model_key, mcfg, "guided_baseline", "federation", N, "primary", grounding, REPS_BASELINE, True, k_doc_default)

        # Guided headline (N ∈ headline_N, both corpora, both groundings)
        for N in part.get("headline_N", []):
            for grounding in ("rag", "nonrag"):
                _emit(model_key, mcfg, "guided_headline", "repair",     N, "primary", grounding, REPS_HEADLINE, True, k_doc_default)
                _emit(model_key, mcfg, "guided_headline", "federation", N, "primary", grounding, REPS_HEADLINE, True, k_doc_default)

        # Guided curve (Repair)
        for N in part.get("repair_curve_N", []):
            for grounding in ("rag", "nonrag"):
                _emit(model_key, mcfg, "guided_curve", "repair", N, "primary", grounding, REPS_CURVE, True, k_doc_default)
        # Guided curve (Federation)
        for N in part.get("federation_curve_N", []):
            for grounding in ("rag", "nonrag"):
                _emit(model_key, mcfg, "guided_curve", "federation", N, "primary", grounding, REPS_CURVE, True, k_doc_default)

        # Ablation (RAG only, no metamodel block)
        for N in part.get("ablation_N", []):
            for corpus in ("repair", "federation"):
                _emit(model_key, mcfg, "ablation", corpus, N, "primary", "rag", REPS_ABLATION, False, k_doc_default)

        # Order sensitivity (Repair only, RAG, alt orderings)
        for N in part.get("order_sensitivity_N", []):
            for ordering in ORDER_ALT_IDS:
                _emit(model_key, mcfg, "order_sensitivity", "repair", N, ordering, "rag", REPS_ORDER, True, k_doc_default)

        # k_doc sweep (Repair only, RAG, per k value)
        for N in part.get("k_doc_sweep_N", []):
            for k in k_doc_grid:
                # Seeds 20..22 keep the sweep paired across k while avoiding
                # exact duplicates of headline seeds 0..2 at the default k=5.
                _emit(model_key, mcfg, f"k_doc_sweep_k{k}", "repair", N, "primary", "rag", REPS_KSWEEP, True, k, seed_start=20)

    _assert_no_duplicate_generation_rows(runs)
    retrieval_checks = _validate_retrieval_depth_grid(cfg)

    selected_model_keys = {
        r.get("extra", {}).get("model_key") for r in runs
    }

    matrix = {
        "campaign_id":     campaign_id,
        "only_enabled":    only_enabled,
        "lane":            lane,
        "generated_at":    datetime.now().isoformat(),
        "hashes":          hashes,
        "encoder_digest":  encoder_digest,
        "models":          {k: {"provider": v["provider"], "model_id": v["model_id"],
                                "enabled": bool(v.get("enabled", True)),
                                "role": v.get("role"),
                                "lane": v.get("lane"),
                                "context_window_tokens": v["context_window_tokens"],
                                "max_output_tokens": v["max_output_tokens"],
                                "reasoning_effort": v.get("reasoning_effort")}
                            for k, v in models.items() if k in selected_model_keys},
        "per_arm_calls":   per_arm,
        "per_model_calls": per_model,
        "per_model_runs":  per_model_runs,
        "per_lane_calls":  per_lane_calls,
        "per_lane_runs":   per_lane_runs,
        "retrieval_depth_checks": retrieval_checks,
        "total_calls":     sum(per_arm.values()),
        "total_runs":      len(runs),
        "runs":            runs,
    }
    if only_enabled:
        _validate_expected_totals(matrix, cfg, lane)
    return matrix


def main() -> int:
    args = _cli()
    matrix = build_matrix(args.campaign_id, only_enabled=args.only_enabled, lane=args.lane)

    label = "enabled-only" if args.only_enabled else "full plan"
    label += f", lane={args.lane}"
    print(f"===== run matrix: {args.campaign_id}  ({label}) =====")
    print(f"  total runs:  {matrix['total_runs']}")
    print(f"  total calls: {matrix['total_calls']}")
    print()
    print("  per model:")
    for m, n in sorted(matrix["per_model_calls"].items(), key=lambda kv: -kv[1]):
        enabled = matrix["models"][m]["enabled"]
        badge = "  " if enabled else " (disabled)"
        print(f"    {m:<24s}{badge} {n:>6d}")
    print()
    print("  per lane:")
    for lane_name, n in sorted(matrix["per_lane_calls"].items()):
        print(f"    {lane_name:<24s} {matrix['per_lane_runs'][lane_name]:>4d} runs  {n:>6d} calls")
    print()
    print("  per arm:")
    for a, n in sorted(matrix["per_arm_calls"].items(), key=lambda kv: -kv[1]):
        print(f"    {a:<28s} {n:>6d}")

    if args.dry_run:
        return 0

    suffix = "_enabled" if args.only_enabled else ""
    if args.lane != "all":
        suffix += f"_{args.lane}"
    out = Path(args.out) if args.out else (
        REPO / f"results/{args.campaign_id}/run_matrix{suffix}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2))
    print(f"\nWrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
