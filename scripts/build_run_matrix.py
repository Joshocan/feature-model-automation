#!/usr/bin/env python3
"""Phase 7.5 — build the full campaign run manifest.

Enumerates every planned run per the brief's §4 arm structure. The output is
a JSON spec directly consumed by ``scripts/run_campaign.py``. It is the
**denominator** for completion reporting throughout Phase 8 — every arm's
progress is measured against this manifest.

Arm structure (from IFS brief §4):

  Guided headline (both corpora, N=10):
      corpora × models × groundings × reps  →  runs of N=10 steps each
  Guided baselines (both corpora, N=1):
      corpora × models × groundings × reps  →  runs of N=1 step
  Guided curves (Repair N∈{5,20,54}, Federation N∈{5,23}):
      per (corpus, N): models × groundings × reps  →  runs of N steps
  Ablation (RAG only, both corpora, N∈{1,10}):
      per (corpus, N): models × 1 × reps  → runs of N steps  (metamodel_block=False)
  Order sensitivity (Repair only, RAG, N∈{10,54}, 2 alt orderings):
      per (corpus, N, ordering): models × 1 × reps  → runs of N steps
  k_doc sweep (Repair only, RAG, N=10, 4 k values):
      per (corpus, N, k_doc): models × 1 × reps  → runs of N steps

Emits a JSON with per-arm run counts and the flat run list.

Usage:
  python scripts/build_run_matrix.py                    # writes to results/<campaign_id>/run_matrix.json
  python scripts/build_run_matrix.py --dry-run          # print totals, no write
  python scripts/build_run_matrix.py --campaign-id ifs-2027-pilot
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# Arm specification
# ─────────────────────────────────────────────────────────────────────────────

# All models are used unless an arm restricts.
ALL_GROUNDINGS = ["rag", "nonrag"]
RAG_ONLY       = ["rag"]

# Repetition counts per arm (from brief §4)
REPS_HEADLINE  = 20   # N=10 headline (H1b)
REPS_BASELINE  = 20   # N=1 baseline
REPS_CURVE     = 5    # curve interior points
REPS_ABLATION  = 20   # ablation arm
REPS_ORDER     = 5    # order sensitivity
REPS_KSWEEP    = 3    # k_doc sweep

# N values
REPAIR_HEADLINE_N       = 10
REPAIR_BASELINE_N       = 1
REPAIR_CURVE_N          = [5, 20, 54]
FEDERATION_HEADLINE_N   = 10
FEDERATION_BASELINE_N   = 1
FEDERATION_CURVE_N      = [5, 23]
ABLATION_N              = [1, 10]
ORDER_N                 = [10, 54]           # Repair only
KSWEEP_N                = 10                 # Repair only

# Alternate orderings for the order-sensitivity arm (Repair)
ORDER_ALT_IDS = ["alt_1", "alt_2"]


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign-id", default="ifs-2027")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=None,
                   help="Output path; default results/<campaign_id>/run_matrix.json")
    return p.parse_args()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config() -> Dict[str, Any]:
    return yaml.safe_load((REPO / "config/experiment.yaml").read_text())


def _hashes() -> Dict[str, str]:
    return {
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
    """Assemble a JSON-ready dict matching :class:`RunConfig` fields."""
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
        "extra":                {"arm": arm_name, "model_key": model_key},
    }


def build_matrix(campaign_id: str) -> Dict[str, Any]:
    cfg = _load_config()
    hashes = _hashes()
    encoder_digest = _encoder_digest()
    models = cfg["generation_models"]
    domain_map = {"federation": "model federation", "repair": "model repair"}
    root_map   = {"federation": "Federation",       "repair": "Repair"}
    k_doc = int(cfg["retrieval"]["k_doc"])

    runs: List[Dict[str, Any]] = []
    per_arm: Dict[str, int] = {}

    def emit(arm: str, corpus: str, N: int, ordering: str, grounding: str,
             reps: int, metamodel: bool, k_doc_val: int | None) -> None:
        for model_key, mcfg in models.items():
            for rep in range(reps):
                seed = rep                # deterministic per-rep seed
                runs.append(make_run_dict(
                    campaign_id=campaign_id, corpus=corpus, N=N,
                    ordering_id=ordering, grounding=grounding,
                    model_key=model_key, model_cfg=mcfg,
                    seed=seed, repetition=rep,
                    metamodel_block=metamodel, k_doc=k_doc_val,
                    domain=domain_map[corpus], root_feature=root_map[corpus],
                    arm_name=arm, hashes=hashes, encoder_digest=encoder_digest,
                ))
        per_arm[arm] = per_arm.get(arm, 0) + reps * len(models) * N

    # ─── Guided headline (N=10, both corpora, both groundings)
    for grounding in ALL_GROUNDINGS:
        emit("guided_headline",  "repair",     REPAIR_HEADLINE_N,     "primary", grounding, REPS_HEADLINE,  True, k_doc)
        emit("guided_headline",  "federation", FEDERATION_HEADLINE_N, "primary", grounding, REPS_HEADLINE,  True, k_doc)

    # ─── Guided baselines (N=1, both corpora, both groundings)
    for grounding in ALL_GROUNDINGS:
        emit("guided_baseline",  "repair",     REPAIR_BASELINE_N,     "primary", grounding, REPS_BASELINE,  True, k_doc)
        emit("guided_baseline",  "federation", FEDERATION_BASELINE_N, "primary", grounding, REPS_BASELINE,  True, k_doc)

    # ─── Guided curves (Repair N∈{5,20,54}, Federation N∈{5,23}, both groundings)
    for N in REPAIR_CURVE_N:
        for grounding in ALL_GROUNDINGS:
            emit("guided_curve", "repair", N, "primary", grounding, REPS_CURVE, True, k_doc)
    for N in FEDERATION_CURVE_N:
        for grounding in ALL_GROUNDINGS:
            emit("guided_curve", "federation", N, "primary", grounding, REPS_CURVE, True, k_doc)

    # ─── Ablation (RAG only, both corpora, N∈{1,10}, metamodel_block=False)
    for corpus in ("repair", "federation"):
        for N in ABLATION_N:
            emit("ablation", corpus, N, "primary", "rag", REPS_ABLATION, False, k_doc)

    # ─── Order sensitivity (Repair only, RAG, N∈{10,54}, 2 alt orderings)
    for N in ORDER_N:
        for ordering in ORDER_ALT_IDS:
            emit("order_sensitivity", "repair", N, ordering, "rag", REPS_ORDER, True, k_doc)

    # ─── k_doc sweep (Repair only, RAG, N=10, 4 k values, 3 reps)
    for k in cfg["retrieval"]["k_doc_grid"]:
        emit("k_doc_sweep_k" + str(k), "repair", KSWEEP_N, "primary", "rag",
             REPS_KSWEEP, True, k)

    return {
        "campaign_id":     campaign_id,
        "generated_at":    None,      # filled by caller
        "hashes":          hashes,
        "encoder_digest":  encoder_digest,
        "models":          {k: {kk: v[kk] for kk in ("provider", "model_id",
                                                     "context_window_tokens",
                                                     "max_output_tokens",
                                                     "reasoning_effort")}
                            for k, v in models.items()},
        "per_arm_calls":   per_arm,
        "total_calls":     sum(per_arm.values()),
        "total_runs":      len(runs),
        "runs":            runs,
    }


def main() -> int:
    args = _cli()
    matrix = build_matrix(args.campaign_id)
    from datetime import datetime
    matrix["generated_at"] = datetime.now().isoformat()

    print(f"===== run matrix: {args.campaign_id} =====")
    print(f"  total runs:  {matrix['total_runs']}")
    print(f"  total calls: {matrix['total_calls']}")
    print()
    print("  per-arm call counts:")
    for arm, n in sorted(matrix["per_arm_calls"].items(), key=lambda kv: -kv[1]):
        print(f"    {arm:<28s} {n:>6d}")

    if args.dry_run:
        return 0

    out = Path(args.out) if args.out else (REPO / f"results/{args.campaign_id}/run_matrix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(matrix, indent=2))
    print(f"\nWrote {out}  ({out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
