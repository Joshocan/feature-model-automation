#!/usr/bin/env python3
"""Zero-call semantic correctness analysis for the frozen pilot cohorts.

The manifest fixes which runs are primary and supplementary. The script keeps
all planned runs in ``run_inventory.csv`` but computes semantic scores only for
completed outputs that pass the frozen FeatureIDE schema/well-formedness gate.

Outputs include raw D15 generated/reference similarities, registered
independent-max precision/recall/F1, a one-to-one redundancy sensitivity,
the registered tau sweep, match multiplicity, and model-level summaries.
No generation-provider calls are made; the matching encoder is loaded only
from the local Hugging Face cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.coverage import extract_nodes  # noqa: E402
from fame.evaluation.local_encoder import (  # noqa: E402
    LocalTransformerEncoder,
    discover_cached_snapshot,
)
from fame.evaluation.semantic import prf_from_similarity  # noqa: E402
from fame.evaluation.wellformed import validate_feature_model  # noqa: E402


DEFAULT_MANIFEST = REPO / "config" / "pilot_correctness_manifest.json"
DEFAULT_OUT = REPO / "results" / "pilot-correctness-analysis-2026-09-23"
TRACE_RE = re.compile(r"Trace:\s*\[\s*(rep_\d+(?:\s*,\s*rep_\d+)*)\s*\]\s*\Z")


def _has_repair_trace(text: str) -> bool:
    """Recognise the trace syntax frozen in the prompt used by these pilots."""
    return TRACE_RE.search(text or "") is not None


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--inventory-only", action="store_true",
        help="Validate the cohort and emit run_inventory.csv without loading the encoder.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    required = {"ground_truth", "xsd", "encoder_model", "tau_primary", "tau_sweep", "sources"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")
    return manifest


def _inventory(manifest: dict[str, Any], xsd: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in manifest["sources"]:
        source_root = (REPO / source["path"]).resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"manifest source does not exist: {source_root}")
        for meta_path in sorted(source_root.glob("*/*/*/run_meta.json")):
            meta = json.loads(meta_path.read_text())
            cfg = meta.get("config", {})
            run_root = meta_path.parent
            fm_gen = run_root / "fm_gen.xml"
            completed = meta.get("completed") is True
            parseable = False
            xsd_valid = False
            description_trace_complete = False
            missing_description_or_trace_count = 0
            errors: list[str] = []
            if fm_gen.exists():
                validation = validate_feature_model(fm_gen, xsd)
                xsd_valid = validation.ok
                errors = list(validation.errors)
                parseable = not any(error.startswith("XML parse error:") for error in errors)
                if parseable:
                    root = ET.parse(fm_gen).getroot()
                    struct = root.find("struct")
                    structural_nodes = [
                        elem for elem in (struct.iter() if struct is not None else [])
                        if elem.tag in {"and", "or", "alt", "feature"} and elem.get("name")
                    ]
                    missing_description_or_trace_count = sum(
                        1 for elem in structural_nodes
                        if elem.find("description") is None
                        or not _has_repair_trace(elem.find("description").text or "")
                    )
                    description_trace_complete = missing_description_or_trace_count == 0
                    if not description_trace_complete:
                        errors.append(
                            f"{missing_description_or_trace_count} named structural nodes lack "
                            "a description with a parseable provenance marker"
                        )
            admissible = (
                completed and fm_gen.exists() and xsd_valid and description_trace_complete
            )
            rows.append({
                "cohort": source["cohort"],
                "primary": bool(source["primary"]),
                "source_campaign": source_root.name,
                "run_id": meta.get("run_id", run_root.name),
                "model_id": cfg.get("model_id"),
                "seed": cfg.get("seed"),
                "N": cfg.get("N"),
                "grounding": cfg.get("grounding"),
                "provider": cfg.get("provider"),
                "reasoning_effort": cfg.get("reasoning_effort"),
                "completed": completed,
                "terminal_status": meta.get("terminal_status"),
                "fm_gen_exists": fm_gen.exists(),
                "parseable": parseable,
                "xsd_valid": xsd_valid,
                "description_trace_complete": description_trace_complete,
                "missing_description_or_trace_count": missing_description_or_trace_count,
                "admissible": admissible,
                "admissibility_errors": " | ".join(errors),
                "run_root": str(run_root.relative_to(REPO)),
                "fm_gen": str(fm_gen.relative_to(REPO)) if fm_gen.exists() else "",
            })
    rows.sort(key=lambda row: (row["cohort"], row["model_id"], int(row["seed"] or 0), row["run_id"]))
    return rows


def _audit_expected(manifest: dict[str, Any], inventory: list[dict[str, Any]]) -> None:
    for cohort, expected in manifest.get("expected_primary", {}).items():
        actual = {
            (row["model_id"], int(row["seed"]))
            for row in inventory if row["primary"] and row["cohort"] == cohort
        }
        wanted = {(model, int(seed)) for model in expected["models"] for seed in expected["seeds"]}
        missing = wanted - actual
        extras = actual - wanted
        duplicates = []
        for key in actual:
            count = sum(
                1 for row in inventory
                if row["primary"] and row["cohort"] == cohort
                and (row["model_id"], int(row["seed"])) == key
            )
            if count > 1:
                duplicates.append(key)
        if missing or extras or duplicates:
            raise ValueError(
                f"cohort {cohort} does not match frozen matrix: missing={sorted(missing)}, "
                f"extras={sorted(extras)}, duplicates={sorted(duplicates)}"
            )


INVENTORY_FIELDS = [
    "cohort", "primary", "source_campaign", "run_id", "model_id", "seed", "N",
    "grounding", "provider", "reasoning_effort", "completed", "terminal_status",
    "fm_gen_exists", "parseable", "xsd_valid", "description_trace_complete",
    "missing_description_or_trace_count", "admissible", "admissibility_errors",
    "run_root", "fm_gen",
]


def _base(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in [
        "cohort", "primary", "source_campaign", "run_id", "model_id", "seed", "N",
        "grounding", "completed", "terminal_status", "admissible",
    ]}


def _summaries(inventory: list[dict[str, Any]], metrics: list[dict[str, Any]], tau: float) -> list[dict[str, Any]]:
    scored = {
        (row["run_id"], row["matching_policy"]): row
        for row in metrics if float(row["tau"]) == tau
    }
    grouped: dict[tuple[str, bool, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        grouped[(row["cohort"], row["primary"], row["model_id"], row["grounding"])].append(row)
    output = []
    for (cohort, primary, model_id, grounding), runs in sorted(grouped.items()):
        for policy in ("independent_max", "one_to_one"):
            values = [
                scored[(r["run_id"], policy)] for r in runs
                if (r["run_id"], policy) in scored
                and "semantic_precision" in scored[(r["run_id"], policy)]
            ]
            mean = lambda key: (sum(float(v[key]) for v in values) / len(values)) if values else ""
            output.append({
                "cohort": cohort,
                "primary": primary,
                "model_id": model_id,
                "grounding": grounding,
                "matching_policy": policy,
                "tau": tau,
                "n_planned": len(runs),
                "n_completed": sum(bool(r["completed"]) for r in runs),
                "n_admissible": sum(bool(r["admissible"]) for r in runs),
                "n_scored": len(values),
                "precision_mean_admissible": mean("semantic_precision"),
                "recall_mean_admissible": mean("semantic_recall"),
                "f1_mean_admissible": mean("semantic_f1"),
            })
    return output


def main() -> int:
    args = _cli()
    manifest_path = args.manifest.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)
    gt_path = (REPO / manifest["ground_truth"]).resolve()
    xsd_path = (REPO / manifest["xsd"]).resolve()

    inventory = _inventory(manifest, xsd_path)
    _audit_expected(manifest, inventory)
    _write_csv(out_dir / "run_inventory.csv", inventory, INVENTORY_FIELDS)
    print(f"inventory: {len(inventory)} runs; {sum(r['admissible'] for r in inventory)} admissible")
    if args.inventory_only:
        return 0

    snapshot = discover_cached_snapshot(manifest["encoder_model"])
    print(f"encoder snapshot: {snapshot.name}")
    encoder = LocalTransformerEncoder(snapshot)
    reference_nodes = extract_nodes(gt_path)
    reference_names = [name for name, _ in reference_nodes]
    reference_embeddings = encoder.encode(
        reference_names, normalize_embeddings=True, convert_to_tensor=False
    )

    match_fields = [
        "cohort", "primary", "source_campaign", "run_id", "model_id", "seed", "N",
        "grounding", "generated_index", "generated_feature", "generated_parent",
        "reference_index", "reference_feature", "reference_parent", "similarity",
        "exact_name_match", "exact_parent_match",
    ]
    metric_fields = [
        "cohort", "primary", "source_campaign", "run_id", "model_id", "seed", "N",
        "grounding", "completed", "terminal_status", "admissible", "matching_policy", "tau",
        "n_generated", "n_reference", "n_generated_matched", "n_reference_matched",
        "n_matched_pairs", "semantic_precision", "semantic_recall", "semantic_f1",
    ]
    multiplicity_fields = [
        "cohort", "primary", "run_id", "model_id", "seed", "tau", "reference_index",
        "reference_feature", "assigned_generated_count", "assigned_generated_features",
    ]
    metrics: list[dict[str, Any]] = []
    multiplicity: list[dict[str, Any]] = []
    match_path = out_dir / "D15_match_scores.csv"
    with match_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=match_fields, extrasaction="ignore")
        writer.writeheader()
        for run_number, row in enumerate(inventory, start=1):
            if not row["admissible"]:
                continue
            generated_nodes = extract_nodes(REPO / row["fm_gen"])
            generated_names = [name for name, _ in generated_nodes]
            generated_embeddings = encoder.encode(
                generated_names, normalize_embeddings=True, convert_to_tensor=False
            )
            similarity = np.asarray(generated_embeddings) @ np.asarray(reference_embeddings).T
            print(
                f"[{run_number}/{len(inventory)}] {row['model_id']} seed={row['seed']} "
                f"matrix={similarity.shape[0]}x{similarity.shape[1]}"
            )
            base = _base(row)
            for gi, (generated_name, generated_parent) in enumerate(generated_nodes):
                for ri, (reference_name, reference_parent) in enumerate(reference_nodes):
                    writer.writerow({
                        **base,
                        "generated_index": gi,
                        "generated_feature": generated_name,
                        "generated_parent": generated_parent or "",
                        "reference_index": ri,
                        "reference_feature": reference_name,
                        "reference_parent": reference_parent or "",
                        "similarity": f"{float(similarity[gi, ri]):.8f}",
                        "exact_name_match": generated_name == reference_name,
                        "exact_parent_match": (
                            generated_parent == reference_parent
                            if generated_name == reference_name else ""
                        ),
                    })
            for tau in manifest["tau_sweep"]:
                for policy in ("independent_max", "one_to_one"):
                    metrics.append({
                        **base,
                        "matching_policy": policy,
                        "tau": tau,
                        **prf_from_similarity(similarity, threshold=float(tau), matching_policy=policy),
                    })
            primary_tau = float(manifest["tau_primary"])
            assignments: dict[int, list[str]] = defaultdict(list)
            for gi, (generated_name, _) in enumerate(generated_nodes):
                ri = int(np.argmax(similarity[gi]))
                if similarity[gi, ri] >= primary_tau:
                    assignments[ri].append(generated_name)
            for ri, names in sorted(assignments.items()):
                multiplicity.append({
                    **base,
                    "tau": primary_tau,
                    "reference_index": ri,
                    "reference_feature": reference_names[ri],
                    "assigned_generated_count": len(names),
                    "assigned_generated_features": " | ".join(names),
                })

    # Explicit blank metric rows preserve failed/inadmissible observations.
    metric_keys = {(row["run_id"], row["matching_policy"], float(row["tau"])) for row in metrics}
    for row in inventory:
        for tau in manifest["tau_sweep"]:
            for policy in ("independent_max", "one_to_one"):
                if (row["run_id"], policy, float(tau)) not in metric_keys:
                    metrics.append({**_base(row), "matching_policy": policy, "tau": tau})
    metrics.sort(key=lambda row: (row["cohort"], row["model_id"], int(row["seed"] or 0), float(row["tau"]), row["matching_policy"]))
    _write_csv(out_dir / "semantic_metrics_by_run.csv", metrics, metric_fields)
    _write_csv(out_dir / "match_multiplicity.csv", multiplicity, multiplicity_fields)

    summaries = _summaries(inventory, metrics, float(manifest["tau_primary"]))
    summary_fields = [
        "cohort", "primary", "model_id", "grounding", "matching_policy", "tau",
        "n_planned", "n_completed", "n_admissible", "n_scored",
        "precision_mean_admissible", "recall_mean_admissible", "f1_mean_admissible",
    ]
    _write_csv(out_dir / "semantic_summary_by_model.csv", summaries, summary_fields)

    protocol = {
        "analysis_id": manifest["analysis_id"],
        "manifest": str(manifest_path.relative_to(REPO)),
        "manifest_sha256": _sha256(manifest_path),
        "ground_truth": str(gt_path.relative_to(REPO)),
        "ground_truth_sha256": _sha256(gt_path),
        "xsd": str(xsd_path.relative_to(REPO)),
        "xsd_sha256": _sha256(xsd_path),
        "encoder_model": manifest["encoder_model"],
        "encoder_snapshot": snapshot.name,
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": importlib.metadata.version("torch"),
        "similarity": "cosine on L2-normalised embeddings",
        "registered_matching_policy": "independent_max",
        "sensitivity_matching_policy": "one_to_one_maximum_cardinality_at_tau",
        "tau_primary": manifest["tau_primary"],
        "tau_sweep": manifest["tau_sweep"],
        "failure_policy": "retain in inventory and metric table; do not assign invented P/R/F1",
        "score_population": (
            "completed, fm_gen-present outputs passing frozen well-formedness/XSD gate "
            "with a description and parseable Repair trace on every named structural node"
        ),
    }
    (out_dir / "analysis_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
