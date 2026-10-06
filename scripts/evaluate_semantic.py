#!/usr/bin/env python3
"""inventory-led semantic evaluation with explicit metric envelopes.

Verifies encoder identity against the recorded pilot revision (D03), then for
every planned run:

  * non-completed → all semantic metrics receive ``status="ineligible"``
  * completed     → cosine matrix vs the frozen reference model, τ-primary
                    metrics, τ-sweep view, dual recall against D21 partition
                    and D4 reach.

Raw pairs (generated × reference cosine similarities) go to ``pairs.csv``
(D15). No thresholding is baked in; τ sweeps re-threshold this file.

Usage
-----

    ./.venv/bin/python scripts/evaluate_semantic.py \
      --inventory results/ifs-2027/analysis/inventory-2026-09-26-v2 \
      --output    results/ifs-2027/analysis/semantic-2026-09-26-v1

The output directory must be a fresh subdirectory under
``results/ifs-2027/analysis``. Existing directories, or the analysis root
itself, are rejected to keep the audit trail intact.
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.inventory import sha256                       # noqa: E402
from fame.evaluation.manifests import read_manifest_doc_ids
from fame.evaluation.coverage import extract_nodes
from fame.evaluation.local_encoder import (                        # noqa: E402
    LocalTransformerEncoder, discover_cached_snapshot, verify_encoder_identity,
)
from fame.evaluation.reachable import (                            # noqa: E402
    load_attribution, load_partition, reach_features, validate_reach_inputs,
)
from fame.evaluation.semantic import SEMANTIC_METRICS, evaluate_semantic  # noqa: E402


CONTRACT_PATH = REPO / "config/evaluation/ifs-2027-v0.1.0.json"
GROUND_TRUTH = {"repair": REPO / "data/ground_truth/repair.xml",
                "federation": REPO / "data/ground_truth/federation.xml"}
ATTRIBUTION = {"repair": REPO / "data/attribution/repair.csv",
               "federation": REPO / "data/attribution/federation.csv"}
PARTITION = {"repair": REPO / "data/feature_partition/repair.csv",
             "federation": REPO / "data/feature_partition/federation.csv"}
RHO_PATH = REPO / "data/calibration/rho.json"
MANIFESTS = {"repair": REPO / "data/raw/repair/manifest_repair.csv",
             "federation": REPO / "data/raw/federation/manifest_fed.csv"}


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="Run inventory directory (contains runs.json, artifacts.json, summary.json)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh subdirectory under results/ifs-2027/analysis/")
    parser.add_argument("--tau-sweep", type=float, nargs="*", default=None,
                        help="τ values to include in sweep (default from contract)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: only score the first N completed runs (smoke test)")
    return parser.parse_args()


def _read_manifest_docids(manifest_csv: Path) -> list[str]:
    return read_manifest_doc_ids(manifest_csv)


def _partition_and_reach_for(corpus: str) -> dict:
    """Load D21 partition, D4 attribution and derive reach for one corpus.

    Reach uses the full corpus manifest, not only retrieved passages.
    It describes annotated corpus availability, not actual evidence exposure.
    """
    partition = load_partition(PARTITION[corpus])
    attribution = load_attribution(ATTRIBUTION[corpus])
    manifest_ids = _read_manifest_docids(MANIFESTS[corpus])
    reference_nodes = extract_nodes(GROUND_TRUTH[corpus])
    validate_reach_inputs(partition, attribution, manifest_ids,
                          [name for name, _ in reference_nodes])
    direct = {f for f, docs in attribution.items() if docs & set(manifest_ids)} & partition.attested
    reach = reach_features(attribution=attribution,
                            corpus_doc_ids=manifest_ids,
                            parents=dict(reference_nodes),
                            only_attested=partition.attested)
    rho_json = json.loads(RHO_PATH.read_text()) if RHO_PATH.is_file() else {}
    rho = len(reach) / len(partition.full)
    recorded = rho_json.get(corpus)
    recorded_rho = recorded.get("rho") if isinstance(recorded, dict) else recorded
    # Historical calibration used direct attribution, not ancestor closure.
    if recorded_rho is not None and not math.isclose(float(recorded_rho), len(direct) / len(partition.full), rel_tol=0, abs_tol=0.00005):
        raise ValueError(f"{corpus}: recorded rho disagrees with historical direct-attribution ratio")
    if isinstance(recorded, dict):
        for key, actual in {"reach_size": len(direct), "F_t": len(partition.full),
                            "F_t_attested": len(partition.attested),
                            "F_t_organising": len(partition.organising)}.items():
            if key in recorded and recorded[key] != actual:
                raise ValueError(f"{corpus}: calibration {key} disagrees with current inputs")
    return dict(partition=partition, reach=reach, rho=rho, manifest_ids=manifest_ids,
                direct=direct, ancestors_added=reach - direct)


def _load_contract() -> dict:
    if not CONTRACT_PATH.is_file():
        raise SystemExit(f"Missing evaluation contract: {CONTRACT_PATH}")
    return json.loads(CONTRACT_PATH.read_text())


def _encoder_or_die(contract: dict) -> tuple[LocalTransformerEncoder, dict]:
    sem = contract["semantic"]
    snapshot = discover_cached_snapshot(sem["encoder_model"])
    identity = verify_encoder_identity(
        snapshot,
        expected_revision=sem.get("recorded_pilot_revision"),
        expected_versions=sem.get("recorded_pilot_versions"),
    )
    if identity["status"] != "ok":
        raise SystemExit(f"Encoder identity check failed [{identity['status']}]: {identity['reason']}")
    return LocalTransformerEncoder(snapshot), identity


def _flatten_env(prefix: str, envelope: dict) -> dict:
    row = {f"{prefix}": envelope.get("value")}
    row[f"{prefix}_status"] = envelope.get("status")
    row[f"{prefix}_reason"] = envelope.get("reason")
    if "numerator" in envelope:
        row[f"{prefix}_numerator"] = envelope.get("numerator")
    if "denominator" in envelope:
        row[f"{prefix}_denominator"] = envelope.get("denominator")
    return row


def _row_to_csv_scalar(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def main() -> int:
    args = _cli()
    contract = _load_contract()
    sem_cfg = contract["semantic"]
    tau_primary = float(sem_cfg["tau_primary"])
    tau_sweep = tuple(args.tau_sweep or sem_cfg.get("tau_sweep", [0.3, 0.4, 0.5, 0.6]))

    inventory = args.inventory.resolve()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        raise SystemExit(f"Choose a new subdirectory under {allowed.relative_to(REPO)}")

    summary = json.loads((inventory / "summary.json").read_text())
    if summary.get("contract_sha256") != sha256(CONTRACT_PATH):
        raise SystemExit("Inventory contract hash differs from current evaluation contract; create a fresh inventory")
    if summary["issues"]:
        raise SystemExit("Resolve inventory integrity issues before evaluation")
    runs = json.loads((inventory / "runs.json").read_text())
    artifacts = json.loads((inventory / "artifacts.json").read_text())
    if len(runs) != summary["planned"]:
        raise SystemExit("Inventory run count mismatch")

    indexed = {a["resolved_path"]: a for a in artifacts
               if a["population"] == "main" and a["snapshot"] == "selected"}

    print(f"Contract:  {CONTRACT_PATH.relative_to(REPO)}")
    print(f"Inventory: {inventory.relative_to(REPO)}")
    print(f"Output:    {output.relative_to(REPO)}")
    print(f"τ_primary: {tau_primary}   τ_sweep: {list(tau_sweep)}")

    per_corpus = {c: {**_partition_and_reach_for(c),
                      "attribution": load_attribution(ATTRIBUTION[c])}
                  for c in GROUND_TRUTH}
    for c, info in per_corpus.items():
        print(f"  {c:10s} |F_t|={len(info['partition'].full):3d}  "
              f"|F_t^att|={len(info['partition'].attested):3d}  "
              f"|reach|={len(info['reach']):3d}  ρ={info['rho']}")

    encoder, identity = _encoder_or_die(contract)
    print(f"Encoder:   {sem_cfg['encoder_model']} @ {identity['revision']}  [{identity['status']}]")

    rows: list[dict] = []
    pairs_rows: list[dict] = []
    scored = 0
    for run in runs:
        base = {k: run[k] for k in ("run_id", "corpus", "model_id", "arm", "N",
                                     "grounding", "ordering_id", "k_doc",
                                     "metamodel_block", "seed", "repetition",
                                     "inventory_status")}
        if not run["completed"]:
            metrics = {k: dict(value=None, status="ineligible",
                                reason="No completed final output") for k in SEMANTIC_METRICS}
            for name, env in metrics.items():
                rows.append({**base, "metric": name, **env})
            continue

        corpus = run["corpus"]
        info = per_corpus.get(corpus)
        if info is None:
            metrics = {k: dict(value=None, status="unsupported",
                                reason=f"No ground-truth artefacts loaded for corpus {corpus!r}")
                       for k in SEMANTIC_METRICS}
            for name, env in metrics.items():
                rows.append({**base, "metric": name, **env})
            continue

        relative = run["resolved_path"] + "/fm_gen.xml"
        path = (REPO / relative).resolve()
        if (not path.is_relative_to(REPO) or relative not in indexed
                or not path.is_file() or sha256(path) != indexed[relative]["sha256"]):
            raise SystemExit(f"Final artefact changed/missing since inventory: {run['run_id']}")

        outcome = evaluate_semantic(
            path, GROUND_TRUTH[corpus],
            encoder=encoder,
            tau_primary=tau_primary, tau_sweep=tau_sweep,
            attested=info["partition"].attested,
            organising=info["partition"].organising,
            reach=info["reach"],
            rho=info["rho"],
            attribution=info["attribution"],
        )
        scored += 1
        for name, env in outcome["metrics"].items():
            rows.append({**base, "metric": name, **env})
        for pair in outcome["pairs"]:
            pairs_rows.append({"run_id": run["run_id"], "corpus": corpus, **pair})

        if args.limit and scored >= args.limit:
            print(f"  --limit {args.limit} reached; stopping early")
            break

    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _row_to_csv_scalar(row.get(k)) for k in fields})

    if pairs_rows:
        pair_fields = list(pairs_rows[0].keys())
        with (output / "pairs.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=pair_fields)
            writer.writeheader()
            for row in pairs_rows:
                writer.writerow(row)

    versions = {}
    for pkg in ("numpy", "transformers", "torch", "sentence-transformers"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None

    result = dict(
        planned_runs=len(runs),
        completed=sum(r["completed"] for r in runs),
        semantic_scored=scored,
        evaluator_errors=sum(1 for r in rows if r["status"] == "evaluator_error"),
        metric_status_counts=dict(Counter(r["status"] for r in rows)),
        raw_pairs=len(pairs_rows),
        tau_primary=tau_primary,
        tau_sweep=list(tau_sweep),
        encoder_identity=identity,
        reach_definition="reference_ancestor_closure_v1",
        historical_calibration_definition="direct_attribution_only",
        reach_calibration={c: dict(reach_size=len(info["reach"]),
                                  direct_size=len(info["direct"]),
                                  ancestors_added=sorted(info["ancestors_added"]),
                                  reach_features=sorted(info["reach"]),
                                  F_t_size=len(info["partition"].full), rho=info["rho"],
                                  manifest_doc_count=len(info["manifest_ids"]))
                           for c, info in per_corpus.items()},
        evaluation_input_sha256={str(p.relative_to(REPO)): sha256(p)
                                 for mapping in (MANIFESTS, PARTITION, ATTRIBUTION, GROUND_TRUTH)
                                 for p in mapping.values()},
        contract_sha256=sha256(CONTRACT_PATH),
        inventory_sha256=sha256(inventory / "runs.json"),
        artifacts_sha256=sha256(inventory / "artifacts.json"),
        implementation_sha256={p: sha256(REPO / p) for p in (
            "fame/evaluation/semantic.py",
            "fame/evaluation/local_encoder.py",
            "fame/evaluation/reachable.py",
            "fame/evaluation/manifests.py",
            "scripts/evaluate_semantic.py",
        )},
        versions=versions,
        publication_ready=False,
        unresolved=["D03 pooled encoder-weight hash review",
                    "D04 statistical family definitions"],
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("encoder_identity", "implementation_sha256", "versions")}, indent=2))
    return 1 if result["evaluator_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
