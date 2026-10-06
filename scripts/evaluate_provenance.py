#!/usr/bin/env python3
"""inventory-led provenance evaluation.

Verifies inventory integrity, then for every planned run:

  * non-completed → all provenance metrics receive ``status="ineligible"``
  * completed     → L0 marker emission, L0 parse rate, L1 referential
                    integrity against the D1 manifest doc_id set, L2
                    hallucination check (vacuous for Non-RAG at N=1), and
                    recency-bias distribution.

Outputs land under a fresh subdirectory of ``results/ifs-2027/analysis``:

  * ``metrics.csv`` / ``metrics.json`` — long-form envelope per (run, metric).
  * ``citations.csv``                   — D16 rows (one per parseable pair
                                          across all runs).
  * ``hallucinated.csv``                — L2 offenders for post-hoc audit.
  * ``summary.json``                    — hashes, status histogram, versions.

Usage
-----

    ./.venv/bin/python scripts/evaluate_provenance.py \
      --inventory results/ifs-2027/analysis/inventory-2026-09-26-v2 \
      --output    results/ifs-2027/analysis/provenance-2026-09-26-v1
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.inventory import sha256                     # noqa: E402
from fame.evaluation.manifests import read_manifest_doc_ids
from fame.evaluation.provenance import (                          # noqa: E402
    PROVENANCE_METRICS, evaluate_provenance,
)


CONTRACT_PATH = REPO / "config/evaluation/ifs-2027-v0.1.0.json"
MANIFESTS = {"repair": REPO / "data/raw/repair/manifest_repair.csv",
             "federation": REPO / "data/raw/federation/manifest_fed.csv"}


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inventory", type=Path, required=True,
                        help="Run inventory directory")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh subdirectory under results/ifs-2027/analysis/")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional: only score the first N completed runs (smoke test)")
    return parser.parse_args()


def _load_known_doc_ids(corpus: str) -> list[str]:
    return read_manifest_doc_ids(MANIFESTS[corpus])


def _row_to_csv_scalar(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value


def main() -> int:
    args = _cli()
    inventory = args.inventory.resolve()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        raise SystemExit(f"Choose a new subdirectory under {allowed.relative_to(REPO)}")

    contract = json.loads(CONTRACT_PATH.read_text())
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

    known_by_corpus = {c: _load_known_doc_ids(c) for c in MANIFESTS}
    for c, ids in known_by_corpus.items():
        print(f"  {c:10s} known doc_ids: {len(ids)}")

    rows: list[dict] = []
    citation_rows: list[dict] = []
    hallu_rows: list[dict] = []
    scored = 0
    for run in runs:
        base = {k: run[k] for k in ("run_id", "corpus", "model_id", "arm", "N",
                                     "grounding", "ordering_id", "k_doc",
                                     "metamodel_block", "seed", "repetition",
                                     "inventory_status")}
        if not run["completed"]:
            for name in PROVENANCE_METRICS:
                rows.append({**base, "metric": name, "value": None,
                             "status": "ineligible",
                             "reason": "No completed final output"})
            continue

        corpus = run["corpus"]
        run_root = REPO / run["resolved_path"]
        fm_gen = run_root / "fm_gen.xml"
        iter_dir = run_root / "fm_iter"
        context_log = run_root / "context_log.jsonl"
        relative = run["resolved_path"] + "/fm_gen.xml"

        if (relative not in indexed or not fm_gen.is_file()
                or sha256(fm_gen) != indexed[relative]["sha256"]):
            raise SystemExit(f"Final artefact changed/missing since inventory: {run['run_id']}")

        outcome = evaluate_provenance(
            fm_gen, iter_dir, context_log,
            known_doc_ids=known_by_corpus.get(corpus, []),
            n_planned_steps=run.get("N"),
        )
        scored += 1
        for name, env in outcome["metrics"].items():
            rows.append({**base, "metric": name, **env})
        for citation in outcome["citations"]:
            citation_rows.append({"run_id": run["run_id"], "corpus": corpus,
                                   **citation})
        for hallu in outcome["hallucinated"]:
            hallu_rows.append({"run_id": run["run_id"], "corpus": corpus,
                                **{k: v for k, v in hallu.items()
                                   if k != "docs_in_context"}})

        if args.limit and scored >= args.limit:
            print(f"  --limit {args.limit} reached; stopping early")
            break

    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")

    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _row_to_csv_scalar(row.get(k)) for k in fields})

    if citation_rows:
        cite_fields = list(citation_rows[0].keys())
        with (output / "citations.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cite_fields)
            writer.writeheader()
            for row in citation_rows:
                writer.writerow(row)

    if hallu_rows:
        h_fields = list(hallu_rows[0].keys())
        with (output / "hallucinated.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=h_fields)
            writer.writeheader()
            for row in hallu_rows:
                writer.writerow(row)

    versions = {}
    for pkg in ("lxml",):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None

    result = dict(
        planned_runs=len(runs),
        completed=sum(r["completed"] for r in runs),
        provenance_scored=scored,
        evaluator_errors=sum(1 for r in rows if r["status"] == "evaluator_error"),
        metric_status_counts=dict(Counter(r["status"] for r in rows)),
        n_citation_rows=len(citation_rows),
        n_hallucinated_rows=len(hallu_rows),
        manifest_sha256={c: sha256(p) for c, p in MANIFESTS.items()},
        known_doc_id_counts={c: len(ids) for c, ids in known_by_corpus.items()},
        contract_sha256=sha256(CONTRACT_PATH),
        inventory_sha256=sha256(inventory / "runs.json"),
        artifacts_sha256=sha256(inventory / "artifacts.json"),
        implementation_sha256={p: sha256(REPO / p) for p in (
            "fame/evaluation/provenance.py",
            "fame/evaluation/manifests.py",
            "fame/evaluation/hallucination.py",
            "fame/evaluation/recency.py",
            "fame/utils/marker_grammar.py",
            "scripts/evaluate_provenance.py",
        )},
        versions=versions,
        publication_ready=False,
        unresolved=["D05 provenance rubric for expert audit"],
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("implementation_sha256", "versions")}, indent=2))
    return 1 if result["evaluator_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
