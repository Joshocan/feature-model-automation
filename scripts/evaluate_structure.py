#!/usr/bin/env python3
"""Offline inventory-led diagnostics, not final admissibility."""
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
from fame.evaluation.inventory import sha256
from fame.evaluation.structural import METRICS, evaluate_structure, metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = (REPO / "results/ifs-2027/analysis").resolve()
    if output.exists() or output == allowed or not output.is_relative_to(allowed):
        parser.error("Choose a new output subdirectory under results/ifs-2027/analysis")
    source = args.inventory.resolve()
    summary = json.loads((source / "summary.json").read_text())
    contract_path = REPO / "config/evaluation/ifs-2027-v0.1.0.json"
    if summary.get("contract_sha256") != sha256(contract_path):
        parser.error("Inventory contract hash differs from current evaluation contract; create a fresh inventory")
    if summary["issues"]:
        parser.error("Resolve inventory integrity issues before evaluation")
    runs = json.loads((source / "runs.json").read_text())
    artifacts = json.loads((source / "artifacts.json").read_text())
    if len(runs) != summary["planned"] or len({r["run_id"] for r in runs}) != len(runs):
        parser.error("Inventory run count/identity mismatch")
    indexed = {a["resolved_path"]: a for a in artifacts if a["population"] == "main" and a["snapshot"] == "selected"}
    xsd = REPO / "prompts/feature-model-schema.xsd"
    if any(r["metamodel_hash"] != sha256(xsd) for r in runs):
        parser.error("Current XSD differs from frozen run schema")
    rows = []
    for run in runs:
        base = {key:run[key] for key in ("run_id", "corpus", "model_id", "arm", "N", "grounding", "ordering_id", "k_doc", "metamodel_block", "seed", "repetition", "inventory_status")}
        if not run["completed"]:
            metrics = {k: metric(status="ineligible", reason="No completed final output; recorded outcome retained") for k in METRICS}
        else:
            relative = run["resolved_path"] + "/fm_gen.xml"
            path = (REPO / relative).resolve()
            if not path.is_relative_to(REPO) or relative not in indexed or not path.is_file() or sha256(path) != indexed[relative]["sha256"]:
                parser.error(f"Final artifact changed/missing since inventory: {run['run_id']}")
            metrics = evaluate_structure(path, xsd, expected_root=run["root_feature"])
        for name, result in metrics.items():
            rows.append({**base, "metric": name, **result})
    output.mkdir(parents=True, exist_ok=False)
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    versions = {}
    for package in ("lxml", "python-sat"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = dict(planned_runs=len(runs), completed_outputs=sum(r["completed"] for r in runs),
        evaluator_errors=sum(r["status"] == "evaluator_error" for r in rows),
        metric_status_counts=dict(Counter(r["status"] for r in rows)),
        publication_ready=False, unresolved=["D02 strict sensitivity gate and FeatureIDE parser"],
        inventory_sha256=sha256(source / "runs.json"), artifacts_sha256=sha256(source / "artifacts.json"),
        xsd_sha256=sha256(xsd), versions=versions, sat_backend="python-sat Solver(name='g3')",
        implementation_sha256={p:sha256(REPO/p) for p in ("fame/evaluation/structural.py", "fame/evaluation/quality_sat.py", "scripts/evaluate_structure.py")})
    for name in ("xsd_valid", "satisfiable", "degenerate"):
        result[name] = {str(value):sum(r["metric"] == name and r["value"] is value for r in rows) for value in (True, False, None)}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 1 if result["evaluator_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
