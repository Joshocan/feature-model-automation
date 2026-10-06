from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_evaluation_outputs import verify


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_verifier_reconciles_all_planned_runs(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "runs.json").write_text(json.dumps([
        {"run_id": "a", "inventory_status": "completed"},
        {"run_id": "b", "inventory_status": "terminal_failed"},
    ]), encoding="utf-8")
    (inventory / "artifacts.json").write_text("[]", encoding="utf-8")
    runs_hash = hashlib.sha256((inventory / "runs.json").read_bytes()).hexdigest()
    artifacts_hash = hashlib.sha256((inventory / "artifacts.json").read_bytes()).hexdigest()
    directories = [tmp_path / name for name in ("struct", "sem", "prov")]
    for directory in directories:
        _write_csv(directory / "metrics.csv", [
            {"run_id": "a", "metric": "m", "status": "ok"},
            {"run_id": "b", "metric": "m", "status": "ineligible"},
        ])
        (directory / "summary.json").write_text(json.dumps({
            "inventory_sha256": runs_hash, "artifacts_sha256": artifacts_hash,
        }), encoding="utf-8")
    aggregate = tmp_path / "aggregate"
    _write_csv(aggregate / "wide.csv", [
        {"run_id": "a", "primary_semantic_eligible": "True",
         "strict_admissible": "False"},
        {"run_id": "b", "primary_semantic_eligible": "False",
         "strict_admissible": "False"},
    ])
    (aggregate / "summary.json").write_text(json.dumps({
        "source_sha256": {name: hashlib.sha256((directory / "metrics.csv").read_bytes()).hexdigest()
                          for name, directory in zip(("structural", "semantic", "provenance"), directories)}
    }), encoding="utf-8")
    result = verify(inventory, *directories, aggregate)
    assert result["planned_runs"] == 2
    assert result["completed_runs"] == 1
    assert result["populations"]["primary_semantic_eligible"] == {"True": 1, "False": 1}

    _write_csv(directories[1] / "metrics.csv", [
        {"run_id": "a", "metric": "m", "status": "ok"}])
    with pytest.raises(ValueError, match="run-ID mismatch"):
        verify(inventory, *directories, aggregate)
