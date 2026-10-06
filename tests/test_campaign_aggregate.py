from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.aggregate_campaign import _metrics_csv_path

from fame.evaluation.campaign_aggregate import (
    DEFAULT_CELL_KEYS,
    add_evaluation_populations,
    degenerate_share,
    group_by_cell,
    infer_metric_columns,
    join_by_run,
    load_long_metrics,
    summarise_campaign,
    summarise_cell,
    variability_report,
)


def test_primary_and_strict_semantic_populations_are_distinct() -> None:
    structural = {f"structural__{key}": True for key in
                  ("xsd_valid", "W1", "W2", "W3", "W4", "W5",
                   "tree_invariants", "identifier_syntax", "xml_envelope")}
    structural.update({f"{key}__status": "ok" for key in structural})
    common = dict(inventory_status="completed", semantic__n_generated=12,
                  semantic__n_generated__status="ok",
                  provenance__feature_trace_coverage=1,
                  provenance__feature_trace_coverage__status="ok",
                  provenance__L1_referential_integrity=1,
                  provenance__L1_referential_integrity__status="ok")
    good = dict(common, **structural)
    bad_xsd = dict(good, structural__xsd_valid=False)
    missing_provenance = dict(good)
    missing_provenance.pop("provenance__L1_referential_integrity__status")
    incomplete = dict(good, inventory_status="terminal_failed")
    rows = add_evaluation_populations([good, bad_xsd, missing_provenance, incomplete])
    assert [(r["primary_semantic_eligible"], r["strict_admissible"]) for r in rows] == [
        (True, True), (True, False), (True, None), (False, False)]


def _write_long_csv(path: Path, rows: list[dict]) -> Path:
    fields = list({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def test_metrics_csv_path_handles_relative_and_absolute_inputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "source" / "metrics.csv"
    assert _metrics_csv_path(Path("source")) == expected
    assert _metrics_csv_path(tmp_path / "source") == expected


def _long_row(run_id: str, metric: str, value, status="ok", **cell):
    row = dict(run_id=run_id, metric=metric, value=value, status=status, reason="")
    row.update(cell)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# load + join
# ─────────────────────────────────────────────────────────────────────────────

def test_load_long_metrics_coerces_numeric(tmp_path: Path) -> None:
    path = _write_long_csv(tmp_path / "m.csv", [
        _long_row("r1", "n_features", 42, corpus="repair"),
        _long_row("r1", "structural_conformance", "true", corpus="repair"),
    ])
    rows = load_long_metrics(path)
    assert rows[0]["value"] == 42
    assert rows[1]["value"] is True


def test_join_by_run_widens_by_source(tmp_path: Path) -> None:
    structural = [
        _long_row("r1", "n_features", 12, corpus="repair", model_id="ds",
                   N=1, grounding="rag", arm="baseline"),
        _long_row("r1", "structural_conformance", True, corpus="repair", model_id="ds",
                   N=1, grounding="rag", arm="baseline"),
    ]
    semantic = [
        _long_row("r1", "semantic_f1_total", 0.72, corpus="repair", model_id="ds",
                   N=1, grounding="rag", arm="baseline"),
    ]
    wide = join_by_run({"structural": structural, "semantic": semantic})
    assert len(wide) == 1
    row = wide[0]
    assert row["structural__n_features"] == 12
    assert row["structural__structural_conformance"] is True
    assert row["semantic__semantic_f1_total"] == 0.72
    assert row["structural__n_features__status"] == "ok"


def test_join_by_run_rejects_duplicate_metric_within_source() -> None:
    dupes = [
        _long_row("r1", "n_features", 12),
        _long_row("r1", "n_features", 13),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        join_by_run({"structural": dupes})


# ─────────────────────────────────────────────────────────────────────────────
# summarise_cell
# ─────────────────────────────────────────────────────────────────────────────

def test_summarise_cell_drops_non_ok_from_mean() -> None:
    rows = [
        {"structural__semantic_f1_total": 0.5, "structural__semantic_f1_total__status": "ok"},
        {"structural__semantic_f1_total": 0.8, "structural__semantic_f1_total__status": "ok"},
        {"structural__semantic_f1_total": None, "structural__semantic_f1_total__status": "ineligible"},
    ]
    summary = summarise_cell(rows, ["structural__semantic_f1_total"])
    assert summary["structural__semantic_f1_total__mean"] == pytest.approx(0.65)
    assert summary["structural__semantic_f1_total__n_ok"] == 2
    assert summary["structural__semantic_f1_total__n_ineligible"] == 1
    assert summary["structural__semantic_f1_total__n_total"] == 3


def test_summarise_cell_no_ok_rows_yields_null_mean() -> None:
    rows = [
        {"col": None, "col__status": "missing_artifact"},
        {"col": None, "col__status": "ineligible"},
    ]
    summary = summarise_cell(rows, ["col"])
    assert summary["col__mean"] is None
    assert summary["col__median"] is None
    assert summary["col__n_ok"] == 0
    assert summary["col__n_missing_artifact"] == 1


def test_summarise_cell_treats_bool_ok_as_1_or_0() -> None:
    rows = [
        {"structural__structural_conformance": True,  "structural__structural_conformance__status": "ok"},
        {"structural__structural_conformance": False, "structural__structural_conformance__status": "ok"},
        {"structural__structural_conformance": True,  "structural__structural_conformance__status": "ok"},
    ]
    summary = summarise_cell(rows, ["structural__structural_conformance"])
    assert summary["structural__structural_conformance__mean"] == pytest.approx(2 / 3)


# ─────────────────────────────────────────────────────────────────────────────
# group_by_cell + summarise_campaign
# ─────────────────────────────────────────────────────────────────────────────

def test_group_by_cell_partitions_runs() -> None:
    wide = [
        {"run_id": "r1", "corpus": "repair", "model_id": "ds", "arm": "a", "N": 1,
         "grounding": "rag", "ordering_id": "o", "k_doc": 5, "metamodel_block": "v1"},
        {"run_id": "r2", "corpus": "repair", "model_id": "ds", "arm": "a", "N": 1,
         "grounding": "rag", "ordering_id": "o", "k_doc": 5, "metamodel_block": "v1"},
        {"run_id": "r3", "corpus": "federation", "model_id": "ds", "arm": "a", "N": 1,
         "grounding": "rag", "ordering_id": "o", "k_doc": 5, "metamodel_block": "v1"},
    ]
    cells = group_by_cell(wide)
    assert len(cells) == 2  # repair vs federation
    n_per_cell = sorted(len(v) for v in cells.values())
    assert n_per_cell == [1, 2]


def test_summarise_campaign_produces_row_per_cell() -> None:
    wide = [
        {"run_id": "r1", "corpus": "repair", "model_id": "ds", "arm": "a", "N": 1,
         "grounding": "rag", "ordering_id": "o", "k_doc": 5, "metamodel_block": "v1",
         "semantic__f1": 0.6, "semantic__f1__status": "ok"},
        {"run_id": "r2", "corpus": "repair", "model_id": "ds", "arm": "a", "N": 1,
         "grounding": "rag", "ordering_id": "o", "k_doc": 5, "metamodel_block": "v1",
         "semantic__f1": 0.8, "semantic__f1__status": "ok"},
    ]
    result = summarise_campaign(wide, ["semantic__f1"])
    assert len(result) == 1
    assert result[0]["corpus"] == "repair"
    assert result[0]["semantic__f1__mean"] == pytest.approx(0.7)
    assert result[0]["semantic__f1__n_ok"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# variability report + degenerate share
# ─────────────────────────────────────────────────────────────────────────────

def test_degenerate_share_counts_only_ok_runs() -> None:
    rows = [
        {"structural__degenerate": True,  "structural__degenerate__status": "ok"},
        {"structural__degenerate": False, "structural__degenerate__status": "ok"},
        {"structural__degenerate": None,  "structural__degenerate__status": "ineligible"},
    ]
    d = degenerate_share(rows)
    assert d["n_ok"] == 2
    assert d["n_degenerate"] == 1
    assert d["degenerate_share"] == 0.5


def test_variability_report_produces_headline_degenerate_share() -> None:
    def base(run_id, degen):
        return dict(run_id=run_id, corpus="repair", model_id="ds", arm="a", N=1,
                    grounding="rag", ordering_id="o", k_doc=5, metamodel_block="v1")
    wide = []
    for i, (deg, feats) in enumerate([(True, 5), (True, 6), (False, 12)]):
        row = base(f"r{i}", deg)
        row["structural__degenerate"] = deg
        row["structural__degenerate__status"] = "ok"
        row["structural__n_features"] = feats
        row["structural__n_features__status"] = "ok"
        wide.append(row)
    report = variability_report(wide, shape_columns=("structural__n_features",),
                                 outcome_columns=())
    assert len(report) == 1
    assert report[0]["n_degenerate"] == 2
    assert report[0]["degenerate_share"] == pytest.approx(2 / 3)
    assert report[0]["structural__n_features__n_ok"] == 3
    assert report[0]["structural__n_features__sd"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# infer_metric_columns
# ─────────────────────────────────────────────────────────────────────────────

def test_infer_metric_columns_ignores_status_and_reason_suffixes() -> None:
    row = {
        "run_id": "r1",
        "structural__n_features": 4,
        "structural__n_features__status": "ok",
        "structural__n_features__reason": "",
        "semantic__f1": 0.7,
        "semantic__f1__status": "ok",
    }
    cols = infer_metric_columns([row])
    assert cols == ["semantic__f1", "structural__n_features"]
