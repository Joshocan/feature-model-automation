"""cross-run aggregation and variability.

Joins the long-form metric CSVs emitted by the structural, semantic
and provenance evaluators into a per-run wide table, then rolls
that table up per campaign cell (corpus × model × N × grounding × arm × …).

Every measurement flows through the ``{value, status, reason}`` envelope.
Aggregators drop non-``ok`` cells from mean/median/SD (so a
``not_applicable`` recall does not zero out a headline), but retain their
counts as ``n_ineligible`` / ``n_missing_artifact`` / … columns so the
denominator story survives.

Nothing here mutates the source CSVs; the aggregator is idempotent and
side-effect-free.
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Cell keys the campaign design cares about. Runs sharing every key form
# one comparable cell. The default excludes seed/repetition (those are
# nested observations within the cell) and run_id.
DEFAULT_CELL_KEYS: Tuple[str, ...] = (
    "corpus", "model_id", "arm", "N", "grounding",
    "metamodel_block", "ordering_id", "k_doc",
)


def _coerce(value: Any) -> Any:
    """CSVs come in as strings; recover numeric and JSON scalars."""
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    s = str(value)
    if s == "":
        return None
    if s.startswith("[") or s.startswith("{"):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return s
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        f = float(s)
        return int(f) if f.is_integer() and "." not in s and "e" not in s.lower() else f
    except ValueError:
        return s


def load_long_metrics(path: Path) -> List[Dict[str, Any]]:
    """Read a long-form metrics.csv from any P2/P3/P4 output directory.

    Rows must carry at minimum ``run_id``, ``metric`` and ``status``. The
    aggregator tolerates extra columns silently.
    """
    if not path.is_file():
        raise FileNotFoundError(f"metrics CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def join_by_run(
    sources: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    id_keys: Sequence[str] = ("run_id",),
    cell_keys: Sequence[str] = DEFAULT_CELL_KEYS,
) -> List[Dict[str, Any]]:
    """Wide-join long-form rows from several sources by run_id.

    ``sources`` maps a source-tag (``"structural"``, ``"semantic"``,
    ``"provenance"``) to its iterable of long rows. The output has one row
    per run and columns named ``{source}__{metric}`` for the value and
    ``{source}__{metric}__status`` for the envelope status. Run-level fields
    (corpus, model, arm, …) are lifted from the first row that carries them.

    Duplicate run_ids within a single source with conflicting metrics raise
    ValueError — that indicates a metrics.csv was generated twice.
    """
    wide: Dict[str, Dict[str, Any]] = {}
    for source, rows in sources.items():
        seen_pairs: set[tuple[str, str]] = set()
        for row in rows:
            rid = row.get("run_id")
            if not rid:
                continue
            metric = row.get("metric")
            if metric is None:
                continue
            key = (rid, metric)
            if key in seen_pairs:
                raise ValueError(
                    f"Duplicate (run_id={rid}, metric={metric}) in source {source!r}"
                )
            seen_pairs.add(key)
            bucket = wide.setdefault(rid, {"run_id": rid})
            # Lift group keys and inventory metadata from whichever row got there first.
            for k in ("corpus", "model_id", "arm", "N", "grounding",
                      "ordering_id", "k_doc", "metamodel_block", "seed",
                      "repetition", "inventory_status", "completed", *cell_keys):
                if k in row and row[k] is not None and bucket.get(k) is None:
                    bucket[k] = row[k]
            col = f"{source}__{metric}"
            bucket[col] = row.get("value")
            bucket[f"{col}__status"] = row.get("status")
            reason = row.get("reason")
            if reason:
                bucket[f"{col}__reason"] = reason
    return [wide[k] for k in sorted(wide)]


def add_evaluation_populations(wide_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Attach declared primary and formal-admissibility sensitivity flags.

    ``None`` means required upstream metrics were not provided. False is an
    observed failure of the gate; it is never a stand-in for missing data.
    """
    checks = ("xsd_valid", "W1", "W2", "W3", "W4", "W5",
              "tree_invariants", "identifier_syntax", "xml_envelope")
    out: List[Dict[str, Any]] = []
    for original in wide_rows:
        row = dict(original)
        completed = row.get("inventory_status") == "completed"
        n_gen = row.get("semantic__n_generated")
        n_status = row.get("semantic__n_generated__status")
        if n_status is None:
            row["primary_semantic_eligible"] = None
        else:
            row["primary_semantic_eligible"] = bool(completed and n_status == "ok" and
                isinstance(n_gen, (int, float)) and n_gen > 0)
        values = [row.get(f"structural__{key}") for key in checks]
        statuses = [row.get(f"structural__{key}__status") for key in checks]
        trace = row.get("provenance__feature_trace_coverage")
        trace_status = row.get("provenance__feature_trace_coverage__status")
        refs = row.get("provenance__L1_referential_integrity")
        refs_status = row.get("provenance__L1_referential_integrity__status")
        if (not completed or any(s == "ok" and v is False for s, v in zip(statuses, values))
                or (trace_status == "ok" and trace != 1)
                or (refs_status == "ok" and refs != 1)):
            row["strict_admissible"] = False
        elif (all(s == "ok" and v is True for s, v in zip(statuses, values))
              and trace_status == refs_status == "ok" and trace == refs == 1):
            row["strict_admissible"] = True
        else:
            row["strict_admissible"] = None
        out.append(row)
    return out


def _numeric_ok_values(rows: Iterable[Mapping[str, Any]], col: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        status = row.get(f"{col}__status")
        value = row.get(col)
        if status != "ok" or value is None:
            continue
        if isinstance(value, bool):
            out.append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            out.append(float(value))
    return out


def _status_counter(rows: Iterable[Mapping[str, Any]], col: str) -> Dict[str, int]:
    counts = Counter(row.get(f"{col}__status") for row in rows)
    return {k: v for k, v in counts.items() if k is not None}


def summarise_cell(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Dict[str, Any]:
    """Per-cell summary: mean / median / SD / n over the *ok* subset only.

    Non-ok cells (ineligible, not_applicable, missing_artifact, unsupported,
    evaluator_error) are retained as separate counts so the denominator is
    never silently reshaped. Boolean-valued metrics get a ``true_share``
    column instead of a mean (the two are numerically equal but named
    differently to keep readers honest).
    """
    n_total = len(rows)
    out: Dict[str, Any] = {"n_runs_in_cell": n_total}
    for col in columns:
        ok_values = _numeric_ok_values(rows, col)
        n_ok = len(ok_values)
        statuses = _status_counter(rows, col)
        out[f"{col}__n_ok"] = n_ok
        out[f"{col}__n_total"] = n_total
        for status, cnt in statuses.items():
            out[f"{col}__n_{status}"] = cnt
        if n_ok == 0:
            out[f"{col}__mean"] = None
            out[f"{col}__median"] = None
            out[f"{col}__sd"] = None
            continue
        out[f"{col}__mean"] = st.fmean(ok_values)
        out[f"{col}__median"] = st.median(ok_values)
        out[f"{col}__sd"] = st.pstdev(ok_values) if n_ok > 1 else 0.0
        out[f"{col}__min"] = min(ok_values)
        out[f"{col}__max"] = max(ok_values)
    return out


def group_by_cell(wide_rows: Iterable[Mapping[str, Any]], cell_keys: Sequence[str] = DEFAULT_CELL_KEYS
                   ) -> Dict[Tuple[Any, ...], List[Mapping[str, Any]]]:
    """Return ``{cell_tuple: [row, ...]}``. ``cell_tuple`` follows cell_keys order."""
    buckets: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in wide_rows:
        key = tuple(row.get(k) for k in cell_keys)
        buckets[key].append(row)
    return dict(buckets)


def summarise_campaign(
    wide_rows: Sequence[Mapping[str, Any]],
    metric_columns: Sequence[str],
    *,
    cell_keys: Sequence[str] = DEFAULT_CELL_KEYS,
) -> List[Dict[str, Any]]:
    """One summary row per cell. Order is deterministic (sorted by cell key)."""
    cells = group_by_cell(wide_rows, cell_keys)
    rows: List[Dict[str, Any]] = []
    for key in sorted(cells.keys(), key=lambda k: tuple("" if v is None else str(v) for v in k)):
        cell_rows = cells[key]
        summary = {name: value for name, value in zip(cell_keys, key)}
        summary.update(summarise_cell(cell_rows, metric_columns))
        rows.append(summary)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Variability report — degenerate share is the headline
# ─────────────────────────────────────────────────────────────────────────────

def degenerate_share(wide_rows: Sequence[Mapping[str, Any]],
                      *, degenerate_col: str = "structural__degenerate"
                      ) -> Dict[str, Any]:
    """Fraction of runs whose ``degenerate`` metric is True.

    Denominator counts only ``status="ok"`` rows for the flag — a run whose
    structural pipeline errored out is not counted as "not degenerate".
    """
    n_ok = 0
    n_deg = 0
    for row in wide_rows:
        if row.get(f"{degenerate_col}__status") != "ok":
            continue
        n_ok += 1
        if bool(row.get(degenerate_col)):
            n_deg += 1
    return dict(n_ok=n_ok, n_degenerate=n_deg,
                degenerate_share=(n_deg / n_ok) if n_ok else None)


def variability_report(
    wide_rows: Sequence[Mapping[str, Any]],
    *,
    cell_keys: Sequence[str] = DEFAULT_CELL_KEYS,
    shape_columns: Sequence[str] = (
        "structural__n_features", "structural__max_depth", "structural__avg_branching",
        "structural__mandatory_ratio",
    ),
    outcome_columns: Sequence[str] = (
        "semantic__semantic_f1_total", "semantic__recall_reach",
        "structural__structural_conformance", "structural__satisfiable",
    ),
    degenerate_col: str = "structural__degenerate",
) -> List[Dict[str, Any]]:
    """Reference-free variability per cell + headline degenerate share."""
    cells = group_by_cell(wide_rows, cell_keys)
    out: List[Dict[str, Any]] = []
    for key in sorted(cells.keys(), key=lambda k: tuple("" if v is None else str(v) for v in k)):
        cell_rows = cells[key]
        row = {name: value for name, value in zip(cell_keys, key)}
        row.update(degenerate_share(cell_rows, degenerate_col=degenerate_col))
        row["n_runs_in_cell"] = len(cell_rows)
        for col in shape_columns:
            values = _numeric_ok_values(cell_rows, col)
            row[f"{col}__sd"] = st.pstdev(values) if len(values) > 1 else (0.0 if values else None)
            row[f"{col}__n_ok"] = len(values)
        for col in outcome_columns:
            values = _numeric_ok_values(cell_rows, col)
            row[f"{col}__mean"] = st.fmean(values) if values else None
            row[f"{col}__sd"] = st.pstdev(values) if len(values) > 1 else (0.0 if values else None)
            row[f"{col}__n_ok"] = len(values)
        out.append(row)
    return out


def write_wide_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = list({k for row in rows for k in row})
    fields.sort()
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(row.get(k)) if isinstance(row.get(k), (dict, list))
                                  else row.get(k)) for k in fields})


def infer_metric_columns(wide_rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Return every ``{source}__{metric}`` column (excluding status/reason)."""
    seen: set[str] = set()
    for row in wide_rows:
        for key in row:
            if "__" not in key:
                continue
            if key.endswith("__status") or key.endswith("__reason"):
                continue
            if key.count("__") != 1:
                continue
            seen.add(key)
    return sorted(seen)
