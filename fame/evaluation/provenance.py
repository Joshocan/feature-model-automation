"""provenance evaluation with explicit metric envelopes.

Reads a completed run's ``fm_gen.xml`` and ``fm_iter/step_*.xml`` checkpoints
plus its ``context_log.jsonl`` and returns:

* **L0 marker emission** — how many feature descriptions carry a parseable
  ``Trace: [id1, id2]`` marker per the frozen grammar
  (:mod:`fame.utils.marker_grammar`).
* **L1 referential integrity** — how many cited doc_ids are members of the
  D1 known-doc set.
* **L2 hallucination** — was each cited doc_id in the run's context at the
  step where the (feature, doc_id) pair was first seen? Vacuous for
  Non-RAG at N=1 (every doc is in every context) → reported as
  ``status="not_applicable"``.
* **Recency bias** — distribution of (citation_step − doc_first_seen_step)
  offsets; ``is_recency_biased`` flags disproportionate zero-offset citing.
* **Trace coverage** — feature-level and citation-pair counts.

Every metric returns the envelope form
``{value, status, reason, numerator, denominator}`` used elsewhere in the
structural and semantic evaluators.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from lxml import etree

from fame.evaluation.hallucination import check_hallucinations
from fame.evaluation.recency import recency_distribution
from fame.utils.marker_grammar import parse_marker

FEATURES = {"and", "or", "alt", "feature"}


PROVENANCE_METRICS = (
    "n_features", "n_features_with_description", "n_features_with_trace",
    "n_features_with_parseable_trace",
    "feature_trace_coverage",
    "L0_marker_emission_compliance",
    "L0_parse_rate",
    "n_citation_pairs", "n_unique_cited_doc_ids",
    "L1_referential_integrity",
    "L2_hallucination_rate", "L2_n_checked", "L2_applicable",
    "recency_mean_offset", "recency_zero_offset_share",
    "recency_max_offset", "recency_negative_offset_count",
    "is_recency_biased",
)


def _envelope(value: Any = None, status: str = "ok", reason: str = "", **counts) -> dict:
    return dict(value=value, status=status, reason=reason, **counts)


def _ineligible(reason: str) -> Dict[str, dict]:
    return {k: _envelope(status="ineligible", reason=reason) for k in PROVENANCE_METRICS}


def _missing(reason: str) -> Dict[str, dict]:
    return {k: _envelope(status="missing_artifact", reason=reason) for k in PROVENANCE_METRICS}


# ─────────────────────────────────────────────────────────────────────────────
# Feature / description extraction from a FeatureIDE XML
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_descriptions(fm_path: Path) -> List[Dict[str, Optional[str]]]:
    """Return ``[{'name', 'description'}, ...]`` in document order.

    ``description`` is ``None`` when the feature has no ``<description>`` child
    or the child has empty text. Root and abstract features are included so
    L0 coverage reflects the whole tree.
    """
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    tree = etree.parse(str(fm_path), parser)
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        return []
    out: List[Dict[str, Optional[str]]] = []
    for node in struct.iter():
        if node.tag not in FEATURES:
            continue
        name = node.get("name")
        if not name:
            continue
        desc_el = node.find("description")
        desc_text = None
        if desc_el is not None and desc_el.text and desc_el.text.strip():
            desc_text = desc_el.text.strip()
        out.append({"name": name, "description": desc_text})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# first_seen_step derivation
# ─────────────────────────────────────────────────────────────────────────────

def _iter_step_files(iter_dir: Path) -> List[tuple[int, Path]]:
    """``[(step_index, path), ...]`` for every ``step_*.xml`` in order."""
    if not iter_dir.is_dir():
        return []
    entries: List[tuple[int, Path]] = []
    for p in sorted(iter_dir.glob("step_*.xml")):
        try:
            idx = int(p.stem.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        entries.append((idx, p))
    return entries


def compute_first_seen_pairs(iter_dir: Path) -> List[Dict[str, Any]]:
    """Compute the earliest step at which each ``(feature, doc_id)`` pair
    appears in an iterate model.

    Returns rows of the form
    ``{"feature_id": name, "cited_doc_id": doc_id, "first_seen_step": step}``,
    suitable input to :mod:`fame.evaluation.hallucination` and
    :mod:`fame.evaluation.recency`.

    Rows without a parseable marker are skipped. Multiple ``(f, d)`` pairs
    from the same feature at the same step produce multiple rows only if the
    doc_ids differ; duplicate ``(f, d)`` pairs at different steps are recorded
    once at the earliest step.
    """
    seen: Dict[tuple[str, str], int] = {}
    for step_idx, path in _iter_step_files(iter_dir):
        try:
            for feat in extract_feature_descriptions(path):
                desc = feat.get("description")
                if not desc:
                    continue
                parsed = parse_marker(desc)
                if not parsed.marker_found:
                    continue
                for doc_id in parsed.doc_ids:
                    key = (feat["name"], doc_id)
                    if key not in seen:
                        seen[key] = step_idx
        except (etree.XMLSyntaxError, OSError):
            continue
    return [{"feature_id": f, "cited_doc_id": d, "first_seen_step": s}
            for (f, d), s in sorted(seen.items())]


def load_context_log(context_log: Path) -> List[dict]:
    if not context_log.is_file():
        return []
    out: List[dict] = []
    with context_log.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_provenance(
    fm_gen_path: Path,
    iter_dir: Path,
    context_log_path: Path,
    *,
    known_doc_ids: Iterable[str] = (),
    n_planned_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute provenance envelopes for a single completed run.

    Returns ``{"metrics": {name: envelope}, "citations": [row, ...],
    "hallucinated": [record, ...], "offsets": [int, ...]}``. The ``citations``
    payload is the D16 row list (one per parseable citation pair); downstream
    analysis re-aggregates from that file.
    """
    if not fm_gen_path.is_file():
        return dict(metrics=_missing(f"generated FM absent: {fm_gen_path}"),
                    citations=[], hallucinated=[], offsets=[])

    try:
        features = extract_feature_descriptions(fm_gen_path)
    except (etree.XMLSyntaxError, OSError) as exc:
        metrics = {k: _envelope(status="evaluator_error",
                                reason=f"{type(exc).__name__}: {exc}")
                   for k in PROVENANCE_METRICS}
        return dict(metrics=metrics, citations=[], hallucinated=[], offsets=[])

    n_features = len(features)
    if n_features == 0:
        metrics = _ineligible("No feature nodes in generated FM")
        metrics["n_features"] = _envelope(0)
        return dict(metrics=metrics, citations=[], hallucinated=[], offsets=[])

    known = set(known_doc_ids)
    n_with_desc = 0
    n_with_marker = 0
    n_with_parseable = 0
    n_pairs = 0
    unique_docs: set[str] = set()
    n_l1_ok_pairs = 0

    for feat in features:
        desc = feat.get("description")
        if not desc:
            continue
        n_with_desc += 1
        parsed = parse_marker(desc)
        if not parsed.marker_found:
            continue
        n_with_marker += 1
        if not parsed.doc_ids:
            continue
        n_with_parseable += 1
        for doc in parsed.doc_ids:
            n_pairs += 1
            unique_docs.add(doc)
            if not known or doc in known:
                n_l1_ok_pairs += 1

    metrics: Dict[str, dict] = {
        "n_features": _envelope(n_features),
        "n_features_with_description": _envelope(n_with_desc),
        "n_features_with_trace": _envelope(n_with_marker),
        "n_features_with_parseable_trace": _envelope(n_with_parseable),
        "feature_trace_coverage": _envelope(
            n_with_parseable / n_features,
            numerator=n_with_parseable, denominator=n_features,
        ),
        "L0_marker_emission_compliance": _envelope(
            n_with_marker / n_features,
            numerator=n_with_marker, denominator=n_features,
        ),
        "L0_parse_rate": (
            _envelope(n_with_parseable / n_with_marker,
                      numerator=n_with_parseable, denominator=n_with_marker)
            if n_with_marker > 0
            else _envelope(status="not_applicable",
                           reason="No markers emitted; parse rate undefined")
        ),
        "n_citation_pairs": _envelope(n_pairs),
        "n_unique_cited_doc_ids": _envelope(len(unique_docs)),
        "L1_referential_integrity": (
            _envelope(n_l1_ok_pairs / n_pairs,
                      numerator=n_l1_ok_pairs, denominator=n_pairs)
            if n_pairs > 0
            else _envelope(status="not_applicable",
                           reason="No citation pairs to check")
        ),
    }

    if not known:
        metrics["L1_referential_integrity"] = _envelope(
            status="unsupported",
            reason="No known_doc_ids provided; cannot compute L1",
        )

    # ─── L2 hallucination + recency require D8 iterates and D9 context log ──
    citations = compute_first_seen_pairs(iter_dir) if iter_dir.is_dir() else []
    context_lines = load_context_log(context_log_path) if context_log_path.is_file() else []

    if n_pairs == 0:
        metrics.update({
            "L2_hallucination_rate": _envelope(status="not_applicable",
                                                reason="No citation pairs to check"),
            "L2_n_checked": _envelope(0),
            "L2_applicable": _envelope(False, reason="No citations"),
            "recency_mean_offset": _envelope(status="not_applicable",
                                              reason="No citation pairs to check"),
            "recency_zero_offset_share": _envelope(status="not_applicable", reason=""),
            "recency_max_offset": _envelope(status="not_applicable", reason=""),
            "recency_negative_offset_count": _envelope(0),
            "is_recency_biased": _envelope(False, reason="No citations"),
        })
        return dict(metrics=metrics, citations=citations, hallucinated=[], offsets=[])

    if not context_lines:
        metrics.update({
            "L2_hallucination_rate": _envelope(status="missing_artifact",
                                                reason=f"context_log absent or empty: {context_log_path}"),
            "L2_n_checked": _envelope(0),
            "L2_applicable": _envelope(False, reason="No context log"),
            "recency_mean_offset": _envelope(status="missing_artifact", reason=""),
            "recency_zero_offset_share": _envelope(status="missing_artifact", reason=""),
            "recency_max_offset": _envelope(status="missing_artifact", reason=""),
            "recency_negative_offset_count": _envelope(status="missing_artifact", reason=""),
            "is_recency_biased": _envelope(status="missing_artifact", reason=""),
        })
        return dict(metrics=metrics, citations=citations, hallucinated=[], offsets=[])

    # L2 — was the cited doc in context at first_seen_step?
    hallu = check_hallucinations(provenance_rows=citations, context_lines=context_lines)

    # L2 vacuous when every step's context is the whole corpus (Non-RAG N=1).
    unique_context_sets = {frozenset(line.get("batch_doc_ids", []) or [])
                            | frozenset(line.get("chunk_doc_ids", []) or [])
                            for line in context_lines}
    n_steps = len({int(line.get("step_index", -1)) for line in context_lines})
    every_context_is_full = (
        n_planned_steps == 1
        or (n_steps == 1 and len(unique_context_sets) == 1
            and len(next(iter(unique_context_sets), frozenset())) >= len(unique_docs) > 0)
    )

    if every_context_is_full and hallu.n_hallucinated == 0:
        metrics["L2_hallucination_rate"] = _envelope(
            status="not_applicable",
            reason="Non-RAG N=1: every context contains every corpus doc; L2 is vacuous")
        metrics["L2_applicable"] = _envelope(False, reason="N=1 vacuous")
    else:
        metrics["L2_hallucination_rate"] = _envelope(
            hallu.hallucination_rate,
            numerator=hallu.n_hallucinated, denominator=hallu.n_checked)
        metrics["L2_applicable"] = _envelope(True)
    metrics["L2_n_checked"] = _envelope(hallu.n_checked)

    # Recency
    rec = recency_distribution(provenance_rows=citations, context_lines=context_lines)
    if rec.n_citations == 0:
        metrics["recency_mean_offset"] = _envelope(status="not_applicable",
            reason="No citations resolved to a batch step; recency undefined")
        metrics["recency_zero_offset_share"] = _envelope(status="not_applicable", reason="")
        metrics["recency_max_offset"] = _envelope(status="not_applicable", reason="")
        metrics["recency_negative_offset_count"] = _envelope(0)
        metrics["is_recency_biased"] = _envelope(False, reason="No offsets")
    else:
        metrics["recency_mean_offset"] = _envelope(rec.mean_offset)
        metrics["recency_zero_offset_share"] = _envelope(
            rec.zero_offset_share,
            numerator=sum(1 for o in rec.offsets if o == 0),
            denominator=rec.n_citations)
        metrics["recency_max_offset"] = _envelope(rec.max_offset)
        metrics["recency_negative_offset_count"] = _envelope(rec.negative_offset_count)
        metrics["is_recency_biased"] = _envelope(rec.is_recency_biased)

    return dict(metrics=metrics, citations=citations,
                hallucinated=[h.__dict__ for h in hallu.hallucinated],
                offsets=rec.offsets)
