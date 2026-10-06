"""Reach subset builder + dual recall (RQ2 calibration).

Reach measures coverage of the annotated reference by the corpus. It is not
an unconditional upper bound on semantic matching or model knowledge.

Definitions (from IFS brief §1):

* **F_t**            — every named feature in the ground-truth model.
* **F_t^attested**   — the subset with per-study citations in D4.
* **F_t^organising** — the framing features the survey authors added,
                       with no per-study citations. F_t = F_t^att ∪ F_t^org.
* **reach(T, C)**    — directly attributed features available in C, together
                       with all their ancestors in the reference hierarchy.

Ancestors can be organising features. Full-corpus reach need not equal the
attested partition; closure does not include unsupported sibling branches.

**Dual recall** — every RQ2 measurement reports recall against three targets:

* against ``reach``    → primary; what is realistically achievable
* against ``F_t``      → labelled bound; documents the natural ceiling
* against ``F_t^att`` and ``F_t^org`` separately → distinguishes matches to
  attributed and unattributed reference features without assuming that an
  unattributed feature lacks all possible evidential support.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeaturePartition:
    """Loaded feature partition (D21) for one corpus.

    ``attested``   — features with at least one attribution entry in D4.
    ``organising`` — features present in D3 but with no attribution.
    """
    attested:   FrozenSet[str]
    organising: FrozenSet[str]

    @property
    def full(self) -> FrozenSet[str]:
        return self.attested | self.organising


@dataclass(frozen=True)
class DualRecall:
    """The four recall numbers demanded by the brief."""
    corpus:               str
    n_extracted_matched:  int      # count of GT features the run recovered
    recall_vs_reach:      float    # primary
    recall_vs_F_t:        float    # bound
    recall_vs_F_t_att:    float
    recall_vs_F_t_org:    float
    reach_size:           int
    F_t_size:             int
    F_t_att_size:         int
    F_t_org_size:         int


# ─────────────────────────────────────────────────────────────────────────────
# Loaders (D4 attribution + D21 partition)
# ─────────────────────────────────────────────────────────────────────────────

def load_attribution(csv_path: Path | str) -> Dict[str, Set[str]]:
    """Read a D4 attribution CSV → ``{gt_feature_id: {doc_id, ...}}``.

    Schema tolerated: any file whose first row lists at least ``gt_feature_id``
    and ``doc_id`` (extra columns like ``source`` are preserved but ignored).
    """
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"attribution CSV not found: {p}")
    out: Dict[str, Set[str]] = {}
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            f = row.get("gt_feature_id") or row.get("feature_id") or row.get("feature_name")
            d = row.get("doc_id")
            if not f or not d:
                continue
            out.setdefault(f, set()).add(d)
    return out


def load_partition(csv_path: Path | str) -> FeaturePartition:
    """Read a D21 feature-partition CSV → :class:`FeaturePartition`.

    Schema tolerated: any file whose first row lists ``gt_feature_id`` and
    ``partition`` (values ``attested`` or ``organising``).
    """
    p = Path(csv_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"partition CSV not found: {p}")
    att: Set[str] = set()
    org: Set[str] = set()
    with open(p, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = row.get("gt_feature_id") or row.get("feature_id") or row.get("feature_name")
            kind = (row.get("partition") or "").strip().lower()
            if not f or not kind:
                continue
            if kind == "attested":
                att.add(f)
            elif kind == "organising":
                org.add(f)
            else:
                raise ValueError(f"unknown partition value {kind!r} for feature {f!r} in {p}")
    return FeaturePartition(attested=frozenset(att), organising=frozenset(org))


# ─────────────────────────────────────────────────────────────────────────────
# Reach subset
# ─────────────────────────────────────────────────────────────────────────────

def reach_features(
    *,
    attribution:      Dict[str, Set[str]],
    corpus_doc_ids:   Iterable[str],
    parents:         Dict[str, Optional[str]],
    only_attested:    Optional[Iterable[str]] = None,
) -> FrozenSet[str]:
    """Upward closure of directly supported seeds in the reference tree.

    only_attested filters seeds, never their ancestors. Parents must include
    every reference node, with None for the root. Invalid links/cycles fail.
    """
    for node in parents:
        seen = set()
        current = node
        while current is not None:
            if current not in parents:
                raise ValueError(f"Missing reference parent: {current}")
            if current in seen:
                raise ValueError(f"Cycle in reference parents at {current}")
            seen.add(current)
            current = parents[current]
    corpus_set = set(corpus_doc_ids)
    if not corpus_set:
        return frozenset()
    reached: Set[str] = set()
    for feature, docs in attribution.items():
        if docs & corpus_set:
            reached.add(feature)
    if only_attested is not None:
        reached &= set(only_attested)
    for feature in tuple(reached):
        if feature not in parents:
            raise ValueError(f"Attributed feature absent from reference: {feature}")
        parent = parents[feature]
        while parent is not None:
            reached.add(parent)
            parent = parents[parent]
    return frozenset(reached)


def rho(reach_size: int, F_t_size: int) -> float:
    """ρ(T, C) = |reach(T,C)| / |F_t|. The reachable-vs-full ratio."""
    if F_t_size <= 0:
        raise ValueError("F_t_size must be positive to compute rho")
    return reach_size / F_t_size


def validate_reach_inputs(partition, attribution, corpus_doc_ids, reference_names):
    """Reject inconsistent full-corpus calibration inputs before scoring."""
    names = list(reference_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Reference must have nonempty, unique feature identifiers")
    if partition.attested & partition.organising:
        raise ValueError("Attested and organising partitions overlap")
    if partition.full != set(names):
        raise ValueError("Partition does not cover exactly the reference feature identifiers")
    attributed = {f for f, docs in attribution.items() if docs}
    if attributed != partition.attested:
        raise ValueError("Attributed features do not equal the attested partition")
    docs = set(corpus_doc_ids)
    if not docs:
        raise ValueError("Corpus manifest has no document IDs")
    unknown = set().union(*attribution.values()) - docs if attribution else set()
    if unknown:
        raise ValueError(f"Attribution contains unknown corpus document IDs: {sorted(unknown)}")


# ─────────────────────────────────────────────────────────────────────────────
# Dual recall reporting
# ─────────────────────────────────────────────────────────────────────────────

def dual_recall(
    *,
    corpus:            str,
    matched_features:  Iterable[str],   # GT feature ids the run matched
    partition:         FeaturePartition,
    reach:             Iterable[str],
) -> DualRecall:
    """Compute the four recall numbers.

    ``matched_features`` is the set of GT feature ids for which the run
    produced at least one accepted match (semantic threshold + parent policy
    live upstream in ``fame.evaluation.semantic``).
    """
    matched = set(matched_features)
    F_t     = set(partition.full)
    F_t_att = set(partition.attested)
    F_t_org = set(partition.organising)
    reach_s = set(reach)

    def _safe_recall(hits: int, total: int) -> float:
        return hits / total if total > 0 else 0.0

    return DualRecall(
        corpus=corpus,
        n_extracted_matched=len(matched & F_t),
        recall_vs_reach=_safe_recall(len(matched & reach_s), len(reach_s)),
        recall_vs_F_t=_safe_recall(len(matched & F_t), len(F_t)),
        recall_vs_F_t_att=_safe_recall(len(matched & F_t_att), len(F_t_att)),
        recall_vs_F_t_org=_safe_recall(len(matched & F_t_org), len(F_t_org)),
        reach_size=len(reach_s),
        F_t_size=len(F_t),
        F_t_att_size=len(F_t_att),
        F_t_org_size=len(F_t_org),
    )
