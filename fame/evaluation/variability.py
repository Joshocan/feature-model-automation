"""Variability measures for generated feature models.

The old design scored feature names and parent-match but ignored *variability
semantics*: a model that recovered every name and parent, marked everything
optional and made every group ``<and>``, would pass conformance, SAT and
parent-match while being a labelled tree rather than a feature model.

These measures close that hole:

**Reference-free** — both corpora participate.
* Group-kind distribution (proportion of ``<and>`` / ``<or>`` / ``<alt>``)
* Mandatory ratio (proportion of non-root features carrying ``mandatory="true"``)
* Degenerate flag: no ``<or>``, no ``<alt>``, no mandatory features anywhere

**Reference-based** — Repair primary; Federation once the ``<alt>`` fix from
the Federation variability correction is in the ground truth.
* Group-kind confusion matrix for matched internal features
* Mandatory agreement (accuracy + direction of error)

The **degenerate share per configuration** is the headline number here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET


GROUP_TAGS = ("and", "or", "alt")


@dataclass(frozen=True)
class GroupKindDistribution:
    """Reference-free variability profile of one generated FM."""
    n_internal:      int             # nodes with at least one child feature/group
    n_and:           int
    n_or:            int
    n_alt:           int
    n_leaves:        int             # named <feature> elements without children
    n_non_root_features: int         # every named node except the root
    n_mandatory:     int             # non-root features with mandatory="true"

    @property
    def and_prop(self) -> float: return _safe_prop(self.n_and, self.n_internal)
    @property
    def or_prop(self)  -> float: return _safe_prop(self.n_or,  self.n_internal)
    @property
    def alt_prop(self) -> float: return _safe_prop(self.n_alt, self.n_internal)
    @property
    def mandatory_ratio(self) -> float:
        return _safe_prop(self.n_mandatory, self.n_non_root_features)

    @property
    def is_degenerate(self) -> bool:
        """No <or>, no <alt>, and no mandatory features — a labelled tree."""
        return (self.n_or == 0 and self.n_alt == 0 and self.n_mandatory == 0)


@dataclass(frozen=True)
class GroupKindConfusion:
    """3×3 confusion matrix for matched internal features (rows: truth, cols: gen)."""
    # keys: (truth_kind, gen_kind) → count.
    counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def total(self) -> int:
        return sum(self.counts.values())

    def accuracy(self) -> float:
        if self.total() == 0:
            return 0.0
        diag = sum(self.counts.get((k, k), 0) for k in GROUP_TAGS)
        return diag / self.total()

    def as_matrix(self) -> List[List[int]]:
        """List of lists in GROUP_TAGS order, for pretty printing."""
        return [[self.counts.get((t, g), 0) for g in GROUP_TAGS] for t in GROUP_TAGS]


@dataclass(frozen=True)
class MandatoryAgreement:
    n_matched_non_root: int
    n_agree:            int
    n_gen_optional_gt_mandatory: int      # extractor omitted mandatory
    n_gen_mandatory_gt_optional: int      # extractor over-marked mandatory

    @property
    def accuracy(self) -> float:
        return _safe_prop(self.n_agree, self.n_matched_non_root)


# ─────────────────────────────────────────────────────────────────────────────
# XML parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse(xml: str | Path | ET.ElementTree) -> ET.Element:
    """Return the root <featureModel> element, tolerating ``<struct>`` wrapping."""
    if isinstance(xml, ET.ElementTree):
        root = xml.getroot()
    elif isinstance(xml, (str, Path)) and (isinstance(xml, Path) or "\n" not in str(xml)):
        p = Path(xml)
        if p.exists():
            root = ET.parse(str(p)).getroot()
        else:
            root = ET.fromstring(str(xml))
    else:
        root = ET.fromstring(xml)
    return root


def _struct_root(root: ET.Element) -> ET.Element:
    """Return the outer ``<struct>`` node or root itself if there is none."""
    if root.tag == "featureModel":
        s = root.find("struct")
        return s if s is not None else root
    return root


def _is_named_group(elem: ET.Element) -> bool:
    return elem.tag in GROUP_TAGS and (elem.get("name") is not None)


def _is_named_feature(elem: ET.Element) -> bool:
    return elem.tag == "feature" and (elem.get("name") is not None)


def _has_children_features(elem: ET.Element) -> bool:
    return any(child.tag in GROUP_TAGS + ("feature",) for child in elem)


# ─────────────────────────────────────────────────────────────────────────────
# Reference-free measures
# ─────────────────────────────────────────────────────────────────────────────

def group_kind_distribution(xml: str | Path) -> GroupKindDistribution:
    """Count internal groups + leaves + mandatory features in one FM."""
    root = _struct_root(_parse(xml))

    # First named node in a depth-first walk is the "root" feature (skip its
    # mandatory count so we don't inflate the denominator).
    n_internal = n_and = n_or = n_alt = 0
    n_leaves = n_non_root = n_mandatory = 0
    seen_root = False

    for elem in root.iter():
        if _is_named_group(elem):
            n_internal += 1
            if elem.tag == "and": n_and += 1
            elif elem.tag == "or": n_or += 1
            elif elem.tag == "alt": n_alt += 1
            if not seen_root:
                seen_root = True
                continue
            n_non_root += 1
            if (elem.get("mandatory") or "").lower() == "true":
                n_mandatory += 1
        elif _is_named_feature(elem):
            if not _has_children_features(elem):
                n_leaves += 1
            if not seen_root:
                seen_root = True
                continue
            n_non_root += 1
            if (elem.get("mandatory") or "").lower() == "true":
                n_mandatory += 1

    return GroupKindDistribution(
        n_internal=n_internal, n_and=n_and, n_or=n_or, n_alt=n_alt,
        n_leaves=n_leaves,
        n_non_root_features=n_non_root,
        n_mandatory=n_mandatory,
    )


def is_degenerate(xml: str | Path) -> bool:
    """Convenience wrapper around :meth:`GroupKindDistribution.is_degenerate`."""
    return group_kind_distribution(xml).is_degenerate


# ─────────────────────────────────────────────────────────────────────────────
# Reference-based measures
# ─────────────────────────────────────────────────────────────────────────────

def _index_by_name(root: ET.Element) -> Dict[str, ET.Element]:
    """Map name → element for every named node under ``root``."""
    out: Dict[str, ET.Element] = {}
    for elem in root.iter():
        n = elem.get("name")
        if n:
            out[n] = elem
    return out


def _group_kind_of(elem: ET.Element) -> Optional[str]:
    """Return 'and'/'or'/'alt' if ``elem`` is an internal group, else None."""
    return elem.tag if elem.tag in GROUP_TAGS else None


def group_kind_confusion(gt_xml: str | Path,
                         gen_xml: str | Path,
                         *,
                         matched_features: Optional[List[str]] = None) -> GroupKindConfusion:
    """3×3 confusion matrix on matched internal features.

    Every feature in ``matched_features`` present as an internal group in BOTH
    trees contributes one cell. If ``matched_features`` is None, features
    with the same name in both trees are used.
    """
    gt_root  = _struct_root(_parse(gt_xml))
    gen_root = _struct_root(_parse(gen_xml))
    gt_index  = _index_by_name(gt_root)
    gen_index = _index_by_name(gen_root)

    if matched_features is None:
        matched_features = sorted(set(gt_index) & set(gen_index))

    counts: Dict[Tuple[str, str], int] = {}
    for name in matched_features:
        gt_elem  = gt_index.get(name)
        gen_elem = gen_index.get(name)
        if gt_elem is None or gen_elem is None:
            continue
        gt_kind  = _group_kind_of(gt_elem)
        gen_kind = _group_kind_of(gen_elem)
        if gt_kind is None or gen_kind is None:
            continue
        key = (gt_kind, gen_kind)
        counts[key] = counts.get(key, 0) + 1
    return GroupKindConfusion(counts=counts)


def mandatory_agreement(gt_xml: str | Path,
                        gen_xml: str | Path,
                        *,
                        matched_features: Optional[List[str]] = None) -> MandatoryAgreement:
    """Accuracy + direction of error for mandatory on matched non-root features."""
    gt_root  = _struct_root(_parse(gt_xml))
    gen_root = _struct_root(_parse(gen_xml))
    gt_index  = _index_by_name(gt_root)
    gen_index = _index_by_name(gen_root)

    if matched_features is None:
        matched_features = sorted(set(gt_index) & set(gen_index))

    # Detect the root name in the GT tree so we can skip it.
    gt_root_name = _first_named_name(gt_root)

    n_matched = n_agree = n_missing = n_over = 0
    for name in matched_features:
        if name == gt_root_name:
            continue
        if name not in gt_index or name not in gen_index:
            continue
        gt_elem  = gt_index[name]
        gen_elem = gen_index[name]
        gt_m  = (gt_elem.get("mandatory")  or "").lower() == "true"
        gen_m = (gen_elem.get("mandatory") or "").lower() == "true"
        n_matched += 1
        if gt_m == gen_m:
            n_agree += 1
        elif gt_m and not gen_m:
            n_missing += 1                # missed a mandatory
        else:
            n_over += 1                   # invented a mandatory
    return MandatoryAgreement(
        n_matched_non_root=n_matched,
        n_agree=n_agree,
        n_gen_optional_gt_mandatory=n_missing,
        n_gen_mandatory_gt_optional=n_over,
    )


def _first_named_name(root: ET.Element) -> Optional[str]:
    for elem in root.iter():
        n = elem.get("name")
        if n:
            return n
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Small helper
# ─────────────────────────────────────────────────────────────────────────────

def _safe_prop(num: int, denom: int) -> float:
    return (num / denom) if denom > 0 else 0.0
