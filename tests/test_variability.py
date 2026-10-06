"""Tests for fame/evaluation/variability.py."""
from __future__ import annotations

import pytest

from fame.evaluation.variability import (
    GROUP_TAGS,
    group_kind_confusion,
    group_kind_distribution,
    is_degenerate,
    mandatory_agreement,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture XMLs
# ─────────────────────────────────────────────────────────────────────────────

DEGENERATE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<featureModel><struct>
  <and name="Root" abstract="true">
    <and name="Group_A" abstract="true">
      <feature name="A1"/>
      <feature name="A2"/>
    </and>
    <and name="Group_B" abstract="true">
      <feature name="B1"/>
    </and>
  </and>
</struct><constraints/></featureModel>
"""

RICH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<featureModel><struct>
  <and name="Root" abstract="true" mandatory="true">
    <alt name="Choice" abstract="true" mandatory="true">
      <feature name="OptionA"/>
      <feature name="OptionB"/>
    </alt>
    <or name="Many" abstract="true">
      <feature name="X" mandatory="true"/>
      <feature name="Y"/>
    </or>
    <feature name="Solo" mandatory="true"/>
  </and>
</struct><constraints/></featureModel>
"""

GT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<featureModel><struct>
  <and name="Root" abstract="true">
    <alt name="Choice" abstract="true">
      <feature name="A"/>
      <feature name="B"/>
    </alt>
    <or name="Many" abstract="true">
      <feature name="X" mandatory="true"/>
      <feature name="Y"/>
    </or>
  </and>
</struct><constraints/></featureModel>
"""

# gen model: everything flattened to <and>, no mandatory, no <or>/<alt>
GEN_ALL_AND = """<?xml version="1.0" encoding="UTF-8"?>
<featureModel><struct>
  <and name="Root" abstract="true">
    <and name="Choice" abstract="true">
      <feature name="A"/>
      <feature name="B"/>
    </and>
    <and name="Many" abstract="true">
      <feature name="X"/>
      <feature name="Y"/>
    </and>
  </and>
</struct><constraints/></featureModel>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Distribution + degeneracy
# ─────────────────────────────────────────────────────────────────────────────

def test_degenerate_model_flagged() -> None:
    d = group_kind_distribution(DEGENERATE_XML)
    assert d.n_or == 0
    assert d.n_alt == 0
    assert d.n_mandatory == 0
    assert d.is_degenerate is True
    assert is_degenerate(DEGENERATE_XML) is True


def test_rich_model_not_degenerate() -> None:
    d = group_kind_distribution(RICH_XML)
    assert d.n_or == 1
    assert d.n_alt == 1
    assert d.n_mandatory > 0
    assert d.is_degenerate is False


def test_group_proportions_sum_to_one_over_internal() -> None:
    d = group_kind_distribution(RICH_XML)
    total = d.and_prop + d.or_prop + d.alt_prop
    assert total == pytest.approx(1.0, abs=1e-9)


def test_mandatory_ratio_excludes_root() -> None:
    """Root's mandatory attribute must not inflate the mandatory ratio."""
    d = group_kind_distribution(RICH_XML)
    # Root is <and> named "Root", mandatory="true" but skipped from denominator.
    # Non-root features: Choice, OptionA, OptionB, Many, X, Y, Solo → 7
    # Non-root mandatories: Choice, X, Solo → 3
    assert d.n_non_root_features == 7
    assert d.n_mandatory == 3
    assert d.mandatory_ratio == pytest.approx(3 / 7)


def test_leaf_count() -> None:
    d = group_kind_distribution(RICH_XML)
    # Leaves (features with no children): OptionA, OptionB, X, Y, Solo → 5
    assert d.n_leaves == 5


# ─────────────────────────────────────────────────────────────────────────────
# Reference-based measures
# ─────────────────────────────────────────────────────────────────────────────

def test_group_kind_confusion_all_and_collapse() -> None:
    """gen flattens alt→and and or→and — the classic failure mode."""
    conf = group_kind_confusion(GT_XML, GEN_ALL_AND)
    # Matched internal names in both trees: Root, Choice, Many
    # Root: gt=and, gen=and  → (and, and)
    # Choice: gt=alt, gen=and → (alt, and)
    # Many: gt=or, gen=and    → (or, and)
    assert conf.counts[("and", "and")] == 1
    assert conf.counts[("alt", "and")] == 1
    assert conf.counts[("or", "and")] == 1
    assert conf.total() == 3
    assert conf.accuracy() == pytest.approx(1 / 3)


def test_group_kind_confusion_perfect_match() -> None:
    conf = group_kind_confusion(GT_XML, GT_XML)
    assert conf.accuracy() == 1.0


def test_group_kind_confusion_as_matrix_shape() -> None:
    conf = group_kind_confusion(GT_XML, GEN_ALL_AND)
    m = conf.as_matrix()
    assert len(m) == 3
    assert all(len(row) == 3 for row in m)


def test_mandatory_agreement_gen_loses_mandatory() -> None:
    """gen has no mandatory anywhere; gt has one (X)."""
    agr = mandatory_agreement(GT_XML, GEN_ALL_AND)
    # Matched non-root: Choice, A, B, Many, X, Y — 6
    assert agr.n_matched_non_root == 6
    # Only X differs (gt mandatory, gen optional)
    assert agr.n_gen_optional_gt_mandatory == 1
    assert agr.n_gen_mandatory_gt_optional == 0
    assert agr.n_agree == 5
    assert agr.accuracy == pytest.approx(5 / 6)


def test_mandatory_agreement_perfect() -> None:
    agr = mandatory_agreement(GT_XML, GT_XML)
    assert agr.accuracy == 1.0
    assert agr.n_gen_optional_gt_mandatory == 0
    assert agr.n_gen_mandatory_gt_optional == 0


def test_group_kind_confusion_skips_missing_names() -> None:
    """Features in only one tree are not in the confusion matrix."""
    gen_extra = """<?xml version="1.0"?>
    <featureModel><struct>
      <and name="Root" abstract="true">
        <alt name="Choice" abstract="true"><feature name="A"/></alt>
        <and name="NewNode" abstract="true"><feature name="Z"/></and>
      </and>
    </struct><constraints/></featureModel>"""
    conf = group_kind_confusion(GT_XML, gen_extra)
    # Common internal groups: Root (and/and), Choice (alt/alt). Many is in gt only.
    assert conf.total() == 2
    assert conf.counts[("and", "and")] == 1
    assert conf.counts[("alt", "alt")] == 1
