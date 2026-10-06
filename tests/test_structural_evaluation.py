from pathlib import Path
import pytest
from fame.evaluation.structural import evaluate_structure
from fame.evaluation.quality_sat import analyze_sat_quality

XSD = Path(__file__).resolve().parents[1] / "prompts/feature-model-schema.xsd"


def fm(tmp_path, body='<and name="Root"><feature name="A"/></and>', rules=""):
    path = tmp_path / "fm.xml"
    path.write_text('<?xml version="1.0"?><featureModel><struct>' + body + '</struct><constraints>' + rules + '</constraints></featureModel>')
    return path


def evaluate(path):
    return evaluate_structure(path, XSD, expected_root="Root")


def test_valid_w5_translation_but_featureide_parser_not_claimed(tmp_path):
    r = evaluate(fm(tmp_path))
    assert r["xsd_valid"]["value"] is True
    assert r["W1"]["value"] is True
    assert r["W5"]["value"] is True
    assert r["featureide_parse"]["status"] == "unsupported"
    assert r["structural_conformance"]["value"] is True
    assert r["satisfiable"]["value"] is True
    assert r["degenerate"]["value"] is True


def test_malformed_and_missing(tmp_path):
    p = tmp_path / "missing.xml"
    assert evaluate(p)["parseable"]["status"] == "missing_artifact"
    p.write_text("<featureModel>")
    r = evaluate(p)
    assert r["parseable"]["value"] is False
    assert r["satisfiable"]["value"] is None


@pytest.mark.parametrize("body", ['<and name="Root"/>', '<and name="Root"><feature name="A"/><feature name="A"/></and>', '<and name="Wrong"><feature name="A"/></and>'])
def test_invalid_structure_never_conforming(tmp_path, body):
    assert evaluate(fm(tmp_path, body))["structural_conformance"]["value"] is False


def test_depth_ignores_descriptions_and_constraints(tmp_path):
    r = evaluate(fm(tmp_path, '<and name="Root"><description>Text</description><and name="B"><feature name="A"/></and></and>'))
    assert r["max_depth"]["value"] == 2
    assert r["n_features"]["value"] == 3
    assert r["avg_branching"]["value"] == 1


def test_unsat_not_zero_dead_features(tmp_path):
    r = evaluate(fm(tmp_path, rules='<rule><not><var>Root</var></not></rule>'))
    assert r["satisfiable"]["value"] is False
    assert r["dead_features"]["value"] is None
    assert r["dead_feature_ratio"]["status"] == "not_applicable"


def test_dead_feature_and_canonical_constraint_counts(tmp_path):
    r = evaluate(fm(tmp_path, rules='<rule><disj><not><var>Root</var></not><not><var>A</var></not></disj></rule><rule><imp><var>A</var><var>Root</var></imp></rule>'))
    assert r["n_excludes"]["value"] == r["n_requires"]["value"] == 1
    assert r["dead_features"]["value"] == ["A"]
    assert r["dead_feature_ratio"]["value"] == 1


@pytest.mark.parametrize("rule", ['<rule><unknown/></rule>', '<rule/>', '<rule><var>Ghost</var></rule>', '<rule><conj><var>A</var></conj></rule>'])
def test_sat_never_silently_discards_bad_rules(tmp_path, rule):
    p = fm(tmp_path, rules=rule)
    assert evaluate(p)["satisfiable"]["status"] == "ineligible"
    with pytest.raises(ValueError):
        analyze_sat_quality(p)


def test_auxiliary_names_do_not_collide_with_features(tmp_path):
    r = evaluate(fm(tmp_path, '<and name="Root"><feature name="not_3"/></and>', '<rule><not><var>not_3</var></not></rule>'))
    assert r["satisfiable"]["value"] is True
    assert r["dead_features"]["value"] == ["not_3"]


def test_missing_schema_is_evaluator_error(tmp_path):
    r = evaluate_structure(fm(tmp_path), tmp_path / "missing.xsd", expected_root="Root")
    assert r["xsd_valid"]["status"] == "evaluator_error"
    assert r["satisfiable"]["value"] is None


def test_boolean_one_and_root_only_denominators(tmp_path):
    r = evaluate(fm(tmp_path, '<and name="Root"><feature name="A" mandatory="1"/></and>'))
    assert r["mandatory_ratio"]["value"] == 1
    assert r["degenerate"]["value"] is False
    r = evaluate(fm(tmp_path, '<feature name="Root"/>'))
    assert r["mandatory_ratio"]["value"] is None
    assert r["max_depth"]["value"] == 0
