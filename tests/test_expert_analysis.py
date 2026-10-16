"""Synthetic, offline checks: no private forms or campaign outputs required."""
import csv

import pytest

from scripts import expert_analysis as e


@pytest.mark.parametrize("source,expected", [
    ("OK", "Yes"), ("Not exactly", "Partially"),
    ("not sure, depends on the context", "Cannot judge"), ("No", "No"),
])
def test_authorised_citation_mapping(source, expected):
    assert e.normalize_citation(source, "Expert B") == expected


def test_mapping_does_not_guess_other_labels():
    with pytest.raises(ValueError, match="unsupported"):
        e.normalize_citation("probably", "Expert B")
    with pytest.raises(ValueError, match="unsupported"):
        e.normalize_citation("OK", "Expert A")


def test_missing_and_negative_decision_are_different():
    assert e.rating("NJ") is None
    assert e.CITATIONS["Cannot judge"] is None
    assert e.CITATIONS["No"] == e.DECISIONS["Reject"] == 0
    for value in (None, "", 0, 6, 2.5):
        with pytest.raises(ValueError):
            e.rating(value)


def test_ordinal_alpha_has_full_domain_and_pairwise_missing():
    assert e.ordinal_alpha([1, 3, 5, 4], [1, 3, 5, None], range(1, 6)) == 1
    assert e.ordinal_alpha([1, 2, 3], [3, 2, 1], range(1, 6)) < 0


def test_kendall_w_tie_correction():
    assert e.kendalls_w([[1, 2, 2, 5], [1, 2, 2, 5]]) == pytest.approx(1)
    assert e.kendalls_w([[1, 2, 3], [3, 2, 1]]) == pytest.approx(0)
    assert e.kendalls_w([[1, 1, 1], [1, 1, 1]]) is None
    # Published example, represented as anonymous item orders only.
    assert e.kendalls_w([[5, 1, 4, 2, 3], [5, 3, 1, 5, 3]]) == pytest.approx(20/38)


def test_kendall_tau_direction_and_missing_pairs():
    assert e.tau([-1, -2, -3, None], [3, 2, 1, 0]) == dict(n=3, tau_b=1.)
    assert e.tau([1, None], [2, 3]) == dict(n=1, tau_b=None)
    assert e.tau([1, 1], [2, 3])["tau_b"] is None


def test_percentile_includes_itself_and_averages_ties():
    assert e.mid_percentile(4, [1, 2, 3, 4, 5]) == 70
    assert e.mid_percentile(2, [1, 2, 2, 3]) == 50
    assert e.mid_percentile(1, [1]) == 50
    with pytest.raises(ValueError, match="contain"):
        e.mid_percentile(2, [1, 3])


def test_degeneracy_ignores_root_mandatory_and_empty_group_leaves(tmp_path):
    path = tmp_path / "model.xml"
    path.write_text('<featureModel><struct><and name="R" mandatory="true">'
                    '<and name="G"><feature name="C"/></and><alt name="Empty"/>'
                    '</and></struct></featureModel>')
    result = e.xml_shape(path)
    assert result["degeneracy"] is True
    assert result["size"] == 4
    assert result["internal_features"] == 2
    assert result["empty_groups_ignored"] == 1
    assert result["mandatory_children"] == 0


@pytest.mark.parametrize("child", [
    '<or name="G"><feature name="C"/></or>',
    '<alt name="G"><feature name="C"/></alt>',
    '<feature name="C" mandatory="true"/>',
])
def test_nondegenerate_alternatives(tmp_path, child):
    path = tmp_path / "model.xml"
    path.write_text('<featureModel><struct><and name="R">' + child + '</and></struct></featureModel>')
    assert not e.xml_shape(path)["degeneracy"]


def test_sanity_gate_uses_published_rounding_and_raises_immediately():
    checks = e.Checks()
    checks.equal("rounded", .527046, .53, 2)
    checks.equal("count", 10, 10)
    with pytest.raises(e.SanityError, match="observed"):
        checks.equal("wrong", .521, .53, 2)
    assert checks.rows[-1]["passed"] is False


def test_input_hashes_use_aliases_and_detect_mutation(tmp_path):
    path = tmp_path / "private-personal-filename.csv"
    path.write_text("a,b\n1,2\n")
    inputs = e.Inputs()
    assert inputs.csv("Expert A source", path) == [{"a": "1", "b": "2"}]
    assert "private-personal" not in str(inputs.rows())
    path.write_text("a,b\n2,3\n")
    with pytest.raises(ValueError, match="Input changed"):
        inputs.verify_unchanged()


def test_frozen_xml_resolves_to_current_checkout(tmp_path):
    path = e.selected_xml(tmp_path, dict(xml_path="/historical/checkouts/feature-model-automation/results/r/fm_gen.xml"))
    assert path == tmp_path / "results/r/fm_gen.xml"
    with pytest.raises(ValueError, match="inside results"):
        e.selected_xml(tmp_path, dict(xml_path="/outside/fm_gen.xml"))


def test_failure_writes_only_checks_not_analysis(tmp_path, monkeypatch):
    def fail(root, inputs, checks):
        checks.equal("published mean", 0.1, 0.9, 3)
    monkeypatch.setattr(e, "analyse", fail)
    assert e.main(["--root", str(tmp_path)]) == 1
    out = tmp_path / "results/expert-study/analysis-final"
    assert {p.name for p in out.iterdir()} == {"summary.md", "sanity_checks.csv"}
    assert "STOPPED" in (out / "summary.md").read_text()
    with pytest.raises(SystemExit):
        e.main(["--root", str(tmp_path)])


def test_csv_missing_is_blank_not_zero(tmp_path):
    path = tmp_path / "ratings.csv"
    e.write_csv(path, [dict(rater="Expert B", Q5=None, decision=0)])
    with path.open() as stream:
        assert list(csv.DictReader(stream)) == [dict(rater="Expert B", Q5="", decision="0")]


def test_duplicate_join_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        e.unique([dict(run_id="r"), dict(run_id="r")], "run_id", "scores")


def test_public_csv_inputs_match_internal_types_and_missingness(tmp_path):
    selection = {"X": {"corpus": "repair"}}
    key = {"C01": dict(code="X", corpus="repair", stratum="agree")}
    rows = [dict(rater=rater, code="X", corpus="repair", Q1=3,Q2=4,Q3=2,Q4=1,
                 Q5=None, decision="Accept",decision_score=2,rank=1) for rater in e.RATERS]
    e.write_csv(tmp_path/"ratings.csv", rows)
    citations = [dict(item_id="C01", **key["C01"], **{
        "Expert A label":"Yes", "Expert A score":2,
        "Expert B label":"Cannot judge", "Expert B score":None})]
    e.write_csv(tmp_path/"citation_judgements.csv", citations)
    forms, result = e.read_public_responses(tmp_path,selection,key,e.Inputs())
    assert forms["Expert B"]["X"]["Q5"] is None
    assert forms["Expert A"]["X"]["decision_score"] == 2
    assert result[0]["Expert B score"] is None
    citations[0]["Expert A score"] = 0
    (tmp_path/"citation_judgements.csv").unlink()
    e.write_csv(tmp_path/"citation_judgements.csv", citations)
    with pytest.raises(ValueError, match="coding mismatch"):
        e.read_public_responses(tmp_path,selection,key,e.Inputs())
