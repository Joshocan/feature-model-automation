"""Evaluation-contract checks; no model imports, downloads or generation calls."""
import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "config/evaluation/ifs-2027-v0.1.0.json"


def test_specification_documents_and_decisions_are_linked():
    spec = json.loads(CONTRACT.read_text())
    assert spec["specification_version"] == "0.1.0"
    for name in spec["documents"]:
        assert (REPO / name).is_file()
        assert "0.1.0" in (REPO / name).read_text()
    decisions = (REPO / "docs/evaluation-decisions.md").read_text()
    for decision in spec["unresolved_decisions"]:
        assert f"### {decision} " in decisions


def test_draft_does_not_claim_freeze_or_authorize_calls():
    spec = json.loads(CONTRACT.read_text())
    assert spec["status"] == "draft"
    assert spec["preregistered"] is False
    assert spec["unresolved_decisions"]
    assert spec["execution"]["publication_freeze_allowed"] is False
    assert spec["execution"]["generation_calls_authorized"] is False
    assert spec["execution"]["current_drivers_enforce_contract"] is False
    assert spec["semantic"]["primary_score_population"] == "completed_extractable_final"
    assert spec["semantic"]["sensitivity_score_population"] == "strict_admissible_final"
    assert spec["statistics"]["allow_equivalence_claim"] is False
    assert spec["statistics"]["allow_numeric_saturation_claim"] is False


def test_inherited_instrument_and_denominator_safeguards():
    spec = json.loads(CONTRACT.read_text())
    pop, sem = spec["population"], spec["semantic"]
    assert sum(pop["expected_by_lane"].values()) == pop["expected_planned_runs"] == 694
    assert pop["pool_pilots_with_main"] is False
    assert {"corpus", "arm", "ordering_id", "k_doc", "metamodel_block"} <= set(pop["group_keys"])
    assert sem["tau_primary"] == 0.4
    assert sem["tau_sweep"] == [0.3, 0.4, 0.5, 0.6]
    assert sem["threshold_operator"] == ">="
    assert sem["preserve_duplicate_occurrences"] is True
    assert spec["failure_policy"]["substitute_last_valid_checkpoint"] is False
    assert spec["failure_policy"]["missing_metric_value"] is None
    assert spec["structure"]["root_depth"] == 0


def test_source_identifiers_are_portable_and_hashes_well_formed():
    spec = json.loads(CONTRACT.read_text())
    # No dependency on large/local archives in the unit suite. Artifact existence
    # and byte hashes are checked separately when the release data is available.
    for name, digest in spec["source_sha256"].items():
        assert not Path(name).is_absolute()
        assert ".." not in Path(name).parts
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert re.fullmatch(r"[0-9a-f]{40}", spec["semantic"]["recorded_pilot_revision"])
