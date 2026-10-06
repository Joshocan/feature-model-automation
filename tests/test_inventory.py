import json

import pytest

from fame.evaluation.inventory import build_inventory, run_config, sha256


def fixture(tmp_path):
    cfg = dict(campaign_id="test", corpus="repair", ordering_id="primary", N=1,
        grounding="rag", model_id="test", provider="fake", seed=0, repetition=0,
        metamodel_block=True, k_doc=5, domain="repair", root_feature="Repair",
        prompt_template_hash="p", metamodel_hash="m", chunks_hash="c",
        encoder_digest="e", max_output_tokens=100, extra={"arm": "guided_baseline"})
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(dict(lane="open_weight", runs=[cfg])))
    contract = dict(campaign_id="test", source_sha256={"matrix.json": sha256(matrix)},
        population=dict(matrices=["matrix.json"], expected_planned_runs=1,
            expected_by_lane={"open_weight": 1}))
    canonical = run_config(cfg)
    rid = canonical.run_id()
    root = tmp_path / f"results/test/repair/{rid}/{rid}"
    return contract, canonical, root


def complete(root, cfg):
    (root / "fm_iter").mkdir(parents=True)
    (root / "fm_iter/step_00.xml").write_text("<featureModel/>")
    (root / "fm_gen.xml").write_text("<featureModel/>")
    (root / "run_meta.json").write_text(json.dumps(dict(run_id=cfg.run_id(),
        config=cfg.canonical_dict(), completed=True, completed_steps=1, planned_steps=1,
        terminal_status="completed", steps=[dict(step_index=0, carry_forward=True)])))


def test_missing_run_is_retained(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    result = build_inventory(tmp_path, spec)
    assert len(result["runs"]) == 1
    assert result["runs"][0]["inventory_status"] == "not_started"
    assert result["runs"][0]["recorded_completed"] is None


def test_completion_requires_all_checkpoints_and_identity(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    complete(root, cfg)
    assert build_inventory(tmp_path, spec)["runs"][0]["completed"] is True
    (root / "fm_iter/step_00.xml").unlink()
    result = build_inventory(tmp_path, spec)
    assert result["runs"][0]["inventory_status"] == "completion_inconsistent"
    assert result["runs"][0]["completed"] is False


def test_corrupt_metadata_not_discarded(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    root.mkdir(parents=True)
    (root / "run_meta.json").write_text("{broken")
    result = build_inventory(tmp_path, spec)
    assert result["runs"][0]["inventory_status"] == "invalid_metadata"
    assert result["issues"][0]["kind"] == "invalid_metadata"


def test_recovery_snapshot_not_another_run_and_files_unchanged(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    complete(root, cfg)
    archive = tmp_path / f"results/recovery_archive/test/repair/{cfg.run_id()}/backup"
    archive.mkdir(parents=True)
    (archive / "run_meta.json").write_text(json.dumps({"terminal_status": "provider_error"}))
    before = {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = build_inventory(tmp_path, spec)
    assert len(result["runs"]) == len(result["recovery_snapshots"]) == 1
    assert result["summary"]["planned"] == 1
    assert before == {str(p): sha256(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert all(len(a["sha256"]) == 64 for a in result["artifacts"])


def test_duplicate_matrix_rows_rejected(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    matrix = tmp_path / "matrix.json"
    data = json.loads(matrix.read_text())
    data["runs"] *= 2
    matrix.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Duplicate planned"):
        build_inventory(tmp_path, spec)


def test_hash_mismatch_and_relocation_are_explicit(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    relocated = tmp_path / "relocated"
    complete(relocated, cfg)
    spec["source_sha256"]["matrix.json"] = "0" * 64
    mapping = {root.relative_to(tmp_path).as_posix(): "relocated"}
    result = build_inventory(tmp_path, spec, mapping)
    assert result["runs"][0]["completed"] is True
    assert result["runs"][0]["resolved_path"] == "relocated"
    assert result["issues"][0]["kind"] == "source_hash_mismatch"


def test_pilot_separate_from_main(tmp_path):
    spec, cfg, root = fixture(tmp_path)
    pilot_root = tmp_path / "pilots/repair/hash/run"
    complete(pilot_root, cfg)
    (tmp_path / "pilots.json").write_text(json.dumps({"sources": [
        {"path": "pilots", "cohort": "n5_rag", "primary": True}]}))
    spec["population"]["pilot_manifest"] = "pilots.json"
    result = build_inventory(tmp_path, spec)
    assert len(result["pilots"]) == len(result["runs"]) == 1
    assert result["runs"][0]["inventory_status"] == "not_started"
