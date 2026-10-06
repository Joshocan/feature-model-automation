"""No-network safety tests for the parallel campaign runner."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.campaign import (
    RunClaim,
    RunClaimed,
    _filter_matrix_rows,
    _run_config_from_row,
)
from scripts.build_run_matrix import build_matrix


def test_runner_annotations_are_not_passed_to_run_config() -> None:
    matrix = build_matrix("ifs-2027", only_enabled=True, lane="astra")
    row = dict(matrix["runs"][0])
    row["_completed"] = False
    cfg = _run_config_from_row(row)
    assert cfg.model_id == "gpt-6-astra"


def test_lane_filter_is_disjoint() -> None:
    combined = build_matrix("ifs-2027", only_enabled=True, lane="all")
    open_matrix = dict(combined)
    open_matrix["runs"] = list(combined["runs"])
    astra_matrix = dict(combined)
    astra_matrix["runs"] = list(combined["runs"])
    _filter_matrix_rows(open_matrix, lane="open_weight", model_keys=[])
    _filter_matrix_rows(astra_matrix, lane="astra", model_keys=[])
    open_ids = {_run_config_from_row(r).run_id() for r in open_matrix["runs"]}
    astra_ids = {_run_config_from_row(r).run_id() for r in astra_matrix["runs"]}
    assert not (open_ids & astra_ids)
    assert len(open_ids) == 644
    assert len(astra_ids) == 50


def test_run_claim_rejects_concurrent_owner(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    with RunClaim(path):
        with pytest.raises(RunClaimed):
            with RunClaim(path):
                pass
