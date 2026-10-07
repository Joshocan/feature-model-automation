"""Small, data-independent controls for release comparison logic."""
from scripts import validate_reproduction as validation


def test_numeric_equivalence_preserves_missingness():
    assert validation.equivalent("0.5", "0.50000001")
    assert validation.equivalent("", "")
    assert not validation.equivalent("", "0")
    assert not validation.equivalent("0.5", "0.51")
    assert not validation.equivalent("accepted", "rejected")


def test_csv_comparison_reports_differences(tmp_path, monkeypatch):
    monkeypatch.setattr(validation, "ROOT", tmp_path)
    old, new = tmp_path / "old.csv", tmp_path / "new.csv"
    old.write_text("run_id,value\na,0.5\nb,\n")
    new.write_text("run_id,value\na,0.50000001\nb,\n")
    assert validation.compare_csv(old, new)["status"] == "equivalent_with_numeric_tolerance"
    new.write_text("run_id,value\na,0.5\nb,0\n")
    result = validation.compare_csv(old, new)
    assert result["status"] == "differences"
    assert result["different_cells"] == 1
    assert result["examples"][0]["column"] == "value"
    new.write_text("run_id,other\na,0.5\n")
    assert validation.compare_csv(old, new)["status"] == "columns_differ"
