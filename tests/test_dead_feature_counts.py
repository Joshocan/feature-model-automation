import csv
import json
import sys

import pytest

from scripts import export_dead_feature_counts as script


def row(rid='a', value='[]', status='ok'):
    return {'run_id': rid, script.SOURCE: value, script.SOURCE+'__status': status,
            script.SOURCE+'__reason': 'original reason'}


def test_zero_positive_and_unavailable_are_distinct():
    source = [row(), row('b', '["A", "B"]'), row('c', '', 'not_applicable'),
              row('d', '', 'ineligible')]
    output = script.derive_counts(source)
    assert [r[script.COUNT] for r in output] == [0, 2, None, None]
    assert output[2][script.COUNT+'__reason'] == 'original reason'
    assert script.COUNT not in source[0]


@pytest.mark.parametrize('value', ['', 'null', '3', '{}', '["A", "A"]', '[null]', '[""]'])
def test_bad_ok_payload_fails(value):
    with pytest.raises(ValueError): script.derive_counts([row(value=value)])


def test_duplicates_and_missing_status_fail():
    with pytest.raises(ValueError): script.derive_counts([row(), row()])
    with pytest.raises(ValueError): script.derive_counts([{'run_id': 'x'}])


def test_cli_roundtrip_preserves_unavailable_and_input(tmp_path, monkeypatch):
    source = tmp_path/'input.csv'
    rows=[row(), row('b','', 'ineligible')]
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    original=source.read_bytes()
    output=tmp_path/'results/ifs-2027/analysis/counts'
    monkeypatch.setattr(script,'REPO',tmp_path)
    monkeypatch.setattr(sys,'argv',['export','--wide',str(source),'--output',str(output)])
    assert script.main()==0
    with (output/'wide.csv').open() as f: exported=list(csv.DictReader(f))
    assert [r[script.COUNT] for r in exported]==['0','']
    assert source.read_bytes()==original
    assert json.loads((output/'summary.json').read_text())['n_unavailable']==1
    with pytest.raises(SystemExit): script.main()
