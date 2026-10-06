from pathlib import Path

import pytest

from scripts import evaluate_provenance as driver


def test_actual_manifests():
    repair = driver._load_known_doc_ids('repair')
    federation = driver._load_known_doc_ids('federation')
    assert len(repair) == len(set(repair)) == 54
    assert len(federation) == len(set(federation)) == 23
    assert 'rep_01' in repair and 'fed_01' in federation
    assert not set(repair).intersection(federation)


def test_bom_and_bibtex_keys(tmp_path, monkeypatch):
    path = tmp_path / 'manifest.csv'
    path.write_text('doc_id;bibtex_key\nrep_01;9\nrep_02;Author2020\n', encoding='utf-8-sig')
    monkeypatch.setitem(driver.MANIFESTS, 'repair', path)
    assert driver._load_known_doc_ids('repair') == ['rep_01', 'rep_02']


@pytest.mark.parametrize('content,reason', [
    ('filename;title\na.pdf;A\n', 'required doc_id'),
    ('doc_id,title\nrep_01,A\n', 'required doc_id'),
    ('doc_id;title\n;A\n', 'blank doc_id'),
    ('doc_id;title\nrep_01;A\nrep_01;B\n', 'duplicate doc_id'),
    ('doc_id;title\n', 'no document IDs'),
])
def test_invalid_manifest_fails(tmp_path, monkeypatch, content, reason):
    path = tmp_path / 'manifest.csv'
    path.write_text(content)
    monkeypatch.setitem(driver.MANIFESTS, 'repair', path)
    with pytest.raises(ValueError, match=reason):
        driver._load_known_doc_ids('repair')


def test_missing_manifest_fails(tmp_path, monkeypatch):
    monkeypatch.setitem(driver.MANIFESTS, 'repair', tmp_path / 'absent.csv')
    with pytest.raises(FileNotFoundError):
        driver._load_known_doc_ids('repair')
