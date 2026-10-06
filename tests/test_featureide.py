import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import evaluate_featureide as script


def test_inventory_selects_only_completed_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'runs.csv'
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run_id', 'completed', 'resolved_path'])
        w.writeheader()
        w.writerows([dict(run_id='a', completed='True', resolved_path='some/path'),
                     dict(run_id='b', completed='False', resolved_path='other')])
    all_rows, rows = script.completed_runs(path)
    assert len(all_rows) == 2
    assert [r['run_id'] for r in rows] == ['a']
    with path.open('a') as f:
        f.write('a,True,some/path\n')
    with pytest.raises(ValueError, match='duplicate'):
        script.completed_runs(path)


@pytest.mark.parametrize('mode', ['rejected', 'runtime', 'timeout', 'bad_payload'])
def test_rejection_is_not_runtime_failure(tmp_path, monkeypatch, mode):
    path = tmp_path / 'model.xml'
    path.write_text('<featureModel/>')
    def run(*args, **kwargs):
        if mode == 'timeout':
            raise subprocess.TimeoutExpired('java', 1)
        return subprocess.CompletedProcess([], 1 if mode == 'runtime' else 0,
            'FEATUREIDE_RESULT ' + json.dumps({'parsed_ok': 'false' if mode == 'bad_payload' else False}), 'diagnostic')
    monkeypatch.setattr(script.subprocess, 'run', run)
    r = script.parse_model(path, 'unused')
    assert r['status'] == ('ok' if mode == 'rejected' else 'evaluator_error')
    assert r['parsed_ok'] is (False if mode == 'rejected' else None)


def test_missing_file_is_not_rejection(tmp_path):
    r = script.parse_model(tmp_path / 'absent.xml', 'unused')
    assert r['status'] == 'missing_artifact'
    assert r['parsed_ok'] is None


def test_real_pinned_library_controls(tmp_path):
    jar = script.REPO / 'tools/featureide/lib/de.ovgu.featureide.lib.fm-v3.10.0.jar'
    if not jar.is_file() or not shutil.which('javac'):
        pytest.skip('Download pinned jar and install JDK for integration test')
    assert script.digest(jar) == script.JAR_SHA256
    subprocess.run(['javac', '-cp', str(jar), '-d', str(tmp_path), str(script.JAVA_SOURCE)], check=True)
    cp = os.pathsep.join([str(tmp_path), str(jar)])
    for corpus, count in [('repair', 100), ('federation', 130)]:
        r = script.parse_model(script.REPO / 'data/ground_truth' / (corpus + '.xml'), cp)
        assert r['status'] == 'ok' and r['parsed_ok'] is True
        assert r['loaded_features'] == count
        assert not r['stderr']
    for text in ['<featureModel>', '<featureModel><struct><and name="R"><feature name="A"/><feature name="A"/></and></struct></featureModel>']:
        path = tmp_path / 'bad.xml'
        path.write_text(text)
        r = script.parse_model(path, cp)
        assert r['status'] == 'ok' and r['parsed_ok'] is False
        assert r['errors'] > 0
