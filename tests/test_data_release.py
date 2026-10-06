import hashlib
import json
from pathlib import Path
import tarfile

import pytest

from scripts.prepare_data_release import bundles, normal_source, release_files


def test_source_resolution():
    assert normal_source('/Users/author/dev/feature-model-automation/results/a.json') == 'results/a.json'
    assert normal_source('results/a.json') == 'results/a.json'


def test_excludes_private_and_backups(tmp_path):
    for n in ['experiments/a.xml', 'expert-study/key.json', 'protocol/release-preparation-backup/key',
              '.git/config', 'bundles/old.tar.gz', 'analysis/.DS_Store']:
        p = tmp_path/n; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('x')
    assert [p.relative_to(tmp_path).as_posix() for p in release_files(tmp_path)] == ['experiments/a.xml']


def setup_candidate(root):
    for name, text in {'README.md': 'Dataset', 'artifact-manifest.json': '{}',
                       'release-validation.json': json.dumps({'failures': []}),
                       'experiments/run.xml': '<model/>', 'analysis/table.csv': 'x\n1\n'}.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
    files=release_files(root)
    (root/'checksums.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}\n' for p in files))


def test_bundle_roundtrip_and_no_overwrite(tmp_path):
    setup_candidate(tmp_path)
    bundles(tmp_path)
    folder=tmp_path/'bundles/nonexpert-candidate-2026-10-06'
    manifest=json.loads((folder/'bundle-manifest.json').read_text())
    assert len(manifest['bundles']) == 3
    for record in manifest['bundles']:
        with tarfile.open(folder/record['file']) as tar:
            for member in tar:
                assert member.mtime == 0
                assert tar.extractfile(member).read() == (tmp_path/member.name).read_bytes()
    with pytest.raises(FileExistsError): bundles(tmp_path)


def test_stale_checksum_refused(tmp_path):
    setup_candidate(tmp_path)
    (tmp_path/'README.md').write_text('Changed')
    with pytest.raises(ValueError, match='Stale checksum'):bundles(tmp_path)
