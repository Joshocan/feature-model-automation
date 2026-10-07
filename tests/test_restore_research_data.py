import hashlib
import json
import pytest
from scripts.restore_research_data import restore, safe_path


def fixture(root):
    root.mkdir()
    (root / 'evidence.txt').write_text('original')
    digest = hashlib.sha256(b'original').hexdigest()
    (root / 'artifact-manifest.json').write_text(json.dumps({'files': [
        {'path': 'evidence.txt', 'source': 'results/test/evidence.txt', 'sha256': digest}]}))
    from scripts.restore_research_data import sha
    (root / 'checksums.sha256').write_text(''.join(
        f'{sha(root / name)}  {name}\n' for name in ['evidence.txt', 'artifact-manifest.json']))


def test_restore_dry_run_copy_and_no_overwrite(tmp_path):
    archive, target = tmp_path / 'archive', tmp_path / 'checkout'
    fixture(archive)
    assert restore(archive, target)['absent_files'] == 1
    assert not target.exists()
    assert restore(archive, target, True)['applied']
    assert restore(archive, target)['absent_files'] == 0
    (target / 'results/test/evidence.txt').write_text('changed')
    with pytest.raises(ValueError, match='overwrite'):
        restore(archive, target, True)


def test_tampering_and_unsafe_paths(tmp_path):
    archive = tmp_path / 'archive'
    fixture(archive)
    (archive / 'evidence.txt').write_text('tampered')
    with pytest.raises(ValueError, match='Checksum'):
        restore(archive, tmp_path / 'dest')
    for name in ['../outside', '/etc/passwd', 'a/../../escape']:
        with pytest.raises(ValueError):
            safe_path(tmp_path, name)
