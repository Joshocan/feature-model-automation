"""Verify a data archive and restore original software-relative paths, without overwrites.

Use a NEW checkout as destination. No generation calls, archive extraction or deletion.
Default is verification/dry-run; --apply copies only absent files after all checks pass.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    rel = PurePosixPath(name)
    if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name:
        raise ValueError(f'Unsafe manifest path: {name}')
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Path escapes root: {name}')
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        # /tmp may itself be a platform symlink; callers resolve the root first.
        raise ValueError(f'Symlink path not accepted: {name}')
    return path


def restore(archive, destination, apply=False, expected_manifest=None):
    archive, destination = archive.resolve(), destination.resolve()
    manifest_path = archive / 'artifact-manifest.json'
    if expected_manifest and sha(manifest_path) != expected_manifest:
        raise ValueError('Manifest does not match the trusted expected SHA256')
    checks = {}
    for line in (archive / 'checksums.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if name in checks:
            raise ValueError('Duplicate checksum path: ' + name)
        checks[name] = digest
        if sha(safe_path(archive, name)) != digest:
            raise ValueError('Checksum mismatch: ' + name)
    if checks.get('artifact-manifest.json') != sha(manifest_path):
        raise ValueError('Manifest missing from checksum catalogue')
    manifest = json.loads(manifest_path.read_text())
    plan, targets = [], set()
    for record in manifest['files']:
        source = safe_path(archive, record['path'])
        if sha(source) != record['sha256']:
            raise ValueError('Original artifact checksum mismatch: ' + record['path'])
        # Software, prompts and configuration come from the compatible software
        # release, not historical copies embedded in the data archive.
        if PurePosixPath(record['source']).parts[0] not in ('data', 'results'):
            continue
        target = safe_path(destination, record['source'])
        if target in targets:
            raise ValueError('Duplicate restore target: ' + record['source'])
        targets.add(target)
        if target.exists():
            if not target.is_file() or sha(target) != record['sha256']:
                raise ValueError('Refusing to overwrite differing file: ' + str(target))
        else:
            plan.append((source, target, record['sha256']))
    if apply:
        for source, target, digest in plan:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out)
            if sha(target) != digest:
                raise ValueError('Copy verification failed: ' + str(target))
    return dict(verified_originals=len(manifest['files']), restorable_data_files=len(targets), absent_files=len(plan), applied=apply,
                manifest_sha256=sha(manifest_path))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--manifest-sha256', help='Trusted hash from release metadata')
    p.add_argument('--apply', action='store_true')
    a = p.parse_args()
    print(json.dumps(restore(a.archive, a.destination, a.apply, a.manifest_sha256), indent=2))
