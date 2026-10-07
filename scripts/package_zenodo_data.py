"""Package a verified non-expert dataset as a ZIP plus README and SHA256SUMS.

No source edits, publication, or inferred approvals. Existing output is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from prepare_data_release import digest, release_files


def package(root, output):
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if output.is_relative_to(root):
        raise ValueError('Output must be outside the source archive')
    expected = {}
    for line in (root / 'checksums.sha256').read_text().splitlines():
        h, name = line.split('  ', 1)
        rel = PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts or name in expected:
            raise ValueError('Unsafe or duplicate checksum entry')
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('Symlink or escaping path: ' + name)
        if digest(path) != h:
            raise ValueError('Stale checksum: ' + name)
        expected[name] = h
    expected['checksums.sha256'] = digest(root / 'checksums.sha256')
    actual = {p.relative_to(root).as_posix() for p in release_files(root)}
    if set(expected) != actual:
        raise ValueError('Release file set differs from checksum catalogue')
    manifest = json.loads((root / 'artifact-manifest.json').read_text())
    for entry in manifest['files']:
        if expected.get(entry['path']) != entry['sha256']:
            raise ValueError('Original evidence changed: ' + entry['path'])
    validation = json.loads((root / 'release-validation.json').read_text())
    if validation.get('failures'):
        raise ValueError('Archive reports technical validation failures')
    print(f'Verified {len(expected)} files; building ZIP.', flush=True)
    output.mkdir(parents=True)
    archive = output / 'feature-model-research-data-2026-10-07-candidate.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        for name in sorted(expected):
            z.write(root / name, 'feature-model-research-data/' + name)
    with zipfile.ZipFile(archive) as z:
        seen = set()
        for member in z.infolist():
            name = member.filename.removeprefix('feature-model-research-data/')
            h = hashlib.sha256()
            with z.open(member) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            if name not in expected or h.hexdigest() != expected[name]:
                raise ValueError('ZIP member verification failed: ' + member.filename)
            seen.add(name)
        if seen != set(expected):
            raise ValueError('ZIP member set differs')
    readme = output / 'README.md'
    with readme.open('x') as f:
        f.write('# Conformance Is Not Correctness: Evaluating LLM-Constructed Feature Models — Research Data\n\n')
        f.write('Local upload candidate prepared 2026-10-07. Not yet publication-approved.\n\n')
        f.write('## Download verification and extraction\n\n')
        f.write('Keep these three files together. Verify with `shasum -a 256 -c SHA256SUMS`,\n')
        f.write('then extract the ZIP into a new directory. Inside `feature-model-research-data/`,\n')
        f.write('run `shasum -a 256 -c checksums.sha256` to verify the extracted contents.\n\n')
        f.write('The ZIP excludes Git history, preparation backups, old bundles and the\n')
        f.write('expert-study directory. It includes automated Astra expert-extension outputs,\n')
        f.write('not expert responses or analysis. No release version or DOI has been assigned\n')
        f.write('by this packaging operation. Review RELEASE_APPROVAL.md before publishing.\n\n')
        f.write('The README below is also preserved inside the ZIP.\n\n---\n\n')
        f.write((root / 'README.md').read_text())
    with (output / 'SHA256SUMS').open('x') as f:
        for path in (archive, readme):
            f.write(f'{digest(path)}  {path.name}\n')
    print(json.dumps({'output': str(output), 'zip_bytes': archive.stat().st_size,
                      'zip_members_verified': len(expected),
                      'publication_ready': validation.get('publication_ready', False)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    package(a.repository, a.output)
