"""Install reviewed release templates without modifying original research evidence."""
import argparse
import shutil
from pathlib import Path

from prepare_data_release import ROOT, digest, read, write, refresh_metadata


def update(root):
    manifest = read(root / 'artifact-manifest.json')
    originals = {r['path']: r['sha256'] for r in manifest['files']}
    for name, expected in originals.items():
        if digest(root / name) != expected:
            raise ValueError('Original evidence changed: ' + name)
    templates = ROOT / 'docs/data-release'
    sources = sorted(p for p in templates.rglob('*') if p.is_file())
    for p in sources:
        if p.relative_to(templates).as_posix() in originals:
            raise ValueError('Template would overwrite original evidence')
    backup = root / 'protocol/release-preparation-backup/licensing-2026-10-06'
    backup.mkdir(parents=True, exist_ok=False)
    targets = [p.relative_to(templates) for p in sources] + [Path(n) for n in (
        'artifact-manifest.json', 'checksums.sha256', 'release-validation.json',
        'CITATION.cff.template', 'CHANGELOG.md')]
    for name in targets:
        if (root / name).is_file():
            (backup / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, backup / name)
    for p in sources:
        write(root, p.relative_to(templates).as_posix(), p.read_text())
    draft = root / 'CITATION.cff.template'
    if draft.exists():
        draft.unlink()  # Exact prior draft is preserved in the backup above.
    report = read(root / 'release-validation.json')
    report['authorship_and_licence_choice'] = {
        'status': 'user supplied 2026-10-06', 'license': 'CC-BY-4.0',
        'authors': ['Joshua Tetteh Ocansey', 'Yngve Lamo', 'Adrian Rutle', 'Fazle Rabbi'],
        'scope': 'original contributions to the extent licensors hold rights'}
    report['remaining_gates'] = [
        'Third-party rights and contributor authority review',
        'Privacy and expert-blinding release timing approval',
        'Historical prompt limitation acceptance or recovery',
        'Final manuscript table/claim reconciliation',
        'Frozen software and clean-environment reproduction test',
        'Author approval of remaining methodological scope',
        'Final metadata, refreshed bundles and publication approval']
    report['publication_ready'] = False
    write(root, 'release-validation.json', report)
    manifest['license_notice'] = 'LICENSE.md'
    manifest['license_scope'] = report['authorship_and_licence_choice']['scope']
    manifest['release_documents_updater_sha256'] = digest(Path(__file__))
    write(root, 'artifact-manifest.json', manifest)
    changelog = root / 'CHANGELOG.md'
    write(root, 'CHANGELOG.md', changelog.read_text() +
          '\n## Licensing and release decisions — 2026-10-06\n\n'
          '- Recorded supplied authors and CC BY 4.0 scope; replaced citation draft.\n'
          '- Added rights, privacy/blinding, prompt limitation, manuscript and reproduction checklists.\n'
          '- Original evidence unchanged; publication approval remains pending.\n'
          '- Previous bundles predate this update and must be regenerated before deposit.\n')
    refresh_metadata(root)
    for name, expected in originals.items():
        assert digest(root / name) == expected, name
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert digest(root / name) == expected, name
    print(f'Updated documents; verified {len(originals)} original files and all release checksums.')
    print('Prior documents backed up; prior bundles preserved but superseded. No publication performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    update(parser.parse_args().repository.resolve())
