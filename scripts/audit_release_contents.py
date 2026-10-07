"""Read-only release-content audit. Findings require human review, not automatic clearance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SECRET = re.compile(rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a new report file')
    paths = sorted(set(git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().strip('\0').split('\0')))
    files, hits, personal, identities, workbooks = [], [], [], [], []
    for name in paths:
        path = ROOT / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        files.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        if SECRET.search(data):
            hits.append(name)
        if b'/Users/' in data:
            personal.append(name)
        if name.startswith(('docs/', 'data/')) and (b'Rep-96' in data or b'Fed-44' in data):
            identities.append(name)
        if path.suffix == '.xlsx':
            with zipfile.ZipFile(path) as z:
                workbooks.append(dict(path=name, comment_parts=[n for n in z.namelist() if 'comment' in n.lower()],
                                      properties_present='docProps/core.xml' in z.namelist()))
    history_hits, scanned = [], 0
    for line in git('rev-list', '--objects', '--all').decode().splitlines():
        oid, _, name = line.partition(' ')
        if not name:
            continue
        kind = git('cat-file', '-t', oid).strip()
        if kind != b'blob':
            continue
        data = git('cat-file', 'blob', oid)
        scanned += 1
        if SECRET.search(data):
            history_hits.append(dict(object=oid, path=name))
    report = dict(status='technical screening only; not privacy/rights approval',
        source_commit=git('rev-parse', 'HEAD').decode().strip(), proposed_files=files,
        credential_pattern_hits=hits, historical_blobs_scanned=scanned,
        historical_credential_pattern_hits=history_hits, personal_path_candidates=personal,
        blinded_code_document_candidates=identities, workbooks=workbooks,
        hosted_ci='workflow is untracked locally; current candidate cannot be validated on GitHub until committed and pushed',
        manual_gates=['Third-party paper excerpts and reference-model rights',
                      'Workbook metadata and embedded comments',
                      'Model identity exposure while expert assessment remains open',
                      'Final selection of uncommitted files for release'])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'proposed_files'}, indent=2))


if __name__ == '__main__':
    main()
